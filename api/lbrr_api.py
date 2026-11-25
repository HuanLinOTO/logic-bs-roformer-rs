#!/usr/bin/env python3
"""HTTP wrapper around the lbrr command-line separator.

The service intentionally shells out to `lbrr`: model loading, CUDA stream
lifetime, and inference stay inside the implementation that already owns them,
while this module handles upload validation, FLAC packaging, concurrency, and
cleanup.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hmac
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
import struct
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

API_VERSION = "0.1.0"
JOB_PREFIX = "lbrr-api-job-"
LOGGER = logging.getLogger("lbrr_api")


@dataclass(frozen=True)
class Settings:
    lbrr_bin: str
    model_dir: Path
    ffmpeg: str
    ffprobe: str
    temp_root: Path | None
    max_duration_seconds: float
    max_upload_bytes: int
    job_timeout_seconds: float
    concurrency: int
    encode_workers: int
    host: str
    port: int
    api_token: str | None
    log_level: str
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    allow_insecure_http: bool = False
    allow_unauthenticated: bool = False

    @property
    def assets_ready(self) -> bool:
        return (
            Path(self.model_dir).joinpath("model.safetensors").is_file()
            and Path(self.model_dir).joinpath("logic_bs_roformer.yaml").is_file()
        )


@dataclass(frozen=True)
class JobResult:
    zip_path: Path
    output_files: tuple[str, ...]
    gain_db: float


class JobError(RuntimeError):
    pass


class UploadTooLarge(Exception):
    pass


def _int_env(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float_env(name: str, default: float, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if raw.lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def load_settings() -> Settings:
    default_encode_workers = min(4, os.cpu_count() or 1)
    temp_root_raw = os.environ.get("LBRR_API_TEMP_DIR", "")
    return Settings(
        lbrr_bin=os.environ.get("LBRR_BIN", "lbrr"),
        model_dir=Path(os.environ.get("LBRR_MODEL_DIR", "assets")).expanduser(),
        ffmpeg=os.environ.get("LBRR_FFMPEG", shutil.which("ffmpeg") or "ffmpeg"),
        ffprobe=os.environ.get("LBRR_FFPROBE", shutil.which("ffprobe") or "ffprobe"),
        temp_root=Path(temp_root_raw).expanduser() if temp_root_raw else None,
        max_duration_seconds=_float_env("LBRR_API_MAX_DURATION_SECONDS", 600.0, 1.0),
        max_upload_bytes=_int_env("LBRR_API_MAX_UPLOAD_BYTES", 512 * 1024 * 1024, 1024),
        job_timeout_seconds=_float_env("LBRR_API_JOB_TIMEOUT_SECONDS", 1800.0, 1.0),
        concurrency=_int_env("LBRR_API_CONCURRENCY", 1, 1),
        encode_workers=_int_env("LBRR_API_ENCODE_WORKERS", default_encode_workers, 1),
        host=os.environ.get("LBRR_API_HOST", "127.0.0.1"),
        port=_int_env("LBRR_API_PORT", 10722, 1),
        api_token=os.environ.get("LBRR_API_TOKEN") or None,
        log_level=os.environ.get("LBRR_API_LOG_LEVEL", "info"),
        ssl_certfile=os.environ.get("LBRR_API_SSL_CERTFILE") or None,
        ssl_keyfile=os.environ.get("LBRR_API_SSL_KEYFILE") or None,
        allow_insecure_http=_bool_env("LBRR_API_ALLOW_INSECURE_HTTP"),
        allow_unauthenticated=_bool_env("LBRR_API_ALLOW_UNAUTHENTICATED"),
    )


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower().strip("[]")
    return normalized in {"localhost", "::1"} or normalized.startswith("127.")


def validate_network_security(settings: Settings) -> None:
    if bool(settings.ssl_certfile) != bool(settings.ssl_keyfile):
        raise RuntimeError("LBRR_API_SSL_CERTFILE and LBRR_API_SSL_KEYFILE must be set together")
    if is_loopback_host(settings.host):
        return
    if settings.api_token is None and not settings.allow_unauthenticated:
        raise RuntimeError(
            "refusing to bind publicly without authentication; set LBRR_API_TOKEN "
            "or explicitly set LBRR_API_ALLOW_UNAUTHENTICATED=true"
        )
    if settings.api_token is not None and not settings.ssl_certfile and not settings.allow_insecure_http:
        raise RuntimeError(
            "refusing to expose bearer credentials over plaintext HTTP; configure "
            "LBRR_API_SSL_CERTFILE/LBRR_API_SSL_KEYFILE or explicitly set LBRR_API_ALLOW_INSECURE_HTTP=true"
        )


def _tail(text: str, max_chars: int = 1200) -> str:
    compact = text.strip()
    if len(compact) <= max_chars:
        return compact
    return "..." + compact[-max_chars:]


def _run_command(command: Sequence[str], timeout: float, label: str) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise JobError(f"{label} executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise JobError(f"{label} timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0:
        detail = _tail("\n".join(part for part in (proc.stderr, proc.stdout) if part))
        raise JobError(f"{label} exited {proc.returncode}: {detail or 'no output'}")
    return proc


def probe_audio_duration(ffprobe: str, audio_path: Path, timeout: float = 30.0) -> float | None:
    """Return declared duration when present, or None for playable streams without it."""
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration:stream=codec_type,duration",
        str(audio_path),
    ]
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"could not probe audio duration: {exc}") from exc
    if proc.returncode != 0:
        raise ValueError(f"not a readable audio file: {_tail(proc.stderr, 300)}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe returned invalid metadata") from exc

    candidates: list[float] = []
    format_duration = data.get("format", {}).get("duration")
    if format_duration not in (None, "N/A"):
        try:
            candidates.append(float(format_duration))
        except (TypeError, ValueError):
            pass
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "audio" or stream.get("duration") in (None, "N/A"):
            continue
        try:
            candidates.append(float(stream["duration"]))
        except (TypeError, ValueError):
            pass

    valid = [value for value in candidates if math.isfinite(value) and value > 0]
    if not valid:
        return None
    return max(valid)


def wav_f32_duration(wav_path: Path, expected_rate: int = 44100, expected_channels: int = 2) -> float:
    """Read the exact frame count of the normalized f32 WAV we pass to lbrr."""
    with wav_path.open("rb") as stream:
        riff = stream.read(12)
        if len(riff) != 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError("normalized audio is not a RIFF/WAVE file")
        data_size: int | None = None
        while True:
            header = stream.read(8)
            if not header:
                break
            if len(header) != 8:
                raise ValueError("normalized WAV has a truncated chunk header")
            chunk_id = header[:4]
            chunk_size = struct.unpack("<I", header[4:])[0]
            if chunk_id == b"data":
                data_size = chunk_size
                break
            stream.seek(chunk_size + (chunk_size % 2), 1)
    if data_size is None:
        raise ValueError("normalized WAV has no audio data chunk")
    bytes_per_frame = expected_channels * 4
    if data_size % bytes_per_frame:
        raise ValueError("normalized WAV has a partial audio frame")
    return data_size / bytes_per_frame / expected_rate


def normalize_input_audio(settings: Settings, source_path: Path, normalized_path: Path) -> float:
    """Decode the input exactly as lbrr would and return its real duration.

    The output cap makes an input whose metadata understates its duration cost
    at most max_duration_seconds of decode work; the exact WAV data size is
    then checked before any GPU work starts.
    """
    command = [
        settings.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-t",
        f"{settings.max_duration_seconds:.6f}",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-c:a",
        "pcm_f32le",
        str(normalized_path),
    ]
    try:
        _run_command(command, settings.job_timeout_seconds, "input decode")
    except JobError as exc:
        raise ValueError(str(exc)) from exc
    duration = wav_f32_duration(normalized_path)
    if duration <= 0:
        raise ValueError("audio duration is zero")
    return duration


def measure_peak_db(settings: Settings, audio_path: Path) -> float:
    command = [
        settings.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(audio_path),
        "-map",
        "0:a:0",
        "-af",
        "astats=measure_overall=Peak_level",
        "-f",
        "null",
        "-",
    ]
    proc = _run_command(command, settings.job_timeout_seconds, f"peak analysis {audio_path.name}")
    match = re.search(r"Peak level dB:\s*(-inf|-?\d+(?:\.\d+)?)", proc.stderr + proc.stdout)
    if not match:
        raise JobError(f"could not measure peak for {audio_path.name}")
    if match.group(1) == "-inf":
        return -math.inf
    return float(match.group(1))


def output_gain_db(settings: Settings, wavs: Sequence[Path], workers: int) -> float:
    """Gain shared by every output, preserving stem balance without clipping."""
    if workers == 1:
        peaks = [measure_peak_db(settings, wav) for wav in wavs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            peaks = list(pool.map(lambda wav: measure_peak_db(settings, wav), wavs))
    peak = max(peaks)
    if peak <= 0:
        return 0.0
    # volumedetect prints hundredths of a dB; leave one extra cent as headroom.
    return -(peak + 0.01)


def convert_wav_to_flac(settings: Settings, wav_path: Path, flac_path: Path, gain_db: float = 0.0) -> None:
    # FLAC cannot represent float samples. s32 with 24 raw bits is the highest
    # bit depth FLAC supports; compression level 12 is ffmpeg's maximum.
    volume_args: list[str] = []
    if gain_db < 0:
        volume_args = ["-af", f"volume={gain_db:.2f}dB"]
    command = [
        settings.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(wav_path),
        *volume_args,
        "-map_metadata",
        "-1",
        "-vn",
        "-c:a",
        "flac",
        "-compression_level",
        "12",
        "-sample_fmt",
        "s32",
        "-bits_per_raw_sample",
        "24",
        str(flac_path),
    ]
    _run_command(command, settings.job_timeout_seconds, f"flac encode {wav_path.name}")
    if not flac_path.is_file() or flac_path.stat().st_size == 0:
        raise JobError(f"flac encode produced no output for {wav_path.name}")


def run_separation_job(settings: Settings, input_path: Path, workdir: Path, mode: str, stems: str | None) -> JobResult:
    lbrr_outdir = workdir / "separated"
    command = [
        settings.lbrr_bin,
        "--separate",
        "--model-dir",
        str(settings.model_dir),
        "--input",
        str(input_path),
        "--outdir",
        str(lbrr_outdir),
        "--mode",
        mode,
        "--no-tui",
        "--no-pause",
    ]
    if mode == "stems" and stems:
        command.extend(["--stems", stems])

    _run_command(command, settings.job_timeout_seconds, "lbrr separation")
    try:
        input_path.unlink()
    except FileNotFoundError:
        pass

    wavs = sorted(lbrr_outdir.glob("*.wav"))
    if not wavs:
        raise JobError("lbrr completed without writing any WAV stems")

    flac_dir = workdir / "flac"
    flac_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, min(settings.encode_workers, len(wavs)))
    gain_db = output_gain_db(settings, wavs, workers)

    def convert_one(wav_path: Path) -> Path:
        flac_path = flac_dir / f"{wav_path.stem}.flac"
        convert_wav_to_flac(settings, wav_path, flac_path, gain_db)
        try:
            wav_path.unlink()
        except FileNotFoundError:
            pass
        return flac_path

    if workers == 1:
        flacs = [convert_one(wav) for wav in wavs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            flacs = list(pool.map(convert_one, wavs))

    zip_path = workdir / "separated-flac.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for flac_path in flacs:
            archive.write(flac_path, arcname=flac_path.name)
    for flac_path in flacs:
        try:
            flac_path.unlink()
        except FileNotFoundError:
            pass
    shutil.rmtree(lbrr_outdir, ignore_errors=True)
    shutil.rmtree(flac_dir, ignore_errors=True)
    return JobResult(zip_path=zip_path, output_files=tuple(path.name for path in flacs), gain_db=gain_db)


DEFAULT_INSTRUMENTS = ("bass", "drums", "other", "vocals", "guitar", "piano")


def load_instrument_names(model_dir: Path) -> list[str]:
    """Read the flow-style instrument list used by the model config."""
    try:
        text = model_dir.joinpath("logic_bs_roformer.yaml").read_text(encoding="utf-8")
    except OSError:
        return list(DEFAULT_INSTRUMENTS)
    match = re.search(r"(?m)^\s*instruments:\s*\[([^\]]*)\]", text)
    if not match:
        return list(DEFAULT_INSTRUMENTS)
    names = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
    return names or list(DEFAULT_INSTRUMENTS)


def validate_stem_selection(spec: str, instruments: Sequence[str]) -> None:
    spec = spec.strip()
    if not spec or spec.lower() == "all":
        return
    for part in re.split(r"[,;\s]+", spec):
        if not part:
            continue
        if part.isdigit():
            if int(part) < len(instruments):
                continue
        elif any(part.lower() == name.lower() for name in instruments):
            continue
        raise ValueError(
            f"unknown stem {part!r}: expected one of {', '.join(instruments)} or an index 0..{len(instruments) - 1}"
        )


def _safe_download_name(filename: str | None, mode: str) -> str:
    raw = Path((filename or "audio").replace("\\", "/")).stem or "audio"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "audio"
    return f"{safe[:80]}-{mode}-flac.zip"


async def save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    written = 0
    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds the {max_bytes} byte limit",
                    )
                output.write(chunk)
    finally:
        await upload.close()
    if written == 0:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    return written


def _cleanup_stale_jobs(settings: Settings) -> None:
    root = settings.temp_root or Path(tempfile.gettempdir())
    try:
        root.mkdir(parents=True, exist_ok=True)
        for child in root.glob(f"{JOB_PREFIX}*"):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
    except OSError as exc:
        LOGGER.warning("could not clean stale job directories under %s: %s", root, exc)


def create_app(settings: Settings) -> FastAPI:
    validate_network_security(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _cleanup_stale_jobs(settings)
        app.state.settings = settings
        app.state.semaphore = asyncio.Semaphore(settings.concurrency)
        yield

    app = FastAPI(title="logic-bs-roformer-rs API", version=API_VERSION, lifespan=lifespan)

    def require_token(request: Request) -> None:
        expected = settings.api_token
        if expected is None:
            return
        supplied: list[str] = []
        authorization = request.headers.get("authorization", "")
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            supplied.append(parts[1])
        api_key = request.headers.get("x-api-key")
        if api_key:
            supplied.append(api_key)
        if any(hmac.compare_digest(candidate.encode(), expected.encode()) for candidate in supplied):
            return
        raise HTTPException(
            status_code=401,
            detail="valid bearer token or X-API-Key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.middleware("http")
    async def guard_separate_upload(request: Request, call_next):
        if request.method != "POST" or request.url.path != "/separate":
            return await call_next(request)
        try:
            require_token(request)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers or {})

        body_limit = settings.max_upload_bytes + 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                return JSONResponse({"detail": "invalid content-length"}, status_code=400)
            if declared > body_limit:
                return JSONResponse({"detail": f"upload exceeds the {settings.max_upload_bytes} byte limit"}, status_code=413)

        received = 0
        inner_receive = request.receive

        async def limited_receive():
            nonlocal received
            message = await inner_receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > body_limit:
                    raise UploadTooLarge
            return message

        request._receive = limited_receive
        try:
            return await call_next(request)
        except UploadTooLarge:
            return JSONResponse({"detail": f"upload exceeds the {settings.max_upload_bytes} byte limit"}, status_code=413)

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "service": "logic-bs-roformer-rs API",
            "version": API_VERSION,
            "separate": "POST /separate (multipart file field: file)",
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        lbrr_path = Path(settings.lbrr_bin)
        lbrr_ready = lbrr_path.is_file() if lbrr_path.is_absolute() or "/" in settings.lbrr_bin else shutil.which(settings.lbrr_bin) is not None
        return {
            "status": "ok" if lbrr_ready and settings.assets_ready else "not_ready",
            "version": API_VERSION,
            "lbrr_ready": lbrr_ready,
            "assets_ready": settings.assets_ready,
            "max_duration_seconds": settings.max_duration_seconds,
            "concurrency": settings.concurrency,
        }

    @app.post("/separate")
    async def separate(
        request: Request,
        file: UploadFile = File(...),
        mode: str = Form("stems"),
        stems: str | None = Form(None),
    ) -> FileResponse:
        require_token(request)
        selected_mode = mode.strip().lower()
        if selected_mode not in {"stems", "vocals"}:
            raise HTTPException(status_code=422, detail="mode must be 'stems' or 'vocals'")
        selected_stems = stems.strip() if stems else None
        if selected_stems is not None and len(selected_stems) > 200:
            raise HTTPException(status_code=422, detail="stems selection is too long")
        if selected_mode == "stems" and selected_stems:
            try:
                validate_stem_selection(selected_stems, load_instrument_names(settings.model_dir))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        temp_root = settings.temp_root or Path(tempfile.gettempdir())
        temp_root.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix=JOB_PREFIX, dir=temp_root))
        original_name = Path((file.filename or "audio").replace("\\", "/")).name
        suffix = Path(original_name).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,16}", suffix):
            suffix = ""
        input_path = workdir / f"input{suffix}"
        handed_to_response = False
        try:
            await save_upload(file, input_path, settings.max_upload_bytes)
            try:
                metadata_duration = await asyncio.to_thread(probe_audio_duration, settings.ffprobe, input_path)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if metadata_duration is not None and metadata_duration >= settings.max_duration_seconds:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"audio duration {metadata_duration:.3f}s is not less than "
                        f"{settings.max_duration_seconds:.3f}s"
                    ),
                )

            normalized_path = workdir / "normalized.wav"
            async with request.app.state.semaphore:
                try:
                    duration = await asyncio.to_thread(normalize_input_audio, settings, input_path, normalized_path)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                try:
                    input_path.unlink()
                except FileNotFoundError:
                    pass
                if duration >= settings.max_duration_seconds:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"decoded audio duration {duration:.3f}s is not less than "
                            f"{settings.max_duration_seconds:.3f}s"
                        ),
                    )

                try:
                    result = await asyncio.to_thread(run_separation_job, settings, normalized_path, workdir, selected_mode, selected_stems)
                except JobError as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc

            download_name = _safe_download_name(file.filename, selected_mode)
            response = FileResponse(
                result.zip_path,
                media_type="application/zip",
                filename=download_name,
                background=BackgroundTask(shutil.rmtree, workdir, True),
                headers={
                    "X-Input-Duration-Seconds": f"{duration:.3f}",
                    "X-Output-Files": ",".join(result.output_files),
                    "X-Zip-Compression": "store",
                    "X-Flac-Compression-Level": "12",
                    "X-Flac-Gain-Db": f"{result.gain_db:.2f}",
                },
            )
            handed_to_response = True
            LOGGER.info("separated %s in %.3fs input: %s", download_name, duration, ",".join(result.output_files))
            return response
        finally:
            if not handed_to_response:
                shutil.rmtree(workdir, ignore_errors=True)

    return app


settings = load_settings()
app = create_app(settings)


def main() -> None:
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level,
        ssl_certfile=settings.ssl_certfile,
        ssl_keyfile=settings.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
