from __future__ import annotations

import io
import json
import math
import shutil
import struct
import subprocess
import sys
import wave
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.lbrr_api import Settings, convert_wav_to_flac, create_app, load_instrument_names, measure_peak_db, output_gain_db

STEM_NAMES = ["0_bass", "1_drums", "2_other", "3_vocals", "4_guitar", "5_piano"]


def wav_bytes(seconds: float = 0.25, sample_rate: int = 44100) -> bytes:
    output = io.BytesIO()
    frames = int(seconds * sample_rate)
    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        samples = bytearray()
        for i in range(frames):
            value = int(10000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            samples.extend(struct.pack("<hh", value, value))
        wav.writeframes(bytes(samples))
    return output.getvalue()


@pytest.fixture()
def fake_lbrr(tmp_path: Path) -> Path:
    script = tmp_path / "fake_lbrr"
    script.write_text(
        """#!/usr/bin/env python3
import math
import struct
import sys
import wave
from pathlib import Path

args = sys.argv[1:]

def value(flag, default=None):
    if flag not in args:
        return default
    return args[args.index(flag) + 1]

outdir = Path(value("--outdir"))
mode = value("--mode", "stems")
outdir.mkdir(parents=True, exist_ok=True)
if mode == "vocals":
    names = ["vocals.wav", "instrumental.wav"]
else:
    names = ["0_bass.wav", "1_drums.wav", "2_other.wav", "3_vocals.wav", "4_guitar.wav", "5_piano.wav"]
for name in names:
    with wave.open(str(outdir / name), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        samples = bytearray()
        for i in range(11025):
            sample = int(8000 * math.sin(2 * math.pi * 330 * i / 44100))
            samples.extend(struct.pack("<hh", sample, sample))
        wav.writeframes(bytes(samples))
print("fake separation complete")
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def make_settings(tmp_path: Path, fake_lbrr: Path, **overrides) -> Settings:
    values = {
        "lbrr_bin": str(fake_lbrr),
        "model_dir": tmp_path / "assets",
        "ffmpeg": shutil.which("ffmpeg") or "ffmpeg",
        "ffprobe": shutil.which("ffprobe") or "ffprobe",
        "temp_root": tmp_path / "jobs",
        "max_duration_seconds": 600.0,
        "max_upload_bytes": 16 * 1024 * 1024,
        "job_timeout_seconds": 20.0,
        "concurrency": 1,
        "encode_workers": 2,
        "host": "127.0.0.1",
        "port": 0,
        "api_token": None,
        "log_level": "error",
    }
    values.update(overrides)
    values["model_dir"].mkdir(parents=True, exist_ok=True)
    values["temp_root"].mkdir(parents=True, exist_ok=True)
    return Settings(**values)


def write_f32_wav(path: Path, left: float = 1.5, right: float = 1.5, frames: int = 4410) -> None:
    samples = struct.pack(f"<{frames * 2}f", *([left, right] * frames))
    data_len = len(samples)
    with path.open("wb") as output:
        output.write(b"RIFF")
        output.write(struct.pack("<I", 36 + data_len))
        output.write(b"WAVEfmt ")
        output.write(struct.pack("<IHHIIHH", 16, 3, 2, 44100, 44100 * 2 * 4, 8, 32))
        output.write(b"data")
        output.write(struct.pack("<I", data_len))
        output.write(samples)


def post_tone(client: TestClient, seconds: float = 0.25, headers: dict[str, str] | None = None, stems: str = "all"):
    return client.post(
        "/separate",
        files={"file": ("tone.wav", wav_bytes(seconds), "audio/wav")},
        data={"mode": "stems", "stems": stems},
        headers=headers or {},
    )


def test_separate_returns_store_only_24bit_flac_zip(tmp_path: Path, fake_lbrr: Path) -> None:
    settings = make_settings(tmp_path, fake_lbrr)
    with TestClient(create_app(settings)) as client:
        response = post_tone(client)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/zip")
    assert response.headers["x-zip-compression"] == "store"
    assert response.headers["x-flac-compression-level"] == "12"

    archive_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(archive_bytes) as archive:
        names = set(archive.namelist())
        assert names == {f"{stem}.flac" for stem in STEM_NAMES}
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        flac_bytes = archive.read("0_bass.flac")

    assert flac_bytes.startswith(b"fLaC")
    probe_input = tmp_path / "result.flac"
    probe_input.write_bytes(flac_bytes)
    proc = subprocess.run(
        [
            settings.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,bits_per_raw_sample",
            "-print_format",
            "json",
            str(probe_input),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    assert stream["codec_name"] == "flac"
    assert stream["bits_per_raw_sample"] == "24"


def test_rejects_audio_at_or_over_duration_limit(tmp_path: Path, fake_lbrr: Path) -> None:
    settings = make_settings(tmp_path, fake_lbrr, max_duration_seconds=0.5)
    with TestClient(create_app(settings)) as client:
        response = post_tone(client, seconds=0.5)

    assert response.status_code == 422
    assert "not less than" in response.json()["detail"]


def test_invalid_stem_selection_is_client_error(tmp_path: Path, fake_lbrr: Path) -> None:
    settings = make_settings(tmp_path, fake_lbrr)
    with TestClient(create_app(settings)) as client:
        response = post_tone(client, stems="kazoo")

    assert response.status_code == 422
    assert "unknown stem" in response.json()["detail"]


def test_overfull_float_output_is_scaled_before_flac(tmp_path: Path, fake_lbrr: Path) -> None:
    settings = make_settings(tmp_path, fake_lbrr)
    wav_path = tmp_path / "hot.wav"
    flac_path = tmp_path / "hot.flac"
    write_f32_wav(wav_path, left=0.5, right=1.5)

    gain_db = output_gain_db(settings, [wav_path], workers=1)
    assert gain_db < 0
    convert_wav_to_flac(settings, wav_path, flac_path, gain_db)
    assert measure_peak_db(settings, flac_path) <= 0


def test_mixed_flow_instrument_names_are_loaded(tmp_path: Path) -> None:
    model_dir = tmp_path / "assets"
    model_dir.mkdir()
    model_dir.joinpath("logic_bs_roformer.yaml").write_text("instruments: ['vocals', drums, \"guitar\"]\n")
    assert load_instrument_names(model_dir) == ["vocals", "drums", "guitar"]


def test_token_is_required_when_configured(tmp_path: Path, fake_lbrr: Path) -> None:
    settings = make_settings(tmp_path, fake_lbrr, api_token="test-secret")
    with TestClient(create_app(settings)) as client:
        assert post_tone(client).status_code == 401
        assert post_tone(client, headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = post_tone(client, headers={"Authorization": "Bearer test-secret"})

    assert response.status_code == 200, response.text
