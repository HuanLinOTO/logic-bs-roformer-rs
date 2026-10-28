//! Minimal WAV reader/writer — 16-bit PCM and 32-bit float, mono/stereo
//! interleave. No external crates so the scaffold stays dependency-free.

#[derive(Debug, Clone)]
pub struct WavData {
    pub sample_rate: u32,
    pub channels: usize,
    /// Interleaved samples, length = frames * channels.
    pub samples: Vec<f32>,
}

pub fn read_wav(path: &std::path::Path) -> Result<WavData, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut r = Reader { b: &bytes, pos: 0 };
    let mut riff = [0u8; 12];
    r.take(&mut riff)?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut fmt: Option<Fmt> = None;
    let mut data: Option<(u16, &[u8])> = None;
    while r.remaining() >= 8 {
        let mut id = [0u8; 4];
        r.take(&mut id)?;
        let size = r.u32()? as usize;
        match &id {
            b"fmt " => {
                let mut f = Fmt::default();
                f.audio_format = r.u16()?;
                f.channels = r.u16()? as usize;
                f.sample_rate = r.u32()?;
                let _byte_rate = r.u32()?;
                let _block_align = r.u16()?;
                f.bits = r.u16()? as usize;
                if size > 16 {
                    r.skip(size - 16)?;
                }
                fmt = Some(f);
            }
            b"data" => {
                let format = fmt.as_ref().map(|f| f.audio_format).unwrap_or(3);
                data = Some((format, r.take_bytes(size)?));
            }
            _ => r.skip(size)?,
        }
    }
    let fmt = fmt.ok_or("missing fmt chunk")?;
    let (format, raw) = data.ok_or("missing data chunk")?;
    let samples = match (format, fmt.bits) {
        (1, 16) => raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        (1, 24) => raw
            .chunks_exact(3)
            .map(|c| {
                let v = ((c[2] as i32) << 24 | (c[1] as i32) << 16 | (c[0] as i32) << 8) >> 8;
                v as f32 / 8388608.0
            })
            .collect(),
        (3, 32) => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        (f, b) => return Err(format!("unsupported wav: format {f}, {b} bits")),
    };
    Ok(WavData {
        sample_rate: fmt.sample_rate,
        channels: fmt.channels,
        samples,
    })
}

#[derive(Default)]
struct Fmt {
    audio_format: u16,
    channels: usize,
    sample_rate: u32,
    bits: usize,
}

pub fn write_wav_f32(path: &std::path::Path, wav: &WavData) -> Result<(), String> {
    use std::io::Write;
    let channels = u16::try_from(wav.channels).map_err(|_| "WAV channel count overflow")?;
    if channels == 0 || wav.sample_rate == 0 || wav.samples.len() % wav.channels != 0 {
        return Err("invalid WAV sample rate, channel count or incomplete frame".into());
    }
    let block_align = channels
        .checked_mul(4)
        .ok_or("WAV block alignment overflow")?;
    let byte_rate = wav
        .sample_rate
        .checked_mul(u32::from(block_align))
        .ok_or("WAV byte rate overflow")?;
    let data_len = wav
        .samples
        .len()
        .checked_mul(4)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or("WAV data size overflow")?;
    let riff_len = data_len.checked_add(36).ok_or("WAV RIFF size overflow")?;
    let mut header = [0u8; 44];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&riff_len.to_le_bytes());
    header[8..16].copy_from_slice(b"WAVEfmt ");
    header[16..20].copy_from_slice(&16u32.to_le_bytes());
    header[20..22].copy_from_slice(&3u16.to_le_bytes());
    header[22..24].copy_from_slice(&channels.to_le_bytes());
    header[24..28].copy_from_slice(&wav.sample_rate.to_le_bytes());
    header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    header[32..34].copy_from_slice(&block_align.to_le_bytes());
    header[34..36].copy_from_slice(&32u16.to_le_bytes());
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data_len.to_le_bytes());
    let write = || -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(&header)?;
        #[cfg(target_endian = "little")]
        {
            // SAFETY: f32 has no padding, all bit patterns are valid, and the
            // immutable byte view cannot outlive the samples it borrows.
            let bytes = unsafe {
                std::slice::from_raw_parts(wav.samples.as_ptr().cast::<u8>(), data_len as usize)
            };
            file.write_all(bytes)?;
        }
        #[cfg(target_endian = "big")]
        {
            let mut bytes = [0u8; 16 * 1024];
            for chunk in wav.samples.chunks(bytes.len() / 4) {
                for (sample, dst) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
                    dst.copy_from_slice(&sample.to_le_bytes());
                }
                file.write_all(&bytes[..chunk.len() * 4])?;
            }
        }
        Ok(())
    };
    write().map_err(|e| format!("write {}: {e}", path.display()))
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, out: &mut [u8]) -> Result<(), String> {
        if self.pos + out.len() > self.b.len() {
            return Err("unexpected EOF".into());
        }
        out.copy_from_slice(&self.b[self.pos..self.pos + out.len()]);
        self.pos += out.len();
        Ok(())
    }
    fn take_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.b.len() {
            return Err("unexpected EOF in data chunk".into());
        }
        let v = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16, String> {
        let mut b = [0u8; 2];
        self.take(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.take(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn skip(&mut self, n: usize) -> Result<(), String> {
        if self.pos + n > self.b.len() {
            self.pos = self.b.len();
            return Ok(());
        }
        self.pos += n;
        Ok(())
    }
    fn remaining(&self) -> usize {
        self.b.len().saturating_sub(self.pos)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                "lbrr-audio-{}-{}.wav",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            )))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn legacy_bytes(wav: &WavData) -> Vec<u8> {
        let data_len = wav.samples.len() * 4;
        let mut out = Vec::with_capacity(44 + data_len);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&3u16.to_le_bytes());
        out.extend_from_slice(&(wav.channels as u16).to_le_bytes());
        out.extend_from_slice(&wav.sample_rate.to_le_bytes());
        out.extend_from_slice(
            &((wav.channels * 4 * wav.sample_rate as usize) as u32).to_le_bytes(),
        );
        out.extend_from_slice(&((wav.channels * 4) as u16).to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        for s in &wav.samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    #[test]
    fn float_wav_is_byte_compatible_and_roundtrips() {
        let values: Vec<f32> = [
            0, 0x80000000, 0x3f800000, 0xbf800000, 1, 0x7f800000, 0x7fc01234, 0xff800001,
        ]
        .into_iter()
        .map(f32::from_bits)
        .collect();
        for channels in [1, 2] {
            let wav = WavData {
                sample_rate: 44100,
                channels,
                samples: values.clone(),
            };
            let path = Temp::new();
            write_wav_f32(&path.0, &wav).unwrap();
            assert_eq!(std::fs::read(&path.0).unwrap(), legacy_bytes(&wav));
            let got = read_wav(&path.0).unwrap();
            assert_eq!(got.sample_rate, wav.sample_rate);
            assert_eq!(got.channels, wav.channels);
            assert_eq!(
                got.samples.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                values.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn pcm16_and_pcm24_decode_unchanged() {
        for bits in [16u16, 24] {
            let path = Temp::new();
            let mut bytes = legacy_bytes(&WavData {
                sample_rate: 48000,
                channels: 1,
                samples: vec![0.0; 3],
            });
            let payload = if bits == 16 {
                vec![0, 128, 0, 0, 255, 127]
            } else {
                vec![0, 0, 128, 0, 0, 0, 255, 255, 127]
            };
            bytes[20..22].copy_from_slice(&1u16.to_le_bytes());
            bytes[34..36].copy_from_slice(&bits.to_le_bytes());
            bytes[40..44].copy_from_slice(&(payload.len() as u32).to_le_bytes());
            bytes.truncate(44);
            bytes.extend(payload);
            std::fs::write(&path.0, bytes).unwrap();
            let got = read_wav(&path.0).unwrap();
            assert_eq!(
                got.samples,
                vec![-1.0, 0.0, 1.0 - 2.0f32.powi(1 - bits as i32)]
            );
        }
    }

    #[test]
    fn invalid_metadata_is_rejected_before_creating_file() {
        for (rate, channels, len) in [
            (0, 2, 2),
            (44100, 0, 0),
            (44100, 2, 3),
            (u32::MAX, 2, 2),
            (44100, usize::MAX, 0),
        ] {
            let path = Temp::new();
            let wav = WavData {
                sample_rate: rate,
                channels,
                samples: vec![0.0; len],
            };
            assert!(write_wav_f32(&path.0, &wav).is_err());
            assert!(!path.0.exists());
        }
    }
}
