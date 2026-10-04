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
    let mut data: Option<(u16, Vec<u8>)> = None;
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
    Ok(WavData { sample_rate: fmt.sample_rate, channels: fmt.channels, samples })
}

#[derive(Default)]
struct Fmt {
    audio_format: u16,
    channels: usize,
    sample_rate: u32,
    bits: usize,
}

pub fn write_wav_f32(path: &std::path::Path, wav: &WavData) -> Result<(), String> {
    let data_len = wav.samples.len() * 4;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    out.extend_from_slice(&(wav.channels as u16).to_le_bytes());
    out.extend_from_slice(&wav.sample_rate.to_le_bytes());
    out.extend_from_slice(&((wav.channels * 4 * wav.sample_rate as usize) as u32).to_le_bytes());
    out.extend_from_slice(&((wav.channels * 4) as u16).to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in &wav.samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, out).map_err(|e| format!("write {}: {e}", path.display()))
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, out: &mut [u8]) -> Result<(), String> {
        if self.pos + out.len() > self.b.len() {
            return Err("unexpected EOF".into());
        }
        out.copy_from_slice(&self.b[self.pos..self.pos + out.len()]);
        self.pos += out.len();
        Ok(())
    }
    fn take_bytes(&mut self, n: usize) -> Result<Vec<u8>, String> {
        if self.pos + n > self.b.len() {
            return Err("unexpected EOF in data chunk".into());
        }
        let v = self.b[self.pos..self.pos + n].to_vec();
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
