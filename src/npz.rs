//! Zero-dependency reader for .npz files written by numpy.savez.
//! Python's zipfile streams with data descriptors, so local headers carry
//! zero sizes — we parse the central directory at the end instead.

use std::collections::BTreeMap;
use std::path::Path;

pub struct Npz {
    pub arrays: BTreeMap<String, Vec<f32>>,
    pub shapes: BTreeMap<String, Vec<usize>>,
}

impl Npz {
    pub fn open(path: &Path) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // find EOCD (PK\x05\x06) scanning back from the end
        let mut eocd = None;
        let mut i = b.len().saturating_sub(22);
        loop {
            if i + 4 <= b.len() && &b[i..i + 4] == b"PK\x05\x06" {
                eocd = Some(i);
                break;
            }
            if i == 0 { break; }
            i -= 1;
        }
        let eocd = eocd.ok_or("no zip EOCD found")?;
        let entries = u16le(&b, eocd + 10) as usize;
        let mut cd = u32le(&b, eocd + 16) as usize;

        let mut arrays = BTreeMap::new();
        let mut shapes = BTreeMap::new();
        for _ in 0..entries {
            if &b[cd..cd + 4] != b"PK\x01\x02" {
                return Err("bad central directory entry".into());
            }
            let method = u16le(&b, cd + 10);
            let comp_size = u32le(&b, cd + 20) as usize;
            let name_len = u16le(&b, cd + 28) as usize;
            let extra_len = u16le(&b, cd + 30) as usize;
            let comment_len = u16le(&b, cd + 32) as usize;
            let local_off = u32le(&b, cd + 42) as usize;
            let name = String::from_utf8_lossy(&b[cd + 46..cd + 46 + name_len]).to_string();
            if let Some(stripped) = name.strip_suffix(".npy") {
                if method != 0 {
                    return Err(format!("{name}: zip method {method} != stored; regenerate with np.savez"));
                }
                // local header: name/extra lengths may differ from central
                let l_name = u16le(&b, local_off + 26) as usize;
                let l_extra = u16le(&b, local_off + 28) as usize;
                let data_start = local_off + 30 + l_name + l_extra;
                let data = &b[data_start..data_start + comp_size];
                let (shape, floats) = parse_npy(data)?;
                arrays.insert(stripped.to_string(), floats);
                shapes.insert(stripped.to_string(), shape);
            }
            cd += 46 + name_len + extra_len + comment_len;
        }
        Ok(Npz { arrays, shapes })
    }

    pub fn f32(&self, name: &str) -> Result<&[f32], String> {
        self.arrays.get(name).map(|v| v.as_slice()).ok_or_else(|| format!("missing array {name}"))
    }
}

fn u16le(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}
fn u32le(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Parse an .npy payload: magic, version, header dict, raw f32 data.
fn parse_npy(data: &[u8]) -> Result<(Vec<usize>, Vec<f32>), String> {
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        return Err("bad npy magic".into());
    }
    let major = data[6];
    let (hlen, hdr_start) = match major {
        1 => (u16le(data, 8) as usize, 10),
        2 => (u32le(data, 8) as usize, 12),
        _ => return Err(format!("npy version {major} unsupported")),
    };
    let header = std::str::from_utf8(&data[hdr_start..hdr_start + hlen]).map_err(|e| e.to_string())?;
    if !header.contains("'descr': '<f4'") {
        return Err(format!("npy dtype not <f4: {header}"));
    }
    // Locate the parenthesized tuple directly: 'shape': (2, 588800), }
    let shape: Vec<usize> = header
        .split("'shape':")
        .nth(1)
        .and_then(|rest| {
            let start = rest.find('(')? + 1;
            let end = rest.find(')')?;
            if end < start {
                return None;
            }
            Some(
                rest[start..end]
                    .split(',')
                    .filter(|p| !p.trim().is_empty())
                    .map(|p| p.trim().parse::<usize>().unwrap_or(0))
                    .collect(),
            )
        })
        .unwrap_or_default();
    let body = &data[hdr_start + hlen..];
    let floats: Vec<f32> = body
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expect: usize = shape.iter().product();
    if floats.len() < expect {
        return Err(format!("npy body {} < shape product {}", floats.len(), expect));
    }
    Ok((shape, floats[..expect].to_vec()))
}