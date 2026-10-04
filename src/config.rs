//! Minimal YAML reader for the pymss training config (logic_bs_roformer.yaml).
//!
//! Zero-dependency: parses only the subset the model export actually uses —
//! nested mappings, block sequences, inline `[a, b]` lists, scalars
//! (int/float/bool/null/str), `!!python/tuple` tags (treated as lists) and
//! `#` comments. Unknown constructs are skipped rather than rejected.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(m) => m.get(key),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(l) => Some(l),
            _ => None,
        }
    }
}

/// Inference-relevant slice of the model config (values verified against the
/// exported checkpoint shapes).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub depth: usize,
    pub num_stems: usize,
    pub heads: usize,
    pub dim_head: usize,
    /// 62 entries, summing to dim_freqs_in.
    pub freqs_per_bands: Vec<usize>,
    pub dim_freqs_in: usize,
    pub stft_n_fft: usize,
    pub stft_hop: usize,
    pub stft_win: usize,
    pub mlp_expansion: usize,
    pub use_shared_bias: bool,
    pub chunk_size: usize,
    pub sample_rate: u32,
    pub instruments: Vec<String>,
}

impl ModelConfig {
    pub fn num_bands(&self) -> usize {
        self.freqs_per_bands.len()
    }
    /// Per-band input width of BandSplit: 2 * freqs * 2 channels.
    pub fn band_dim_in(&self, band: usize) -> usize {
        2 * self.freqs_per_bands[band] * 2
    }
    pub fn parse(text: &str) -> Result<Self, String> {
        let root = parse_yaml(text)?;
        let audio = root.get("audio").ok_or("missing audio section")?;
        let model = root.get("model").ok_or("missing model section")?;
        let training = root.get("training").ok_or("missing training section")?;
        let freqs: Vec<usize> = model
            .get("freqs_per_bands")
            .and_then(|v| v.as_list())
            .ok_or("missing freqs_per_bands")?
            .iter()
            .map(|v| v.as_i64().unwrap() as usize)
            .collect();
        let cfg = ModelConfig {
            dim: req(model, "dim")?,
            depth: req(model, "depth")?,
            num_stems: req(model, "num_stems")?,
            heads: req(model, "heads")?,
            dim_head: req(model, "dim_head")?,
            freqs_per_bands: freqs,
            dim_freqs_in: req(model, "dim_freqs_in")?,
            stft_n_fft: req(model, "stft_n_fft")?,
            stft_hop: req(model, "stft_hop_length")?,
            stft_win: req(model, "stft_win_length")?,
            mlp_expansion: req(model, "mlp_expansion_factor")?,
            use_shared_bias: model.get("use_shared_bias").and_then(|v| v.as_bool()).unwrap_or(false),
            chunk_size: req(audio, "chunk_size")?,
            sample_rate: req::<u32>(audio, "sample_rate")?,
            instruments: training
                .get("instruments")
                .and_then(|v| v.as_list())
                .ok_or("missing instruments")?
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect(),
        };
        let freq_sum: usize = cfg.freqs_per_bands.iter().sum();
        if freq_sum != cfg.dim_freqs_in {
            return Err(format!("freqs_per_bands sum {freq_sum} != dim_freqs_in {}", cfg.dim_freqs_in));
        }
        Ok(cfg)
    }
}

fn req<T>(map: &Value, key: &str) -> Result<T, String>
where
    T: std::convert::From<ValueNum>,
{
    let v = map.get(key).ok_or_else(|| format!("missing key {key}"))?;
    ValueNum::of(v).map(T::from).ok_or_else(|| format!("key {key} not numeric"))
}

/// Helper so `req` can convert Int/Float to usize/u32 generically.
pub struct ValueNum(i64);

impl ValueNum {
    fn of(v: &Value) -> Option<ValueNum> {
        v.as_i64().map(ValueNum)
    }
}
macro_rules! num_from {
    ($t:ty) => {
        impl From<ValueNum> for $t {
            fn from(n: ValueNum) -> $t {
                n.0 as $t
            }
        }
    };
}
num_from!(usize);
num_from!(u32);

// ---------------------------------------------------------------------------
// Tiny indentation-based YAML subset parser
// ---------------------------------------------------------------------------

pub fn parse_yaml(text: &str) -> Result<Value, String> {
    let lines: Vec<Line> = text
        .lines()
        .enumerate()
        .filter_map(|(i, raw)| {
            let no_comment = strip_comment(raw);
            let trimmed = no_comment.trim_end();
            if trimmed.trim().is_empty() {
                return None;
            }
            let indent = trimmed.len() - trimmed.trim_start().len();
            Some(Line { num: i + 1, indent, text: trimmed.trim_start().to_string() })
        })
        .collect();
    let mut pos = 0usize;
    let v = parse_block(&lines, &mut pos, 0)?;
    Ok(v)
}

struct Line {
    num: usize,
    indent: usize,
    text: String,
}

fn parse_block(lines: &[Line], pos: &mut usize, indent: usize) -> Result<Value, String> {
    if *pos >= lines.len() {
        return Ok(Value::Null);
    }
    if lines[*pos].text.starts_with("- ") || lines[*pos].text == "-" {
        // sequence
        let mut items = Vec::new();
        while *pos < lines.len() && lines[*pos].indent == indent {
            let l = &lines[*pos];
            if !(l.text.starts_with("- ") || l.text == "-") {
                break;
            }
            let rest = l.text[1..].trim_start();
            *pos += 1;
            if rest.is_empty() {
                items.push(parse_block(lines, pos, indent + 2)?);
            } else {
                items.push(parse_scalar_or_inline(rest)?);
            }
        }
        return Ok(Value::List(items));
    }
    // mapping
    let mut map = BTreeMap::new();
    while *pos < lines.len() {
        let l = &lines[*pos];
        if l.indent < indent {
            break;
        }
        if l.indent > indent {
            return Err(format!("line {}: unexpected indent", l.num));
        }
        let (key, rest) = l
            .text
            .split_once(':')
            .ok_or_else(|| format!("line {}: expected 'key:'", l.num))?;
        let key = key.trim().to_string();
        let rest = rest.trim();
        *pos += 1;
        if rest.is_empty() {
            // nested block (or null)
            if *pos < lines.len() && lines[*pos].indent > indent {
                let child_indent = lines[*pos].indent;
                map.insert(key, parse_block(lines, pos, child_indent)?);
            } else {
                map.insert(key, Value::Null);
            }
        } else {
            map.insert(key, parse_scalar_or_inline(rest)?);
        }
    }
    Ok(Value::Map(map))
}

/// Parses scalars, inline lists `[a, b]` and tagged values (`!!python/tuple`
/// payload on the same line is handled by the caller emitting sequence items).
fn parse_scalar_or_inline(s: &str) -> Result<Value, String> {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix("!!python/tuple") {
        let inner = inner.trim();
        if inner.is_empty() {
            return Ok(Value::Null); // block sequence follows
        }
        return parse_scalar_or_inline(inner);
    }
    if s.starts_with('[') && s.ends_with(']') {
        let items = s[1..s.len() - 1]
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(parse_scalar_or_inline)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Value::List(items));
    }
    if s.starts_with('\'') && s.ends_with('\'') {
        return Ok(Value::Str(s[1..s.len() - 1].to_string()));
    }
    if s.starts_with('"') && s.ends_with('"') {
        return Ok(Value::Str(s[1..s.len() - 1].to_string()));
    }
    Ok(match s {
        "null" | "~" | "" => Value::Null,
        "true" | "True" => Value::Bool(true),
        "false" | "False" => Value::Bool(false),
        _ => {
            if let Ok(i) = s.parse::<i64>() {
                Value::Int(i)
            } else if let Ok(f) = s.parse::<f64>() {
                Value::Float(f)
            } else {
                Value::Str(s.to_string())
            }
        }
    })
}

/// Removes trailing `# comment` (naive: only when '#' is preceded by
/// whitespace or starts the token — good enough for this config).
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_squote = false;
    let mut in_dquote = false;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'\'' if !in_dquote => in_squote = !in_squote,
            b'"' if !in_squote => in_dquote = !in_dquote,
            b'#' if !in_squote && !in_dquote && (i == 0 || bytes[i - 1] == b' ') => {
                return &line[..i];
            }
            _ => {}
        }
    }
    line
}
