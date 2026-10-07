//! safetensors loader (zero-dependency) for the exported BS-RoFormer weights.
//!
//! Layout contract (§3 of the plan): tensors are consumed in torch row-major
//! [out, in] order directly by the GEMM kernels; shared biases
//! (linear_62_bias_0 / linear_64_bias_0) are deduplicated at load time.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

pub struct SafeTensors {
    pub metas: BTreeMap<String, TensorMeta>,
    data: Vec<u8>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        if header_len > 64 << 20 {
            return Err("safetensors header too large".into());
        }
        let mut header = vec![0u8; header_len];
        f.read_exact(&mut header).map_err(|e| e.to_string())?;
        let mut data = Vec::new();
        f.read_to_end(&mut data).map_err(|e| e.to_string())?;
        let metas = parse_header(&header)?;
        Ok(SafeTensors { metas, data })
    }

    /// Copy one fp32 tensor out as host memory (row-major as stored).
    pub fn f32_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        let m = self.metas.get(name).ok_or_else(|| format!("missing tensor {name}"))?;
        if m.dtype != "F32" {
            return Err(format!("{name}: expected F32, got {}", m.dtype));
        }
        let bytes = m.end as usize - m.start as usize;
        if bytes != m.shape.iter().product::<usize>() * 4 {
            return Err(format!("{name}: byte length mismatch"));
        }
        let src = &self.data[m.start as usize..m.end as usize];
        Ok(src
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Minimal JSON parser for the safetensors header.
// ---------------------------------------------------------------------------

fn parse_header(bytes: &[u8]) -> Result<BTreeMap<String, TensorMeta>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
    let mut p = P { s: text.as_bytes(), i: 0 };
    p.ws();
    let v = p.value()?;
    let ValueRef::Obj(entries) = v else {
        return Err("header root is not an object".into());
    };
    let mut out = BTreeMap::new();
    for (k, v) in entries {
        if k == "__metadata__" {
            continue;
        }
        let ValueRef::Obj(fields) = v else {
            return Err(format!("{k}: entry is not an object"));
        };
        let mut dtype = String::new();
        let mut shape = Vec::new();
        let mut offs = [0u64; 2];
        for (fk, fv) in fields {
            match fk.as_str() {
                "dtype" => {
                    let ValueRef::Str(s) = fv else { return Err("dtype not str".into()) };
                    dtype = s.to_string();
                }
                "shape" => {
                    let ValueRef::Arr(a) = fv else { return Err("shape not arr".into()) };
                    shape = a
                        .iter()
                        .map(|x| match x {
                            ValueRef::Num(n) => Ok(*n as usize),
                            _ => Err("shape elem not num".to_string()),
                        })
                        .collect::<Result<_, _>>()?;
                }
                "data_offsets" => {
                    let ValueRef::Arr(a) = fv else { return Err("offsets not arr".into()) };
                    if a.len() != 2 {
                        return Err("offsets len != 2".into());
                    }
                    for (j, x) in a.iter().enumerate() {
                        let ValueRef::Num(n) = x else { return Err("offset not num".into()) };
                        offs[j] = *n as u64;
                    }
                }
                _ => {}
            }
        }
        out.insert(k, TensorMeta { dtype, shape, start: offs[0], end: offs[1] });
    }
    Ok(out)
}

enum ValueRef<'a> {
    Null,
    Bool(bool),
    Num(f64),
    Str(&'a str),
    Arr(Vec<ValueRef<'a>>),
    Obj(Vec<(String, ValueRef<'a>)>),
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn value(&mut self) -> Result<ValueRef<'a>, String> {
        self.ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => {
                let s = self.string()?;
                Ok(ValueRef::Str(s))
            }
            Some(b't') => {
                self.lit_word("true")?;
                Ok(ValueRef::Bool(true))
            }
            Some(b'f') => {
                self.lit_word("false")?;
                Ok(ValueRef::Bool(false))
            }
            Some(b'n') => {
                self.lit_word("null")?;
                Ok(ValueRef::Null)
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(format!("unexpected char at {}", self.i)),
        }
    }
    fn lit_word(&mut self, word: &str) -> Result<(), String> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(())
        } else {
            Err(format!("bad literal at {}", self.i))
        }
    }
    fn number(&mut self) -> Result<ValueRef<'a>, String> {
        let start = self.i;
        while let Some(c) = self.peek() {
            if c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' || c.is_ascii_digit() {
                self.i += 1;
            } else {
                break;
            }
        }
        let s: &'a [u8] = self.s;
        std::str::from_utf8(&s[start..self.i])
            .ok()
            .and_then(|t| t.parse::<f64>().ok())
            .map(ValueRef::Num)
            .ok_or_else(|| format!("bad number at {start}"))
    }
    fn string(&mut self) -> Result<&'a str, String> {
        let s: &'a [u8] = self.s;
        let mut i = self.i;
        if s.get(i) != Some(&b'"') {
            return Err("expected string".into());
        }
        i += 1;
        let start = i;
        while i < s.len() {
            match s[i] {
                b'\\' => i += 2,
                b'"' => {
                    let out = std::str::from_utf8(&s[start..i]).map_err(|e| e.to_string())?;
                    self.i = i + 1;
                    return Ok(out);
                }
                _ => i += 1,
            }
        }
        Err("unterminated string".into())
    }
    fn array(&mut self) -> Result<ValueRef<'a>, String> {
        self.i += 1; // '['
        let mut items = Vec::new();
        loop {
            self.ws();
            if self.peek() == Some(b']') {
                self.i += 1;
                return Ok(ValueRef::Arr(items));
            }
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {}
                _ => return Err(format!("expected , or ] at {}", self.i)),
            }
        }
    }
    fn object(&mut self) -> Result<ValueRef<'a>, String> {
        self.i += 1; // '{'
        let mut items = Vec::new();
        loop {
            self.ws();
            if self.peek() == Some(b'}') {
                self.i += 1;
                return Ok(ValueRef::Obj(items));
            }
            let k = self.string()?.to_string();
            self.ws();
            if self.peek() != Some(b':') {
                return Err("expected :".into());
            }
            self.i += 1;
            let v = self.value()?;
            items.push((k, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {}
                _ => return Err(format!("expected , or }} at {}", self.i)),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Model weight groups (§3 mapping), host side. GPU upload happens in model.rs.
// ---------------------------------------------------------------------------

/// Attention block weights for one axis-transformer layer.
#[derive(Debug, Clone, Default)]
pub struct AttnWeights {
    pub norm_gamma: Vec<f32>,       // [256]  (pre-attention RMSNorm gamma)
    pub to_qkv_w: Vec<f32>,         // [1536, 256]
    pub to_qkv_bias: Vec<f32>,      // [1536] (shared bias values, per-layer copy)
    pub to_gates_w: Vec<f32>,       // [8, 256]
    pub to_gates_b: Vec<f32>,       // [8]
    pub to_out_w: Vec<f32>,         // [256, 512]
    pub to_out_bias: Vec<f32>,      // [256] (shared)
}

#[derive(Debug, Clone, Default)]
pub struct FfWeights {
    pub gamma0: Vec<f32>,           // [256]
    pub w1: Vec<f32>,               // [1024, 256]
    pub b1: Vec<f32>,               // [1024]
    pub w2: Vec<f32>,               // [256, 1024]
    pub b2: Vec<f32>,               // [256]
}

/// One depth step: time transformer + freq transformer. Confirmed from the
/// checkpoint layout: NO per-transformer output norm (norm_output was off at
/// training time); the only trailing norm is the global final_norm.
#[derive(Debug, Clone, Default)]
pub struct TransformerLayer {
    pub time: (AttnWeights, FfWeights),
    pub freq: (AttnWeights, FfWeights),
}

pub struct ModelWeights {
    pub shared_qkv_bias: Vec<f32>,  // [1536]
    pub shared_out_bias: Vec<f32>,  // [256]
    pub layers: Vec<TransformerLayer>, // 12
    pub final_norm_gamma: Vec<f32>,
    // BandSplit: per band [gamma(dim_in), W(256,dim_in), b(256)]
    pub band_gamma: Vec<Vec<f32>>,
    pub band_w: Vec<Vec<f32>>,
    pub band_b: Vec<Vec<f32>>,
    // MaskEstimator: per stem per band [W1(1024,256), b1(1024), W2(dim_in*2,1024), b2(dim_in*2)]
    pub mask_w1: Vec<Vec<Vec<f32>>>, // [stem][band]
    pub mask_b1: Vec<Vec<Vec<f32>>>,
    pub mask_w2: Vec<Vec<Vec<f32>>>,
    pub mask_b2: Vec<Vec<Vec<f32>>>,
}

impl ModelWeights {
    /// Load and validate everything against the config. Every one of the 1989
    /// checkpoint keys must be consumed exactly once (§7 risk control).
    pub fn load(st: &SafeTensors, cfg: &crate::config::ModelConfig) -> Result<Self, String> {
        let mut used = std::collections::BTreeSet::new();
        let take = |st: &SafeTensors, used: &mut std::collections::BTreeSet<String>, name: &str| -> Result<Vec<f32>, String> {
            if !used.insert(name.to_string()) {
                return Err(format!("duplicate key use: {name}"));
            }
            st.f32_tensor(name)
        };

        let shared_qkv_bias = take(st, &mut used, "linear_62_bias_0")?;
        let shared_out_bias = take(st, &mut used, "linear_64_bias_0")?;

        let expect_len = |v: &[f32], n: usize, what: &str| -> Result<(), String> {
            if v.len() != n {
                Err(format!("{what}: len {} != {n}", v.len()))
            } else {
                Ok(())
            }
        };
        let dim_inner = cfg.heads * cfg.dim_head; // 512
        expect_len(&shared_qkv_bias, 3 * dim_inner, "shared_qkv_bias")?;
        expect_len(&shared_out_bias, cfg.dim, "shared_out_bias")?;

        let mut layers = Vec::with_capacity(cfg.depth);
        for l in 0..cfg.depth {
            let mut axes = Vec::with_capacity(2);
            for a in 0..2 {
                let p = format!("layers.{l}.{a}.layers.0.0");
                let attn = AttnWeights {
                    norm_gamma: take(st, &mut used, &format!("{p}.norm.gamma"))?,
                    to_qkv_w: take(st, &mut used, &format!("{p}.to_qkv.weight"))?,
                    to_qkv_bias: shared_qkv_bias.clone(),
                    to_gates_w: take(st, &mut used, &format!("{p}.to_gates.weight"))?,
                    to_gates_b: take(st, &mut used, &format!("{p}.to_gates.bias"))?,
                    to_out_w: take(st, &mut used, &format!("{p}.to_out.0.weight"))?,
                    to_out_bias: shared_out_bias.clone(),
                };
                expect_len(&attn.norm_gamma, cfg.dim, "norm_gamma")?;
                expect_len(&attn.to_qkv_w, 3 * dim_inner * cfg.dim, "to_qkv_w")?;
                expect_len(&attn.to_gates_w, cfg.heads * cfg.dim, "to_gates_w")?;
                expect_len(&attn.to_out_w, cfg.dim * dim_inner, "to_out_w")?;
                let fp = format!("layers.{l}.{a}.layers.0.1.net");
                let ff = FfWeights {
                    gamma0: take(st, &mut used, &format!("{fp}.0.gamma"))?,
                    w1: take(st, &mut used, &format!("{fp}.1.weight"))?,
                    b1: take(st, &mut used, &format!("{fp}.1.bias"))?,
                    w2: take(st, &mut used, &format!("{fp}.4.weight"))?,
                    b2: take(st, &mut used, &format!("{fp}.4.bias"))?,
                };
                let ff_dim = cfg.dim * cfg.mlp_expansion;
                expect_len(&ff.gamma0, cfg.dim, "ff gamma")?;
                expect_len(&ff.w1, ff_dim * cfg.dim, "ff w1")?;
                expect_len(&ff.b1, ff_dim, "ff b1")?;
                expect_len(&ff.w2, cfg.dim * ff_dim, "ff w2")?;
                expect_len(&ff.b2, cfg.dim, "ff b2")?;
                // Per-layer bias copies share storage with the global
                // linear_62/64_bias_0 (use_shared_bias=True). Verify they are
                // byte-equal, then drop — we use the shared copy.
                let qkv_b = take(st, &mut used, &format!("{p}.to_qkv.bias"))?;
                if qkv_b != shared_qkv_bias {
                    return Err(format!("{p}.to_qkv.bias != linear_62_bias_0"));
                }
                let out_b = take(st, &mut used, &format!("{p}.to_out.0.bias"))?;
                if out_b != shared_out_bias {
                    return Err(format!("{p}.to_out.0.bias != linear_64_bias_0"));
                }
                // rotary freqs: identical across all layers per axis; verify
                // against the analytic 1/10000^(2i/dim_head) and drop.
                let freqs = take(st, &mut used, &format!("layers.{l}.{a}.layers.0.0.rotary_embed.freqs"))?;
                expect_len(&freqs, cfg.dim_head / 2, "rotary freqs")?;
                for (i, f) in freqs.iter().enumerate() {
                    let want = 1.0 / 10000.0f32.powf((2.0 * i as f32) / cfg.dim_head as f32);
                    if (f - want).abs() > 1e-6 {
                        return Err(format!("rotary freqs[{i}] {f} != analytic {want}"));
                    }
                }
                axes.push((attn, ff));
            }
            let mut axes=axes.into_iter();
            layers.push(TransformerLayer {
                time: axes.next().expect("time layer"),
                freq: axes.next().expect("frequency layer"),
            });
        }
        let final_norm_gamma = take(st, &mut used, "final_norm.gamma")?;
        expect_len(&final_norm_gamma, cfg.dim, "final_norm")?;

        let nb = cfg.num_bands();
        let mut band_gamma = Vec::with_capacity(nb);
        let mut band_w = Vec::with_capacity(nb);
        let mut band_b = Vec::with_capacity(nb);
        for b in 0..nb {
            let dim_in = cfg.band_dim_in(b);
            band_gamma.push(take(st, &mut used, &format!("band_split.to_features.{b}.0.gamma"))?);
            band_w.push(take(st, &mut used, &format!("band_split.to_features.{b}.1.weight"))?);
            band_b.push(take(st, &mut used, &format!("band_split.to_features.{b}.1.bias"))?);
            expect_len(band_gamma.last().unwrap(), dim_in, "band gamma")?;
            expect_len(band_w.last().unwrap(), cfg.dim * dim_in, "band w")?;
            expect_len(band_b.last().unwrap(), cfg.dim, "band b")?;
        }

        let mut mask_w1 = vec![vec![Vec::new(); nb]; cfg.num_stems];
        let mut mask_b1 = vec![vec![Vec::new(); nb]; cfg.num_stems];
        let mut mask_w2 = vec![vec![Vec::new(); nb]; cfg.num_stems];
        let mut mask_b2 = vec![vec![Vec::new(); nb]; cfg.num_stems];
        let mlp_inner = cfg.dim * cfg.mlp_expansion; // 1024
        for s in 0..cfg.num_stems {
            for b in 0..nb {
                let dim_in = cfg.band_dim_in(b);
                let p = format!("mask_estimators.{s}.to_freqs.{b}.0");
                mask_w1[s][b] = take(st, &mut used, &format!("{p}.0.weight"))?;
                mask_b1[s][b] = take(st, &mut used, &format!("{p}.0.bias"))?;
                mask_w2[s][b] = take(st, &mut used, &format!("{p}.2.weight"))?;
                mask_b2[s][b] = take(st, &mut used, &format!("{p}.2.bias"))?;
                expect_len(&mask_w1[s][b], mlp_inner * cfg.dim, "mask w1")?;
                expect_len(&mask_b1[s][b], mlp_inner, "mask b1")?;
                expect_len(&mask_w2[s][b], dim_in * 2 * mlp_inner, "mask w2")?;
                expect_len(&mask_b2[s][b], dim_in * 2, "mask b2")?;
            }
        }

        // Full-coverage assertion: every checkpoint key consumed.
        let unconsumed: Vec<&String> = st.metas.keys().filter(|k| !used.contains(*k)).collect();
        if !unconsumed.is_empty() {
            return Err(format!("{} unconsumed keys, e.g. {unconsumed:?}", unconsumed.len()));
        }

        Ok(ModelWeights {
            shared_qkv_bias,
            shared_out_bias,
            layers,
            final_norm_gamma,
            band_gamma,
            band_w,
            band_b,
            mask_w1,
            mask_b1,
            mask_w2,
            mask_b2,
        })
    }
}
