//! Explicit per-axis backend requests. Auto defaults remain conservative.
use serde::Serialize;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AttentionBackendRequest {
    #[default]
    Auto,
    Cudnn,
    Handwritten,
}
impl FromStr for AttentionBackendRequest {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "cudnn" => Ok(Self::Cudnn),
            "handwritten" => Ok(Self::Handwritten),
            _ => Err(format!("unknown attention backend {s}")),
        }
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Ff1BackendRequest {
    #[default]
    Auto,
    Handwritten,
    CublasltErf,
    HandwrittenAsync,
}
impl FromStr for Ff1BackendRequest {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "handwritten" => Ok(Self::Handwritten),
            "cublaslt-erf" => Ok(Self::CublasltErf),
            "handwritten-async" => Ok(Self::HandwrittenAsync),
            _ => Err(format!("unknown FF1 backend {s}")),
        }
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeMode {
    #[default]
    Split,
    Fused,
}
impl FromStr for RopeMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "split" => Ok(Self::Split),
            "fused" => Ok(Self::Fused),
            _ => Err(format!("unknown Q/K RoPE mode {s}")),
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct InferenceOptions {
    pub attention: [Option<AttentionBackendRequest>; 2],
    pub ff1: Ff1BackendRequest,
    pub rope: RopeMode,
}
#[derive(Debug, Clone, Serialize)]
pub struct AttentionSelection {
    pub requested: AttentionBackendRequest,
    pub selected: AttentionBackendRequest,
    pub fallback_reason: Option<String>,
    pub source: &'static str,
}
impl InferenceOptions {
    pub fn selected_ff1(&self) -> Ff1BackendRequest {
        if self.ff1 == Ff1BackendRequest::Auto {
            Ff1BackendRequest::Handwritten
        } else {
            self.ff1
        }
    }

    pub fn attention_request(&self, axis: usize, no_cudnn: bool) -> AttentionSelection {
        let cli = self.attention[axis];
        let requested = cli.unwrap_or_default();
        let disabled = cli.is_none() && no_cudnn;
        AttentionSelection {
            requested,
            selected: if disabled {
                AttentionBackendRequest::Handwritten
            } else {
                requested
            },
            fallback_reason: disabled
                .then(|| "disabled by LBRR_NO_CUDNN (presence semantics)".into()),
            source: if cli.is_some() {
                "cli"
            } else if disabled {
                "environment"
            } else {
                "default"
            },
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_cli_overrides_environment_per_axis() {
        let mut o = InferenceOptions::default();
        o.attention[0] = Some(AttentionBackendRequest::Cudnn);
        assert_eq!(
            o.attention_request(0, true).selected,
            AttentionBackendRequest::Cudnn
        );
        assert_eq!(
            o.attention_request(1, true).selected,
            AttentionBackendRequest::Handwritten
        );
        o.attention[0] = Some(AttentionBackendRequest::Auto);
        assert_eq!(
            o.attention_request(0, true).selected,
            AttentionBackendRequest::Auto
        );
    }
    #[test]
    fn unknown_backends_fail_instead_of_auto() {
        assert!("flash".parse::<AttentionBackendRequest>().is_err());
        assert!("tanh".parse::<Ff1BackendRequest>().is_err());
    }
}
