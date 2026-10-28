//! Output plan: which stems a separation run writes, and how they combine.
//!
//! One definition shared by the drag-and-drop TUI, the `--mode` / `--stems`
//! flags and the writer in [`crate::separate`], so "detailed stems" and
//! "vocals / instrumental" can never drift apart.
//!
//! Everything here is pure: no filesystem probing, no GPU, no terminal. That
//! keeps the mapping unit-testable on machines without a model or a device.

use std::path::{Path, PathBuf};

/// Instrument order shipped with `logic_bs_roformer.yaml`. Only used when the
/// config cannot be read: dry runs, and the TUI before the model is loaded.
pub const DEFAULT_INSTRUMENTS: [&str; 6] = ["bass", "drums", "other", "vocals", "guitar", "piano"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// One file per instrument (six by default), named `<index>_<stem>.wav`.
    Stems,
    /// Two files: the vocal stem plus the sum of everything else.
    Vocals,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Mode, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "stems" | "stem" | "detailed" | "detail" | "6" => Ok(Mode::Stems),
            "vocals" | "vocal" | "inst" | "instrumental" | "karaoke" | "2" => Ok(Mode::Vocals),
            other => Err(format!("bad output mode {other:?}: expected stems|vocals")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Mode::Stems => "detailed stems",
            Mode::Vocals => "vocals / instrumental",
        }
    }
}

/// One file the writer should produce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// A single model stem, written as-is.
    Stem { index: usize, file: String },
    /// Sum of several model stems, e.g. instrumental = everything but vocals.
    Mix { stems: Vec<usize>, file: String },
}

impl Target {
    pub fn file(&self) -> &str {
        match self {
            Target::Stem { file, .. } | Target::Mix { file, .. } => file,
        }
    }

    pub fn sources(&self) -> &[usize] {
        match self {
            Target::Stem { index, .. } => std::slice::from_ref(index),
            Target::Mix { stems, .. } => stems,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub mode: Mode,
    /// Stem indices written when `mode == Mode::Stems`; ignored otherwise.
    pub stems: Vec<usize>,
}

impl Plan {
    pub fn all_stems(instruments: &[String]) -> Plan {
        Plan { mode: Mode::Stems, stems: (0..stem_count(instruments)).collect() }
    }

    pub fn vocals(instruments: &[String]) -> Plan {
        Plan { mode: Mode::Vocals, stems: (0..stem_count(instruments)).collect() }
    }

    /// Instrument names, falling back to the bundled defaults when the config
    /// is unavailable (dry run without assets).
    pub fn stem_names(instruments: &[String]) -> Vec<String> {
        if instruments.is_empty() {
            DEFAULT_INSTRUMENTS.iter().map(|s| (*s).to_string()).collect()
        } else {
            instruments.to_vec()
        }
    }

    /// Files this plan writes, in a stable order.
    pub fn targets(&self, instruments: &[String]) -> Vec<Target> {
        let names = Self::stem_names(instruments);
        match self.mode {
            Mode::Stems => {
                let mut stems: Vec<usize> = self.stems.clone();
                stems.retain(|&i| i < names.len());
                stems.sort_unstable();
                stems.dedup();
                stems
                    .into_iter()
                    .map(|index| {
                        let name = names.get(index).map(|s| s.as_str()).unwrap_or("stem");
                        Target::Stem { index, file: format!("{index}_{name}.wav") }
                    })
                    .collect()
            }
            Mode::Vocals => {
                let v = vocals_index(&names);
                let rest: Vec<usize> = (0..names.len()).filter(|&i| i != v).collect();
                let mut out = vec![Target::Stem { index: v, file: "vocals.wav".to_string() }];
                if !rest.is_empty() {
                    out.push(Target::Mix { stems: rest, file: "instrumental.wav".to_string() });
                }
                out
            }
        }
    }

    /// One-line description used by logs, dry runs and the TUI summary.
    pub fn summary(&self, instruments: &[String]) -> String {
        let files: Vec<String> = self.targets(instruments).iter().map(|t| t.file().to_string()).collect();
        format!("{} ({})", self.mode.label(), files.join(", "))
    }
}

pub fn stem_count(instruments: &[String]) -> usize {
    if instruments.is_empty() { DEFAULT_INSTRUMENTS.len() } else { instruments.len() }
}

/// The vocal stem index: the name containing "vocal", else the fourth stem of
/// a six-stem model, else the first one.
pub fn vocals_index(names: &[String]) -> usize {
    names
        .iter()
        .position(|n| n.to_ascii_lowercase().contains("vocal"))
        .unwrap_or(if names.len() == 6 { 3 } else { 0 })
}

/// Parse `--stems`: `all`, comma/space separated names, or indices.
pub fn parse_stems(spec: &str, instruments: &[String]) -> Result<Vec<usize>, String> {
    let names = Plan::stem_names(instruments);
    let spec = spec.trim();
    if spec.is_empty() || spec.eq_ignore_ascii_case("all") {
        return Ok((0..names.len()).collect());
    }
    let mut selected = Vec::new();
    for part in spec.split(|c: char| c == ',' || c == ';' || c == ' ') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let index = match part.parse::<usize>() {
            Ok(n) => n,
            Err(_) => names
                .iter()
                .position(|n| n.eq_ignore_ascii_case(part))
                .ok_or_else(|| format!("unknown stem {part:?}: expected one of {} or an index", names.join(", ")))?,
        };
        if index >= names.len() {
            return Err(format!("stem index {index} out of range (0..{})", names.len()));
        }
        if !selected.contains(&index) {
            selected.push(index);
        }
    }
    if selected.is_empty() {
        return Err("no stems selected".into());
    }
    selected.sort_unstable();
    Ok(selected)
}

/// Output directory for one input. Dragged files land next to the source so
/// the explorer window the user dropped from is where the results appear;
/// `--input` keeps the historical cwd-relative `separated/`.
pub fn default_outdir(input: &Path, beside_input: bool) -> PathBuf {
    if beside_input {
        let stem = input
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "output".to_string());
        input.parent().unwrap_or_else(|| Path::new(".")).join(format!("{stem}_separated"))
    } else {
        PathBuf::from("separated")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        DEFAULT_INSTRUMENTS.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn detailed_plan_keeps_selected_stems() {
        let plan = Plan { mode: Mode::Stems, stems: vec![3, 1] };
        let targets = plan.targets(&names());
        assert_eq!(targets.len(), 2, "one file per selected stem");
        assert_eq!(targets[0].file(), "1_drums.wav");
        assert_eq!(targets[1].file(), "3_vocals.wav");
    }

    #[test]
    fn vocals_plan_is_vocal_plus_instrumental_mix() {
        let plan = Plan::vocals(&names());
        let targets = plan.targets(&names());
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].file(), "vocals.wav");
        assert_eq!(targets[0].sources(), &[3]);
        assert_eq!(targets[1].file(), "instrumental.wav");
        assert_eq!(targets[1].sources(), &[0, 1, 2, 4, 5]);
    }

    #[test]
    fn vocals_mode_ignores_stem_selection() {
        let plan = Plan { mode: Mode::Vocals, stems: vec![] };
        assert_eq!(plan.targets(&names()).len(), 2);
    }

    #[test]
    fn stem_spec_accepts_names_indices_and_all() {
        assert_eq!(parse_stems("all", &names()).unwrap(), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(parse_stems("vocals, drums", &names()).unwrap(), vec![1, 3]);
        assert_eq!(parse_stems("5,0", &names()).unwrap(), vec![0, 5]);
        assert_eq!(parse_stems("VOCALS", &names()).unwrap(), vec![3]);
        assert!(parse_stems("kazoo", &names()).is_err());
        assert!(parse_stems("99", &names()).is_err());
    }

    #[test]
    fn mode_aliases_parse() {
        assert_eq!(Mode::parse("detailed").unwrap(), Mode::Stems);
        assert_eq!(Mode::parse("2").unwrap(), Mode::Vocals);
        assert!(Mode::parse("nope").is_err());
    }

    #[test]
    fn default_outdir_is_beside_dragged_input() {
        let out = default_outdir(Path::new("D:/Music/song.mp3"), true);
        assert_eq!(out, PathBuf::from("D:/Music/song_separated"));
        assert_eq!(default_outdir(Path::new("song.mp3"), false), PathBuf::from("separated"));
    }
}
