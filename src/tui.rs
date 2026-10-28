//! Interactive output picker for drag-and-drop launches.
//!
//! Dropping an audio file onto `lbrr` on Windows (and `./lbrr song.mp3` on
//! Linux) passes the path as a positional argument. The first interactive
//! screen asks what to write before the GPU pipeline starts:
//!
//!   * detailed stems  — one WAV per instrument
//!   * vocals / instrumental — two WAVs, the second is the sum of the rest
//!
//! and, in detailed mode, which of the stems to keep. A single vertical
//! cursor covers both sections plus the start row, so arrow keys + Enter are
//! enough; Space toggles tracks and `a`/`n` select all / none.
//!
//! The picker is skipped whenever stdin/stdout is not a terminal (CI, pipes),
//! which is why it never has to invent a "quiet" rendering mode.

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

use crossterm::cursor;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{execute, queue};

use crate::plan::{Mode, Plan};

/// True when a full-screen picker can be shown.
pub fn available() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

/// Windows consoles default to an OEM code page; the picker prints Chinese
/// labels, so ask for UTF-8 before the first frame.
#[cfg(windows)]
pub fn force_utf8_console() {
    unsafe extern "system" {
        fn SetConsoleOutputCP(code_page: u32) -> i32;
        fn SetConsoleCP(code_page: u32) -> i32;
    }
    // SAFETY: both calls only change console state owned by this process.
    unsafe {
        SetConsoleOutputCP(65001);
        SetConsoleCP(65001);
    }
}

#[cfg(not(windows))]
pub fn force_utf8_console() {}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Row {
    Mode(Mode),
    Track(usize),
    Start,
}

struct Ui {
    mode: Mode,
    selected: Vec<bool>,
    cursor: usize,
    note: String,
}

impl Ui {
    fn new(stem_count: usize) -> Ui {
        Ui { mode: Mode::Stems, selected: vec![true; stem_count], cursor: 0, note: String::new() }
    }

    fn rows(&self) -> Vec<Row> {
        let mut rows = vec![Row::Mode(Mode::Stems), Row::Mode(Mode::Vocals)];
        if self.mode == Mode::Stems {
            rows.extend((0..self.selected.len()).map(Row::Track));
        }
        rows.push(Row::Start);
        rows
    }

    fn shrink_cursor(&mut self) {
        let rows = self.rows().len();
        if self.cursor >= rows {
            self.cursor = rows - 1;
        }
    }

    fn plan(&self) -> Plan {
        Plan {
            mode: self.mode,
            stems: (0..self.selected.len()).filter(|&i| self.selected[i]).collect(),
        }
    }
}

/// Restores the terminal even when the picker unwinds (Ctrl+C, panic).
struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        let mut out = io::stdout();
        let _ = execute!(out, cursor::Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

/// Show the picker. `Ok(None)` means the user cancelled.
pub fn choose(inputs: &[PathBuf], outdir: &Path, instruments: &[String]) -> io::Result<Option<Plan>> {
    if !available() {
        return Ok(None);
    }
    force_utf8_console();
    let names = Plan::stem_names(instruments);
    let mut ui = Ui::new(names.len());
    let mut stdout = io::stdout();

    terminal::enable_raw_mode()?;
    {
        execute!(stdout, EnterAlternateScreen, cursor::Hide)?;
        let _guard = Guard;
        loop {
            draw(&mut stdout, &ui, inputs, outdir, &names)?;
            stdout.flush()?;

            let Event::Key(key) = event::read()? else { continue };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            let rows = ui.rows();
            let row = rows.get(ui.cursor).copied();
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => return Ok(None),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(None),
                KeyCode::Up | KeyCode::Char('k') => ui.cursor = ui.cursor.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => {
                    if ui.cursor + 1 < rows.len() {
                        ui.cursor += 1;
                    }
                }
                KeyCode::Home => ui.cursor = 0,
                KeyCode::End => ui.cursor = rows.len().saturating_sub(1),
                KeyCode::Left => {
                    ui.mode = Mode::Stems;
                    ui.shrink_cursor();
                    ui.note.clear();
                }
                KeyCode::Right => {
                    ui.mode = Mode::Vocals;
                    ui.shrink_cursor();
                    ui.note.clear();
                }
                KeyCode::Tab => {
                    // Jump between the two sections and the start row.
                    ui.cursor = match row {
                        Some(Row::Mode(_)) => 2.min(ui.rows().len() - 1),
                        Some(Row::Track(_)) => ui.rows().len() - 1,
                        _ => 0,
                    };
                }
                KeyCode::Char('a') if ui.mode == Mode::Stems => {
                    ui.selected.iter_mut().for_each(|s| *s = true);
                    ui.note.clear();
                }
                KeyCode::Char('n') if ui.mode == Mode::Stems => {
                    ui.selected.iter_mut().for_each(|s| *s = false);
                }
                KeyCode::Char(' ') | KeyCode::Enter => match row {
                    Some(Row::Mode(mode)) => {
                        ui.mode = mode;
                        ui.shrink_cursor();
                        ui.note.clear();
                    }
                    Some(Row::Track(index)) => ui.selected[index] = !ui.selected[index],
                    Some(Row::Start) => {
                        if ui.mode == Mode::Stems && !ui.selected.iter().any(|&s| s) {
                            ui.note = "至少选择一个音轨（a 全选，n 全不选）".to_string();
                        } else {
                            return Ok(Some(ui.plan()));
                        }
                    }
                    None => {}
                },
                _ => {}
            }
        }
    }
}

fn draw(
    out: &mut impl Write,
    ui: &Ui,
    inputs: &[PathBuf],
    outdir: &Path,
    names: &[String],
) -> io::Result<()> {
    let rows = ui.rows();
    queue!(
        out,
        cursor::MoveTo(0, 0),
        Clear(ClearType::All),
        SetForegroundColor(Color::Cyan),
        SetAttribute(Attribute::Bold),
        Print("lbrr · BS-RoFormer 音源分离"),
        SetAttribute(Attribute::Reset),
        ResetColor,
        Print("\r\n")
    )?;

    queue!(out, SetForegroundColor(Color::DarkGrey), Print("─".repeat(56)), ResetColor, Print("\r\n"))?;
    queue!(out, SetAttribute(Attribute::Bold), Print("输入"), SetAttribute(Attribute::Reset), Print("\r\n"))?;
    for (i, input) in inputs.iter().take(4).enumerate() {
        queue!(out, Print(format!("  {}. {}\r\n", i + 1, input.display())))?;
    }
    if inputs.len() > 4 {
        queue!(out, Print(format!("  … 共 {} 个文件\r\n", inputs.len())))?;
    }
    if inputs.len() == 1 {
        queue!(out, Print(format!("  输出目录  {}\r\n", outdir.display())))?;
    } else {
        queue!(out, Print("  输出目录  每个文件旁边的 <文件名>_separated/\r\n"))?;
    }

    queue!(out, Print("\r\n"), SetAttribute(Attribute::Bold), Print("输出模式"), SetAttribute(Attribute::Reset), Print("\r\n"))?;
    for (index, mode) in [Mode::Stems, Mode::Vocals].into_iter().enumerate() {
        let focused = ui.cursor == index;
        let marker = if ui.mode == mode { "(•)" } else { "( )" };
        let detail = match mode {
            Mode::Stems => format!("详细音轨 — 每个 stem 一个文件（{} 个）", names.len()),
            Mode::Vocals => "人声 / 伴奏 — vocals + instrumental 两个文件".to_string(),
        };
        cursor_line(out, focused, format!("   {marker} {detail}"))?;
    }

    queue!(out, Print("\r\n"), SetAttribute(Attribute::Bold), Print("输出音轨"), SetAttribute(Attribute::Reset), Print("\r\n"))?;
    if ui.mode == Mode::Stems {
        for (index, name) in names.iter().enumerate() {
            let selected = ui.selected.get(index).copied().unwrap_or(false);
            let box_mark = if selected { "[x]" } else { "[ ]" };
            let focused = rows.get(ui.cursor) == Some(&Row::Track(index));
            cursor_line(out, focused, format!("   {box_mark} {index}  {name}"))?;
        }
    } else {
        let v = crate::plan::vocals_index(names);
        queue!(out, Print(format!("   人声   {}\r\n", names.get(v).map(|s| s.as_str()).unwrap_or("vocals"))))?;
        queue!(out, Print(format!("   伴奏   其余 {} 个 stem 混合\r\n", names.len().saturating_sub(1))))?;
    }

    queue!(out, Print("\r\n"))?;
    let start_focused = rows.get(ui.cursor) == Some(&Row::Start);
    cursor_line(out, start_focused, "   ▶ 开始分离".to_string())?;

    if !ui.note.is_empty() {
        queue!(out, SetForegroundColor(Color::Yellow), Print(format!("   {}\r\n", ui.note)), ResetColor)?;
    }
    queue!(
        out,
        Print("\r\n"),
        SetForegroundColor(Color::DarkGrey),
        Print("   ↑↓/jk 移动  ←→ 切换模式  Enter/空格 选择/开始  a 全选  n 全不选  Esc 取消\r\n"),
        ResetColor
    )?;
    Ok(())
}

fn cursor_line(out: &mut impl Write, focused: bool, line: String) -> io::Result<()> {
    if focused {
        queue!(
            out,
            SetForegroundColor(Color::Black),
            SetAttribute(Attribute::Reverse),
            Print(format!("{line} ")),
            SetAttribute(Attribute::Reset),
            ResetColor,
            Print("\r\n")
        )
    } else {
        queue!(out, Print(format!("{line}\r\n")))
    }
}

/// Keep a console window opened by Explorer on screen after the run, so a
/// drag-and-drop user can read the result instead of watching it vanish.
pub fn pause() {
    if !io::stdin().is_terminal() {
        return;
    }
    force_utf8_console();
    print!("\r\n按 Enter 关闭窗口…");
    io::stdout().flush().ok();
    let mut line = String::new();
    io::stdin().read_line(&mut line).ok();
}
