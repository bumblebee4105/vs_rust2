//! ML Class Launcher — one-click Python environment + classroom IDE.
//!
//! Flow for a student:
//!   1. Start the app. The setup window opens automatically if needed.
//!   2. Pick an interpreter (or let it download a managed Python 3.12
//!      via Miniconda), click "Set up environment".
//!   3. The app creates the environment, installs requirements.txt,
//!      runs a self-test, and then the editor is ready.
//!   4. Browse the project in the file tree, edit code in tabs, press Run.
//!      Saved plots (confusion matrices, ...) appear in the tree and can
//!      be viewed in the app.

mod bootstrap;
mod runner;
mod terminal;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use eframe::egui;

const STARTER_CODE: &str = r#"# Welcome! Press Run (or Ctrl+Enter) to execute this script.
# Files you save (plots, csv, ...) appear in the file tree on the left.
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("My first plot")
plt.savefig("my_first_plot.png")   # <- click it in the file tree to view it!

print("Hello from Python!")
print("NumPy version:", np.__version__)

# PyTorch quick check (if installed):
try:
    import torch
    print("PyTorch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if hasattr(torch.backends, "mps"):
        print("mps (Apple GPU):", torch.backends.mps.is_available())
except ImportError:
    pass

# Later in the lesson, e.g. a confusion matrix:
# from sklearn.metrics import ConfusionMatrixDisplay
# ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
# plt.savefig("confusion_matrix.png")
"#;

/// Folders never shown in the file tree.
const SKIP_DIRS: &[&str] = &[
    "__pycache__", ".git", ".venv", "venv", "env", "node_modules", ".idea",
];
/// Extensions opened in the image viewer.
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg"];
/// Max file sizes the viewer/editor will open.
const MAX_TEXT_BYTES: u64 = 5 * 1024 * 1024;
const MAX_IMAGE_BYTES: u64 = 32 * 1024 * 1024;

/// Messages from background threads to the UI.
enum Msg {
    Log(String),
    SetupDone(Result<(), String>),
    RunLine(String),
    RunDone(i32),
    EnvCheck(bool),
    Candidates(Vec<bootstrap::Candidate>),
}

enum TabKind {
    Code { content: String },
    Image { bytes: Arc<[u8]>, tex: Option<egui::TextureHandle> },
}

/// Which bottom panel is shown.
#[derive(PartialEq, Clone, Copy)]
enum BottomTab {
    Console,
    Terminal,
}

struct Tab {
    path: PathBuf,
    kind: TabKind,
    dirty: bool,
}

impl Tab {
    fn title(&self) -> String {
        let name = self
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if self.dirty {
            format!("{name} •")
        } else {
            name
        }
    }
}

struct App {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,

    project_dir: PathBuf,
    tabs: Vec<Tab>,
    active: Option<usize>,

    picker_open: bool,
    picker_dir: PathBuf,
    new_file_name: String,

    console: String,
    ready: bool,
    setting_up: bool,
    setup_open: bool,
    candidates: Vec<bootstrap::Candidate>,
    /// Index into candidates; candidates.len() = "download managed Python".
    selected: usize,
    running: Option<runner::RunningChild>,
    fit_image: bool,

    bottom_tab: BottomTab,
    term: Option<terminal::Terminal>,
    term_err: Option<String>,
    term_has_focus: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_theme(egui::ThemePreference::Dark);

        let (tx, rx) = channel();
        let project_dir = runner::scripts_dir();
        let _ = std::fs::create_dir_all(&project_dir);

        let mut app = App {
            tx,
            rx,
            project_dir: project_dir.clone(),
            tabs: Vec::new(),
            active: None,
            picker_open: false,
            picker_dir: project_dir.clone(),
            new_file_name: String::new(),
            console: String::new(),
            ready: false,
            setting_up: false,
            setup_open: false,
            candidates: Vec::new(),
            selected: 0,
            running: None,
            fit_image: true,
            bottom_tab: BottomTab::Console,
            term: None,
            term_err: None,
            term_has_focus: false,
        };

        // First run: drop a starter main.py into the project and open it.
        let main_py = project_dir.join("main.py");
        if !main_py.exists() {
            let _ = std::fs::write(&main_py, STARTER_CODE);
        }
        app.open_path(&main_py);

        // Background: environment check + interpreter discovery.
        let tx = app.tx.clone();
        thread::spawn(move || {
            let _ = tx.send(Msg::EnvCheck(runner::env_ready()));
            let _ = tx.send(Msg::Candidates(bootstrap::discover_interpreters()));
        });

        app
    }

    fn log(&mut self, line: &str) {
        self.console.push_str(line);
        self.console.push('\n');
        const MAX: usize = 200_000;
        if self.console.len() > MAX {
            let cut = self.console.len() - MAX / 2;
            let boundary = self.console.ceil_char_boundary(cut);
            self.console.drain(..boundary);
        }
    }

    // ------------------------------------------------------------------
    // Files & tabs
    // ------------------------------------------------------------------

    fn open_path(&mut self, path: &Path) {
        // Already open? Just activate it.
        if let Some(i) = self.tabs.iter().position(|t| t.path == path) {
            self.active = Some(i);
            return;
        }
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        let kind = if IMAGE_EXTS.contains(&ext.as_str()) {
            if size > MAX_IMAGE_BYTES {
                self.log(&format!("Image too large to view: {}", path.display()));
                return;
            }
            match std::fs::read(path) {
                Ok(bytes) => TabKind::Image {
                    bytes: Arc::from(bytes.into_boxed_slice()),
                    tex: None,
                },
                Err(e) => {
                    self.log(&format!("Could not open {}: {e}", path.display()));
                    return;
                }
            }
        } else {
            if size > MAX_TEXT_BYTES {
                self.log(&format!("File too large to edit: {}", path.display()));
                return;
            }
            match std::fs::read_to_string(path) {
                Ok(content) => TabKind::Code { content },
                Err(_) => {
                    self.log(&format!("Not a text file, not opening: {}", path.display()));
                    return;
                }
            }
        };
        self.tabs.push(Tab {
            path: path.to_path_buf(),
            kind,
            dirty: false,
        });
        self.active = Some(self.tabs.len() - 1);
    }

    fn save_tab(&mut self, idx: usize) {
        let mut err = None;
        if let Some(tab) = self.tabs.get_mut(idx) {
            if let TabKind::Code { content } = &tab.kind {
                if tab.dirty {
                    match std::fs::write(&tab.path, content) {
                        Ok(()) => tab.dirty = false,
                        Err(e) => err = Some(format!("Could not save {}: {e}", tab.path.display())),
                    }
                }
            }
        }
        if let Some(e) = err {
            self.log(&e);
        }
    }

    fn close_tab(&mut self, idx: usize) {
        self.save_tab(idx); // never lose student work
        self.tabs.remove(idx);
        self.active = match self.active {
            Some(a) if a == idx => None,
            Some(a) if a > idx => Some(a - 1),
            other => other,
        };
        if self.active.is_none() && !self.tabs.is_empty() {
            self.active = Some(self.tabs.len() - 1);
        }
    }

    fn create_file(&mut self) {
        let name = self.new_file_name.trim();
        if name.is_empty() {
            return;
        }
        let path = self.project_dir.join(name);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if !path.exists() {
            let _ = std::fs::write(&path, "");
        }
        self.new_file_name.clear();
        self.open_path(&path);
    }

    fn set_project_dir(&mut self, dir: PathBuf) {
        self.project_dir = dir.clone();
        self.picker_dir = dir;
        self.tabs.clear();
        self.active = None;
        // Restart the terminal later so it starts in the new project folder.
        if let Some(t) = &mut self.term {
            t.kill();
        }
        self.term = None;
        self.log(&format!("Project folder: {}", self.project_dir.display()));
        let main_py = self.project_dir.join("main.py");
        if main_py.exists() {
            self.open_path(&main_py);
        }
    }

    // ------------------------------------------------------------------
    // File tree
    // ------------------------------------------------------------------

    /// Render a directory; returns a file path when one is clicked.
    fn tree_ui(&mut self, ui: &mut egui::Ui, dir: &Path, depth: usize) -> Option<PathBuf> {
        if depth > 12 {
            return None;
        }
        let mut entries: Vec<(bool, PathBuf)> = std::fs::read_dir(dir)
            .map(|rd| {
                rd.filter_map(Result::ok)
                    .map(|e| (e.path().is_dir(), e.path()))
                    .collect()
            })
            .unwrap_or_default();
        entries.sort_by(|a, b| (b.0, a.1.file_name().map(|n| n.to_owned()))
            .cmp(&(a.0, b.1.file_name().map(|n| n.to_owned()))));

        let mut clicked = None;
        for (is_dir, path) in entries {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            if is_dir {
                if SKIP_DIRS.contains(&name.as_ref()) {
                    continue;
                }
                let resp = egui::CollapsingHeader::new(format!("📁 {name}"))
                    .id_salt(&path)
                    .show(ui, |ui| self.tree_ui(ui, &path, depth + 1));
                if let Some(p) = resp.body_returned.flatten() {
                    clicked = Some(p);
                }
            } else {
                let is_active = self
                    .active
                    .and_then(|a| self.tabs.get(a))
                    .is_some_and(|t| t.path == path);
                let icon = if IMAGE_EXTS.contains(
                    &path
                        .extension()
                        .map(|e| e.to_string_lossy().to_lowercase())
                        .unwrap_or_default()
                        .as_str(),
                ) {
                    "🖼"
                } else {
                    "📄"
                };
                if ui
                    .selectable_label(is_active, format!("{icon} {name}"))
                    .clicked()
                {
                    clicked = Some(path);
                }
            }
        }
        clicked
    }

    // ------------------------------------------------------------------
    // Setup & run (unchanged pipeline)
    // ------------------------------------------------------------------

    fn managed_label() -> String {
        format!(
            "Download managed Python {} (Miniconda)",
            bootstrap::MANAGED_PYTHON_SERIES
        )
    }

    fn start_setup(&mut self) {
        if self.setting_up {
            return;
        }
        self.setting_up = true;
        let choice = self.candidates.get(self.selected).map(|c| c.path.clone());
        let tx = self.tx.clone();
        thread::spawn(move || {
            let tx_log = tx.clone();
            let log = move |s: String| {
                let _ = tx_log.send(Msg::Log(s));
            };
            let result = (|| -> Result<(), String> {
                match &choice {
                    Some(path) => {
                        log(format!("Using interpreter: {}", path.display()));
                        runner::create_venv(path, &log)?;
                    }
                    None => {
                        log("No interpreter selected — using managed Miniconda.".into());
                        runner::create_conda_env(&log)?;
                    }
                }
                runner::install_packages(&log)?;
                runner::self_test(&log)?;
                Ok(())
            })();
            let _ = tx.send(Msg::SetupDone(result));
        });
    }

    fn start_run(&mut self) {
        if self.running.is_some() || !self.ready {
            return;
        }
        let Some(active) = self.active else { return };
        let is_code = matches!(self.tabs[active].kind, TabKind::Code { .. });
        if !is_code {
            self.log("Open a .py file to run it.");
            return;
        }
        self.save_tab(active);
        let path = self.tabs[active].path.clone();
        self.log(&format!("\n--- running {} ---", path.display()));
        self.bottom_tab = BottomTab::Console; // show the output where it lands

        let (tx_lines, rx_lines) = channel::<String>();
        let cwd = self.project_dir.clone();
        match runner::run_script(&path, &cwd, tx_lines) {
            Ok((handle, waiter)) => {
                self.running = Some(handle);
                let tx = self.tx.clone();
                thread::spawn(move || {
                    while let Ok(line) = rx_lines.recv() {
                        if tx.send(Msg::RunLine(line)).is_err() {
                            return;
                        }
                    }
                    let code = waiter.join().unwrap_or(-1);
                    let _ = tx.send(Msg::RunDone(code));
                });
            }
            Err(e) => self.log(&e),
        }
    }

    fn process_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Log(line) => self.log(&line),
                Msg::RunLine(line) => self.log(&line),
                Msg::RunDone(code) => {
                    self.log(&format!("--- finished (exit code {code}) ---\n"));
                    self.running = None;
                }
                Msg::SetupDone(Ok(())) => {
                    self.setting_up = false;
                    self.ready = true;
                    self.setup_open = false;
                    self.log("Setup complete — the editor is ready, press Run!");
                }
                Msg::SetupDone(Err(e)) => {
                    self.setting_up = false;
                    self.setup_open = true;
                    self.log(&format!("Setup failed: {e}"));
                }
                Msg::EnvCheck(ready) => {
                    self.ready = ready;
                    if !ready {
                        self.setup_open = true;
                    }
                }
                Msg::Candidates(list) => {
                    self.candidates = list;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Windows
    // ------------------------------------------------------------------

    fn setup_window(&mut self, ctx: &egui::Context) {
        let mut open = self.setup_open;
        egui::Window::new("Environment setup")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label("Choose which Python to use for the class environment:");
                ui.add_space(4.0);

                let managed = Self::managed_label();
                let current = self
                    .candidates
                    .get(self.selected)
                    .map(|c| c.label())
                    .unwrap_or_else(|| managed.clone());
                egui::ComboBox::from_id_salt("interpreter")
                    .selected_text(current)
                    .width(520.0)
                    .show_ui(ui, |ui| {
                        for (i, c) in self.candidates.iter().enumerate() {
                            ui.selectable_value(&mut self.selected, i, c.label());
                        }
                        ui.selectable_value(&mut self.selected, self.candidates.len(), &managed);
                    });
                if self.candidates.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "No Python found on this machine — a managed one will be downloaded.",
                        )
                        .weak(),
                    );
                }

                ui.add_space(8.0);
                ui.label(egui::RichText::new(runner::gpu_note()).weak());
                ui.label(
                    egui::RichText::new(format!(
                        "Packages: {}",
                        runner::requirements().join(", ")
                    ))
                    .weak(),
                );
                ui.add_space(8.0);

                let label = if self.setting_up {
                    "Setting up... (watch the console)"
                } else {
                    "Set up environment"
                };
                if ui
                    .add_enabled(!self.setting_up, egui::Button::new(label))
                    .clicked()
                {
                    self.start_setup();
                }
                ui.label(
                    egui::RichText::new(
                        "Creates the environment, installs the packages, then runs a self-test.",
                    )
                    .weak()
                    .small(),
                );
            });
        self.setup_open = open;
    }

    fn folder_picker(&mut self, ctx: &egui::Context) {
        let mut open = self.picker_open;
        let mut chosen: Option<PathBuf> = None;
        egui::Window::new("Open project folder")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size([520.0, 380.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                // Direct path entry
                let mut text = self.picker_dir.display().to_string();
                ui.horizontal(|ui| {
                    ui.label("Path:");
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut text).desired_width(f32::INFINITY),
                    );
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        let p = PathBuf::from(text.trim());
                        if p.is_dir() {
                            self.picker_dir = p;
                        }
                    }
                });
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    if ui.button("⬆ Up").clicked() {
                        if let Some(parent) = self.picker_dir.parent() {
                            self.picker_dir = parent.to_path_buf();
                        }
                    }
                    if ui.button("Home").clicked() {
                        if let Some(home) = dirs::home_dir() {
                            self.picker_dir = home;
                        }
                    }
                    if ui.button("Default scripts folder").clicked() {
                        self.picker_dir = runner::scripts_dir();
                    }
                });
                ui.separator();

                // Subfolder list
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&self.picker_dir)
                        .map(|rd| {
                            rd.filter_map(Result::ok)
                                .map(|e| e.path())
                                .filter(|p| p.is_dir())
                                .filter(|p| {
                                    !p.file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy()
                                        .starts_with('.')
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    dirs.sort();
                    if dirs.is_empty() {
                        ui.weak("(no subfolders)");
                    }
                    for d in dirs {
                        let name = d.file_name().unwrap_or_default().to_string_lossy();
                        if ui.selectable_label(false, format!("📁 {name}")).clicked() {
                            self.picker_dir = d;
                        }
                    }
                });

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(format!("Selected: {}", self.picker_dir.display()));
                });
                if ui.button("Open this folder").clicked() {
                    chosen = Some(self.picker_dir.clone());
                }
            });
        self.picker_open = open;
        if let Some(dir) = chosen {
            self.set_project_dir(dir);
            self.picker_open = false;
        }
    }

    // ------------------------------------------------------------------
    // Integrated terminal
    // ------------------------------------------------------------------

    fn terminal_ui(&mut self, ui: &mut egui::Ui) {
        // Spawn lazily on first visit.
        if self.term.is_none() && self.term_err.is_none() {
            let bins = runner::env_bin_dirs();
            let cwd = self.project_dir.clone();
            match terminal::Terminal::spawn(&bins, &cwd, 24, 110, ui.ctx().clone()) {
                Ok(t) => self.term = Some(t),
                Err(e) => self.term_err = Some(e),
            }
        }
        if let Some(e) = self.term_err.clone() {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("Terminal failed to start: {e}"));
            if ui.button("Retry").clicked() {
                self.term_err = None;
            }
            return;
        }
        let Some(term) = &mut self.term else { return };
        term.poll();

        let font = egui::FontId::monospace(13.0);
        let (char_w, line_h) = ui.fonts(|f| {
            let w = f
                .layout_no_wrap(
                    "M".repeat(32),
                    font.clone(),
                    egui::Color32::WHITE,
                )
                .size()
                .x
                / 32.0;
            (w.max(1.0), f.row_height(&font))
        });

        let avail = ui.available_size_before_wrap();
        let cols = (avail.x / char_w).floor().clamp(20.0, 500.0) as u16;
        let rows = (avail.y / line_h).floor().clamp(4.0, 200.0) as u16;
        term.resize(rows, cols);

        let (resp, painter) = ui.allocate_painter(avail, egui::Sense::click());
        let rect = resp.rect;
        painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(14, 14, 18));

        if resp.clicked() {
            resp.request_focus();
        }
        self.term_has_focus = resp.has_focus();

        // Scrollback with the mouse wheel.
        if resp.hovered() {
            let dy = ui.input(|i| i.raw_scroll_delta.y);
            if dy.abs() > 0.1 {
                let lines = (dy / line_h).round() as isize;
                term.scroll = (term.scroll as isize + lines).clamp(0, 4000) as usize;
            }
        }

        // Keyboard input → PTY.
        if resp.has_focus() && term.alive {
            let events = ui.input(|i| i.events.clone());
            let mut buf: Vec<u8> = Vec::new();
            for ev in &events {
                match ev {
                    egui::Event::Text(t) => {
                        buf.extend_from_slice(t.replace('\r', "").as_bytes());
                    }
                    egui::Event::Paste(t) => {
                        buf.extend_from_slice(t.replace("\r\n", "\n").as_bytes());
                    }
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => {
                        if let Some(b) = term_key_bytes(*key, modifiers) {
                            buf.extend_from_slice(&b);
                        }
                    }
                    _ => {}
                }
            }
            if !buf.is_empty() {
                term.send(&buf);
            }
        }

        // Render the vt100 screen cell-grid with colors.
        let default_fg = egui::Color32::from_rgb(212, 212, 216);
        let screen = term.screen();
        let (cursor_row, cursor_col) = screen.cursor_position();
        for row in 0..rows {
            let y = rect.min.y + row as f32 * line_h;
            if y + line_h > rect.max.y + 0.5 {
                break;
            }
            let mut col = 0u16;
            while col < cols {
                // Collect a run of cells with the same style.
                let start = col;
                let (mut fg, mut bg, mut bold) = (None, None, false);
                let mut text = String::new();
                while col < cols {
                    let (c_fg, c_bg, c_bold, s) = match screen.cell(row, col) {
                        Some(cell) => (
                            term_color(cell.fgcolor(), cell.bold()),
                            term_color(cell.bgcolor(), false),
                            cell.bold(),
                            cell.contents(),
                        ),
                        None => (None, None, false, String::new()),
                    };
                    if col == start {
                        fg = c_fg;
                        bg = c_bg;
                        bold = c_bold;
                        text.push_str(if s.is_empty() { " " } else { &s });
                        col += 1;
                    } else if c_fg == fg && c_bg == bg && c_bold == bold {
                        text.push_str(if s.is_empty() { " " } else { &s });
                        col += 1;
                    } else {
                        break;
                    }
                }
                let x = rect.min.x + start as f32 * char_w;
                if x >= rect.max.x {
                    continue;
                }
                if let Some(bg) = bg {
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(x, y),
                            egui::vec2(char_w * (col - start) as f32, line_h),
                        ),
                        0.0,
                        bg,
                    );
                }
                painter.text(
                    egui::pos2(x, y),
                    egui::Align2::LEFT_TOP,
                    text,
                    font.clone(),
                    fg.unwrap_or(default_fg),
                );
            }
        }

        // Cursor.
        if term.alive && term.scroll == 0 && resp.has_focus() {
            let cx = rect.min.x + cursor_col as f32 * char_w;
            let cy = rect.min.y + cursor_row as f32 * line_h;
            if cx + char_w <= rect.max.x && cy + line_h <= rect.max.y {
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(cx, cy + line_h - 2.0),
                        egui::vec2(char_w, 2.0),
                    ),
                    0.0,
                    egui::Color32::from_rgb(180, 180, 190),
                );
            }
        }

        if !term.alive {
            painter.text(
                egui::pos2(rect.min.x + 6.0, rect.max.y - 6.0),
                egui::Align2::LEFT_BOTTOM,
                "[shell exited — press Restart shell above]",
                egui::FontId::monospace(12.0),
                egui::Color32::YELLOW,
            );
        }
    }
}

/// Translate an egui key press into the bytes a terminal expects.
fn term_key_bytes(key: egui::Key, mods: &egui::Modifiers) -> Option<Vec<u8>> {
    use egui::Key::*;
    // Ctrl+letter → control codes (SIGINT etc.).
    if mods.ctrl && !mods.shift && !mods.alt {
        let code = match key {
            A => 0x01,
            B => 0x02,
            C => 0x03,
            D => 0x04,
            E => 0x05,
            F => 0x06,
            G => 0x07,
            H => 0x08,
            K => 0x0b,
            L => 0x0c,
            N => 0x0e,
            P => 0x10,
            R => 0x12,
            T => 0x14,
            U => 0x15,
            W => 0x17,
            Z => 0x1a,
            _ => return None,
        };
        return Some(vec![code]);
    }
    let bytes: &[u8] = match key {
        Enter => b"\r",
        Backspace => b"\x7f",
        Tab => b"\t",
        Escape => b"\x1b",
        ArrowUp => b"\x1b[A",
        ArrowDown => b"\x1b[B",
        ArrowRight => b"\x1b[C",
        ArrowLeft => b"\x1b[D",
        Home => b"\x1b[H",
        End => b"\x1b[F",
        Delete => b"\x1b[3~",
        PageUp => b"\x1b[5~",
        PageDown => b"\x1b[6~",
        Insert => b"\x1b[2~",
        _ => return None,
    };
    Some(bytes.to_vec())
}

/// Map a vt100 color to an egui color (None = terminal default).
fn term_color(c: vt100::Color, bold: bool) -> Option<egui::Color32> {
    const ANSI: [egui::Color32; 8] = [
        egui::Color32::from_rgb(0, 0, 0),
        egui::Color32::from_rgb(205, 49, 49),
        egui::Color32::from_rgb(13, 188, 121),
        egui::Color32::from_rgb(229, 229, 16),
        egui::Color32::from_rgb(36, 114, 200),
        egui::Color32::from_rgb(188, 63, 188),
        egui::Color32::from_rgb(17, 168, 205),
        egui::Color32::from_rgb(229, 229, 229),
    ];
    const BRIGHT: [egui::Color32; 8] = [
        egui::Color32::from_rgb(102, 102, 102),
        egui::Color32::from_rgb(241, 76, 76),
        egui::Color32::from_rgb(35, 209, 139),
        egui::Color32::from_rgb(245, 245, 67),
        egui::Color32::from_rgb(59, 142, 234),
        egui::Color32::from_rgb(214, 112, 214),
        egui::Color32::from_rgb(41, 184, 219),
        egui::Color32::from_rgb(255, 255, 255),
    ];
    match c {
        vt100::Color::Default => None,
        vt100::Color::Rgb(r, g, b) => Some(egui::Color32::from_rgb(r, g, b)),
        vt100::Color::Idx(i) => Some(match i {
            0..=7 if bold => BRIGHT[i as usize],
            0..=7 => ANSI[i as usize],
            8..=15 => BRIGHT[(i - 8) as usize],
            16..=231 => {
                let i = i - 16;
                let conv = |x: u8| if x == 0 { 0 } else { 55 + 40 * x };
                egui::Color32::from_rgb(conv(i / 36), conv((i / 6) % 6), conv(i % 6))
            }
            _ => {
                let g = 8 + (i.saturating_sub(232)) * 10;
                egui::Color32::from_rgb(g, g, g)
            }
        }),
    }
}

fn lang_for(path: &Path) -> &'static str {    match path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .as_deref()
    {
        Some("py") => "py",
        Some("rs") => "rust",
        Some("md") => "md",
        Some("js") => "js",
        Some("ts") => "ts",
        Some("json") => "json",
        Some("toml") => "toml",
        _ => "",
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_messages();
        if self.setting_up || self.running.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        // Global editor shortcuts are suspended while typing in the terminal.
        if !self.term_has_focus {
            if ctx.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.command) {
                self.start_run();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::S) && i.modifiers.command) {
                if let Some(a) = self.active {
                    self.save_tab(a);
                }
            }
        }

        self.setup_window(ctx);
        self.folder_picker(ctx);

        // ------------------------------------------------------------------
        // Toolbar
        // ------------------------------------------------------------------
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.setting_up, egui::Button::new("Setup..."))
                    .on_hover_text("Environment setup (interpreter choice, reinstall)")
                    .clicked()
                {
                    self.setup_open = true;
                }
                if ui
                    .button("Open folder...")
                    .on_hover_text("Open a project folder")
                    .clicked()
                {
                    self.picker_dir = self.project_dir.clone();
                    self.picker_open = true;
                }

                ui.separator();

                let can_run = self.ready
                    && self.running.is_none()
                    && !self.setting_up
                    && self
                        .active
                        .and_then(|a| self.tabs.get(a))
                        .is_some_and(|t| matches!(t.kind, TabKind::Code { .. }));
                if ui
                    .add_enabled(can_run, egui::Button::new("▶ Run"))
                    .on_hover_text("Run the active file (Ctrl+Enter)")
                    .clicked()
                {
                    self.start_run();
                }
                if ui
                    .add_enabled(self.running.is_some(), egui::Button::new("■ Stop"))
                    .clicked()
                {
                    if let Some(handle) = &self.running {
                        runner::stop_script(handle);
                    }
                }

                ui.separator();
                let status_text = if self.running.is_some() {
                    "running..."
                } else if self.setting_up {
                    "setting up environment..."
                } else if self.ready {
                    "environment ready"
                } else {
                    "setup needed"
                };
                ui.label(status_text);
            });
            ui.add_space(4.0);
        });

        // ------------------------------------------------------------------
        // File tree
        // ------------------------------------------------------------------
        egui::SidePanel::left("file_tree")
            .resizable(true)
            .default_width(210.0)
            .min_width(140.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.heading(
                        self.project_dir
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "project".into()),
                    );
                });
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_file_name)
                            .hint_text("new file...")
                            .desired_width(120.0),
                    );
                    if ui.button("＋").on_hover_text("Create file").clicked() {
                        self.create_file();
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let dir = self.project_dir.clone();
                        if let Some(path) = self.tree_ui(ui, &dir, 0) {
                            self.open_path(&path);
                        }
                    });
            });

        // ------------------------------------------------------------------
        // Bottom panel: Console | Terminal
        // ------------------------------------------------------------------
        egui::TopBottomPanel::bottom("console")
            .resizable(true)
            .default_height(220.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.bottom_tab, BottomTab::Console, "Console");
                    let term_label = if self
                        .term
                        .as_ref()
                        .is_some_and(|t| t.unread_output && self.bottom_tab != BottomTab::Terminal)
                    {
                        "Terminal •"
                    } else {
                        "Terminal"
                    };
                    ui.selectable_value(&mut self.bottom_tab, BottomTab::Terminal, term_label);
                    ui.separator();
                    match self.bottom_tab {
                        BottomTab::Console => {
                            if ui.button("Clear").clicked() {
                                self.console.clear();
                            }
                            ui.separator();
                            ui.label("Project:");
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(self.project_dir.display().to_string())
                                        .monospace()
                                        .weak(),
                                )
                                .selectable(true),
                            );
                        }
                        BottomTab::Terminal => {
                            if ui
                                .button("Restart shell")
                                .on_hover_text(
                                    "Kill the shell and start a fresh one in the project folder",
                                )
                                .clicked()
                            {
                                if let Some(t) = &mut self.term {
                                    t.kill();
                                }
                                self.term = None;
                                self.term_err = None;
                            }
                            ui.separator();
                            let env_note = if runner::env_bin_dirs().is_empty() {
                                "no managed environment yet — run Setup to give this shell the class Python"
                            } else {
                                "python/pip in this shell = the managed environment"
                            };
                            ui.label(egui::RichText::new(env_note).weak().small());
                        }
                    }
                });
                ui.separator();
                match self.bottom_tab {
                    BottomTab::Console => {
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(&self.console).monospace(),
                                    )
                                    .selectable(true)
                                    .wrap(),
                                );
                            });
                    }
                    BottomTab::Terminal => self.terminal_ui(ui),
                }
            });

        // ------------------------------------------------------------------
        // Editor / viewer with tabs
        // ------------------------------------------------------------------
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.tabs.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.weak("Open a file from the tree, or create one on the left.");
                });
                return;
            }

            // Tab bar
            let mut close_idx = None;
            ui.horizontal_wrapped(|ui| {
                for (i, tab) in self.tabs.iter().enumerate() {
                    let selected = self.active == Some(i);
                    if ui.selectable_label(selected, tab.title()).clicked() {
                        self.active = Some(i);
                    }
                    if ui.small_button("✕").clicked() {
                        close_idx = Some(i);
                    }
                    ui.separator();
                }
            });
            if let Some(i) = close_idx {
                self.close_tab(i);
            }
            ui.separator();

            let Some(active) = self.active.filter(|&a| a < self.tabs.len()) else {
                return;
            };

            let active_path = self.tabs[active].path.clone();
            let mut deferred_log: Option<String> = None;

            match &mut self.tabs[active].kind {
                TabKind::Code { content } => {
                    let path = active_path.clone();
                    let theme = egui_extras::syntax_highlighting::CodeTheme::from_memory(
                        ui.ctx(),
                        ui.style(),
                    );
                    let lang = lang_for(&path);
                    let mut layouter = |ui: &egui::Ui, code: &str, wrap_width: f32| {
                        let mut job = if lang.is_empty() {
                            egui::text::LayoutJob::simple(
                                code.to_string(),
                                egui::FontId::monospace(14.0),
                                ui.style().visuals.text_color(),
                                wrap_width,
                            )
                        } else {
                            egui_extras::syntax_highlighting::highlight(
                                ui.ctx(),
                                ui.style(),
                                &theme,
                                code,
                                lang,
                            )
                        };
                        job.wrap.max_width = wrap_width;
                        ui.fonts(|f| f.layout_job(job))
                    };

                    let resp = egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(content)
                                .font(egui::TextStyle::Monospace)
                                .code_editor()
                                .desired_width(f32::INFINITY)
                                .layouter(&mut layouter),
                        )
                    });
                    if resp.inner.changed() {
                        self.tabs[active].dirty = true;
                    }
                }
                TabKind::Image { bytes, tex } => {
                    let path_label = active_path.display().to_string();
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.fit_image, "Fit to width");
                        ui.label(egui::RichText::new(&path_label).weak().small());
                    });
                    if tex.is_none() {
                        match image::load_from_memory(bytes) {
                            Ok(img) => {
                                let rgba = img.to_rgba8();
                                let size = [rgba.width() as usize, rgba.height() as usize];
                                let color =
                                    egui::ColorImage::from_rgba_unmultiplied(size, &rgba);
                                *tex = Some(ui.ctx().load_texture(
                                    &path_label,
                                    color,
                                    egui::TextureOptions::LINEAR,
                                ));
                            }
                            Err(e) => {
                                deferred_log = Some(format!("Could not decode image: {e}"));
                            }
                        }
                    }
                    if let Some(texture) = tex {
                        let avail = ui.available_size();
                        let native = texture.size_vec2();
                        let size = if self.fit_image && native.x > 0.0 {
                            let scale = (avail.x / native.x).min(4.0);
                            native * scale
                        } else {
                            native
                        };
                        egui::ScrollArea::both().show(ui, |ui| {
                            ui.image((texture.id(), size));
                        });
                    }
                }
            }
            if let Some(msg) = deferred_log {
                self.log(&msg);
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 780.0])
            .with_min_inner_size([800.0, 500.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ML Class Launcher",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
