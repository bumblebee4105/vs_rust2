//! ML Class Launcher — one-click Python environment + code runner.
//!
//! Flow for a student:
//!   1. Start the app. The setup window opens automatically if needed.
//!   2. Pick an interpreter (or let it download a managed Python 3.12
//!      via Miniconda), click "Set up environment".
//!   3. The app creates the environment, installs requirements.txt,
//!      runs a self-test, and then the editor is ready.
//!   4. Write code, press Run, see output in the console.

mod bootstrap;
mod runner;

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Duration;

use eframe::egui;

const STARTER_CODE: &str = r#"# Welcome! Press Run (or Ctrl+Enter) to execute this script.
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("My first plot")
plt.savefig("my_first_plot.png")

print("Hello from Python!")
print("NumPy version:", np.__version__)
print("Saved my_first_plot.png in the scripts folder.")

# PyTorch quick check (if installed):
try:
    import torch
    print("PyTorch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if hasattr(torch.backends, "mps"):
        print("mps (Apple GPU):", torch.backends.mps.is_available())
except ImportError:
    pass
"#;

/// Messages from background threads to the UI.
enum Msg {
    Log(String),
    SetupDone(Result<(), String>),
    RunLine(String),
    RunDone(i32),
    EnvCheck(bool), // env exists?
    Candidates(Vec<bootstrap::Candidate>),
}

struct App {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,

    code: String,
    file_name: String,
    console: String,

    ready: bool,
    setting_up: bool,
    setup_open: bool,
    candidates: Vec<bootstrap::Candidate>,
    /// Index into candidates; candidates.len() = "download managed Python".
    selected: usize,
    running: Option<runner::RunningChild>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_theme(egui::ThemePreference::Dark);

        let (tx, rx) = channel();
        let app = App {
            tx,
            rx,
            code: STARTER_CODE.to_string(),
            file_name: "main.py".to_string(),
            console: String::new(),
            ready: false,
            setting_up: false,
            setup_open: false,
            candidates: Vec::new(),
            selected: 0,
            running: None,
        };

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
        // Keep the console from growing without bound.
        const MAX: usize = 200_000;
        if self.console.len() > MAX {
            let cut = self.console.len() - MAX / 2;
            let boundary = self.console.ceil_char_boundary(cut);
            self.console.drain(..boundary);
        }
    }

    fn scripts_dir(&self) -> PathBuf {
        let dir = runner::scripts_dir();
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn script_files(&self) -> Vec<String> {
        let mut files: Vec<String> = std::fs::read_dir(self.scripts_dir())
            .map(|rd| {
                rd.filter_map(Result::ok)
                    .filter_map(|e| e.file_name().to_str().map(str::to_string))
                    .filter(|n| n.ends_with(".py"))
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        files
    }

    fn load_file(&mut self, name: &str) {
        let path = self.scripts_dir().join(name);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                self.code = text;
                self.file_name = name.to_string();
            }
            Err(e) => self.log(&format!("Could not open {}: {e}", path.display())),
        }
    }

    fn save_file(&mut self) -> Result<PathBuf, String> {
        let name = if self.file_name.trim().is_empty() {
            "main.py".to_string()
        } else if self.file_name.ends_with(".py") {
            self.file_name.trim().to_string()
        } else {
            format!("{}.py", self.file_name.trim())
        };
        self.file_name = name.clone();
        let path = self.scripts_dir().join(&name);
        std::fs::write(&path, &self.code)
            .map_err(|e| format!("Could not save {}: {e}", path.display()))?;
        Ok(path)
    }

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
        let path = match self.save_file() {
            Ok(p) => p,
            Err(e) => {
                self.log(&e);
                return;
            }
        };
        self.log(&format!("\n--- running {} ---", self.file_name));

        let (tx_lines, rx_lines) = channel::<String>();
        match runner::run_script(&path, tx_lines) {
            Ok((handle, waiter)) => {
                self.running = Some(handle);
                let tx = self.tx.clone();
                thread::spawn(move || {
                    // Forward output lines, then report the exit code.
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
                    // Default to "managed" only if nothing usable was found.
                    self.selected = if list.is_empty() { 0 } else { 0 }; // newest first
                    self.candidates = list;
                }
            }
        }
    }

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
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_messages();
        if self.setting_up || self.running.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }

        // Ctrl+Enter runs the script.
        if ctx.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.command) {
            self.start_run();
        }

        self.setup_window(ctx);

        // ------------------------------------------------------------------
        // Toolbar
        // ------------------------------------------------------------------
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.setting_up, egui::Button::new("Setup..."))
                    .on_hover_text("Open environment setup (interpreter choice, reinstall)")
                    .clicked()
                {
                    self.setup_open = true;
                }

                ui.separator();

                ui.label("File:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.file_name)
                        .desired_width(130.0),
                );
                egui::ComboBox::from_id_salt("open_file")
                    .selected_text("open...")
                    .show_ui(ui, |ui| {
                        let mut picked = None;
                        for name in self.script_files() {
                            if ui.selectable_label(self.file_name == name, &name).clicked() {
                                picked = Some(name);
                            }
                        }
                        if let Some(name) = picked {
                            self.load_file(&name);
                        }
                    });
                if ui.button("Save").clicked() {
                    match self.save_file() {
                        Ok(p) => self.log(&format!("Saved {}", p.display())),
                        Err(e) => self.log(&e),
                    }
                }
                if ui.button("New").clicked() {
                    self.code = STARTER_CODE.to_string();
                    self.file_name = "main.py".to_string();
                }

                ui.separator();

                let can_run = self.ready && self.running.is_none() && !self.setting_up;
                if ui
                    .add_enabled(can_run, egui::Button::new("▶ Run"))
                    .on_hover_text("Run this script (Ctrl+Enter)")
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
                    "environment ready (self-test passed)"
                } else {
                    "setup needed"
                };
                ui.label(status_text);
            });
            ui.add_space(4.0);
        });

        // ------------------------------------------------------------------
        // Console
        // ------------------------------------------------------------------
        egui::TopBottomPanel::bottom("console")
            .resizable(true)
            .default_height(220.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Console");
                    if ui.button("Clear").clicked() {
                        self.console.clear();
                    }
                    ui.separator();
                    ui.label("Scripts folder:");
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(self.scripts_dir().display().to_string())
                                .monospace()
                                .weak(),
                        )
                        .selectable(true),
                    );
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.add(
                            egui::Label::new(egui::RichText::new(&self.console).monospace())
                                .selectable(true)
                                .wrap(),
                        );
                    });
            });

        // ------------------------------------------------------------------
        // Code editor
        // ------------------------------------------------------------------
        egui::CentralPanel::default().show(ctx, |ui| {
            let theme =
                egui_extras::syntax_highlighting::CodeTheme::from_memory(ui.ctx(), ui.style());
            let mut layouter = |ui: &egui::Ui, code: &str, wrap_width: f32| {
                let mut job = egui_extras::syntax_highlighting::highlight(
                    ui.ctx(),
                    ui.style(),
                    &theme,
                    code,
                    "py",
                );
                job.wrap.max_width = wrap_width;
                ui.fonts(|f| f.layout_job(job))
            };

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.code)
                        .font(egui::TextStyle::Monospace)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .layouter(&mut layouter),
                );
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 750.0])
            .with_min_inner_size([700.0, 450.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ML Class Launcher",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
