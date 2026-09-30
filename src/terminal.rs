//! Integrated PTY terminal, docked in the bottom panel.
//!
//! Spawns a real shell (cmd.exe on Windows, $SHELL/bash on unix) attached to a
//! pseudo-terminal, pre-configured so that `python` / `pip` resolve to the
//! app's managed environment and the working directory is the open project.
//! Terminal emulation is handled by `vt100`, the PTY by `portable-pty`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};

use portable_pty::{native_pty_system, Child, CommandBuilder, MasterPty, PtySize};

pub enum TermMsg {
    Output(Vec<u8>),
    Exited,
}

pub struct Terminal {
    parser: vt100::Parser,
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    master: Box<dyn MasterPty + Send>,
    child: Arc<Mutex<Box<dyn Child + Send + Sync>>>,
    rx: Receiver<TermMsg>,
    pub alive: bool,
    /// Current scrollback offset (0 = bottom / live view).
    pub scroll: usize,
    /// Set when the shell prints something while scrolled away from the bottom.
    pub unread_output: bool,
}

impl Terminal {
    /// Spawn a shell in `cwd` with `env_bins` prepended to PATH.
    pub fn spawn(
        env_bins: &[PathBuf],
        cwd: &Path,
        rows: u16,
        cols: u16,
        ctx: eframe::egui::Context,
    ) -> Result<Self, String> {
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| format!("openpty failed: {e}"))?;

        let shell = shell_path();
        let mut cmd = CommandBuilder::new(&shell);
        cmd.cwd(cwd);
        cmd.env("TERM", "xterm-256color");
        cmd.env("COLORTERM", "truecolor");

        // PATH: managed environment first, so `python`/`pip` hit it directly.
        if !env_bins.is_empty() {
            let sep = if cfg!(windows) { ";" } else { ":" };
            let old = std::env::var("PATH").unwrap_or_default();
            let prefix = env_bins
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(sep);
            cmd.env("PATH", format!("{prefix}{sep}{old}"));
        }

        // Same import semantics as the Run button.
        let cwd_s = cwd.to_string_lossy().into_owned();
        let sep = if cfg!(windows) { ";" } else { ":" };
        match std::env::var("PYTHONPATH") {
            Ok(old) if !old.is_empty() => {
                cmd.env("PYTHONPATH", format!("{cwd_s}{sep}{old}"));
            }
            _ => {
                cmd.env("PYTHONPATH", &cwd_s);
            }
        }

        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| format!("failed to spawn {}: {e}", shell.display()))?;
        drop(pair.slave);

        let mut reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| format!("pty reader: {e}"))?;
        let writer = pair.master.take_writer().map_err(|e| format!("pty writer: {e}"))?;

        let child = Arc::new(Mutex::new(child));
        let (tx, rx) = channel();
        {
            let child = child.clone();
            std::thread::spawn(move || {
                let mut buf = [0u8; 8192];
                loop {
                    match reader.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            if tx.send(TermMsg::Output(buf[..n].to_vec())).is_err() {
                                return;
                            }
                            ctx.request_repaint();
                        }
                        Err(_) => break,
                    }
                }
                if let Ok(mut c) = child.lock() {
                    let _ = c.wait();
                }
                let _ = tx.send(TermMsg::Exited);
                ctx.request_repaint();
            });
        }

        Ok(Self {
            parser: vt100::Parser::new(rows, cols, 2000),
            writer: Arc::new(Mutex::new(writer)),
            master: pair.master,
            child,
            rx,
            alive: true,
            scroll: 0,
            unread_output: false,
        })
    }

    /// Drain pending PTY output into the screen model. Returns true if anything changed.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                TermMsg::Output(bytes) => {
                    self.parser.process(&bytes);
                    if self.scroll > 0 {
                        self.unread_output = true;
                    }
                    changed = true;
                }
                TermMsg::Exited => {
                    self.alive = false;
                    changed = true;
                }
            }
        }
        changed
    }

    pub fn screen(&mut self) -> &vt100::Screen {
        self.parser.set_scrollback(self.scroll);
        self.parser.screen()
    }

    pub fn size(&self) -> (u16, u16) {
        self.parser.screen().size()
    }

    pub fn resize(&mut self, rows: u16, cols: u16) {
        if (rows, cols) == self.size() || rows == 0 || cols == 0 {
            return;
        }
        self.parser.set_size(rows, cols);
        let _ = self.master.resize(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        });
    }

    /// Send keystroke bytes to the shell and snap back to the live view.
    pub fn send(&mut self, bytes: &[u8]) {
        if !self.alive {
            return;
        }
        self.scroll = 0;
        self.unread_output = false;
        if let Ok(mut w) = self.writer.lock() {
            let _ = w.write_all(bytes);
            let _ = w.flush();
        }
    }

    pub fn kill(&mut self) {
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
        }
        self.alive = false;
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Pick the user's interactive shell.
fn shell_path() -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from("cmd.exe")
    }
    #[cfg(not(windows))]
    {
        if let Ok(sh) = std::env::var("SHELL") {
            let p = PathBuf::from(&sh);
            if p.exists() {
                return p;
            }
        }
        for cand in ["/bin/bash", "/bin/zsh", "/bin/sh"] {
            if Path::new(cand).exists() {
                return PathBuf::from(cand);
            }
        }
        PathBuf::from("/bin/sh")
    }
}
