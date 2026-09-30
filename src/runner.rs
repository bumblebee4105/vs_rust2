//! Environment creation (venv for an existing interpreter, conda env for the
//! managed Miniconda), GPU-aware package installation, the post-install
//! self-test, and running student scripts with live output streaming.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::bootstrap::{self, silent_command};

/// Fallback package list when no requirements.txt ships next to the executable.
pub const DEFAULT_REQUIREMENTS: &str = "numpy\npandas\nmatplotlib\nscikit-learn\ntorch\n";

/// PyTorch wheel index for CUDA 12.6 (most compatible current CUDA build).
const TORCH_CUDA_INDEX: &str = "https://download.pytorch.org/whl/cu126";
/// PyTorch wheel index for CPU-only builds (much smaller than PyPI's CUDA stack).
const TORCH_CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";

fn venv_dir() -> PathBuf {
    bootstrap::app_dir().join("venv")
}

fn conda_env_dir() -> PathBuf {
    bootstrap::app_dir().join("env")
}

/// The python binary of the app's environment (conda env wins if present).
pub fn env_python() -> Option<PathBuf> {
    let candidates = if cfg!(windows) {
        [
            conda_env_dir().join("python.exe"),
            venv_dir().join("Scripts").join("python.exe"),
        ]
    } else {
        [
            conda_env_dir().join("bin").join("python"),
            venv_dir().join("bin").join("python"),
        ]
    };
    candidates.into_iter().find(|p| p.exists())
}

pub fn env_ready() -> bool {
    env_python().is_some()
}

/// Directories to prepend to PATH so `python` and `pip` in a shell resolve
/// to the app's managed environment. Empty when no environment exists yet.
pub fn env_bin_dirs() -> Vec<PathBuf> {
    if cfg!(windows) {
        // Conda env: python.exe sits at the env root, pip in Scripts.
        let conda = conda_env_dir().join("python.exe");
        if conda.exists() {
            return vec![conda_env_dir(), conda_env_dir().join("Scripts")];
        }
        let venv = venv_dir().join("Scripts").join("python.exe");
        if venv.exists() {
            return vec![venv_dir().join("Scripts")];
        }
    } else {
        let conda_bin = conda_env_dir().join("bin");
        if conda_bin.join("python").exists() {
            return vec![conda_bin];
        }
        let venv_bin = venv_dir().join("bin");
        if venv_bin.join("python").exists() {
            return vec![venv_bin];
        }
    }
    Vec::new()
}

/// Folder where student scripts are stored (and where scripts run, so
/// relative file paths like "data.csv" behave the same for everyone).
pub fn scripts_dir() -> PathBuf {
    bootstrap::app_dir().join("scripts")
}

/// Run a command to completion, forwarding each output line to `log`.
fn run_logged(cmd: &mut Command, log: &dyn Fn(String)) -> Result<(), String> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Could not start process: {e}"))?;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let t_out = thread::spawn(move || {
        let mut lines = Vec::new();
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            lines.push(line);
        }
        lines
    });
    let err_lines: Vec<String> = BufReader::new(stderr)
        .lines()
        .map_while(Result::ok)
        .collect();

    let status = child.wait().map_err(|e| e.to_string())?;
    for line in t_out.join().unwrap_or_default() {
        log(line);
    }
    for line in err_lines {
        log(line);
    }
    if status.success() {
        Ok(())
    } else {
        Err(format!("Process exited with {status}"))
    }
}

/// Create a stdlib venv from a user-chosen interpreter.
pub fn create_venv(base_python: &Path, log: &dyn Fn(String)) -> Result<(), String> {
    if venv_dir().exists() {
        log("Removing previous virtual environment...".into());
        std::fs::remove_dir_all(venv_dir()).map_err(|e| e.to_string())?;
    }
    log("Creating virtual environment...".into());
    run_logged(
        silent_command(base_python)
            .arg("-m")
            .arg("venv")
            .arg(venv_dir()),
        log,
    )?;
    log("Virtual environment ready.".into());
    Ok(())
}

/// Create the managed conda environment (Python 3.12) via Miniconda.
pub fn create_conda_env(log: &dyn Fn(String)) -> Result<(), String> {
    bootstrap::ensure_miniconda(log)?;
    if conda_env_dir().exists() {
        log("Removing previous conda environment...".into());
        std::fs::remove_dir_all(conda_env_dir()).map_err(|e| e.to_string())?;
    }
    log(format!(
        "Creating conda environment (python={})...",
        bootstrap::MANAGED_PYTHON_SERIES
    ));
    // conda-forge only: no Anaconda "defaults" channel, so no license worries.
    run_logged(
        silent_command(&bootstrap::conda_exe()).args([
            "create",
            "-y",
            "-p",
        ]).arg(conda_env_dir()).args([
            "-c",
            "conda-forge",
            "--override-channels",
            &format!("python={}", bootstrap::MANAGED_PYTHON_SERIES),
            "pip",
        ]),
        log,
    )?;
    log("Conda environment ready.".into());
    Ok(())
}

// ---------------------------------------------------------------------------
// Requirements + GPU-aware installation
// ---------------------------------------------------------------------------

/// The package list: requirements.txt next to the executable wins,
/// otherwise the built-in default.
pub fn requirements() -> Vec<String> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let req = dir.join("requirements.txt");
            if let Ok(text) = std::fs::read_to_string(&req) {
                let pkgs: Vec<String> = text
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .map(str::to_string)
                    .collect();
                if !pkgs.is_empty() {
                    return pkgs;
                }
            }
        }
    }
    DEFAULT_REQUIREMENTS
        .lines()
        .map(str::to_string)
        .collect()
}

/// The bare package name of a requirement line ("torch>=2.0" -> "torch").
fn base_name(req: &str) -> &str {
    req.split(|c: char| "=<>~[; ".contains(c))
        .next()
        .unwrap_or(req)
}

/// The Python import name for a requirement line
/// ("scikit-learn" -> "sklearn", "Pillow" -> "PIL", ...).
pub fn import_name(req: &str) -> String {
    let base = base_name(req);
    let lowered = base.to_ascii_lowercase();
    let mapped = match lowered.as_str() {
        "scikit-learn" => "sklearn",
        "opencv-python" | "opencv-python-headless" => "cv2",
        "pillow" => "PIL",
        "pyyaml" => "yaml",
        "beautifulsoup4" => "bs4",
        "python-dateutil" => "dateutil",
        "pygame" => "pygame",
        other => other,
    };
    mapped.replace('-', "_")
}

fn is_torch_pkg(req: &str) -> bool {
    matches!(base_name(req), "torch" | "torchvision" | "torchaudio")
}

/// True if an NVIDIA GPU with a working driver is present.
fn detect_nvidia() -> bool {
    if cfg!(target_os = "macos") {
        return false;
    }
    let candidates: &[&str] = if cfg!(windows) {
        &["nvidia-smi", r"C:\Windows\System32\nvidia-smi.exe"]
    } else {
        &["nvidia-smi", "/usr/bin/nvidia-smi"]
    };
    candidates.iter().any(|c| {
        silent_command(Path::new(c))
            .arg("-L")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// One-line explanation for the setup screen.
pub fn gpu_note() -> String {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "Apple Silicon: PyTorch with MPS (GPU) acceleration — included in the standard wheel."
            .into()
    } else if cfg!(target_os = "macos") {
        "Intel Mac: CPU-only PyTorch.".into()
    } else if detect_nvidia() {
        "NVIDIA GPU detected: PyTorch CUDA 12.6 wheels will be installed (large download, ~2.5 GB)."
            .into()
    } else {
        "No NVIDIA GPU found: CPU-only PyTorch (much smaller download, runs everywhere).".into()
    }
}

/// Install the requirements into the environment, GPU-aware:
/// torch/torchvision/torchaudio get the right wheel index for the hardware.
pub fn install_packages(log: &dyn Fn(String)) -> Result<(), String> {
    let python = env_python().ok_or("No environment found — create it first.")?;
    let pkgs = requirements();
    log(format!("Installing packages: {}", pkgs.join(", ")));
    log("This can take a while on the first run, especially PyTorch...".into());

    run_logged(
        silent_command(&python)
            .args(["-m", "pip", "install", "--upgrade", "pip"]),
        log,
    )?;

    let (torch, rest): (Vec<&String>, Vec<&String>) =
        pkgs.iter().partition(|p| is_torch_pkg(p));

    if !torch.is_empty() && !cfg!(target_os = "macos") {
        // macOS gets torch from PyPI (MPS included); elsewhere pick an index.
        let index = if detect_nvidia() {
            log("NVIDIA GPU found — installing CUDA-enabled PyTorch.".into());
            TORCH_CUDA_INDEX
        } else {
            log("No NVIDIA GPU — installing CPU-only PyTorch.".into());
            TORCH_CPU_INDEX
        };
        let mut cmd = silent_command(&python);
        cmd.args(["-m", "pip", "install"]);
        for t in &torch {
            cmd.arg(t);
        }
        cmd.arg("--index-url").arg(index);
        run_logged(&mut cmd, log)?;
    }

    let rest: Vec<&String> = if cfg!(target_os = "macos") {
        pkgs.iter().collect() // macOS: everything from PyPI in one pass
    } else {
        rest
    };
    if !rest.is_empty() {
        let mut cmd = silent_command(&python);
        cmd.args(["-m", "pip", "install"]);
        for p in &rest {
            cmd.arg(p);
        }
        run_logged(&mut cmd, log)?;
    }
    log("All packages installed.".into());
    Ok(())
}

// ---------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------

const SELFTEST_PY: &str = r#"import importlib, sys
modules = sys.argv[1:]
fails = []
for name in modules:
    try:
        m = importlib.import_module(name)
        print(f"SELFTEST OK   {name} {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"SELFTEST FAIL {name}: {e}")
        fails.append(name)
if "torch" in modules and "torch" not in fails:
    import torch
    t = torch.rand(512, 512)
    print(f"SELFTEST OK   torch-matmul sum={(t @ t).sum().item():.1f}")
    print(f"SELFTEST INFO torch-cuda={torch.cuda.is_available()}")
    if sys.platform == "darwin":
        print(f"SELFTEST INFO torch-mps={torch.backends.mps.is_available()}")
print("SELFTEST " + ("PASS" if not fails else "FAIL"))
sys.exit(1 if fails else 0)
"#;

/// Verify the environment by importing every requirement and exercising
/// torch (matmul + accelerator availability). Runs automatically after setup.
pub fn self_test(log: &dyn Fn(String)) -> Result<(), String> {
    let python = env_python().ok_or("No environment found.")?;
    let dir = scripts_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let script = dir.join("_selftest.py");
    std::fs::write(&script, SELFTEST_PY).map_err(|e| e.to_string())?;

    let modules: Vec<String> = requirements().iter().map(|r| import_name(r)).collect();
    log("Running self-test...".into());
    let mut cmd = silent_command(&python);
    cmd.arg(&script);
    for m in &modules {
        cmd.arg(m);
    }
    run_logged(&mut cmd, log).map_err(|_| "Self-test FAILED (see console above).".to_string())?;
    log("Self-test passed — environment is healthy.".into());
    Ok(())
}

// ---------------------------------------------------------------------------
// Running student scripts
// ---------------------------------------------------------------------------

/// A handle to a running student script, so the Stop button can kill it.
pub type RunningChild = Arc<Mutex<Option<Child>>>;

/// Spawn a script with the environment's python. Output lines stream to `tx`.
///
/// `cwd` is the student's project folder: it becomes the working directory
/// (so `open("data.csv")` finds files next to the script) and is prepended to
/// PYTHONPATH (so sibling files like `helpers.py` import as modules).
pub fn run_script(
    script: &Path,
    cwd: &Path,
    tx: Sender<String>,
) -> Result<(RunningChild, thread::JoinHandle<i32>), String> {
    let python = env_python().ok_or("No environment found — run setup first.")?;

    // Prepend the project folder to any existing PYTHONPATH.
    let pythonpath = match std::env::var_os("PYTHONPATH") {
        Some(existing) if !existing.is_empty() => {
            let mut v = cwd.as_os_str().to_owned();
            v.push(if cfg!(windows) { ";" } else { ":" });
            v.push(existing);
            v
        }
        _ => cwd.as_os_str().to_owned(),
    };

    let mut child = silent_command(&python)
        .arg("-u") // unbuffered, so prints show up immediately
        .arg(script)
        .current_dir(cwd)
        .env("PYTHONPATH", pythonpath)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Could not start Python: {e}"))?;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let handle: RunningChild = Arc::new(Mutex::new(Some(child)));

    let tx_err = tx.clone();
    let t_out = thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let t_err = thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if tx_err.send(format!("[stderr] {line}")).is_err() {
                break;
            }
        }
    });

    let handle2 = Arc::clone(&handle);
    let waiter = thread::spawn(move || {
        let _ = t_out.join();
        let _ = t_err.join();
        let mut guard = handle2.lock().unwrap();
        match guard.as_mut() {
            Some(child) => {
                let code = child.wait().map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
                *guard = None;
                code
            }
            None => -1, // killed via Stop
        }
    });
    Ok((handle, waiter))
}

/// Kill the running script, if any.
pub fn stop_script(handle: &RunningChild) {
    if let Some(child) = handle.lock().unwrap().as_mut() {
        let _ = child.kill();
    }
}
