//! Interpreter discovery (with versions) and the managed Python path
//! (a silent Miniconda install, so conda manages the environment).
//!
//! Everything the app manages lives in a per-user data folder, so no admin
//! rights are needed on any platform:
//!
//!   Windows : %LOCALAPPDATA%\ml-class-launcher\
//!   macOS   : ~/Library/Application Support/ml-class-launcher/
//!   Linux   : ~/.local/share/ml-class-launcher/

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

pub const MIN_PYTHON: (u32, u32) = (3, 10);
pub const MANAGED_PYTHON_SERIES: &str = "3.12";
const USER_AGENT: &str = "ml-class-launcher";

/// Root folder for everything the app manages.
pub fn app_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ml-class-launcher")
}

/// Where Miniconda gets installed.
pub fn miniconda_dir() -> PathBuf {
    app_dir().join("miniconda")
}

/// A usable interpreter found on the machine.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub path: PathBuf,
    pub version: String,
    pub source: String,
}

impl Candidate {
    pub fn label(&self) -> String {
        format!("Python {} — {} ({})", self.version, self.path.display(), self.source)
    }
}

// ---------------------------------------------------------------------------
// Interpreter discovery
// ---------------------------------------------------------------------------

/// Parse a `Python x.y.z` version string; return None if too old.
fn parse_version(text: &str) -> Option<String> {
    let version = text.trim().strip_prefix("Python ")?;
    let mut parts = version.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    if (major, minor) >= MIN_PYTHON {
        Some(version.to_string())
    } else {
        None
    }
}

/// Check a candidate binary: must run and be >= MIN_PYTHON.
fn probe(path: &Path) -> Option<String> {
    let out = silent_command(path).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    parse_version(&text)
}

/// Hide extra console windows on Windows.
pub fn silent_command(program: &Path) -> Command {
    #[allow(unused_mut)] // only mutated on Windows
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

fn push_candidate(list: &mut Vec<Candidate>, path: PathBuf, source: &str) {
    if !path.exists() {
        return;
    }
    // Skip files we already have (by canonical path).
    let canon = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
    if list
        .iter()
        .any(|c| fs::canonicalize(&c.path).unwrap_or_else(|_| c.path.clone()) == canon)
    {
        return;
    }
    if let Some(version) = probe(&path) {
        list.push(Candidate {
            path,
            version,
            source: source.to_string(),
        });
    }
}

/// Add all files matching a name filter inside `dir` as candidates.
fn scan_dir(list: &mut Vec<Candidate>, dir: &Path, pred: impl Fn(&str) -> bool, source: &str) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.filter_map(Result::ok) {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if pred(name) {
            push_candidate(list, entry.path(), source);
        }
    }
}

fn is_python_name(name: &str) -> bool {
    let n = name.strip_suffix(".exe").unwrap_or(name);
    n == "python" || n == "python3" || (n.starts_with("python3.") && n[8..].chars().all(|c| c.is_ascii_digit()))
}

/// Find every usable Python on the machine: PATH, OS-specific install
/// locations, and existing conda installations.
pub fn discover_interpreters() -> Vec<Candidate> {
    let mut list: Vec<Candidate> = Vec::new();

    // 1. PATH
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            scan_dir(&mut list, &dir, is_python_name, "PATH");
        }
    }

    let home = dirs::home_dir().unwrap_or_default();

    #[cfg(target_os = "windows")]
    {
        // `py -0p` lists every interpreter registered with the py launcher.
        if let Ok(out) = silent_command(Path::new("py")).arg("-0p").output() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                if let Some(path) = line.split_whitespace().last() {
                    let path = PathBuf::from(path);
                    if path.extension().is_some_and(|e| e == "exe") {
                        push_candidate(&mut list, path, "py launcher");
                    }
                }
            }
        }
        // Standard python.org installs: %LOCALAPPDATA%\Programs\Python\Python3xx\
        if let Some(local) = dirs::data_local_dir() {
            let base = local.join("Programs").join("Python");
            if let Ok(rd) = fs::read_dir(&base) {
                for entry in rd.filter_map(Result::ok) {
                    push_candidate(&mut list, entry.path().join("python.exe"), "python.org");
                }
            }
        }
        for conda in ["miniconda3", "anaconda3"] {
            push_candidate(&mut list, home.join(conda).join("python.exe"), "conda");
        }
        push_candidate(
            &mut list,
            PathBuf::from(r"C:\ProgramData\miniconda3\python.exe"),
            "conda",
        );
    }

    #[cfg(target_os = "macos")]
    {
        push_candidate(&mut list, PathBuf::from("/usr/bin/python3"), "system");
        for dir in ["/opt/homebrew/bin", "/usr/local/bin"] {
            scan_dir(&mut list, Path::new(dir), is_python_name, "homebrew");
        }
        let fw = Path::new("/Library/Frameworks/Python.framework/Versions");
        if let Ok(rd) = fs::read_dir(fw) {
            for entry in rd.filter_map(Result::ok) {
                push_candidate(&mut list, entry.path().join("bin/python3"), "python.org");
            }
        }
        for conda in ["miniconda3", "anaconda3", "opt/miniconda3", "opt/anaconda3"] {
            push_candidate(&mut list, home.join(conda).join("bin/python"), "conda");
        }
    }

    #[cfg(target_os = "linux")]
    {
        for dir in ["/usr/bin", "/usr/local/bin"] {
            scan_dir(&mut list, Path::new(dir), is_python_name, "system");
        }
        for conda in ["miniconda3", "anaconda3", "opt/miniconda3", "opt/anaconda3"] {
            push_candidate(&mut list, home.join(conda).join("bin/python"), "conda");
        }
    }

    // The app's own managed Miniconda base, if already installed.
    let managed = miniconda_python();
    push_candidate(&mut list, managed, "managed miniconda");

    // Sort newest first.
    list.sort_by(|a, b| b.version.cmp(&a.version));
    list
}

/// The base python of the app's Miniconda (after install).
pub fn miniconda_python() -> PathBuf {
    if cfg!(windows) {
        miniconda_dir().join("python.exe")
    } else {
        miniconda_dir().join("bin").join("python")
    }
}

/// The conda executable of the app's Miniconda.
pub fn conda_exe() -> PathBuf {
    if cfg!(windows) {
        miniconda_dir().join("Scripts").join("conda.exe")
    } else {
        miniconda_dir().join("bin").join("conda")
    }
}

// ---------------------------------------------------------------------------
// Managed install: Miniconda
// ---------------------------------------------------------------------------

fn miniconda_installer_url() -> Option<String> {
    let base = "https://repo.anaconda.com/miniconda";
    let file = if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "Miniconda3-latest-Windows-x86_64.exe"
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "Miniconda3-latest-MacOSX-arm64.sh"
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        "Miniconda3-latest-MacOSX-x86_64.sh"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "Miniconda3-latest-Linux-x86_64.sh"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "Miniconda3-latest-Linux-aarch64.sh"
    } else {
        return None;
    };
    Some(format!("{base}/{file}"))
}

fn http_get(url: &str) -> Result<ureq::Response, String> {
    ureq::get(url)
        .set("User-Agent", USER_AGENT)
        .call()
        .map_err(|e| format!("HTTP request failed ({url}): {e}"))
}

/// Download `url` to `dest` with progress logging.
fn download(url: &str, dest: &Path, log: &dyn Fn(String)) -> Result<(), String> {
    let response = http_get(url)?;
    let total: u64 = response
        .header("content-length")
        .and_then(|h| h.parse().ok())
        .unwrap_or(0);
    let mut reader = response.into_reader();
    let mut file =
        fs::File::create(dest).map_err(|e| format!("Could not create {}: {e}", dest.display()))?;
    let mut buf = [0u8; 256 * 1024];
    let mut downloaded: u64 = 0;
    let mut next_report: u64 = 20 * 1024 * 1024;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("Download error: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("Write error: {e}"))?;
        downloaded += n as u64;
        if downloaded >= next_report {
            if total > 0 {
                log(format!(
                    "  ... {} / {} MB ({}%)",
                    downloaded / 1_048_576,
                    total / 1_048_576,
                    downloaded * 100 / total
                ));
            } else {
                log(format!("  ... {} MB", downloaded / 1_048_576));
            }
            next_report = downloaded + 20 * 1024 * 1024;
        }
    }
    Ok(())
}

/// Ensure the app's Miniconda is installed; install it silently if not.
pub fn ensure_miniconda(log: &dyn Fn(String)) -> Result<(), String> {
    if conda_exe().exists() {
        log("Miniconda already installed.".into());
        return Ok(());
    }
    let url = miniconda_installer_url()
        .ok_or_else(|| "No Miniconda build for this platform.".to_string())?;
    let file_name = url.rsplit('/').next().unwrap().to_string();
    let tmp = std::env::temp_dir().join(&file_name);

    log(format!("Downloading Miniconda ({file_name})..."));
    download(&url, &tmp, log)?;
    log("Installing Miniconda (silent)...".into());

    let dest = miniconda_dir();
    if dest.exists() {
        let _ = fs::remove_dir_all(&dest);
    }
    let status = run_miniconda_installer(&tmp, &dest)?;
    let _ = fs::remove_file(&tmp);

    if !status.success() || !conda_exe().exists() {
        return Err(format!("Miniconda installer exited with {status}"));
    }
    log(format!("Miniconda installed to {}", dest.display()));
    Ok(())
}

#[cfg(target_os = "windows")]
fn run_miniconda_installer(installer: &Path, dest: &Path) -> Result<std::process::ExitStatus, String> {
    // NSIS silent install; /D must be last and unquoted (spaces are fine
    // because it is last). `start /wait` is needed because the installer
    // detaches. raw_arg avoids Command's quote-mangling.
    use std::os::windows::process::CommandExt;
    let mut cmd = Command::new("cmd");
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    cmd.raw_arg(format!(
        "/c start /wait \"\" \"{}\" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D={}",
        installer.display(),
        dest.display()
    ));
    cmd.status()
        .map_err(|e| format!("Could not run Miniconda installer: {e}"))
}

#[cfg(not(target_os = "windows"))]
fn run_miniconda_installer(installer: &Path, dest: &Path) -> Result<std::process::ExitStatus, String> {
    silent_command(Path::new("sh"))
        .arg(installer)
        .arg("-b") // batch (no prompts)
        .arg("-p")
        .arg(dest)
        .status()
        .map_err(|e| format!("Could not run Miniconda installer: {e}"))
}
