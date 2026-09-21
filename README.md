# ML Class Launcher

A single Rust executable that gives students a complete, identical Python machine-learning
environment — from a light MacBook to a battle station — with **one setup click**:

1. **Finds every Python on the machine** (PATH, python.org installs, py-launcher entries,
   Homebrew, existing conda installs). If several are found, the student picks one from a
   dropdown. If none are found (or "managed" is chosen), it **silently installs Miniconda**
   and creates a conda environment with **Python 3.12** — conda does the environment
   management, the app just tells it what to install.
2. **Creates the environment** — a stdlib `venv` for a chosen existing interpreter, or a
   conda env (`conda-forge` channel only, so no Anaconda licensing worries) for the
   managed path.
3. **Installs the shipped `requirements.txt`** (default: numpy, pandas, matplotlib,
   scikit-learn, torch) with **GPU-aware PyTorch selection**:
   - NVIDIA GPU detected (via `nvidia-smi`) → CUDA 12.6 wheels (`download.pytorch.org/whl/cu126`)
   - No NVIDIA GPU → CPU-only wheels (`/whl/cpu`) — saves a ~2.5 GB download on Linux
   - Apple Silicon → standard wheel, which includes **MPS (Apple GPU)** support
     (the Apple *Neural Engine* is not exposed to PyTorch; MPS is the accelerator)
4. **Self-tests the environment** — imports every package from requirements.txt, does a
   torch matmul, and reports `torch.cuda.is_available()` / `torch.backends.mps.is_available()`.
   Only when the self-test passes does the editor unlock.
5. **Acts as a mini code editor** — Python syntax highlighting, Run button (or Ctrl+Enter),
   Stop button, live output console.

Works on Windows (x64), macOS (Apple Silicon + Intel), and Linux (x64, ARM64).
No admin rights needed anywhere.

## Deliberately skipped: Visual Studio Build Tools

PyTorch, numpy, pandas, etc. ship as **prebuilt wheels** — pip never compiles anything on
a student machine, so no C++ toolchain is needed. Build Tools (~7 GB) would only matter
for building packages from source, which this setup never does. (Only exception: a
requirement with no wheel for the platform — avoid those in requirements.txt.)

## Building

You need [Rust](https://rustup.rs) installed. Then:

```bash
cargo build --release
```

The single self-contained binary ends up in `target/release/ml-class-launcher`
(`.exe` on Windows). Distribute it **together with `requirements.txt`** — that file next
to the executable controls what gets installed.

## Using it in class

1. Student starts the app → the **Environment setup** window opens automatically.
2. They pick an interpreter (newest found is preselected; "Download managed Python 3.12"
   is always an option) and click **Set up environment**. The console shows progress.
3. After install, the **self-test** runs; when it passes, the editor is ready.
4. Write or open a `.py` file, press **▶ Run**. **■ Stop** kills a stuck script.

Everything lives in one per-user folder (easy to delete for a full reset):

| Platform | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\ml-class-launcher\` |
| macOS | `~/Library/Application Support/ml-class-launcher/` |
| Linux | `~/.local/share/ml-class-launcher/` |

Student scripts live in the `scripts/` subfolder (path shown in the console bar) and run
with that folder as working directory — `open("data.csv")` and `plt.savefig("plot.png")`
behave identically for everyone.

## Customizing for your lesson

- **Packages**: edit the shipped `requirements.txt` before distributing. Lines like
  `torch`, `torchvision`, `torchaudio` (with or without version pins) are automatically
  routed to the correct CUDA/CPU/Mac wheel index; everything else comes from PyPI.
- **Starter code**: edit `STARTER_CODE` in `src/main.rs`.
- **Managed Python version**: `MANAGED_PYTHON_SERIES` in `src/bootstrap.rs` (default 3.12).
- **CUDA variant**: `TORCH_CUDA_INDEX` in `src/runner.rs` (default cu126, the most
  driver-compatible current build; cu130/cu132 exist for newer toolkits).

## Troubleshooting

- **Self-test fails**: the console shows which package failed to import — usually a typo
  in requirements.txt or a package without a wheel for that platform.
- **CUDA wheel install fails / runs out of disk**: remove torch lines from
  requirements.txt, or the app falls back cleanly — delete the app folder and re-run setup.
- **Weird state**: delete the app folder and click Set up environment — a full reset.
- **Classroom firewall**: needs access to `repo.anaconda.com` (only for the managed path),
  `pypi.org` / `files.pythonhosted.org`, and `download.pytorch.org` (for torch).

## Project layout

```
src/main.rs      GUI: setup window (interpreter picker), editor, console
src/bootstrap.rs interpreter discovery + silent Miniconda install
src/runner.rs    venv/conda-env creation, GPU-aware pip install, self-test, run/stop
requirements.txt package list shipped with the app
```
