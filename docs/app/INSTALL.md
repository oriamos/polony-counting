# Installing Polony Review (one command, no admin rights)

Polony Review is a local web app: it runs on **your own computer** and opens in your normal web browser.
Nothing is uploaded anywhere. The installer puts a **Polony Review** icon on your Desktop.

## Install

**Windows** (open *PowerShell*: Start menu, type `powershell`, Enter):

```powershell
irm https://raw.githubusercontent.com/oriamos/polony-counting/claude/polony-hitl-app/install/install.ps1 | iex
```

**macOS** (open *Terminal*: Cmd+Space, type `terminal`, Enter) **and Linux**:

```bash
curl -fsSL https://raw.githubusercontent.com/oriamos/polony-counting/claude/polony-hitl-app/install/install.sh | bash
```

Alternative if you cloned the repository (`git clone https://github.com/oriamos/polony-counting`, `git checkout claude/polony-hitl-app`):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install\install.ps1       # Windows
```
```bash
bash install/install.sh                                                        # macOS / Linux
```

Add `-DryRun` (Windows) or `--dry-run` (`curl ... | bash -s -- --dry-run`) to see what would be done without changing anything.

### Prerequisites

None. You do not need Python, Node, Git or administrator rights, and the installer never uses `sudo`.
If Python 3.10 to 3.13 is already on the machine it is used; otherwise the installer downloads the small tool
[`uv`](https://astral.sh/uv) (from `https://astral.sh/uv/install.ps1` or `install.sh`, kept inside the install folder, your
`PATH` is not modified) and with it a standalone **Python 3.12** (about 30 MB). It tells you before it does so.
The app code is fetched with `git` if present, otherwise as a zip/tar.gz of the branch from `github.com`.
Internet access to `github.com`, `pypi.org` (and `astral.sh` / `github.com` release files if Python must be downloaded) is needed once.

### What is installed where, and how big

| What | Windows | macOS / Linux |
|---|---|---|
| Program (code, private Python environment, logs, settings) | `%LOCALAPPDATA%\PolonyReview` | `~/PolonyReview` |
| **Your data** (analyses, results, exports) | `Documents\PolonyReviewData` | `~/PolonyReviewData` |
| Icon | Desktop + Start Menu `Polony Review.lnk` (+ `Stop Polony Review`) | macOS: `~/Applications/Polony Review.app` and a copy on the Desktop; Linux: `Polony Review.desktop` on the Desktop and in the app menu |

Disk space: about 650 MB for the program (measured on Linux) (mostly scientific Python libraries), plus your data.
Nothing is written outside these folders. The installation takes a few minutes (mostly downloading libraries).

### First start

Double-click the icon. The **first start can take up to a minute** (heavy libraries are loaded); later starts take a few seconds.
Your browser opens `http://127.0.0.1:8765/` (the launcher picks another free port automatically if 8765 is taken and
remembers it). The server keeps running in the background after you close the browser tab, so you can reopen the page by
clicking the icon again. It is never shut down automatically (a running detection must not be interrupted).

## Stop

* Windows: Start Menu > Polony Review > **Stop Polony Review**.
* macOS: open `~/Applications/Stop Polony Review.app`.
* Any system: `~/PolonyReview/bin/polony-review stop` (macOS/Linux) or
  `%LOCALAPPDATA%\PolonyReview\venv\Scripts\python.exe -m polony_app.launcher stop` (Windows).
  `... status` shows whether it is running.

Restarting or shutting down the computer also stops it.

## Update

Run the same install command again. It stops the running app, fetches the newest code, updates the libraries and
recreates the icons. Your data is untouched.

## Uninstall

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\PolonyReview\app\install\uninstall.ps1"           # Windows
```
```bash
bash ~/PolonyReview/app/install/uninstall.sh                                                                             # macOS / Linux
```

This removes the program, the Python environment and the icons but **keeps your data folder**. Add `-PurgeData`
(Windows) or `--purge-data` (macOS/Linux) to delete the data as well. `-DryRun` / `--dry-run` only lists what would be removed.

## Optional: shared database (several people see the same analyses)

During the install you are asked: *"Use a shared database? Paste the connection string (input is hidden) or press Enter for local-only."*
Press Enter to stay local (default). The connection string (a `postgresql://...` URL that contains a password, handed to you by the
project owner, see [SHARED_DATABASE.md](SHARED_DATABASE.md)) is typed without echo, never shown, never logged and never put on a command
line; it is saved only in `config.json` in the install folder (file permissions: owner only; on Windows the access list is restricted to your user).

* **Original TIFF files always stay on your computer.** Only analyses, counts, review history and the SHA-256 of each original are shared;
  to open an analysis made elsewhere you attach the matching original file.
* Unattended installs: set `POLONY_DATABASE_URL` before running the installer (it is saved the same way and removed from the environment).
  No prompt is shown when there is no terminal or `POLONY_CI=1`.
* Re-running the installer **keeps** an existing database setting; to replace it answer *n* at the prompt or set `POLONY_DATABASE_REPLACE=1`.
* Change it later: `~/PolonyReview/bin/polony-review set-database` / `clear-database` (macOS/Linux), Windows: Start Menu > Polony Review >
  *Set shared database* (or `...\venv\Scripts\python.exe -m polony_app.launcher set-database` / `clear-database`), then stop and start the app.
  `... status` shows `Shared database: connected (postgresql://user:***@host/db)` or `Local only`.
* Never copy `config.json` to another computer (it holds the password and this computer's id); run the installer there instead.

## Optional: AI detector (Spotiflow + PyTorch)

The default install detects spots with **Big-FISH** and needs no PyTorch. The **AI detector** ([Spotiflow](https://github.com/weigertlab/spotiflow))
adds the `spotiflow` and `hybrid` detection engines (see [TWO_ENGINE.md](TWO_ENGINE.md)). It is **opt-in** and never installed unless you ask:

```bash
curl -fsSL <install.sh URL> | bash -s -- --with-ai          # macOS / Linux  (--without-ai: never ask)
```
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install\install.ps1 -WithAi      # Windows (saved script)
$env:POLONY_WITH_AI = '1'; irm <install.ps1 URL> | iex                                # Windows one-liner
```
Interactively (a terminal is present, not CI) the installer asks *"Install the AI detector ... [y/N]"* (default **No**). Non-interactive: `POLONY_WITH_AI=1`.
You can also add it later by re-running the installer with the flag.

| Platform | What is installed | Extra download / disk |
|---|---|---|
| Windows, Linux (x86-64), no NVIDIA GPU | PyTorch **CPU** wheels from `download.pytorch.org/whl/cpu` | measured on Linux: venv 646 MB -> 1.8 GB |
| Windows, Linux with an NVIDIA GPU (`nvidia-smi` found) | asks whether to use the **CUDA 12.6** build (`POLONY_AI_CUDA=1` for unattended) | about 2.5 GB or more |
| macOS Apple Silicon | default PyPI wheels, uses the Apple GPU (MPS) when available | about 0.8 GB |
| Intel Macs, Windows on ARM | not available (no current PyTorch wheels); Big-FISH keeps working | none |

`torch` and `torchvision` are installed **together as a matching, pinned pair** (torch 2.14.1 / torchvision 0.29.1; a mismatched torchvision breaks
Spotiflow with `torchvision::nms does not exist`), then `spotiflow==0.6.5` (`install/extra-ai-requirements.txt`), then an import check that also
calls a torchvision operator. If anything fails the installer says so and **Big-FISH still works**. The result is recorded as `"ai_enabled"` in `config.json`.

**Model:** the first AI detection needs the pretrained model (about 88 MB, downloaded once into `~/PolonyReview/models/spotiflow`). To fetch it now,
or on a computer that will later be offline: `~/PolonyReview/bin/polony-review prepare-model [--model general]` (Windows:
`...\venv\Scripts\python.exe -m polony_app.launcher prepare-model`); it prints the model's sha256. For a computer without internet copy the model
folder (`general/` with `config.yaml` and the weights) into that directory, or set `POLONY_MODEL_DIR`.
`polony-review status` shows `AI detector: installed (spotiflow x, torch y, device cpu|cuda|mps)` or `not installed`.

**Why opt-in and not bundled:** PyTorch makes the installation about 3x larger here (and several GB with CUDA) and the install much slower, while most
reviews only need Big-FISH. Uninstalling removes the AI packages together with the program, but **keeps the downloaded models** (`~/PolonyReview/models`)
unless you pass `--purge-data` / `-PurgeData`.

## Security note

The app has **no login**. It listens on `127.0.0.1` (this computer only) and cannot be reached from the network, but anyone
who can use your computer account can open it and see your data. Do not expose the port with port-forwarding or tunnels.

## Troubleshooting

* **Windows: "running scripts is disabled on this system"** when running a saved `.ps1`: use the form
  `powershell -NoProfile -ExecutionPolicy Bypass -File install\install.ps1` (the `irm ... | iex` one-liner is not affected).
* **Windows SmartScreen** may warn the first time you double-click the icon or a downloaded script ("More info" > "Run anyway").
  The icon is a normal shortcut to the private Python in the install folder; nothing is signed.
* **macOS Gatekeeper** ("cannot be opened because the developer cannot be verified"): the installer removes the quarantine flag, so this
  should not appear. If it does: right-click the app > **Open** > **Open**, or run `xattr -dr com.apple.quarantine "$HOME/Applications/Polony Review.app"`.
  If macOS offers to install "command line developer tools" during the install, you can click *Cancel*: the installer then downloads its own Python.
* **Nothing happens / the browser does not open:** open the log (`~/PolonyReview/logs/launcher.log`, `server.log`; Windows
  `%LOCALAPPDATA%\PolonyReview\logs`) and open `http://127.0.0.1:8765/` yourself. Run `... launcher status` to see the actual port.
  A message box also explains a failed start.
* **Installer fails while installing libraries:** check your internet connection / proxy, then run the install command again (it resumes).
* **Linux: "ensurepip is not available"**: the installer falls back to a downloaded Python automatically.
* **Antivirus/corporate laptops** may block downloads from `astral.sh`, `github.com` or `pypi.org`; ask IT to allow them or install Python 3.12 yourself first.
* **Reinstall from scratch:** uninstall (keeps data) and install again.

## For developers

* The installer serves the prebuilt UI in `src/polony_app/webdist/` (committed) when `frontend/dist` does not exist. **Whenever the
  frontend changes, run `make web` and commit `src/polony_app/webdist`**, otherwise installed copies show an outdated UI.
* `make package-icons` regenerates the icons in `install/assets/` from `install/make_icons.py` (Pillow).
* `install/extra-requirements.txt` lists libraries installed in addition to `requirements-app.txt` (currently `openpyxl`, which the Excel export imports).
* Launcher: `python -m polony_app.launcher [launch|stop|status] [--no-browser]`, `--self-test`. Environment: `POLONY_HOME`, `POLONY_DATA_DIR`, `POLONY_PORT`, `POLONY_CI=1`.
* Installer test hooks: `POLONY_SOURCE=<checkout>` (copy a local checkout instead of downloading), `POLONY_FORCE_UV=1` (force the download-Python path), `POLONY_FORCE_OS=mac|linux`, `--shortcuts-only`.
* Tests: `tests/app/test_installer_*.py` (unit + script checks; `POLONY_E2E_INSTALL=1` also runs a full install in a temp HOME) and `.github/workflows/install-test.yml` (Windows / macOS / Linux runners).

## What has been tested

* **Linux (this repository's development machine):** the AI extra was installed end to end (CPU wheels), `status` and `prepare-model` (real 88 MB download) ran, and a hybrid detection ran through the installed app; full install into a temporary HOME from a local checkout (system Python 3.12, and the
  `uv` path), launcher start, `/api/health`, UI at `/`, stop, re-run (idempotent), uninstall keeping data; shellcheck clean; PowerShell scripts parse-checked with PowerShell 7.4 on Linux.
* **macOS:** the `.app` bundle structure is generated and checked on Linux (Info.plist, executable bit, icns); running it on macOS is verified only by the CI workflow.
* **Windows:** the PowerShell scripts are parse-checked only; actual installation, `.lnk` creation and launching are verified only by the CI workflow
  (`install-test`, Windows PowerShell 5.1 and PowerShell 7). The AI extra on Windows/macOS can be CI-verified later by running the workflow manually with `with_ai` ticked. Until that workflow is green on the branch, treat Windows and macOS as untested on real machines.
