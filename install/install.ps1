# Polony Review installer for Windows (PowerShell 5.1 and 7+). No administrator rights needed.
#
#   irm https://raw.githubusercontent.com/oriamos/polony-counting/main/install/install.ps1 | iex
#
# Re-running it updates the installation. Options when run from a saved file:
#   powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 [-DryRun] [-NoLaunch] [-WithAi|-WithoutAi] [-Ref NAME] [-Python PATH]
# Environment: POLONY_HOME (default %LOCALAPPDATA%\PolonyReview), POLONY_DATA_DIR (default Documents\PolonyReviewData),
#   POLONY_SOURCE (local checkout to copy instead of downloading), POLONY_REPO, POLONY_REF,
#   POLONY_CI=1 (no browser, no prompts; used by the CI tests), POLONY_DRYRUN=1 (same as -DryRun).
#
# Steps: 1) fetch the code into <home>\app (git if present, else a GitHub zip), 2) find Python 3.10-3.13 or, if
# none, download `uv` (https://astral.sh/uv) and with it a standalone Python 3.12 into <home>, 3) create
# <home>\venv and pip install requirements-app.txt, 4) self-test, 5) create the Desktop + Start Menu shortcuts.
# Everything is inside one function that is only called on the last line, so a truncated download runs nothing.

function Install-PolonyReview {
    param([string[]]$Argv)

    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'   # makes Invoke-WebRequest much faster on PowerShell 5.1
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }

    # ---- options ---------------------------------------------------------------------------------
    $repo = if ($env:POLONY_REPO) { $env:POLONY_REPO } else { 'oriamos/polony-counting' }
    $ref = if ($env:POLONY_REF) { $env:POLONY_REF } else { 'main' }
    $dry = ($env:POLONY_DRYRUN -and $env:POLONY_DRYRUN -ne '0')
    $ci = ($env:POLONY_CI -and $env:POLONY_CI -ne '0')
    $noLaunch = $false
    $shortcutsOnly = $false
    $withAi = ''
    $pyArg = ''
    for ($i = 0; $i -lt $Argv.Count; $i++) {
        switch -Regex ($Argv[$i]) {
            '^-{1,2}DryRun$' { $dry = $true }
            '^-{1,2}NoLaunch$' { $noLaunch = $true }
            '^-{1,2}WithAi$' { $withAi = '1' }
            '^-{1,2}WithoutAi$' { $withAi = '0' }
            '^-{1,2}ShortcutsOnly$' { $shortcutsOnly = $true; $noLaunch = $true }
            '^-{1,2}Ref$' { $i++; $ref = $Argv[$i] }
            '^-{1,2}Python$' { $i++; $pyArg = $Argv[$i] }
            default { throw "Unknown option: $($Argv[$i])" }
        }
    }
    if ($ci) { $noLaunch = $true }
    if (-not $withAi -and $env:POLONY_WITH_AI) { $withAi = $env:POLONY_WITH_AI }
    # optional shared database URL: taken out of the environment right away so no other process inherits it
    $dbUrl = $env:POLONY_DATABASE_URL
    Remove-Item Env:\POLONY_DATABASE_URL -ErrorAction SilentlyContinue

    $ph = if ($env:POLONY_HOME) { $env:POLONY_HOME } else { Join-Path $env:LOCALAPPDATA 'PolonyReview' }
    $docs = [Environment]::GetFolderPath('MyDocuments')
    if (-not $docs) { $docs = $HOME }
    $data = if ($env:POLONY_DATA_DIR) { $env:POLONY_DATA_DIR } else { Join-Path $docs 'PolonyReviewData' }
    $app = Join-Path $ph 'app'
    $venv = Join-Path $ph 'venv'
    $vpy = Join-Path $venv 'Scripts\python.exe'
    $vpyw = Join-Path $venv 'Scripts\pythonw.exe'
    $steps = 7

    function Say($m) { Write-Host $m }
    function Step($n, $m) { Write-Host ''; Write-Host "[$n/$steps] $m" -ForegroundColor Cyan }
    function Plan($m) { Write-Host "  [dry-run] $m" -ForegroundColor Yellow }
    function Warn($m) { Write-Host "WARNING: $m" -ForegroundColor Yellow }
    function Fail($m) {
        Write-Host ''
        Write-Host "ERROR: $m" -ForegroundColor Red
        Write-Host 'Nothing is broken by this: you can run the installer again.' -ForegroundColor Red
        throw $m
    }
    # Run a program, show its output, return the exit code (stderr output must not become an exception on 5.1).
    function Invoke-Show([string]$exe, [string[]]$a) {
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        try { & $exe @a 2>&1 | ForEach-Object { Write-Host "    $_" }; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $old }
        return $code
    }
    # Run a program quietly; returns @{Code=..; Out=..}
    function Invoke-Quiet([string]$exe, [string[]]$a) {
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        try { $o = & $exe @a 2>$null; $code = $LASTEXITCODE } catch { $o = ''; $code = 1 } finally { $ErrorActionPreference = $old }
        return @{ Code = $code; Out = (($o | Out-String).Trim()) }
    }
    function Has($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

    Say 'Polony Review installer (Windows)'
    Say "  install folder : $ph"
    Say "  your data      : $data   (never deleted by updates or the uninstaller unless you ask)"
    $srcText = if ($env:POLONY_SOURCE) { $env:POLONY_SOURCE } else { "github.com/$repo ($ref)" }
    Say "  source         : $srcText"
    if ($dry) { Say '  DRY RUN: nothing will be changed.' }

    $pyCode = "import sys; v=sys.version_info[:2]; assert (3,10)<=v<=(3,13); print(sys.executable)"

    if (-not $shortcutsOnly) {
        # ---- 1. code ----------------------------------------------------------------------------
        Step 1 "Getting the app code into $app"
        if ($dry) {
            Plan "stop a running Polony Review; fetch code -> $app"
        } else {
            if ((Test-Path $vpy) -and (Test-Path $app)) {
                Say '  stopping a running Polony Review (if any) so the update takes effect'
                $env:POLONY_HOME = $ph
                [void](Invoke-Quiet $vpy @('-m', 'polony_app.launcher', 'stop'))
            }
            New-Item -ItemType Directory -Force -Path $ph | Out-Null
            Push-Location $ph
            $tmp = Join-Path $ph 'app.new'
            if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
            $done = $false
            if ($env:POLONY_SOURCE) {
                if (-not (Test-Path (Join-Path $env:POLONY_SOURCE 'src\polony_app'))) { Fail "POLONY_SOURCE=$($env:POLONY_SOURCE) is not a Polony checkout" }
                Say "  copying the local checkout $($env:POLONY_SOURCE)"
                $null = & robocopy $env:POLONY_SOURCE $tmp /E /NFL /NDL /NJH /NJS /NP /XD .git node_modules __pycache__
                if ($LASTEXITCODE -ge 8) { Fail "copying the checkout failed (robocopy exit code $LASTEXITCODE)" }
                $done = $true
            } elseif ((Has 'git') -and (Test-Path (Join-Path $app '.git'))) {
                Say '  updating the existing copy with git'
                $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
                & git -C $app fetch --depth 1 origin $ref 2>&1 | Out-Null
                $c1 = $LASTEXITCODE
                if ($c1 -eq 0) { & git -C $app checkout -q -f FETCH_HEAD 2>&1 | Out-Null; $c1 = $LASTEXITCODE }
                $ErrorActionPreference = $old
                if ($c1 -eq 0) { $done = $true; $tmp = $null } else { Warn 'git update failed; downloading a fresh copy instead' }
            } elseif (Has 'git') {
                Say "  cloning https://github.com/$repo.git ($ref)"
                $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
                & git clone -q --depth 1 --branch $ref "https://github.com/$repo.git" $tmp 2>&1 | Out-Null
                $c2 = $LASTEXITCODE
                $ErrorActionPreference = $old
                if ($c2 -eq 0) { $done = $true } else { Warn 'git clone failed; downloading the zip archive instead'; if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp } }
            }
            if (-not $done) {
                $url = "https://github.com/$repo/archive/refs/heads/$ref.zip"
                $zip = Join-Path $ph 'app.zip'
                $ex = Join-Path $ph 'app.unzip'
                Say "  downloading $url"
                try { Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip } catch { Fail "download of $url failed: $($_.Exception.Message)" }
                if (Test-Path $ex) { Remove-Item -Recurse -Force $ex }
                Expand-Archive -Path $zip -DestinationPath $ex -Force
                $inner = Get-ChildItem $ex | Where-Object { $_.PSIsContainer } | Select-Object -First 1
                if (-not $inner) { Fail 'the downloaded archive is empty' }
                Move-Item $inner.FullName $tmp
                Remove-Item -Recurse -Force $ex, $zip
            }
            if ($tmp) {
                if (Test-Path $app) { Remove-Item -Recurse -Force $app }
                Move-Item $tmp $app
            }
            Pop-Location
            if (-not (Test-Path (Join-Path $app 'requirements-app.txt'))) { Fail 'the downloaded code has no requirements-app.txt' }
        }

        # ---- 2. python + venv -------------------------------------------------------------------
        Step 2 "Preparing Python (3.10-3.13) and a private virtual environment in $venv"
        function Find-Python {
            if ($env:POLONY_FORCE_UV) { return $null }   # test hook: exercise the "download Python with uv" path
            $cands = @()
            if ($pyArg) { $cands += , @($pyArg) }
            if (Has 'py') { foreach ($v in '3.12', '3.11', '3.13', '3.10') { $cands += , @('py', "-$v") } }
            $cands += , @('python')
            $cands += , @('python3')
            foreach ($c in $cands) {
                if (-not (Has $c[0])) { continue }
                $rest = @(); if ($c.Count -gt 1) { $rest = $c[1..($c.Count - 1)] }
                $r = Invoke-Quiet $c[0] ($rest + @('-c', $pyCode))
                if ($r.Code -eq 0 -and $r.Out -and (Test-Path $r.Out)) { return $r.Out }
            }
            return $null
        }
        if ($dry) {
            $p = Find-Python
            if ($p) { Plan "use existing Python at $p -> venv $venv" }
            else { Plan "no Python 3.10-3.13 found: would download uv (astral.sh/uv/install.ps1) into $ph\uv-bin and Python 3.12 into $ph\python" }
        } else {
            $reuse = $false
            if (Test-Path $vpy) {
                $r = Invoke-Quiet $vpy @('-c', $pyCode)
                $r2 = Invoke-Quiet $vpy @('-m', 'pip', '--version')
                if ($r.Code -eq 0 -and $r2.Code -eq 0) { $reuse = $true; Say '  reusing the existing environment' }
            }
            if (-not $reuse) {
                $made = $false
                $p = Find-Python
                if ($p) {
                    if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
                    $r = Invoke-Quiet $p @('-m', 'venv', $venv)
                    $r2 = if (Test-Path $vpy) { Invoke-Quiet $vpy @('-m', 'pip', '--version') } else { @{ Code = 1 } }
                    if ($r.Code -eq 0 -and $r2.Code -eq 0) { $made = $true; Say "  using the Python found at $p" }
                    else { Warn "could not create a virtual environment with $p; using a downloaded Python instead" }
                }
                if (-not $made) {
                    $uvDir = Join-Path $ph 'uv-bin'
                    $uv = Join-Path $uvDir 'uv.exe'
                    if (-not (Test-Path $uv)) {
                        Say "  No suitable Python found. Downloading the small 'uv' tool from https://astral.sh/uv/install.ps1"
                        Say "  (it is installed only inside $uvDir; your PATH is not modified)."
                        New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
                        $inst = Join-Path $ph 'uv-install.ps1'
                        try { Invoke-WebRequest -UseBasicParsing -Uri 'https://astral.sh/uv/install.ps1' -OutFile $inst } catch { Fail "could not download the uv installer: $($_.Exception.Message)" }
                        $env:UV_UNMANAGED_INSTALL = $uvDir
                        $env:UV_NO_MODIFY_PATH = '1'
                        $shell = if (Has 'powershell') { 'powershell' } else { 'pwsh' }
                        $code = Invoke-Show $shell @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $inst)
                        Remove-Item -Force $inst -ErrorAction SilentlyContinue
                        if ($code -ne 0 -or -not (Test-Path $uv)) { Fail 'the uv installer failed' }
                    }
                    Say '  Downloading a standalone Python 3.12 with uv (about 30 MB) ...'
                    if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
                    $env:UV_PYTHON_INSTALL_DIR = Join-Path $ph 'python'
                    $code = Invoke-Show $uv @('venv', '--seed', '--python', '3.12', $venv)
                    if ($code -ne 0 -or -not (Test-Path $vpy)) { Fail 'uv could not set up Python 3.12' }
                }
            }
        }

        # ---- 3. dependencies --------------------------------------------------------------------
        Step 3 "Installing the app's libraries (a few minutes the first time; downloads roughly 300 MB)"
        if ($dry) {
            Plan "$vpy -m pip install -r $app\requirements-app.txt -r $app\install\extra-requirements.txt"
            Plan "write $venv\Lib\site-packages\polony_review.pth pointing at $app\src"
        } else {
            [void](Invoke-Quiet $vpy @('-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', 'pip'))
            $code = Invoke-Show $vpy @('-m', 'pip', 'install', '-q', '--disable-pip-version-check', '-r', (Join-Path $app 'requirements-app.txt'), '-r', (Join-Path $app 'install\extra-requirements.txt'))
            if ($code -ne 0) { Fail 'installing the libraries failed (see the messages above)' }
            $sp = (Invoke-Quiet $vpy @('-c', 'import sysconfig; print(sysconfig.get_paths()[''purelib''])')).Out
            Set-Content -Path (Join-Path $sp 'polony_review.pth') -Value (Join-Path $app 'src') -Encoding ASCII
        }

        # ---- 4. checks --------------------------------------------------------------------------
        Step 4 'Checking the installation'
        if ($dry) {
            Plan "create $data, write $ph\config.json, import bigfish/tifffile/fastapi, run 'python -m polony_app.launcher --self-test'"
        } else {
            New-Item -ItemType Directory -Force -Path $data, (Join-Path $ph 'logs') | Out-Null
            $env:POLONY_HOME = $ph
            $env:POLONY_CFG_DATA = $data
            $code = Invoke-Show $vpy @((Join-Path $app 'install\write_config.py'))
            if ($code -ne 0) { Fail 'could not write config.json' }
            $code = Invoke-Show $vpy @((Join-Path $app 'install\check_imports.py'))
            if ($code -ne 0) { Fail 'the installed libraries cannot be imported' }
            Say '  starting a temporary server to test it (the first start takes a while) ...'
            $env:POLONY_CI = '1'
            $code = Invoke-Show $vpy @('-m', 'polony_app.launcher', '--self-test')
            if (-not $ci) { Remove-Item Env:\POLONY_CI -ErrorAction SilentlyContinue }
            if ($code -ne 0) { Fail 'the self-test failed' }
        }
    }

    # ---- 5. shared database (optional) ----------------------------------------------------------
    Step 5 'Shared database (optional)'
    function Set-SharedDatabase {
        $script = Join-Path $app 'install\set_database.py'
        $env:POLONY_HOME = $ph
        $url = $dbUrl
        $replace = @()
        $interactive = $false
        try { $interactive = (-not $ci) -and [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected) } catch { }
        $cur = Invoke-Quiet $vpy @($script, 'status')
        if ($cur.Code -eq 0) {
            if ($url) {
                if ($env:POLONY_DATABASE_REPLACE) { $replace = @('--replace') }
                else { Say "  A shared database is already configured ($($cur.Out)): kept it (set POLONY_DATABASE_REPLACE=1 to replace it)."; return }
            } elseif ($interactive) {
                $ans = Read-Host "  A shared database is already configured ($($cur.Out)). Keep it? [Y/n]"
                if ($ans -notmatch '^(n|no)$') { Say '  kept the existing shared database'; return }
                $replace = @('--replace')
            } else { Say "  Shared database: already configured ($($cur.Out)), unchanged."; return }
        }
        if (-not $url) {
            if (-not $interactive) {
                Say '  Local only (no prompt in non-interactive mode). Later: Start Menu > Polony Review > Set shared database'
                return
            }
            $sec = Read-Host '  Use a shared database? Paste the connection string (input is hidden) or press Enter for local-only' -AsSecureString
            if ($sec.Length -eq 0) { Say '  Local only. You can add a shared database later (Start Menu > Polony Review > Set shared database).'; return }
            $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
            try { $url = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
        }
        # the URL goes to python through stdin only (not an argument, not the environment)
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        try { $url | & $vpy $script 'set' @replace 2>&1 | ForEach-Object { Write-Host "    $_" }; $c = $LASTEXITCODE } finally { $ErrorActionPreference = $old }
        $url = $null
        if ($c -ne 0 -and $c -ne 3) { Warn 'the connection string was not saved (see above)' }
    }
    if ($dry) {
        Plan "ask (hidden input) for an optional shared-database connection string, or use POLONY_DATABASE_URL; saved only in $ph\config.json (current user only); an existing one is kept"
    } elseif (-not $shortcutsOnly) {
        Set-SharedDatabase
    }

    # ---- 6. AI detector (optional) ---------------------------------------------------------------
    Step 6 'AI detector (optional)'
    function Install-AiDetector {
        $script = Join-Path $app 'install\ai_extra.py'
        $interactive = $false
        try { $interactive = (-not $ci) -and [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected) } catch { }
        $want = $withAi
        if (-not $want) {
            if (-not $interactive) { Say '  Skipped (Big-FISH detection works without it). Add it later: re-run this installer with -WithAi'; return }
            [void](Invoke-Show $vpy @($script, 'describe'))
            $ans = Read-Host '  Install the AI detector (Spotiflow + PyTorch, large download ~0.8-2.5 GB depending on platform)? [y/N]'
            if ($ans -notmatch '^(y|yes)$') { Say '  Skipped (Big-FISH only). Add it later with -WithAi.'; return }
            $want = '1'
        }
        if ($want -eq '0') { Say '  Skipped (-WithoutAi).'; return }
        $cuda = 'no'
        if ($env:POLONY_AI_CUDA -eq '1') { $cuda = 'yes' }
        elseif ((Has 'nvidia-smi') -and $interactive) {
            $a2 = Read-Host '  An NVIDIA GPU was found. Use the CUDA build of PyTorch (about 2.5 GB instead of ~0.8 GB)? [y/N]'
            if ($a2 -match '^(y|yes)$') { $cuda = 'yes' }
        }
        Say '  Downloading and installing PyTorch + Spotiflow (this can take a long time) ...'
        $env:POLONY_HOME = $ph
        $code = Invoke-Show $vpy @($script, 'install', '--cuda', $cuda)
        if ($code -ne 0) { Warn 'the AI detector could not be installed; Big-FISH detection still works' }
    }
    if ($dry) {
        if ($withAi -eq '1') { Plan 'install the AI detector: matching torch+torchvision (CPU wheels from download.pytorch.org/whl/cpu; CUDA only if asked), then spotiflow (install\extra-ai-requirements.txt), then import check; record ai_enabled in config.json' }
        else { Plan "ask 'Install the AI detector (Spotiflow + PyTorch, ~0.8-2.5 GB)? [y/N]' (default no; skipped without a terminal/CI unless -WithAi or POLONY_WITH_AI=1)" }
    } elseif (-not $shortcutsOnly) {
        Install-AiDetector
    }

    # ---- 5. shortcuts ---------------------------------------------------------------------------
    Step 7 "Creating the 'Polony Review' icon"
    $ico = Join-Path $app 'install\assets\PolonyReview.ico'
    $desktop = [Environment]::GetFolderPath('Desktop')
    $startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'Polony Review'
    $lnkDesktop = Join-Path $desktop 'Polony Review.lnk'
    $lnkStart = Join-Path $startMenu 'Polony Review.lnk'
    $lnkStop = Join-Path $startMenu 'Stop Polony Review.lnk'
    function New-Shortcut($path, $arguments, $description) {
        $dir = Split-Path $path
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        $ws = New-Object -ComObject WScript.Shell
        $s = $ws.CreateShortcut($path)
        $s.TargetPath = $vpyw
        $s.Arguments = $arguments
        $s.WorkingDirectory = $ph
        $s.IconLocation = "$ico,0"
        $s.Description = $description
        $s.WindowStyle = 7
        $s.Save()
    }
    if ($dry) {
        Plan "create $lnkDesktop, $lnkStart and $lnkStop (target: $vpyw -m polony_app.launcher)"
    } else {
        New-Shortcut $lnkDesktop '-m polony_app.launcher' 'Review and approve polony counts (runs locally in your browser)'
        New-Shortcut $lnkStart '-m polony_app.launcher' 'Review and approve polony counts (runs locally in your browser)'
        New-Shortcut $lnkStop '-m polony_app.launcher stop' 'Stop the Polony Review background server'
        $lnkDb = Join-Path $startMenu 'Set shared database.lnk'
        $ws = New-Object -ComObject WScript.Shell
        $sc = $ws.CreateShortcut($lnkDb)
        $sc.TargetPath = $vpy                       # needs a console window for the hidden prompt
        $sc.Arguments = '-m polony_app.launcher set-database'
        $sc.WorkingDirectory = $ph
        $sc.IconLocation = "$ico,0"
        $sc.Description = 'Connect Polony Review to a shared database (paste the connection string)'
        $sc.Save()
        Say "  created $lnkDesktop and Start Menu entries"
    }

    Write-Host ''
    Say 'Done.'
    Say "  Start it from the 'Polony Review' icon on your Desktop (first start takes up to a minute)."
    Say "  Stop it:    Start Menu > Polony Review > Stop Polony Review"
    Say '  Update:     run the same install command again.'
    Say "  Uninstall:  powershell -NoProfile -ExecutionPolicy Bypass -File `"$app\install\uninstall.ps1`"   (add -PurgeData to also delete $data)"
    Say "  Your results are stored in: $data"
    if (-not $dry -and -not $noLaunch) {
        Say ''
        Say 'Starting Polony Review now ...'
        $env:POLONY_HOME = $ph
        [void](Invoke-Show $vpy @('-m', 'polony_app.launcher'))
    }
}

Install-PolonyReview -Argv $args
