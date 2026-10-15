# Polony Review uninstaller for Windows (PowerShell 5.1 and 7+).
# Removes the app, its Python environment and the shortcuts. Your data folder is KEPT unless -PurgeData is given.
#   powershell -NoProfile -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\PolonyReview\app\install\uninstall.ps1" [-PurgeData] [-DryRun]

function Uninstall-PolonyReview {
    param([string[]]$Argv)
    $ErrorActionPreference = 'Stop'
    $purge = $false
    $dry = ($env:POLONY_DRYRUN -and $env:POLONY_DRYRUN -ne '0')
    foreach ($a in $Argv) {
        switch -Regex ($a) {
            '^-{1,2}PurgeData$' { $purge = $true }
            '^-{1,2}DryRun$' { $dry = $true }
            default { throw "Unknown option: $a" }
        }
    }
    $ph = if ($env:POLONY_HOME) { $env:POLONY_HOME } else { Join-Path $env:LOCALAPPDATA 'PolonyReview' }
    $docs = [Environment]::GetFolderPath('MyDocuments')
    if (-not $docs) { $docs = $HOME }
    $data = if ($env:POLONY_DATA_DIR) { $env:POLONY_DATA_DIR } else { Join-Path $docs 'PolonyReviewData' }
    $cfg = Join-Path $ph 'config.json'
    if (-not $env:POLONY_DATA_DIR -and (Test-Path $cfg)) {
        try { $c = Get-Content $cfg -Raw | ConvertFrom-Json; if ($c.data_dir) { $data = $c.data_dir } } catch { }
    }
    $vpy = Join-Path $ph 'venv\Scripts\python.exe'
    $desktop = [Environment]::GetFolderPath('Desktop')
    $startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'Polony Review'

    function Remove-Thing($p) {
        if (Test-Path $p) {
            if ($dry) { Write-Host "  [dry-run] would remove $p" -ForegroundColor Yellow }
            else { Remove-Item -Recurse -Force $p; Write-Host "  removed $p" }
        }
    }

    Write-Host "Uninstalling Polony Review from $ph"
    if ($dry) {
        Write-Host '  [dry-run] would stop a running server' -ForegroundColor Yellow
    } elseif (Test-Path $vpy) {
        $env:POLONY_HOME = $ph
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & $vpy -m polony_app.launcher stop 2>&1 | ForEach-Object { Write-Host "  $_" }
        $ErrorActionPreference = $old
    }
    Remove-Thing (Join-Path $desktop 'Polony Review.lnk')
    Remove-Thing $startMenu
    if ($ph.TrimEnd('\', '/').Length -le 3) { throw "Refusing to remove '$ph'" }
    # do not stay inside the folder we delete (this script itself lives there)
    Set-Location $env:TEMP
    if (-not $purge -and (Test-Path (Join-Path $ph 'models'))) {
        # keep the downloaded AI models (~90 MB each) unless -PurgeData; everything else goes
        Get-ChildItem -LiteralPath $ph -Force | Where-Object { $_.Name -ne 'models' } | ForEach-Object { Remove-Thing $_.FullName }
        Write-Host "  kept the downloaded AI models in $(Join-Path $ph 'models') (use -PurgeData to delete them too)"
    } else {
        Remove-Thing $ph
    }
    if ($purge) {
        if ($data.TrimEnd('\', '/').Length -le 3) { throw "Refusing to remove '$data'" }
        Remove-Thing $data
    } elseif (Test-Path $data) {
        Write-Host "  kept your data in $data (delete it yourself, or re-run with -PurgeData)"
    }
    Write-Host 'Done.'
}

Uninstall-PolonyReview -Argv $args
