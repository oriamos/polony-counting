# CI assertions for the Windows installer. Usage: check_windows.ps1 -Mode shortcuts|run|ai|uninstall
param([Parameter(Mandatory = $true)][string]$Mode)
$ErrorActionPreference = 'Stop'
$ph = if ($env:POLONY_HOME) { $env:POLONY_HOME } else { Join-Path $env:LOCALAPPDATA 'PolonyReview' }
$vpy = Join-Path $ph 'venv\Scripts\python.exe'
$vpyw = Join-Path $ph 'venv\Scripts\pythonw.exe'
$desktop = [Environment]::GetFolderPath('Desktop')
$startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'Polony Review'
function Ok($m) { Write-Host "OK: $m" }
function Bad($m) { Write-Host "FAIL: $m"; exit 1 }
function Launcher([string[]]$a) {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $o = & $vpy -m polony_app.launcher @a 2>&1 | Out-String
    $c = $LASTEXITCODE
    $ErrorActionPreference = $old
    return @{ Code = $c; Out = $o.Trim() }
}
function Read-Lnk($path) {
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
    return @{ Target = $s.TargetPath; Args = $s.Arguments; Icon = $s.IconLocation; Dir = $s.WorkingDirectory }
}

switch ($Mode) {
    'shortcuts' {
        foreach ($f in $vpy, $vpyw, (Join-Path $ph 'config.json'), (Join-Path $ph 'app\src\polony_app\webdist\index.html')) {
            if (-not (Test-Path $f)) { Bad "missing $f" }
        }
        Ok 'install folder complete'
        foreach ($l in (Join-Path $desktop 'Polony Review.lnk'), (Join-Path $startMenu 'Polony Review.lnk'), (Join-Path $startMenu 'Stop Polony Review.lnk')) {
            if (-not (Test-Path $l)) { Bad "shortcut missing: $l" }
            $s = Read-Lnk $l
            Write-Host "  $l -> $($s.Target) $($s.Args)  icon=$($s.Icon)"
            if (-not $s.Target.EndsWith('pythonw.exe')) { Bad "target is not pythonw.exe: $($s.Target)" }
            if (-not (Test-Path $s.Target)) { Bad "target does not exist: $($s.Target)" }
            if (-not $s.Args.StartsWith('-m polony_app.launcher')) { Bad "unexpected arguments: $($s.Args)" }
            $ico = $s.Icon.Split(',')[0]
            if (-not ($ico.EndsWith('.ico') -and (Test-Path $ico))) { Bad "icon missing: $($s.Icon)" }
        }
        Ok 'Desktop and Start Menu shortcuts are valid'
        $r = Launcher @('--self-test'); Write-Host $r.Out
        if ($r.Code -ne 0) { Bad 'self-test failed' }
    }
    'run' {
        # start exactly what the Desktop shortcut starts: pythonw.exe -m polony_app.launcher (no console window)
        $s = Read-Lnk (Join-Path $desktop 'Polony Review.lnk')
        Start-Process -FilePath $s.Target -ArgumentList $s.Args -WorkingDirectory $s.Dir -WindowStyle Hidden -Wait
        $url = $null
        for ($i = 0; $i -lt 60 -and -not $url; $i++) {
            $r = Launcher @('status')
            if ($r.Code -eq 0 -and $r.Out -match 'running: (http://\S+)') { $url = $Matches[1] } else { Start-Sleep -Seconds 2 }
        }
        if (-not $url) { Get-Content (Join-Path $ph 'logs\*.log') -ErrorAction SilentlyContinue | Select-Object -Last 60; Bad 'server not running after launching the shortcut target' }
        $h = Invoke-RestMethod -UseBasicParsing "${url}api/health"
        if (-not $h.ok) { Bad 'health failed' }
        $page = Invoke-WebRequest -UseBasicParsing $url
        if ($page.Content -notmatch '<div id="root">') { Bad 'UI not served at /' }
        Ok "launched: $url (health + UI)"
        [void](Start-Process -FilePath $vpyw -ArgumentList '-m polony_app.launcher' -WindowStyle Hidden -Wait)   # second launch reuses
        $r = Launcher @('status'); if ($r.Out -notmatch [regex]::Escape($url)) { Bad "second launch changed the server: $($r.Out)" }
        $r = Launcher @('stop'); Write-Host $r.Out
        Start-Sleep -Seconds 2
        $r = Launcher @('status'); if ($r.Code -eq 0) { Bad 'still running after stop' }
        Ok 'stopped'
    }
    'ai' {
        $r = Launcher @('status'); Write-Host $r.Out
        if ($r.Out -notmatch 'AI detector: installed \(spotiflow') { Bad 'AI detector not reported as installed' }
        $r = Launcher @('prepare-model'); Write-Host ($r.Out -split "`n" | Select-Object -Last 3)
        if ($r.Code -ne 0 -or $r.Out -notmatch 'sha256: [0-9a-f]{64}') { Bad 'prepare-model failed' }
        if ((Get-Content (Join-Path $ph 'config.json') -Raw) -notmatch '"ai_enabled":\s*true') { Bad 'ai_enabled not recorded' }
        Ok 'AI detector installed and model prepared'
    }
    'uninstall' {
        $un = Join-Path $ph 'app\install\uninstall.ps1'
        & powershell -NoProfile -ExecutionPolicy Bypass -File $un
        if ($LASTEXITCODE -ne 0) { Bad 'uninstall failed' }
        if (Test-Path $ph) { Bad 'install folder still there' }
        if (Test-Path (Join-Path $desktop 'Polony Review.lnk')) { Bad 'desktop shortcut still there' }
        if (Test-Path $startMenu) { Bad 'start menu folder still there' }
        $docs = [Environment]::GetFolderPath('MyDocuments')
        $data = if ($env:POLONY_DATA_DIR) { $env:POLONY_DATA_DIR } else { Join-Path $docs 'PolonyReviewData' }
        if (-not (Test-Path $data)) { Bad 'user data was deleted' }
        Ok 'uninstalled, data kept'
    }
    default { Bad "unknown mode $Mode" }
}
