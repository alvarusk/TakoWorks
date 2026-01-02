param(
    [string]$Repo = "alvarusk/takoworks",
    [string]$InstallDir = (Join-Path $PSScriptRoot "TakoWorks"),
    [string]$AssetPattern = "TakoWorks_win64.zip",
    [switch]$AllowPrerelease
)

$ErrorActionPreference = "Stop"

function Assert-InstallWritable {
    param([string]$TargetPath)
    $parent = Split-Path -Path $TargetPath -Parent
    if (-not $parent) { $parent = $TargetPath }
    if (-not (Test-Path -Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force -ErrorAction Stop | Out-Null
    }
    $probe = Join-Path $parent ("._takoworks_probe_" + [guid]::NewGuid().ToString("N"))
    try {
        Set-Content -Path $probe -Value "probe" -ErrorAction Stop
        Remove-Item -Path $probe -Force -ErrorAction SilentlyContinue
    } catch {
        throw "No write permission at '$parent'. Run PowerShell as Administrator or choose a writable -InstallDir (e.g. `$env:LOCALAPPDATA\\TakoWorks`)."
    }
}

Assert-InstallWritable -TargetPath $InstallDir

function Get-AuthHeaders {
    $headers = @{ "User-Agent" = "TakoWorks-Updater" }
    if ($env:GITHUB_TOKEN) {
        $headers["Authorization"] = "Bearer $($env:GITHUB_TOKEN)"
    }
    return $headers
}

Write-Host "Checking releases for $Repo..."
$releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases" -Headers (Get-AuthHeaders)

if (-not $AllowPrerelease) {
    $releases = $releases | Where-Object { -not $_.prerelease -and -not $_.draft }
} else {
    $releases = $releases | Where-Object { -not $_.draft }
}

$release = $releases | Sort-Object { [datetime]$_.published_at } -Descending | Select-Object -First 1
if (-not $release) { throw "No release found for $Repo." }

$asset = $release.assets | Where-Object { $_.name -like $AssetPattern } | Select-Object -First 1
if (-not $asset) { throw "No asset matching '$AssetPattern' in release $($release.tag_name)." }

$tmpDir = Join-Path ([IO.Path]::GetTempPath()) ("takoworks_update_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmpDir | Out-Null
$zipPath = Join-Path $tmpDir $asset.name

Write-Host "Downloading $($asset.name) from release $($release.tag_name)..."
# Para repos privados, usa el endpoint de assets con Accept octet-stream (browser_download_url puede dar 404)
$assetUri = "https://api.github.com/repos/$Repo/releases/assets/$($asset.id)"
$headers = Get-AuthHeaders
$headers["Accept"] = "application/octet-stream"
Invoke-WebRequest -Uri $assetUri -Headers $headers -OutFile $zipPath

$extractDir = Join-Path $tmpDir "extracted"
Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

$payloadRoot = $extractDir

$timestamp = Get-Date -Format "yyyyMMddHHmmss"
if (Test-Path $InstallDir) {
    $backupDir = "$InstallDir.bak.$timestamp"
    Write-Host "Existing install found. Backing up to $backupDir"
    Move-Item -Path $InstallDir -Destination $backupDir -Force
}

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Write-Host "Copying files into $InstallDir"
Copy-Item -Path (Join-Path $payloadRoot "*") -Destination $InstallDir -Recurse -Force

Write-Host "Update completed. Installed release $($release.tag_name) to $InstallDir"
Write-Host "Cleanup temp folder: $tmpDir (removed automatically on reboot if you skip manual cleanup)."
