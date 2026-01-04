# Validate AppxManifest.xml with makeappx (schema check only).
param(
    [string]$Manifest = "dist_installer/AppxManifest.xml",
    [string]$AssetsDir = "dist_installer/assets",
    [string]$BuildDir = "dist/TakoWorks"
)

function Find-MakeAppx {
    param([string]$PreferredArch = "x64")

    $cmd = Get-Command makeappx -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $roots = @(
        "C:\Program Files (x86)\Windows Kits\10\bin",
        "C:\Program Files\Windows Kits\10\bin"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $candidates = Get-ChildItem -Path $root -Recurse -Filter makeappx.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\$PreferredArch\\" } |
            Sort-Object FullName -Descending
        if (-not $candidates) {
            $candidates = Get-ChildItem -Path $root -Recurse -Filter makeappx.exe -ErrorAction SilentlyContinue |
                Sort-Object FullName -Descending
        }
        if ($candidates) { return $candidates[0].FullName }
    }
    throw "makeappx.exe not found on this machine"
}

$resolvedManifest = Resolve-Path -LiteralPath $Manifest -ErrorAction Stop
$resolvedAssets = Resolve-Path -LiteralPath $AssetsDir -ErrorAction Stop
$resolvedBuildDir = Resolve-Path -LiteralPath $BuildDir -ErrorAction Stop

$xml = [xml](Get-Content -LiteralPath $resolvedManifest.Path)
$nsm = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
$nsm.AddNamespace("d", "http://schemas.microsoft.com/appx/manifest/foundation/windows10")
$nsm.AddNamespace("uap", "http://schemas.microsoft.com/appx/manifest/uap/windows10")

# Collect asset paths referenced in manifest
$assetPaths = @()
$logoNode = $xml.SelectSingleNode("//d:Properties/d:Logo", $nsm)
if ($logoNode) { $assetPaths += $logoNode.InnerText }

$visual = $xml.SelectSingleNode("//uap:VisualElements", $nsm)
if ($visual) {
    foreach ($attrName in @("Square44x44Logo","Square150x150Logo")) {
        $attr = $visual.Attributes[$attrName]
        if ($attr) { $assetPaths += $attr.Value }
    }
    $defaultTile = $xml.SelectSingleNode("//uap:DefaultTile", $nsm)
    if ($defaultTile) {
        foreach ($attrName in @("Square310x310Logo","Wide310x150Logo")) {
            $attr = $defaultTile.Attributes[$attrName]
            if ($attr) { $assetPaths += $attr.Value }
        }
    }
}
$assetPaths = $assetPaths | Where-Object { $_ -and $_ -ne "" } | Select-Object -Unique

# Quick existence check in assets dir
foreach ($path in $assetPaths) {
    $norm = $path -replace '^[./\\]+',''
    $candidate = Join-Path $resolvedAssets.Path ($norm -replace '^assets[\\/]', '')
    if (-not (Test-Path $candidate)) {
        throw "Missing asset referenced by manifest: $path (expected at $candidate)"
    }
}

$makeappxPath = Find-MakeAppx
Write-Host "Using makeappx at: $makeappxPath"
Write-Host "Validating manifest by packing temp MSIX..."

# Build a minimal temp directory to validate quickly
$tmpDir = Join-Path ([IO.Path]::GetTempPath()) ("msix_validate_" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmpDir | Out-Null

# Copy manifest and assets
$tmpManifest = Join-Path $tmpDir "AppxManifest.xml"
Copy-Item -LiteralPath $resolvedManifest.Path -Destination $tmpManifest -Force
$dstAssets = Join-Path $tmpDir "assets"
New-Item -ItemType Directory -Path $dstAssets | Out-Null
Copy-Item -Path (Join-Path $resolvedAssets.Path "*") -Destination $dstAssets -Recurse -Force

# Create stub executable so pack does not complain
$exeName = $xml.Package.Applications.Application.Executable
if (-not $exeName) { $exeName = "app.exe" }
$exePath = Join-Path $tmpDir $exeName
$exeDir = Split-Path $exePath
if (-not (Test-Path $exeDir)) { New-Item -ItemType Directory -Path $exeDir | Out-Null }
if (-not (Test-Path $exePath)) { Set-Content -Path $exePath -Value "" -Encoding Byte }

$tmpOut = Join-Path ([IO.Path]::GetTempPath()) "manifest_validate.msix"
if (Test-Path $tmpOut) { Remove-Item $tmpOut -Force }

& $makeappxPath pack /o /nv /nfv /d $tmpDir /p $tmpOut
$exit = $LASTEXITCODE
if (Test-Path $tmpOut) { Remove-Item $tmpOut -Force }
Remove-Item $tmpDir -Recurse -Force
if ($exit -ne 0) {
    throw "makeappx pack (validation) failed with exit code $exit"
}
Write-Host "Manifest validation succeeded."
