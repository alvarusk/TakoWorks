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
if (-not (Test-Path $exePath)) { [IO.File]::WriteAllBytes($exePath, [byte[]]@()) }

$tmpOut = Join-Path ([IO.Path]::GetTempPath()) "manifest_validate.msix"
if (Test-Path $tmpOut) { Remove-Item $tmpOut -Force }

$valOut = Join-Path ([IO.Path]::GetTempPath()) ("makeappx_validate_" + [guid]::NewGuid().ToString() + ".out.log")
$valErr = Join-Path ([IO.Path]::GetTempPath()) ("makeappx_validate_" + [guid]::NewGuid().ToString() + ".err.log")

try {
    $proc = Start-Process -FilePath $makeappxPath `
        -ArgumentList @("pack", "/o", "/nv", "/nfv", "/d", $tmpDir, "/p", $tmpOut) `
        -RedirectStandardOutput $valOut `
        -RedirectStandardError $valErr `
        -PassThru -WindowStyle Hidden
    $proc.WaitForExit()

    Write-Host "---- makeappx validation stdout (tail 200) ----"
    if (Test-Path $valOut) { Get-Content $valOut -Tail 200 | Write-Host }
    Write-Host "---- makeappx validation stderr (tail 200) ----"
    if (Test-Path $valErr) { Get-Content $valErr -Tail 200 | Write-Host }

    if ($proc.ExitCode -ne 0) {
        $stdout = @()
        $stderr = @()
        if (Test-Path $valOut) { $stdout = Get-Content $valOut }
        if (Test-Path $valErr) { $stderr = Get-Content $valErr }
        $all = $stdout + $stderr
        $lastProcessing = $all | Where-Object { $_ -like 'Processing "*' } | Select-Object -Last 1
        $codeMatch = $all | Select-String -Pattern '0x[0-9A-Fa-f]{8}' | Select-Object -Last 1
        $errorCode = if ($codeMatch) { $codeMatch.Matches[0].Value } else { "<not-detected>" }
        $lastProcessingText = if ($lastProcessing) { $lastProcessing } else { "<not-found>" }

        $hint = "Review manifest-declared files and the last payload line above."
        if ($errorCode -ieq "0x8007007b") {
            $hint = "0x8007007B indicates invalid filename/path syntax. Verify asset names and manifest paths (no trailing dots/spaces, reserved names, or malformed separators)."
        }

        throw @"
makeappx validation pack failed with exit code $($proc.ExitCode).
Windows error code: $errorCode
Last payload processed: $lastProcessingText
Hint: $hint
Logs:
  stdout: $valOut
  stderr: $valErr
"@
    }

    Write-Host "Manifest validation succeeded."
}
finally {
    if (Test-Path $tmpOut) { Remove-Item $tmpOut -Force }
    if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }
    if (Test-Path $valOut) { Remove-Item $valOut -Force }
    if (Test-Path $valErr) { Remove-Item $valErr -Force }
}
