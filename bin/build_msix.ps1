# Build an MSIX from a PyInstaller folder using makeappx with path validation.
param(
    [string]$Manifest = "dist_installer/AppxManifest.xml",
    [string]$AssetsDir = "dist_installer/assets",
    [string]$BuildDir = "dist/TakoWorks",
    [string]$OutFile = "TakoWorks.msix",
    [switch]$ExcludeTransformersPythonSources = $true
)

$ErrorActionPreference = "Stop"

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

function Test-AppxRelativePath {
    param([string]$RelPath)

    $maxLen = 255
    if ($RelPath.Length -gt $maxLen) {
        return "Relative path length $($RelPath.Length) exceeds $maxLen characters"
    }

    $segments = $RelPath -split "[\\/]"
    $reserved = @(
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    )

    foreach ($segment in $segments) {
        if (-not $segment) { return "Empty path segment" }
        if ($segment -match '[<>:"|?*]') {
            return "Invalid character in path segment '$segment'"
        }
        if ($segment.TrimEnd([char[]]@(" ", ".")) -ne $segment) {
            return "Path segment ends with space/dot: '$segment'"
        }
        $baseName = [IO.Path]::GetFileNameWithoutExtension($segment)
        if ($reserved -contains $baseName.ToUpperInvariant()) {
            return "Reserved Windows name used in path segment '$segment'"
        }
    }

    return $null
}

$resolvedManifest = Resolve-Path -LiteralPath $Manifest -ErrorAction Stop
$resolvedAssets = Resolve-Path -LiteralPath $AssetsDir -ErrorAction Stop
$resolvedBuildDir = Resolve-Path -LiteralPath $BuildDir -ErrorAction Stop
$outPath = [IO.Path]::GetFullPath($OutFile)

$makeappxPath = Find-MakeAppx
Write-Host "Using makeappx at: $makeappxPath"
Write-Host "Build directory: $($resolvedBuildDir.Path)"
Write-Host "Manifest: $($resolvedManifest.Path)"
Write-Host "Assets: $($resolvedAssets.Path)"
Write-Host "Output MSIX: $outPath"

if (Test-Path $outPath) { Remove-Item -LiteralPath $outPath -Force }

$stagingRoot = Join-Path ([IO.Path]::GetTempPath()) ("msix_stage_" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $stagingRoot | Out-Null
Write-Host "Staging payload at: $stagingRoot"

$files = Get-ChildItem -LiteralPath $resolvedBuildDir.Path -Recurse -File -Force
$pathStats = New-Object System.Collections.Generic.List[object]
$skipped = New-Object System.Collections.Generic.List[object]
$copiedCount = 0
$policySkipped = 0

foreach ($file in $files) {
    $rel = [IO.Path]::GetRelativePath($resolvedBuildDir.Path, $file.FullName).Replace("/", "\")
    $pathStats.Add([pscustomobject]@{ Rel = $rel; Len = $rel.Length }) | Out-Null

    # PyInstaller includes many transformers source files under _internal.
    # Runtime uses bundled bytecode, so excluding these sources keeps package payload cleaner.
    if ($ExcludeTransformersPythonSources -and $rel -match '^(?:_internal|internal)\\transformers\\.*\.py$') {
        $policySkipped++
        $skipped.Add([pscustomobject]@{ Rel = $rel; Reason = "Excluded transformers source .py for MSIX packaging" }) | Out-Null
        continue
    }

    $reason = Test-AppxRelativePath -RelPath $rel
    if ($reason) {
        $skipped.Add([pscustomobject]@{ Rel = $rel; Reason = $reason }) | Out-Null
        continue
    }

    $dst = Join-Path $stagingRoot $rel
    $dstDir = Split-Path -Parent $dst
    if (-not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    }
    Copy-Item -LiteralPath $file.FullName -Destination $dst -Force
    $copiedCount++
}

Write-Host "Copied files: $copiedCount"
Write-Host "Skipped files: $($skipped.Count)"
Write-Host "Policy-skipped transformers .py files: $policySkipped"
Write-Host "Top 20 longest relative paths:"
$pathStats | Sort-Object Len -Descending | Select-Object -First 20 | ForEach-Object {
    Write-Host ("  {0,4}  {1}" -f $_.Len, $_.Rel)
}

if ($skipped.Count -gt 0) {
    Write-Warning "Some files were skipped because they are invalid for Appx naming rules."
    $skipped | Select-Object -First 50 | ForEach-Object {
        Write-Warning ("Skipped: {0} :: {1}" -f $_.Rel, $_.Reason)
    }
}

# Ensure installer-specific manifest and assets are present in staging.
Copy-Item -LiteralPath $resolvedManifest.Path -Destination (Join-Path $stagingRoot "AppxManifest.xml") -Force
$dstAssets = Join-Path $stagingRoot "assets"
if (-not (Test-Path $dstAssets)) {
    New-Item -ItemType Directory -Path $dstAssets | Out-Null
}
Copy-Item -Path (Join-Path $resolvedAssets.Path "*") -Destination $dstAssets -Recurse -Force

Write-Host "Running makeappx pack..."
& $makeappxPath pack /o /d $stagingRoot /p $outPath
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    throw "makeappx failed with exit code $exitCode (staging at $stagingRoot)"
}

Write-Host "MSIX build succeeded: $outPath"
Remove-Item -LiteralPath $stagingRoot -Recurse -Force
