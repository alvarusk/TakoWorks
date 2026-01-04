# Validate AppxManifest.xml with makeappx (schema check only).
param(
    [string]$Manifest = "dist_installer/AppxManifest.xml",
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
$resolvedBuildDir = Resolve-Path -LiteralPath $BuildDir -ErrorAction Stop

$makeappxPath = Find-MakeAppx
Write-Host "Using makeappx at: $makeappxPath"
Write-Host "Validating manifest by packing temp MSIX..."

$tmpManifest = Join-Path $resolvedBuildDir.Path "AppxManifest.xml"
Copy-Item -LiteralPath $resolvedManifest.Path -Destination $tmpManifest -Force

$tmpOut = Join-Path ([IO.Path]::GetTempPath()) "manifest_validate.msix"
if (Test-Path $tmpOut) { Remove-Item $tmpOut -Force }

& $makeappxPath pack /o /d $resolvedBuildDir.Path /p $tmpOut
$exit = $LASTEXITCODE
if (Test-Path $tmpOut) { Remove-Item $tmpOut -Force }
if ($exit -ne 0) {
    throw "makeappx pack (validation) failed with exit code $exit"
}
Write-Host "Manifest validation succeeded."
