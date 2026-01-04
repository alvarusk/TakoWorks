# Validate AppxManifest.xml with makeappx (schema check only).
param(
    [string]$Manifest = "dist_installer/AppxManifest.xml"
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
$makeappxPath = Find-MakeAppx
Write-Host "Using makeappx at: $makeappxPath"
Write-Host "Validating manifest: $resolvedManifest"

& $makeappxPath validate /m $resolvedManifest
if ($LASTEXITCODE -ne 0) {
    throw "makeappx validate failed with exit code $LASTEXITCODE"
}
Write-Host "Manifest validation succeeded."
