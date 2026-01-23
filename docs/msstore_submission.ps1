# Partner Center Submission API (no StoreBroker).
# Env vars required:
#   MSSTORE_TENANT_ID, MSSTORE_CLIENT_ID, MSSTORE_CLIENT_SECRET, MSSTORE_APP_ID

param(
  [string]$AppId = $env:MSSTORE_APP_ID,
  [string]$MsixPath = "TakoWorks.msix"
)

$ErrorActionPreference = "Stop"

if (-not $AppId) {
  Write-Error "STORE_APP_ID is required."
  exit 1
}
if (-not $env:MSSTORE_TENANT_ID -or -not $env:MSSTORE_CLIENT_ID -or -not $env:MSSTORE_CLIENT_SECRET) {
  Write-Error "MSSTORE_TENANT_ID / MSSTORE_CLIENT_ID / MSSTORE_CLIENT_SECRET are required."
  exit 1
}

if ($env:MSSTORE_PUBLISHER_ID) {
  Write-Host "MSSTORE_PUBLISHER_ID: $env:MSSTORE_PUBLISHER_ID"
}

function Test-Truthy([string]$value) {
  if (-not $value) { return $false }
  switch ($value.ToLowerInvariant()) {
    "1" { return $true }
    "true" { return $true }
    "yes" { return $true }
    "y" { return $true }
    default { return $false }
  }
}

$forceNewSubmission = Test-Truthy $env:MSSTORE_FORCE_NEW_SUBMISSION
if ($forceNewSubmission) {
  Write-Host "MSSTORE_FORCE_NEW_SUBMISSION enabled."
}

$msix = $null
if (Test-Path $MsixPath) {
  $msix = Get-Item $MsixPath
} else {
  $msix = Get-ChildItem -Recurse -Path "." -Filter "*.msix" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
}

if (-not $msix) {
  Write-Error "MSIX not found (expected $MsixPath or latest *.msix under repo)."
  exit 1
}

function Get-MsixVersion([string]$path) {
  try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
    $zip = [System.IO.Compression.ZipFile]::OpenRead($path)
    try {
      $entry = $zip.Entries | Where-Object { $_.FullName -ieq "AppxManifest.xml" } | Select-Object -First 1
      if (-not $entry) { return $null }
      $reader = New-Object System.IO.StreamReader($entry.Open())
      try {
        $xml = [xml]$reader.ReadToEnd()
        return $xml.Package.Identity.Version
      } finally {
        $reader.Dispose()
      }
    } finally {
      $zip.Dispose()
    }
  } catch {
    return $null
  }
}

$msixVersion = Get-MsixVersion -path $msix.FullName
if ($msixVersion) {
  Write-Host "MSIX AppxManifest version: $msixVersion"
} else {
  Write-Host "MSIX AppxManifest version: (unknown)"
}

# Use a versioned filename to avoid stale metadata when reusing the same MSIX name.
$uploadPath = $msix.FullName
$uploadName = $msix.Name
if ($msixVersion) {
  $uploadName = "TakoWorks_$msixVersion.msix"
  $uploadPath = Join-Path $env:TEMP $uploadName
  Copy-Item -LiteralPath $msix.FullName -Destination $uploadPath -Force
}
Write-Host "Uploading MSIX file: $uploadName"

function Get-AccessToken {
  $tenant = $env:MSSTORE_TENANT_ID
  $clientId = $env:MSSTORE_CLIENT_ID
  $clientSecret = $env:MSSTORE_CLIENT_SECRET
  $body = @{
    grant_type = "client_credentials"
    client_id = $clientId
    client_secret = $clientSecret
    scope = "https://manage.devcenter.microsoft.com/.default"
  }
  $resp = Invoke-RestMethod -Method Post -Uri "https://login.microsoftonline.com/$tenant/oauth2/v2.0/token" -Body $body
  return $resp.access_token
}

function Invoke-PartnerApi([string]$method, [string]$path, $body = $null) {
  $token = Get-AccessToken
  $headers = @{
    Authorization = "Bearer $token"
    "Content-Type" = "application/json"
  }
  $uri = "https://manage.devcenter.microsoft.com/v1.0/my/applications/$AppId/$path"
  if ($body -ne $null) {
    return Invoke-RestMethod -Method $method -Uri $uri -Headers $headers -Body ($body | ConvertTo-Json -Depth 8)
  }
  return Invoke-RestMethod -Method $method -Uri $uri -Headers $headers
}

# Root endpoint (for listing applications)
function Invoke-PartnerApiRoot([string]$method, [string]$path, $body = $null) {
  $token = Get-AccessToken
  $headers = @{
    Authorization = "Bearer $token"
    "Content-Type" = "application/json"
  }
  $uri = "https://manage.devcenter.microsoft.com/v1.0/my/$path"
  if ($body -ne $null) {
    return Invoke-RestMethod -Method $method -Uri $uri -Headers $headers -Body ($body | ConvertTo-Json -Depth 8)
  }
  return Invoke-RestMethod -Method $method -Uri $uri -Headers $headers
}

# Validate AppId early and show accessible apps when possible
$appIdInput = $AppId
try {
  $apps = Invoke-PartnerApiRoot -method "GET" -path "applications"
  if ($apps -and $apps.value) {
    $match = $apps.value | Where-Object {
      $_.id -eq $appIdInput -or $_.applicationId -eq $appIdInput -or $_.productId -eq $appIdInput
    } | Select-Object -First 1
    if (-not $match) {
      Write-Warning "MSSTORE_APP_ID not found in accessible applications for this tenant/app registration."
      Write-Host "Accessible applications (first 10):"
      $apps.value |
        Select-Object -First 10 id, primaryName, productId, packageIdentityName, publisherId |
        Format-Table -AutoSize | Out-String -Width 200 | Write-Host
    } else {
      Write-Host "Resolved application: $($match.primaryName) (id=$($match.id))"
      if ($match.id -and $match.id -ne $appIdInput) {
        Write-Host "Using application id from Partner Center: $($match.id)"
        $AppId = $match.id
      }
      if ($env:MSSTORE_PUBLISHER_ID -and $match.publisherId -and $match.publisherId -ne $env:MSSTORE_PUBLISHER_ID) {
        Write-Warning "MSSTORE_PUBLISHER_ID does not match Partner Center publisherId for this app."
      }
    }
  }
} catch {
  Write-Warning "Could not list applications from Partner Center API. Verify tenant/app access. Details: $_"
}

# Create or reuse a submission
$submissionId = $null
try {
  $submission = Invoke-PartnerApi -method "POST" -path "submissions"
  $submissionId = $submission.id
} catch {
  if ($forceNewSubmission) {
    Write-Error "Create submission failed and reuse is disabled. Discard any in-progress submission and retry. Details: $_"
    exit 1
  }
  Write-Host "Create submission failed; trying to reuse an in-progress submission..."
  try {
    $subs = Invoke-PartnerApi -method "GET" -path "submissions"
    if ($subs -and $subs.value) {
      $existing = $subs.value | Where-Object { $_.status -in @("InProgress","PendingCommit") } | Sort-Object lastModifiedDate -Descending | Select-Object -First 1
      if ($existing) {
        $submissionId = $existing.id
        Write-Host "Reusing submission $submissionId (status $($existing.status))"
      }
    }
  } catch {
    Write-Error "Partner Center API error. Check MSSTORE_APP_ID, tenant, and app registration permissions. Details: $_"
    exit 1
  }
}
if (-not $submissionId) {
  Write-Error "Failed to create or reuse a submission."
  exit 1
}

# Get submission to fetch upload URL
$submission = Invoke-PartnerApi -method "GET" -path "submissions/$submissionId"
$fileUploadUrl = $submission.fileUploadUrl
if (-not $fileUploadUrl) {
  Write-Error "fileUploadUrl not found in submission."
  exit 1
}

# Upload MSIX to the provided SAS URL
Write-Host "Uploading MSIX to SAS URL..."
$uploadHeaders = @{
  "x-ms-blob-type" = "BlockBlob"
}
Invoke-RestMethod -Method Put -Uri $fileUploadUrl -InFile $uploadPath -ContentType "application/octet-stream" -Headers $uploadHeaders

# Refresh submission to pick up any package ids created after upload
$submissionAfterUpload = $null
for ($i = 0; $i -lt 6; $i++) {
  try {
    $submissionAfterUpload = Invoke-PartnerApi -method "GET" -path "submissions/$submissionId"
    $pkgFound = $null
    if ($submissionAfterUpload -and $submissionAfterUpload.applicationPackages) {
      $pkgFound = $submissionAfterUpload.applicationPackages |
        Where-Object { $_.fileName -eq $uploadName } | Select-Object -First 1
    }
    if ($pkgFound) {
      break
    }
  } catch {
    Write-Warning "Could not refresh submission after upload (attempt $($i + 1)). Details: $_"
    $submissionAfterUpload = $null
  }
  Start-Sleep -Seconds 5
}
if ($submissionAfterUpload) {
  $submission = $submissionAfterUpload
}

# Update submission with package file name
$packages = @()
$matched = $false
$existingByName = @{}
if ($submission.applicationPackages) {
  foreach ($p in $submission.applicationPackages) {
    if ($p.fileName) {
      $existingByName[$p.fileName.ToLowerInvariant()] = $p
    }
    $packages += $p
  }
}

$uploadKey = $uploadName.ToLowerInvariant()
if ($existingByName.ContainsKey($uploadKey)) {
  $existing = $existingByName[$uploadKey]
  if (-not $existing.id) {
    Write-Error "Existing package '$uploadName' is missing id in submission. Discard the in-progress submission and retry."
    exit 1
  }
  $existing.fileStatus = "Uploaded"
  $matched = $true
}

if (-not $packages) {
  $packages = @(@{ fileName = $uploadName; fileStatus = "Uploaded" })
} elseif (-not $matched) {
  $packages += @{ fileName = $uploadName; fileStatus = "Uploaded" }
}

$submission.packageDeliveryOptions = @{
  packageRollout = @{
    isPackageRollout = $false
  }
}
$submission.applicationPackages = $packages

$updated = Invoke-PartnerApi -method "PUT" -path "submissions/$submissionId" -body $submission
if ($updated -and $updated.applicationPackages) {
  Write-Host "Submission packages (after upload):"
  $updated.applicationPackages |
    Select-Object fileName, version, fileStatus, architecture |
    Format-Table -AutoSize | Out-String -Width 200 | Write-Host
}

if ($msixVersion) {
  $found = $false
  for ($i = 0; $i -lt 6; $i++) {
    try {
      Start-Sleep -Seconds 10
      $check = Invoke-PartnerApi -method "GET" -path "submissions/$submissionId"
      $pkg = $check.applicationPackages | Where-Object { $_.fileName -eq $uploadName } | Select-Object -First 1
      if ($pkg -and $pkg.version -eq $msixVersion) {
        Write-Host "Partner Center ingestion shows version $($pkg.version) for $uploadName."
        $found = $true
        break
      }
    } catch {
      Write-Warning "Could not verify ingestion yet: $_"
      break
    }
  }
  if (-not $found) {
    Write-Warning "Partner Center still reports a different version for $uploadName. It may update after ingestion finishes."
  }
}

# Commit submission
Invoke-PartnerApi -method "POST" -path "submissions/$submissionId/commit"

Write-Host "Submission created and committed for AppId $AppId."
