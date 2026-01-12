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

# Create or reuse a submission
$submissionId = $null
try {
  $submission = Invoke-PartnerApi -method "POST" -path "submissions"
  $submissionId = $submission.id
} catch {
  Write-Host "Create submission failed; trying to reuse an in-progress submission..."
  $subs = Invoke-PartnerApi -method "GET" -path "submissions"
  if ($subs -and $subs.value) {
    $existing = $subs.value | Where-Object { $_.status -in @("InProgress","PendingCommit") } | Sort-Object lastModifiedDate -Descending | Select-Object -First 1
    if ($existing) {
      $submissionId = $existing.id
      Write-Host "Reusing submission $submissionId (status $($existing.status))"
    }
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
Invoke-RestMethod -Method Put -Uri $fileUploadUrl -InFile $msix.FullName -ContentType "application/octet-stream" -Headers $uploadHeaders

# Update submission with package file name
$packages = @(
  @{
    fileName = $msix.Name
    fileStatus = "Uploaded"
  }
)

$submission.packageDeliveryOptions = @{
  packageRollout = @{
    isPackageRollout = $false
  }
}
$submission.applicationPackages = $packages

Invoke-PartnerApi -method "PUT" -path "submissions/$submissionId" -body $submission

# Commit submission
Invoke-PartnerApi -method "POST" -path "submissions/$submissionId/commit"

Write-Host "Submission created and committed for AppId $AppId."
