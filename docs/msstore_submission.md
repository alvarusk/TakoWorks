# MS Store submission (Windows, MSIX) - Partner Center API workflow

Template files to automate MS Store submission and upload for a Flutter MSIX
build using the Partner Center Submission API (no StoreBroker).

Files in this folder:
- `docs/msstore_submission_workflow.yml`: GitHub Actions workflow template.
- `docs/msstore_submission.ps1`: PowerShell script invoked by the workflow (expects `TakoWorks.msix` in repo root).

## Prereqs (one-time)
1) Create an Azure AD app (client credentials flow).
2) Grant access to Microsoft Store submission API (Partner Center).
3) Capture these values:
   - Tenant ID
   - Client ID
   - Client Secret
   - Store App ID (Partner Center app ID, not the package identity)

## Secrets required by the workflow
- `MSSTORE_TENANT_ID`
- `MSSTORE_CLIENT_ID`
- `MSSTORE_CLIENT_SECRET`
- `MSSTORE_APP_ID`

## How it works (high level)
1) Build Windows release.
2) Create MSIX package (your build pipeline should output a signed `.msix`).
3) Create a new submission via Partner Center API.
4) Upload MSIX to the SAS URL returned by the API.
5) Commit the submission for certification.

## Notes
- The script uses the Partner Center REST API at
  `https://manage.devcenter.microsoft.com/v1.0/my/applications/{appId}`.
- Keep `msix_config` in `pubspec.yaml` aligned with your Store identity.
