# TakoWorks: Windows update via MS Store (based on VoiceX)

Goal: Windows-only update entry point that opens the MS Store listing. Android
update flow is not used.

## 1) Dependencies
- `package_info_plus` (version label in UI, optional)
- `url_launcher` (open MS Store link)

Add to `pubspec.yaml` if missing.

## 2) Version helper (optional but used in VoiceX)
Create `lib/utils/app_version.dart`:

```dart
import 'package:flutter/foundation.dart';
import 'package:package_info_plus/package_info_plus.dart';

class AppVersion {
  AppVersion._();

  static final Future<String> _versionFuture = _readVersion();
  static Future<String> load() => _versionFuture;

  static Future<String> _readVersion() async {
    try {
      final info = await PackageInfo.fromPlatform();
      final build = info.buildNumber.isEmpty ? '' : '+${info.buildNumber}';
      return '${info.version}$build';
    } catch (e) {
      if (kDebugMode) {
        debugPrint('[app_version] error reading version: $e');
      }
      return '';
    }
  }
}
```

## 3) Store URL
Pick a single store URL (Windows only):
- Primary: `ms-windows-store://pdp/?ProductId=YOUR_PRODUCT_ID`
- Fallback: `https://apps.microsoft.com/store/detail/YOUR_PRODUCT_ID`

## 4) Settings UI (Windows only)
Add a small update section with a button. Example:

```dart
Future<void> _launchUpdater() async {
  final primary = Uri.parse('ms-windows-store://pdp/?ProductId=YOUR_PRODUCT_ID');
  final fallback = Uri.parse('https://apps.microsoft.com/store/detail/YOUR_PRODUCT_ID');
  final ok = await launchUrl(primary, mode: LaunchMode.externalApplication);
  if (!ok) {
    await launchUrl(fallback, mode: LaunchMode.externalApplication);
  }
}
```

In the settings page:

```dart
if (Platform.isWindows) ...[
  const Text('Updates', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
  const SizedBox(height: 8),
  FilledButton.icon(
    onPressed: _launchUpdater,
    icon: const Icon(Icons.system_update),
    label: const Text('Update (Windows)'),
  ),
]
```

## 5) Optional: show version in splash or about
Use `AppVersion.load()` and render it in the UI, as done in VoiceX splash.

## 6) Android update check: remove/skip
Do not add `in_app_update` and do not call any Android update checks. Keep the
button Windows-only.

## 7) Store identity alignment
Make sure your Windows package identity matches the MS Store listing
(ProductId/Identity). If you use `msix_config`, keep `identity_name` aligned
with the Store app registration.

## Reference locations in VoiceX
- `lib/utils/app_version.dart` (version helper)
- `lib/settings/settings_page.dart` (Windows update button)
- `lib/projects/projects_page.dart` (Android update check; skip in TakoWorks)
