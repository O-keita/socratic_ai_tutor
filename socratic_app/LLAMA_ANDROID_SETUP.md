# Critical Android Setup for On-Device LLM

## ⚠️ Why This Matters
The rewritten `llm_service.dart` will **crash silently** if these steps are not completed. Android has strict asset handling, permissions, and library requirements for GGUF models.

---

## STEP 1: Asset Bundling (Required)

### Problem
Flutter's `pubspec.yaml` asset system has a ~100MB practical limit. GGUF models are 2GB+.

### Solution: Manual APK Asset Inclusion

**1. Create the directory structure:**
```bash
android/app/src/main/assets/models/
```

**2. Place your model in the correct location:**
```
android/
  app/
    src/
      main/
        assets/
          models/
            socratic-q4_k_m.gguf  ← Model goes HERE
```

**3. Do NOT add it to pubspec.yaml**
Do NOT add this to your `pubspec.yaml`:
```yaml
assets:
  - assets/models/socratic-q4_k_m.gguf  # WRONG - too large
```

Instead, the build system will automatically include `android/app/src/main/assets/` in the APK.

### Verify
After building, check the APK:
```bash
unzip -l build/app/outputs/flutter-apk/app-release.apk | grep socratic
```
Output should show: `assets/models/socratic-q4_k_m.gguf`

---

## STEP 2: Verify libllama.so is Bundled

The `llama_flutter_android` package must include the native C++ library.

### Check
**In `android/app/build.gradle.kts` or `build.gradle`:**
Ensure `llama_flutter_android` is in dependencies:
```gradle
dependencies {
    implementation 'dev.your_package:llama_flutter_android:latest'
    // or if you're using the GitHub version:
    // implementation 'com.github.user:llama_flutter_android:version'
}
```

### Verify After Build
Extract the APK and check:
```bash
unzip -l build/app/outputs/flutter-apk/app-release.apk | grep libllama
```

If no `libllama.so` appears in `lib/armeabi-v7a/` or `lib/arm64-v8a/`, the package wasn't bundled correctly. This is the #1 cause of silent failures.

---

## STEP 3: Android Permissions

**In `android/app/src/main/AndroidManifest.xml`:**
```xml
<manifest>
  <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
  <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
  
  <application>
    <!-- Your app config -->
  </application>
</manifest>
```

### Runtime Permissions (Android 6.0+)
If targeting `targetSdkVersion >= 23`, request permissions at runtime:
```dart
import 'package:permission_handler/permission_handler.dart';

Future<bool> requestStoragePermissions() async {
  final status = await Permission.storage.request();
  return status.isGranted;
}
```

---

## STEP 4: Model File Requirements

### File Format
- **Extension:** `.gguf` (Gemmini Unified Format)
- **Quantization:** `q4_k_m` (4-bit, medium calibration)
  - Works on mid-range Android (Snapdragon 665+)
  - ~5GB RAM minimum
- **Source:** Must be a proper chat model or fine-tuned for chat
  - ❌ Base models (Llama 2 base, Qwen base) will produce nonsense
  - ✅ Chat models (Llama 2 Chat, Qwen Chat, Mistral Instruct)

### Verify Model
Before deploying, test locally:
```bash
# Using llama.cpp CLI
./main -m socratic-q4_k_m.gguf -p "What is AI?" -t 4 -n 50
```

If it produces garbage output locally, it will fail in the app.

---

## STEP 5: Build Configuration

**`android/app/build.gradle.kts`**
```kotlin
android {
    compileSdkVersion 34  // Android 14
    
    defaultConfig {
        minSdkVersion 24  // Android 7.0
        targetSdkVersion 34
    }

    packagingOptions {
        // Ensure native libraries are included
        pickFirst 'lib/arm64-v8a/libc++_shared.so'
        pickFirst 'lib/armeabi-v7a/libc++_shared.so'
    }
}
```

---

## STEP 6: Build and Test

### Clean Build
```bash
cd frontend
flutter clean
rm -rf android/.gradle android/app/build
flutter pub get
```

### Debug Build (Emulator/Real Device)
```bash
flutter run -v
```

### Release Build
```bash
flutter build apk --release
# or
flutter build appbundle --release
```

### Verify Model Path at Runtime
Check logcat:
```bash
adb logcat | grep LLMService
```

Look for:
```
LLMService: Data directory: /data/data/com.yourapp/app_flutter
LLMService: Model already exists at /data/data/com.yourapp/app_flutter/socratic-q4_k_m.gguf (2450.5 MB)
LLMService: ✅ Initialization complete
```

---

## STEP 7: Device Requirements

### Minimum Device Specs
- **RAM:** 4GB (2GB technically possible, but unstable)
- **Storage:** 5GB free (for model + cache)
- **Processor:** Snapdragon 665 or better (or equivalent MediaTek Helio G)
- **Android:** 7.0+ (API 24+)

### Testing
Test on real device, not emulator:
- Emulator may have different memory layout
- Native library loading may behave differently

---

## STEP 8: Debugging Initialization Failures

### If `initialize()` Returns False

**Check 1: Model File Exists**
```dart
import 'dart:io';
final appSupport = await getApplicationSupportDirectory();
final modelPath = '${appSupport.path}/socratic-q4_k_m.gguf';
print("Model exists: ${File(modelPath).existsSync()}");
```

**Check 2: Free Space**
```dart
import 'dart:io';
final stat = await FileStat.stat(appSupport.path);
print("Free space: ${stat.size / 1024 / 1024}MB");
```

**Check 3: Logcat Errors**
```bash
adb logcat | grep "llama\|LLMService\|native"
```

**Check 4: Asset Loading**
Verify asset is accessible:
```dart
try {
  final data = await rootBundle.load('assets/models/socratic-q4_k_m.gguf');
  print("Asset loaded: ${data.lengthInBytes} bytes");
} catch (e) {
  print("Asset load error: $e");
}
```

---

## STEP 9: Common Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `initialize()` returns false immediately | Asset not bundled | Copy to `android/app/src/main/assets/` |
| `initialize()` hangs indefinitely | libllama.so missing | Rebuild, verify package includes native lib |
| Crashes with "file not found" | Wrong file path | Use `getApplicationSupportDirectory()` |
| OOM crash during init | Model too large or device too weak | Use q4_k_m quantization, test on real device |
| Generates nonsense responses | Wrong model type (base instead of chat) | Use a proper chat/instruction-tuned model |
| First 5 tokens missing from stream | Stream race condition (FIXED) | This is solved in the rewritten code |
| All tokens show together, no streaming | Token buffering issue (FIXED) | This is solved in the rewritten code |

---

## STEP 10: Environment Validation Checklist

Before deploying, verify:

```bash
# ✅ Model file size
ls -lh android/app/src/main/assets/models/socratic-q4_k_m.gguf

# ✅ APK contains model
unzip -l build/app/outputs/flutter-apk/app-release.apk | grep socratic

# ✅ APK contains native library
unzip -l build/app/outputs/flutter-apk/app-release.apk | grep libllama

# ✅ Device specs
adb shell getprop ro.build.version.sdk  # Should be ≥24
adb shell cat /proc/meminfo | head -3    # Should be ≥2000000 KB

# ✅ Storage available
adb shell df /data | tail -1             # Should have ≥5GB free
```

---

## What the Fixed Code Does Differently

1. **Disk space check** before copying model
2. **File existence validation** before loading
3. **4096 context window** instead of 2048 (fixes truncation)
4. **Fixed streaming race condition** (tokens subscribed before yield)
5. **Proper marker handling** (doesn't strip `<|im_end|>` prematurely)
6. **Initialization failure flag** to prevent infinite retry loops
7. **Better error messages** with actionable diagnostics
8. **Timeout protection** on initialize call

---

## Deployment Checklist

- [ ] Model file copied to `android/app/src/main/assets/models/`
- [ ] Model is `q4_k_m` quantization
- [ ] Model is chat-tuned (not base model)
- [ ] Test device has ≥4GB RAM, ≥5GB free storage
- [ ] `llama_flutter_android` in pubspec.yaml
- [ ] Permissions added to AndroidManifest.xml
- [ ] Clean build: `flutter clean && flutter pub get`
- [ ] Release build successful: `flutter build apk --release`
- [ ] APK verified to contain model and libllama.so
- [ ] `initialize()` called with timeout: `.timeout(Duration(minutes: 5))`
- [ ] First test on real device (not emulator)
- [ ] Check logcat for "✅ Initialization complete"

---

## Reference: Chat Message Format (ChatML)

The code uses `template: 'chatml'` which expects this format:

```
<|im_start|>system
You are a Socratic AI tutor. Ask one focused guiding question.
<|im_end|>
<|im_start|>user
What is a neural network?
<|im_end|>
<|im_start|>assistant
What is a simplified system in nature that processes information?
<|im_end|>
```

If your model doesn't support this, the generation will fail silently. For non-ChatML models, use:
- `template: 'llama2'` for Llama 2 models
- `template: 'mistral'` for Mistral models
- Consult the model's README for the correct template
