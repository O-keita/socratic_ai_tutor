# How to Verify This Works on a Real Android Device

## CHECKLIST: Verification on Real Android Device

### ✅ Phase 1: Pre-Build Setup (5 min)

- [ ] Read `LLAMA_ANDROID_SETUP.md` section 1
- [ ] Model file: `android/app/src/main/assets/models/socratic-q4_k_m.gguf` exists
- [ ] Model size: 2-5GB (expected for q4_k_m)
- [ ] Model type: Chat-tuned (verified locally if possible)
- [ ] `pubspec.yaml` does NOT list this in assets: (too large)

### ✅ Phase 2: Build Verification (10 min)

Run:
```bash
cd /home/omar/school/Capstone/socratic_ai_tutor/frontend
flutter clean
flutter pub get
flutter build apk --release
```

Expected output:
```
✓ Built build/app/outputs/flutter-apk/app-release.apk
```

Then verify APK contents:
```bash
# Check for model
unzip -l build/app/outputs/flutter-apk/app-release.apk | grep "socratic"
# Expected: assets/models/socratic-q4_k_m.gguf

# Check for native library
unzip -l build/app/outputs/flutter-apk/app-release.apk | grep "libllama"
# Expected: lib/arm64-v8a/libllama.so (or armeabi-v7a)
```

- [ ] Model appears in APK listing
- [ ] libllama.so appears in APK listing
- [ ] APK size reasonable (2-3GB if model is 2GB)
- [ ] No build warnings about assets

### ✅ Phase 3: Device Capabilities (5 min)

Connect real Android device via USB. Run:

```bash
# Check Android version (must be 7.0+, API 24+)
adb shell getprop ro.build.version.sdk
# Expected: ≥24

# Check RAM (must be 4GB+)
adb shell cat /proc/meminfo | head -3
# Expected: MemTotal ≥3500000 KB

# Check storage (must have 5GB+ free)
adb shell df /data | tail -1
# Expected: Available ≥5000000 KB

# Check processor (check if Snapdragon 665+)
adb shell getprop ro.hardware
# Expected: sdm665, sdm778, kona, or similar (avoid: sdm636, msm8998)
```

- [ ] Android version ≥7.0
- [ ] RAM ≥4GB
- [ ] Free storage ≥5GB
- [ ] Device model: Snapdragon 665 or newer

### ✅ Phase 4: Install App (3 min)

```bash
# Install APK
adb install -r build/app/outputs/flutter-apk/app-release.apk
# Expected: Success
```

- [ ] Installation succeeds
- [ ] App icon appears on device home screen

### ✅ Phase 5: Permission Check (2 min)

```bash
# Check storage permission granted
adb shell pm list permissions | grep -i "WRITE_EXTERNAL_STORAGE"
```

If permission not granted:
- Open app
- Go to Settings > Permissions > Storage
- Grant permission
- Return to app

- [ ] Storage permission granted

### ✅ Phase 6: Initialization Test (10 min)

Start monitoring logs:
```bash
adb logcat -c
adb logcat | grep "LLMService"
```

In the app, navigate to the Socratic Tutor screen or trigger initialization:

```dart
// Pseudo-code - exact location depends on your app
final llm = SocraticLlmService();
final initialized = await llm.initialize().timeout(Duration(minutes: 5));
```

**Expected logcat output (watch for these messages):**

```
LLMService: Data directory: /data/data/com.yourapp/app_flutter
LLMService: Copying model from assets...
LLMService: Loaded 2450.5 MB from assets
LLMService: Model copied in 45000ms
LLMService: Initializing LlamaController...
LLMService: ✅ LlamaController ready
LLMService: ✅ Initialization complete. Model: /data/data/com.yourapp/app_flutter/socratic-q4_k_m.gguf
```

**Timing expectations:**
- First model copy: 2-6 minutes (one-time)
- Subsequent startups: Skip copy, init in 5-10 seconds

- [ ] See "Data directory:" message
- [ ] See "Model copied in Xms" (or "already exists")
- [ ] See "✅ LlamaController ready"
- [ ] See "✅ Initialization complete"
- [ ] ⏱️ Total time: 2-6 minutes
- [ ] ❌ NOT: "Error initializing", "File not found", OOM crash

### ✅ Phase 7: Streaming Generation Test (15 min)

In the app, ask a question: **"What is machine learning?"**

Expected behavior:

**Logcat:**
```
LLMService: Generating stream...
LLMService: ✅ Stream done in 8450ms
```

**UI:**
- Tokens appear one-at-a-time (not all at once)
- First token appears within 3 seconds
- Full response within 15 seconds
- Example response: "Can you think of any systems in nature that learn from experience?"

**Response properties:**
- Single coherent sentence
- Ends naturally (doesn't cut off mid-word)
- First word is not garbage
- No repetition
- Asks a guiding question (Socratic style)

- [ ] First token within 3 seconds
- [ ] Tokens appear incrementally (not buffered)
- [ ] Full response within 15 seconds
- [ ] Response is a single meaningful question
- [ ] No missing tokens at start
- [ ] No garbage characters

### ✅ Phase 8: Non-Streaming Test (10 min)

Ask another question: **"What is artificial intelligence?"**

Expected behavior (same as above, but response collected first):
```dart
final response = await llm.generateSocraticResponse("What is artificial intelligence?");
print(response);
// Output: "What examples of intelligence can you think of in the real world?"
```

- [ ] Response appears all at once
- [ ] Response is coherent and contextual
- [ ] Takes 10-15 seconds
- [ ] No errors in logcat

### ✅ Phase 9: Multi-Turn Dialogue Test (20 min)

Ask a series of questions, building on previous responses:

1. **Question 1:** "What is a neural network?"
   - App remembers context
   
2. **Question 2:** "How do they learn?"
   - Response should reference previous discussion
   - Should ask follow-up, not repeat previous question
   
3. **Question 3:** "Can you give an example?"
   - Response should build on history

- [ ] Each response is unique (not repeating)
- [ ] Responses reference previous context
- [ ] All 3 questions answered in ~45 seconds
- [ ] No errors or crashes

### ✅ Phase 10: Stress Test (10 min)

Rapid consecutive requests:

```dart
for (int i = 0; i < 5; i++) {
  final resp = await llm.generateSocraticResponse("Question $i?");
  print("Response $i: $resp");
}
```

Expected behavior:
- Requests queue and process one-at-a-time
- All requests complete successfully
- No OOM crash
- Device doesn't overheat
- App remains responsive

- [ ] All 5 requests complete
- [ ] No crashes
- [ ] Device temperature normal (~35-40°C)
- [ ] Battery drain acceptable

### ✅ Phase 11: Error Recovery (5 min)

Force-stop and restart app:

```bash
adb shell am force-stop com.yourapp.package
# Wait 2 seconds
# Restart app
```

Expected behavior:
- App restarts
- Initialization skips model copy (already exists)
- Next request works immediately

- [ ] App restarts successfully
- [ ] Model init faster 2nd time (skip copy)
- [ ] Requests work after restart

### ✅ Phase 12: Cleanup

```bash
# Stop logcat
Ctrl+C

# Uninstall if needed
adb uninstall com.yourapp.package

# Close ADB
adb disconnect
```

---

## PASS/FAIL MATRIX

| Phase | Test | Pass | Fail | Action |
|-------|------|------|------|--------|
| 1 | Setup files | ✓ | ✗ | Check LLAMA_ANDROID_SETUP.md |
| 2 | Build | ✓ | ✗ | `flutter clean && build apk` |
| 2 | APK model | ✓ | ✗ | Check `android/app/src/main/assets/` |
| 2 | APK libllama | ✓ | ✗ | Check `llama_flutter_android` in pubspec |
| 3 | Device API | ✓ | ✗ | Requires Android 7.0+ |
| 3 | Device RAM | ✓ | ✗ | Requires 4GB+, can't proceed |
| 3 | Device storage | ✓ | ✗ | Free 5GB space, then retry |
| 4 | Install | ✓ | ✗ | Check storage permission |
| 5 | Permission | ✓ | ✗ | Grant in app Settings |
| 6 | Init | ✓ | ✗ | Check logcat for errors |
| 7 | Stream tokens | ✓ | ✗ | Model may be base (not chat) |
| 8 | Non-stream | ✓ | ✗ | Check logcat, may need timeout longer |
| 9 | Dialogue | ✓ | ✗ | Context window limit (use <4 messages) |
| 10 | Stress | ✓ | ✗ | Device may be underpowered |
| 11 | Recovery | ✓ | ✗ | Check initialization state |

---

## TROUBLESHOOTING QUICK REFERENCE

### Problem: "Model already exists" but app won't respond to queries

**Check:**
```bash
adb shell ls -l /data/data/com.yourapp/app_flutter/
# Should show: socratic-q4_k_m.gguf (2GB+)

adb shell cat /proc/meminfo | grep MemFree
# Check if enough free RAM
```

**Fix:** Device may not have enough RAM to load model. Need 3.5GB+ free.

---

### Problem: First 5 tokens missing from response

**Status:** FIXED in production code. If still occurring, check you're using the latest `llm_service.dart`.

---

### Problem: Response gets cut off mid-sentence

**Possible causes:**
1. Context window too small (set to 2048 instead of 4096)
2. maxTokens too low (set to 50 instead of 100)
3. Model reaching limit because of long history

**Check:**
```dart
// In code, verify context is 4096
contextSize: 4096,  // Should be this

// Verify maxTokens is sufficient
maxTokens: 100,     // Should be at least this
```

---

### Problem: Nonsense output ("bababababa" or loops)

**Likely cause:** Model is base model, not chat-tuned

**Fix:** 
- Verify model locally: `./main -m model.gguf -p "hello" -n 50 -t 4`
- If output is gibberish locally, wrong model
- Get a chat-tuned model (Qwen-Chat, Llama-Chat, Mistral-Instruct)

---

### Problem: App crashes with OOM

**Check:**
```bash
adb shell cat /proc/meminfo | head -3
```

**If MemFree < 2000000 KB:** Device doesn't have enough RAM

**Fix:** Close other apps or use lower quantization (q3_k instead of q4_k_m)

---

### Problem: Initialization never completes (spinner forever)

**Check:**
```bash
adb logcat | grep LLMService
# Should see status messages
```

**If no messages:** Native library (libllama.so) not found in APK
- Verify: `unzip -l build/.../app-release.apk | grep libllama`

**If stuck on "Copying model":** File I/O slow or storage full
- Check storage: `adb shell df /data`
- Free up space if <5GB available

---

### Problem: Permission denied errors

**Solution:**
- Open app
- Go to Settings > Apps > [Your App] > Permissions
- Grant "Files and Media" or "Storage" permission
- Return to app, retry

---

### Problem: App crashes with logcat showing "dlopen failed"

**Cause:** Native library (libllama.so) not loading

**Check ABI match:**
```bash
adb shell getprop ro.product.cpu.abi
# Expected: arm64-v8a or armeabi-v7a

unzip -l build/.../app-release.apk | grep libllama
# Should have matching lib/[ABI]/libllama.so
```

**Fix:** Rebuild APK, verify ABI matches device

---

## SUCCESS CRITERIA

You're done when you can:

✅ Build APK that contains model and libllama.so

✅ Install on real device (Snapdragon 665+)

✅ App initializes in 2-6 minutes

✅ First request generates response in 10-15 seconds

✅ Tokens appear one-at-a-time (streaming)

✅ Response is a coherent guiding question

✅ Multi-turn dialogue preserves context

✅ 5+ consecutive requests work without crash

✅ Device temperature stable, no thermal throttling

✅ All logcat messages show "✅" status

---

## Next Steps After Verification

1. ✅ All checks pass? → Deploy to testers
2. ✅ Stability confirmed? → Release to production
3. ❌ Something fails? → Check troubleshooting section
4. ❌ Still stuck? → Review `LLAMA_PRODUCTION_REVIEW.md` detailed analysis
