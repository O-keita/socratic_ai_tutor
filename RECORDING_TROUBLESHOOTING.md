# Recording Troubleshooting Guide

Common issues during video recording and how to fix them.

---

## 🐛 Common Issues & Solutions

### Issue 1: App Crashes or Freezes During Recording

**Symptoms:** App crashes when asking questions or toggling offline mode

**Solutions:**
1. **Restart the app** before recording
2. **Clear app cache**: Settings → Apps → Socratic AI Tutor → Storage → Clear Cache
3. **Close other apps** to free up RAM
4. **Reduce background tasks**: Enable "Battery Saver" mode
5. **Use a device with 4GB+ RAM** (if using 2GB device, expect freezes)

**If it keeps crashing:**
- Record shorter segments and compile them
- Test each feature individually first
- Use a different device if available

---

### Issue 2: Slow Response Times (> 10 seconds)

**Symptoms:** Chat responses take longer than 5-7 seconds

**Causes:**
- Device is low on RAM
- Many background apps running
- Model is initializing (first response is slower)
- Network latency (if online mode)

**Solutions:**
1. **Restart device** before recording
2. **Enable Airplane Mode** before toggling to Offline (removes network latency)
3. **Close all other apps**
4. **Let the model warm up**: Ask a question before recording, wait 10 sec, then start recording
5. **Use offline mode** for consistent 5-7s performance

**If still slow:**
- Edit the video to speed up waiting time (2x speed)
- Or mention in narration: "Response time varies based on device load"

---

### Issue 3: Audio Problems

**Symptoms:** Narration is too quiet, muffled, or has background noise

**Solutions:**
1. **Use a quiet room** — close windows, turn off fans
2. **Speak clearly** — not too fast, not too slow
3. **Keep phone away from mouth** — about 6 inches
4. **Check microphone** — on top or bottom of phone (not covered)
5. **Record narration separately** — easier to edit later

**Best Practice:**
- Record video + on-device audio
- Re-record narration in quiet environment
- Edit audio + video together in CapCut/DaVinci Resolve

---

### Issue 4: Video Quality Too Low

**Symptoms:** Video is blurry, pixelated, or choppy

**Solutions:**
1. **Clean screen** — remove fingerprints
2. **Adjust brightness** — not too dark, not too bright
3. **Use native resolution** — don't scale the recording
4. **Close live wallpapers** — can cause lag
5. **Record at 1080p or higher** if possible

**If using scrcpy:**
```bash
scrcpy --record video.mp4 --video-codec=h264
# or for better quality:
scrcpy --record video.mp4 -m 1920  # max dimension 1920
```

---

### Issue 5: Recording Stops in Middle

**Symptoms:** Recording suddenly stops and you lose footage

**Causes:**
- Device ran out of storage
- App crashed mid-recording
- USB connection disconnected (if using ADB)
- Battery critical

**Solutions:**
1. **Check storage space** — need at least 2GB free
2. **Close other recording apps** — Discord, Spotify, etc.
3. **Check USB cable** — if using desktop recording
4. **Enable airplane mode** — reduces interruptions
5. **Full charge device** before recording

**Prevention:**
- Record in short segments (30 sec - 1 min each)
- Save frequently
- Have a backup device ready

---

### Issue 6: Offline Mode Won't Work

**Symptoms:** Toggle to Offline but still shows "Online" or responses fail

**Causes:**
- Model not downloaded yet
- Model corrupted or incomplete
- Insufficient disk space
- App bug

**Solutions:**
1. **Check model is downloaded**:
   - Settings → Manage Local Model
   - Should show ✅ "Model Downloaded"
2. **Verify download size**:
   - Should be ~460 MB
   - If less, model is incomplete — re-download
3. **Free up storage**:
   - Delete old photos/videos
   - Need at least 500MB free
4. **Restart app** and try again

**If model still won't download:**
- Use **online mode only** for video
- Mention in narration: "App downloads model on first launch"
- Still demonstrates the core functionality

---

### Issue 7: Quiz or Playground Won't Load

**Symptoms:** Quiz page shows blank or Playground code won't run

**Solutions:**
1. **Refresh the page** — go back, re-open
2. **Restart app** — close completely, reopen
3. **Check connectivity** — if using online features
4. **Clear app cache** — Settings → Apps → Storage → Clear Cache
5. **Playground needs WiFi first time** — to download Pyodide (~8MB)

**Quick Fix:**
- Skip that section in video
- Or record it separately when it's working
- You have 5 minutes, can trim slower features

---

### Issue 8: Narration Too Soft or Loud

**Symptoms:** Narration recorded but barely audible or distorted

**Solutions:**
1. **Test audio levels** — record 10 seconds, play back
2. **Speak louder** — aim for 70-80 dB (normal conversation)
3. **Closer to mic** — about 6 inches from device mic
4. **Slower speech** — easier to understand, better for video
5. **Use external mic** — if available, better quality

**Best Option:**
- Record video silently first
- Then record separate audio in quiet place
- Sync them in editor (CapCut/DaVinci)

---

## 🎯 Quick Workarounds

**If something breaks during recording:**

| Problem | Workaround |
|---------|-----------|
| Chat hangs | Skip to next question |
| Offline fails | Use online mode, mention it connects to backend |
| Quiz won't load | Skip to Glossary or Playground |
| Playground error | Move to dark mode demo |
| Audio is bad | Re-record narration separately |
| Slow response | Edit with 2x speed or mention network latency |
| Video freezes | Record in short segments, compile later |

---

## ✅ Quality Checklist

Before submitting, verify:

- [ ] Video is 5 minutes or under
- [ ] All sections flow smoothly
- [ ] Narration is clear and audible (≥ 70dB)
- [ ] No long pauses or dead air
- [ ] App features work as shown
- [ ] File size < 500MB
- [ ] Video plays on desktop AND phone
- [ ] Share link works and is public/accessible

---

## 📞 Last Resort Solutions

**If recording fails:**

1. **Use existing screenshots** + screen recording of app
   - Record just voice narration
   - Edit in video editor with screenshots as background

2. **Record multiple short clips**
   - 10-15 second segments
   - Compile into full video
   - Edit transitions between clips

3. **Screen recording from emulator**
   - Android Emulator has built-in video recording
   - Go to: More → Screen Record
   - Better stability than physical device

---

## 🎬 Pro Tips for Smooth Recording

✅ **Before you hit record:**
- Fully charge device (at least 80%)
- Close all unnecessary apps
- Enable Do Not Disturb
- Clean the screen
- Test mic/audio
- Do a 30-second practice run

✅ **During recording:**
- Speak slowly and clearly
- Pause 2 seconds between sections
- Point at what you're explaining
- Mention numbers/metrics as you show them

✅ **After recording:**
- Watch it fully before editing
- Check audio is synced
- Trim dead air (silence, loading)
- Add text overlays with key metrics
- Export as MP4 H.264 for compatibility

---

## 📊 Expected Performance Benchmarks

For reference during recording:

| Task | Expected Time |
|------|---------|
| App launch | 2-3 seconds |
| Chat response (online) | 4-8.5 seconds |
| Chat response (offline ARM64) | 5-7 seconds |
| Model download (first launch) | 5-10 minutes |
| Quiz load | 1-2 seconds |
| Playground first load | 2-3 seconds (needs WiFi) |
| Playground code run | 1-2 seconds |

**If times are significantly longer:**
- Device might be overloaded
- Restart and try again
- Or edit video to speed up

---

Good luck with your recording! You've got this. 🎬✨
