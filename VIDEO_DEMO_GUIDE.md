# 5-Minute Demo Video Guide
## Socratic AI Tutor - Capstone Submission

Quick reference for recording and submitting your demo video.

---

## 📹 Video Flow (5 Minutes)

| Time | Section | What to Show | Key Points |
|------|---------|-------------|-----------|
| 0:00-0:15 | **Intro** | Home screen with 4 tabs | "Socratic AI tutor for DS/ML, works offline" |
| 0:15-1:00 | **Chat: Socratic Questioning** | 3 questions (conceptual, code, casual) | AI guides with questions, never direct answers |
| 1:00-1:45 | **Offline Mode** | Toggle offline, ask question, AI responds | Same quality offline, completely on-device |
| 1:45-2:30 | **Quizzes** | Select course, show quiz, answer questions | Adaptive difficulty based on performance |
| 2:30-3:15 | **Python Playground** | Write simple code, run it | Full Python environment (Pyodide/WebAssembly) |
| 3:15-4:00 | **Glossary & Profile** | Search term, show definition, show progress | All data saved locally |
| 4:00-4:45 | **Performance & Dark Mode** | Toggle dark mode, mention metrics | Response times: 4-8.5s online, 5-7s offline |
| 4:45-5:00 | **Closing** | Summary & links | Deploy link, GitHub repo |

---

## 🎬 Recording Options

### Option A: Built-in Android Recording (Easiest)
```
Settings → Advanced → Screen Recording
→ Start recording
→ Narrate as you demo
→ Stop when done
```
✅ No extra tools needed
⏱️ Fastest option

### Option B: Desktop Recording (Better Control)
```bash
# Install scrcpy: https://github.com/Genymobile/scrcpy
scrcpy --record video.mp4

# Record narration separately (Audacity, GarageBand)
# Edit together in CapCut or DaVinci Resolve
```
✅ Better quality
✅ Can re-record audio if needed

### Option C: OBS Studio (Professional)
```
Download: https://obsproject.com
Setup: Android stream via ADB
Record: Video + Audio
Export: MP4
```
✅ Professional quality
✅ Full control over audio/video

---

## 🎤 Narration Script (Use as Reference)

**[0:00-0:15] Intro**
> "This is the Socratic AI Tutor — a hybrid offline-first mobile app for learning Data Science and Machine Learning. It uses AI to guide you through questions, and it works completely offline on Android devices."

**[0:15-1:00] Chat Feature**
> "The key feature is the Socratic chat. Let me ask some questions.
> First: 'What is gradient descent?'
> [Wait 5-7 sec]
> Notice: The AI asked a guiding question instead of defining it. This is the Socratic method — it forces you to think and discover concepts yourself.
> [Ask 2-3 more questions to show variety]
> All responses took 5-7 seconds, which is great for ARM64 devices."

**[1:00-1:45] Offline Mode**
> "Now here's the coolest part — this all works completely offline.
> [Toggle to Offline]
> The AI now runs entirely on the device, no internet needed. Same response quality, same performance. This is possible because we compiled an entire LLM inference engine directly into the app."

**[1:45-2:30] Quizzes**
> "The app includes courses with adaptive quizzes. The difficulty adjusts based on how you perform."

**[2:30-3:15] Playground**
> "You can run Python code directly in the app using Pyodide."

**[3:15-4:00] Glossary & Profile**
> "There's a glossary with ML/DS terms, and your progress is tracked locally."

**[4:00-4:45] Performance**
> "We tested this on three different devices:
> - Emulator (x86): 8.5 seconds
> - Huawei P Smart (4GB ARM64): 5-7 seconds offline, 6.4 online
> - Modern phones (6GB+ ARM64): 4-7 seconds online
> Consistent, reliable performance across hardware."

**[4:45-5:00] Closing**
> "Socratic AI Tutor — hybrid offline-first learning.
> Deployed at socratic.hx-ai.org
> GitHub: github.com/O-keita/socratic_ai_tutor
> Thanks for watching!"

---

## ✂️ Quick Editing Tips

**Free Video Editors:**
- **CapCut** (Mobile app - easiest)
- **DaVinci Resolve** (Desktop - professional, free)
- **Adobe Premiere Express** (Web-based)

**Quick Edits:**
- Speed up long loading times (1.5-2x speed)
- Cut pauses or fumbles
- Add text overlays:
  - "Response time: 5.2 seconds"
  - "Running offline"
  - "Device: Huawei P Smart"
- Keep under 5 minutes total

---

## 🎯 What NOT to Include

❌ Sign-up/login screen (too long, not interesting)
❌ Model download process (takes 5-10 min, already shown in README)
❌ Admin backend features (user-focused, not backend)
❌ Long loading times (edit them out or speed up)
❌ Dark/unclear narration (speak clearly!)

---

## 📤 Upload & Share

### YouTube (Recommended)
1. Upload video
2. Set to "Unlisted" (not "Private")
3. Copy video URL
4. Share in Canvas

### Google Drive
1. Upload video
2. Right-click → Share
3. Set to "Anyone with link can view"
4. Copy link
5. Share in Canvas

### OneDrive / Dropbox
Same process as Google Drive

---

## 📋 Pre-Submission Checklist

**Video Content:**
- [ ] 5 minutes or under
- [ ] All major features shown
- [ ] No sign-up/login screen
- [ ] Narration is clear
- [ ] Performance metrics mentioned
- [ ] Closing with GitHub + deployment links

**Technical:**
- [ ] Video is MP4 (H.264)
- [ ] File size < 500MB
- [ ] Plays on multiple devices
- [ ] Share link is set to "viewable"

**Canvas Submission:**
- [ ] Video link
- [ ] GitHub URL: https://github.com/O-keita/socratic_ai_tutor
- [ ] Backend: https://socratic.hx-ai.org/
- [ ] Brief description: "Demonstrates Socratic questioning, offline mode, quizzes, playground, performance across devices"

---

## 🚀 Recording Timeline

**Recommended Order:**
1. Install app + log in (done beforehand)
2. Record full 5-minute demo (no edits yet)
3. Review and re-record sections if needed
4. Edit (cut, speed up, add text)
5. Export as MP4
6. Upload to YouTube/Drive
7. Get shareable link
8. Submit in Canvas

**Time Estimate:**
- Record + review: 15-30 min
- Edit: 15-30 min
- Upload + share: 5 min
- **Total: 35-65 minutes**

---

## 💡 Pro Tips

✅ **Test everything first** — Make sure all features work smoothly
✅ **Do a practice run** — Record a quick 1-2 min test first
✅ **Speak naturally** — Not scripted, but prepared
✅ **Point at screen** — Gesture while explaining features
✅ **Mention metrics** — "5-7 seconds response time" sounds impressive
✅ **Show the toggle** — Offline mode is the coolest feature
✅ **Keep energy up** — You're demonstrating something cool!

---

**Need help?** Refer to this guide while recording. Good luck! 🎬
