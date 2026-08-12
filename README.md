# VISION

A voice-first personal AI assistant for macOS — talk to it naturally, and it can control your Mac, browse and interact with real apps and websites, search the web, run code, and remember what you've told it, all while showing a live glowing orb that reacts to what it's doing.

Built on Gemini Live for native, low-latency voice-to-voice conversation (not a cascaded transcribe → text-model → text-to-speech pipeline).

## Features

- **Voice-first conversation** — wake word activation ("Vision"), natural interruption/barge-in, no push-to-talk
- **Computer Use** — real GUI automation: VISION can move the mouse, click, type, scroll, and take/zoom screenshots to complete tasks across apps and websites (e.g. "open Spotify and play a song"), using a look-act-look loop that watches the screen after every action
- **System control** — open/close apps, adjust volume, run Shortcuts, send notifications, read calendar/email, all via AppleScript
- **Web search** — real-time answers via Tavily, shown visually with images
- **Code execution** — runs Python and reports real output
- **File access** — read/list/search files, sandboxed to your home directory
- **Screen & camera awareness** — VISION can watch your screen or camera and describe what it sees
- **Memory** — durable facts, semantic search over past conversations, recent context in every turn
- **Visual HUD** — a Three.js particle orb that visibly shifts (idle / listening / processing / speaking) plus panels for code, search results, and live webpage previews
- **Acoustic cues** — short synthesized tones (thinking / success / error / waiting) that cover the gap while VISION is working
- **Confirmation gating** — risky actions (closing an app, running a Shortcut, executing code, starting a GUI-automation task) require a one-time confirmation token before they run

## Requirements

- **macOS** — the system-control and GUI-automation layers are built on AppleScript and Quartz, so this only runs on a Mac
- **Python 3.10+**
- API keys:
  - `GEMINI_API_KEY` — **required**, powers the entire voice pipeline (Gemini Live)
  - `TAVILY_API_KEY` — required for the `web_search` tool to work
  - `OPENROUTER_API_KEY` — only needed if you run the experimental text-chat pipeline (`brain/router.py`) directly; not used by the app you actually launch
- macOS permissions (grant these in **System Settings → Privacy & Security**):
  - **Microphone** — for voice input
  - **Screen Recording** — for screenshots and screen-watching
  - **Accessibility** — required for Computer Use (mouse/keyboard control) to actually work; without it, clicks/keystrokes are silently dropped by macOS
  - **Automation** — for AppleScript-based app control

## Setup

```bash
git clone https://github.com/directorlagbajaweb/vision-ai-agent.git
cd vision-ai-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here   # optional, legacy pipeline only
```

Then grant the macOS permissions listed above to whichever app/terminal launches Python (System Settings → Privacy & Security), and run:

```bash
python3 main.py
```

Say **"Vision"** to wake it up. Say **"shut down"**, **"go to sleep"**, or **"goodbye"** to put it back to sleep, or use the mute button in the HUD to pause listening without ending the session.

## How it's put together

```
main.py                    Entry point — opens the HUD window, starts the voice backend
config.py                  API keys, model routing, paths

voice/
  vision_live.py           The real, running voice pipeline (Gemini Live native audio),
                            tool-calling, Look-Act-Look loop, session/reconnect lifecycle
  acoustic_cues.py         Synthesized thinking/success/error/waiting tones
  wake_word.py, stt.py     Dormant wake-word detection (Whisper-based STT — only used
                            while dormant, not part of the live conversation path)
  tts.py                   Unused by the live pipeline (kept for the legacy text pipeline)

system_control/
  dispatcher.py            Gatekeeper for every action — confirmation-token system
  mac_actions.py           AppleScript-based app/volume/calendar/email control
  computer_control.py      GUI automation: mouse/keyboard via Quartz, screenshot capture,
                            session/step-limit guard, app whitelist
  computer_control_selftest.py   Standalone verification script (no live GUI clicks)
  file_tools.py            Sandboxed file read/list/search
  code_exec.py             Sandboxed-ish Python execution
  web_search.py            Tavily web search

brain/                     Experimental text-chat pipeline (OpenRouter) — real code,
                            but NOT wired into main.py; kept for future use
memory/
  db.py                    SQLite conversation history + durable facts
  semantic.py               Chroma-based semantic search over past conversations

ui/
  static/                  The HUD: orb.js (Three.js particle orb), app.js, index.html
  ui.py                    Alternate pywebview entry point (bridges to brain/, unused by main.py)
```

## Computer Use: how the safety model works

Starting a GUI-automation task (`start_computer_use`) normally requires a one-time confirmation — VISION proposes a goal, you approve it out loud, and only then can it start moving the mouse and typing. Once approved, a task is capped at 15 actions before it must stop and check in again.

**Known accepted risk tradeoff:** a short whitelist of apps (`BENIGN_APPS` in `system_control/computer_control.py`) lets tasks start with *no* confirmation at all when one of those apps is already frontmost. This trades safety for convenience for low-stakes apps (media players, browsers) — edit that list directly if you want to change the balance. There's currently no ongoing check of which app is frontmost *during* an already-active session, so a confirmed (or whitelisted) session isn't restricted to staying inside the app that justified it. If you want tighter guarantees here, that's the first thing to change.

## Testing

```bash
python3 -m system_control.computer_control_selftest
```

Verifies the confirmation-token flow, coordinate math, and the app whitelist — without performing any real clicks or keystrokes (those need a human watching a real screen to judge).

## Status

This is an actively evolving personal project, not a polished release. The `brain/` text-chat pipeline exists in the repo but isn't part of the app you actually run — everything under `voice/` is what's live.
