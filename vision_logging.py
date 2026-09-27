"""
vision_logging.py
Makes VISION diagnosable when it is packaged as a .app.

Three separate jobs, all of which have to be set up before anything else
runs:

  1. Real file logging. Everything the app prints -- and every one of the
     ~100 bare print() calls scattered through the codebase -- is teed into
     LOG_FILE_PATH. Inside a .app there is no attached terminal, so stdout
     goes to /dev/null and print() output simply evaporates; that is why
     the logs directory was created but always empty.

  2. Nothing-gets-swallowed exception handling. Uncaught exceptions on the
     main thread, in any other thread, and inside asyncio callbacks all end
     up in the log instead of a discarded stderr. The asyncio one matters
     most: a QueueFull raised inside a call_soon_threadsafe callback is
     reported *only* through the asyncio exception handler.

  3. A freeze watchdog. A hang leaves no crash report, so there is nothing
     to read afterwards -- the evidence has to be collected while the app
     is still stuck. A daemon thread watches a heartbeat that the event
     loop updates; when the loop stops ticking it dumps the stack of every
     thread to the log. That turns "it froze and I found nothing" into a
     stack trace naming the exact blocking call.

Also installs a SIGUSR1 handler, so a stuck VISION can be made to dump its
stacks on demand without killing it:

    kill -USR1 $(pgrep -f VISION)
"""

import atexit
import faulthandler
import logging
import os
import signal
import sys
import threading
import time
import traceback
from logging.handlers import RotatingFileHandler

import config

# How long the asyncio event loop may go without ticking before the
# watchdog treats it as hung and dumps every thread's stack. The loop
# heartbeat fires every second, so this is ~15 missed beats -- long enough
# that a slow blocking call (a 10s AppleScript on the loop) doesn't trip
# it, short enough to catch a real freeze while it is still happening.
LOOP_STALL_SECONDS = 15.0
WATCHDOG_POLL_SECONDS = 5.0

# Once stuck, VISION usually stays stuck. Re-dumping every 5s would fill
# the log with the same stacks, so back off between dumps.
STALL_REDUMP_SECONDS = 60.0

_MAX_BYTES = 8 * 1024 * 1024
_BACKUP_COUNT = 3

log = logging.getLogger("vision")

_configured = False
_heartbeat = {"at": 0.0, "started": False}
_fault_log = None  # kept open for the lifetime of the process for faulthandler


class _StreamToLog:
    """File-like shim that forwards print() output into the logging system.

    Bare print() is how this codebase logs, and rewriting ~100 call sites
    would be a much bigger change than making print() actually go
    somewhere. Line-buffered so a partial write doesn't produce a log
    record without its newline.
    """

    def __init__(self, level, mirror=None):
        self._level = level
        self._mirror = mirror
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, text):
        if self._mirror is not None:
            try:
                self._mirror.write(text)
            except Exception:
                pass
        if not text:
            return len(text) if text else 0
        with self._lock:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    log.log(self._level, line)
        return len(text)

    def flush(self):
        if self._mirror is not None:
            try:
                self._mirror.flush()
            except Exception:
                pass
        with self._lock:
            if self._buf.strip():
                log.log(self._level, self._buf)
            self._buf = ""

    def isatty(self):
        return False

    def fileno(self):
        # faulthandler and subprocess want a real descriptor. Hand them the
        # mirror's if there is one; otherwise say we have none rather than
        # returning a bogus number.
        if self._mirror is not None and hasattr(self._mirror, "fileno"):
            return self._mirror.fileno()
        raise OSError("no fileno")


def dump_all_stacks(reason: str):
    """Writes every live thread's stack to the log.

    This is the payload of the whole module: for a hang it is the only
    evidence there will ever be, since a deadlocked process never produces
    a crash report.
    """
    try:
        frames = sys._current_frames()
        names = {t.ident: t.name for t in threading.enumerate()}
        lines = [f"===== THREAD DUMP: {reason} ====="]
        for ident, frame in frames.items():
            lines.append(f"--- thread {names.get(ident, '?')} ({ident}) ---")
            lines.extend(
                s.rstrip("\n") for s in traceback.format_stack(frame)
            )
        lines.append("===== END THREAD DUMP =====")
        log.error("\n".join(lines))
        for h in log.handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        # A diagnostic must never be the thing that takes the app down.
        try:
            log.exception("Thread dump failed")
        except Exception:
            pass


def heartbeat():
    """Called from the asyncio loop to say 'still ticking'."""
    _heartbeat["at"] = time.monotonic()
    _heartbeat["started"] = True


async def heartbeat_loop():
    """Long-running task that keeps the watchdog fed. Runs on the event
    loop, so it stops ticking exactly when the loop stops -- which is the
    condition worth detecting."""
    import asyncio

    heartbeat()
    while True:
        await asyncio.sleep(1.0)
        heartbeat()


def _watchdog():
    last_dump = 0.0
    while True:
        time.sleep(WATCHDOG_POLL_SECONDS)
        if not _heartbeat["started"]:
            continue
        age = time.monotonic() - _heartbeat["at"]
        if age < LOOP_STALL_SECONDS:
            continue
        now = time.monotonic()
        if now - last_dump < STALL_REDUMP_SECONDS:
            continue
        last_dump = now
        dump_all_stacks(
            f"event loop has not ticked for {age:.1f}s -- VISION appears hung"
        )


def _install_excepthooks():
    def _sys_hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Uncaught exception on main thread",
                     exc_info=(exc_type, exc, tb))

    def _thread_hook(args):
        if issubclass(args.exc_type, SystemExit):
            return
        log.critical(
            f"Uncaught exception in thread {args.thread.name if args.thread else '?'}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook

    try:
        # Python 3.12+: exceptions never retrieved from a Task. Without this
        # a core task can die and the only notice is a GC-time message to a
        # stderr that goes nowhere.
        def _unraisable(args):
            log.error(f"Unraisable exception in {args.object!r}",
                      exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        sys.unraisablehook = _unraisable
    except Exception:
        pass


def install_asyncio_exception_handler(loop):
    """Routes asyncio's own error reporting into the log.

    Matters specifically because out_queue.put_nowait() is scheduled with
    call_soon_threadsafe from the realtime audio callback: when the queue
    is full, the resulting QueueFull surfaces *only* here.
    """
    def handler(loop_, context):
        message = context.get("message", "asyncio error")
        exc = context.get("exception")
        detail = ", ".join(
            f"{k}={v!r}" for k, v in context.items()
            if k not in ("message", "exception")
        )
        if exc is not None:
            log.error(f"[asyncio] {message} ({detail})",
                      exc_info=(type(exc), exc, exc.__traceback__))
        else:
            log.error(f"[asyncio] {message} ({detail})")

    loop.set_exception_handler(handler)


def setup(component: str = "vision"):
    """Idempotent. Call as early as possible -- before webview, before any
    module that prints at import time."""
    global _configured, _fault_log
    if _configured:
        return config.LOG_FILE_PATH
    _configured = True

    path = config.LOG_FILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # Third-party libraries are chatty at INFO and would bury VISION's own
    # lines; their warnings still come through.
    for noisy in ("httpx", "httpcore", "urllib3", "websockets", "slack_sdk",
                  "chromadb", "sentence_transformers", "PIL", "asyncio",
                  "google_genai", "google.genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # ...except asyncio, whose warnings are exactly the hang evidence wanted.
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    frozen = bool(getattr(sys, "frozen", False))

    if frozen:
        # Compiled dependencies (whisper.cpp/ggml, PortAudio, OpenCV,
        # WebKit) write to file descriptors 1 and 2 directly, bypassing
        # sys.stdout entirely -- a Python-level shim never sees them. In a
        # .app those descriptors point at /dev/null, so the most valuable
        # output for a native freeze or a device error is discarded. Point
        # the real descriptors at a file. Separate from vision.log because
        # rotating that file would leave these descriptors writing to the
        # rotated-away inode.
        try:
            native = open(path.parent / "vision-native.log", "ab", buffering=0)
            os.dup2(native.fileno(), 1)
            os.dup2(native.fileno(), 2)
        except Exception:
            log.exception("Could not redirect native stdio")

    # Running from a terminal, keep printing to it as well as to the file;
    # inside a .app there is nothing to mirror to.
    mirror_out = None if frozen else sys.__stdout__
    mirror_err = None if frozen else sys.__stderr__
    sys.stdout = _StreamToLog(logging.INFO, mirror_out)
    sys.stderr = _StreamToLog(logging.ERROR, mirror_err)

    _install_excepthooks()

    # A hard crash in native code (PortAudio, WebKit, OpenCV) never reaches
    # Python's excepthook. faulthandler catches those signals and writes a
    # C-level traceback. Separate file: it needs a real descriptor, which
    # the logging handler does not expose safely.
    try:
        _fault_log = open(path.parent / "vision-faulthandler.log", "a", buffering=1)
        faulthandler.enable(file=_fault_log, all_threads=True)
        # On-demand dump for a live hang: kill -USR1 <pid>
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, file=_fault_log,
                                  all_threads=True, chain=True)
    except Exception:
        log.exception("Could not enable faulthandler")

    if hasattr(signal, "SIGUSR2"):
        try:
            signal.signal(
                signal.SIGUSR2,
                lambda *_: dump_all_stacks("SIGUSR2 requested"),
            )
        except Exception:
            pass

    threading.Thread(target=_watchdog, name="vision-watchdog",
                     daemon=True).start()

    atexit.register(lambda: log.info("=== VISION exiting ==="))

    log.info("=" * 70)
    log.info(f"=== VISION starting: component={component} pid={os.getpid()} "
             f"frozen={frozen} python={sys.version.split()[0]} ===")
    log.info(f"=== log file: {path} ===")
    log.info("=" * 70)

    return path
