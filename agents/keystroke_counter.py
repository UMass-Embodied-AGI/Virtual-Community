"""
KeystrokeCounter — runs pynput in a subprocess to avoid conflicts with
Genesis's macOS event loop (which causes a Trace/BPT trap when both share
the same process).
"""
import multiprocessing as mp
from collections import defaultdict
from threading import Lock

from pynput.keyboard import Key, KeyCode


# ---------------------------------------------------------------------------
# Serialization helpers (Key/KeyCode → str and back) for cross-process IPC
# ---------------------------------------------------------------------------

def _serialize_key(key):
    if isinstance(key, Key):
        return f"Key.{key.name}"
    if isinstance(key, KeyCode):
        if key.char is not None:
            return f"char:{key.char}"
        if key.vk is not None:
            return f"vk:{key.vk}"
    return None


def _deserialize_key(s):
    if s is None:
        return None
    if s.startswith("Key."):
        return getattr(Key, s[4:], None)
    if s.startswith("char:"):
        return KeyCode(char=s[5:])
    if s.startswith("vk:"):
        return KeyCode(vk=int(s[3:]))
    return None


# ---------------------------------------------------------------------------
# Subprocess entry point — runs entirely in a child process with no Genesis
# ---------------------------------------------------------------------------

def _listener_worker(event_queue: mp.Queue, stop_event: mp.Event):
    """Target for the listener subprocess. Sends serialized keys to event_queue."""
    from pynput.keyboard import Listener

    def on_press(key):
        serialized = _serialize_key(key)
        if serialized is not None:
            event_queue.put(serialized)

    with Listener(on_press=on_press):
        stop_event.wait()   # block until main process signals shutdown


# ---------------------------------------------------------------------------
# Public interface — mirrors the old KeystrokeCounter API
# ---------------------------------------------------------------------------

class KeystrokeCounter:
    """
    Drop-in replacement for the pynput-based KeystrokeCounter.

    Starts pynput in a child process (via mp.Process) so that it never
    shares the macOS event loop with the Genesis viewer.
    """

    def __init__(self):
        ctx = mp.get_context("spawn")
        self._event_queue: mp.Queue = ctx.Queue()
        self._stop_event: mp.Event = ctx.Event()
        self._process = ctx.Process(
            target=_listener_worker,
            args=(self._event_queue, self._stop_event),
            daemon=True,
        )
        # Internal state (kept for __getitem__ compatibility)
        self._lock = Lock()
        self._key_count_map = defaultdict(lambda: 0)
        self._press_list: list = []

    # ------------------------------------------------------------------
    # Context-manager protocol (mirrors pynput Listener)
    # ------------------------------------------------------------------

    def __enter__(self):
        self._process.start()
        return self

    def __exit__(self, *args):
        self._stop_event.set()
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()

    # ------------------------------------------------------------------
    # Drain the IPC queue into local state
    # ------------------------------------------------------------------

    def _drain(self):
        while not self._event_queue.empty():
            try:
                raw = self._event_queue.get_nowait()
                key = _deserialize_key(raw)
                if key is not None:
                    with self._lock:
                        self._key_count_map[key] += 1
                        self._press_list.append(key)
            except Exception:
                break

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_press_events(self) -> list:
        self._drain()
        with self._lock:
            events = list(self._press_list)
            self._press_list = []
            return events

    def clear(self):
        self._drain()
        with self._lock:
            self._key_count_map = defaultdict(lambda: 0)
            self._press_list = []

    def __getitem__(self, key):
        self._drain()
        with self._lock:
            return self._key_count_map[key]


if __name__ == "__main__":
    import time

    print("Press keys (Ctrl-C to quit)…")
    with KeystrokeCounter() as counter:
        try:
            while True:
                time.sleep(1 / 60)
                events = counter.get_press_events()
                if events:
                    print("events:", events)
        except KeyboardInterrupt:
            pass
