"""Simple speech output with a safe no-audio fallback."""

from __future__ import annotations

import queue
import threading
from typing import Optional


try:
    import pyttsx3
except Exception:  # pragma: no cover - optional dependency behavior
    pyttsx3 = None


class SpeechEngine:
    """Background speech queue so detections do not block camera inference."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and pyttsx3 is not None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 155)
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def say(self, text: str) -> None:
        if self.enabled:
            self._queue.put(text)
        else:
            print(f"[speech disabled] {text}")

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            self._engine.say(text)
            self._engine.runAndWait()
            self._queue.task_done()
