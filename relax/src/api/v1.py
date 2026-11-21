"""osu! API v1 — rate-limited legacy key client.

Only used for `get_replay`, which v2 does not expose.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from collections import deque
from typing import Optional

import requests
from dotenv import load_dotenv


API_V1 = "https://osu.ppy.sh/api"


class RateLimiter:
    """Simple sliding-window limiter: N tokens per W seconds."""

    def __init__(self, n_per_window: int = 50, window_s: float = 60.0):
        self.n = n_per_window
        self.w = window_s
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                while self._stamps and now - self._stamps[0] > self.w:
                    self._stamps.popleft()
                if len(self._stamps) < self.n:
                    self._stamps.append(now)
                    return
                wait = self.w - (now - self._stamps[0]) + 0.05
            time.sleep(max(0.01, wait))


class OsuApiV1:
    """osu! API v1 wrapper with token-bucket rate limit.

    Default 50 requests per 60s (the published limit is "around 60"; stay conservative).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        n_per_minute: int = 50,
        user_agent: str = "new-era-relax/0.1 (research)",
        session: Optional[requests.Session] = None,
    ):
        load_dotenv(dotenv_path=os.environ.get("NER_ENV_FILE", ".env"))
        self.api_key = api_key or os.environ["OSU_API_V1_KEY"]
        self.limiter = RateLimiter(n_per_minute, 60.0)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _get(self, path: str, params: dict) -> requests.Response:
        self.limiter.acquire()
        p = dict(params)
        p["k"] = self.api_key
        for attempt in range(3):
            r = self.session.get(f"{API_V1}{path}", params=p, timeout=30)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            return r
        return r

    def get_replay_bytes(
        self,
        *,
        beatmap_id: int,
        user_id: int,
        mode: int = 0,
        mods: Optional[int] = None,
    ) -> Optional[bytes]:
        """Return the raw .osr-bytes payload (just the lzma'd replay data, not a full .osr).

        osu! v1 get_replay returns ONLY the replay-frames block (base64). To get a
        true .osr file we would need to synthesize the header. For per-click feature
        extraction we only need the frames + player metadata, so we store the raw
        base64-decoded bytes and parse them with our own helper.

        Returns None on 401/403/404/replay-unavailable.
        """
        params = {"b": beatmap_id, "u": user_id, "m": mode, "type": "id"}
        if mods is not None:
            params["mods"] = mods
        r = self._get("/get_replay", params)
        if r.status_code in (401, 403, 404):
            return None
        if r.status_code != 200:
            return None
        try:
            js = r.json()
        except Exception:
            return None
        if not isinstance(js, dict):
            return None
        content = js.get("content")
        if not content:
            # {"error": "..."} shape
            return None
        try:
            return base64.b64decode(content)
        except Exception:
            return None

    def get_beatmap(self, beatmap_id: int) -> Optional[dict]:
        r = self._get("/get_beatmaps", {"b": beatmap_id})
        if r.status_code != 200:
            return None
        js = r.json()
        return js[0] if isinstance(js, list) and js else None
