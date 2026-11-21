"""osu! API v2 — OAuth client_credentials wrapper.

Scope: public. Enough for beatmap search, beatmap metadata, and leaderboards.

Reference: https://osu.ppy.sh/docs/index.html
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


OSU_API_BASE = "https://osu.ppy.sh/api/v2"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"


@dataclass
class _Token:
    access_token: str
    expires_at: float  # unix seconds


class OsuApiV2:
    """Thin OAuth client_credentials wrapper.

    Thread-safe: shares a token across callers, refreshes on expiry.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        *,
        session: Optional[requests.Session] = None,
        user_agent: str = "new-era-relax/0.1 (research)",
    ):
        load_dotenv(dotenv_path=os.environ.get("NER_ENV_FILE", ".env"))
        self.client_id = client_id or os.environ["OSU_CLIENT_ID"]
        self.client_secret = client_secret or os.environ["OSU_CLIENT_SECRET"]
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        self._token: Optional[_Token] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ oauth
    def _ensure_token(self) -> str:
        with self._lock:
            now = time.time()
            if self._token is None or self._token.expires_at - 60 <= now:
                r = requests.post(
                    TOKEN_URL,
                    json={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "grant_type": "client_credentials",
                        "scope": "public",
                    },
                    timeout=30,
                )
                r.raise_for_status()
                js = r.json()
                self._token = _Token(
                    access_token=js["access_token"],
                    expires_at=now + float(js.get("expires_in", 3600)),
                )
            return self._token.access_token

    def _get(self, path: str, params: Optional[dict] = None) -> requests.Response:
        token = self._ensure_token()
        url = f"{OSU_API_BASE}{path}"
        # 3 tries with backoff on 429/5xx
        for attempt in range(3):
            r = self.session.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            if r.status_code == 401:
                # force refresh
                with self._lock:
                    self._token = None
                token = self._ensure_token()
                continue
            if r.status_code == 429 or 500 <= r.status_code < 600:
                sleep_s = min(60, 2 ** attempt * 2)
                time.sleep(sleep_s)
                continue
            return r
        return r  # last response

    # ------------------------------------------------------------------ endpoints
    def get_beatmap(self, beatmap_id: int) -> Optional[Dict[str, Any]]:
        r = self._get(f"/beatmaps/{beatmap_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def get_beatmapset(self, beatmapset_id: int) -> Optional[Dict[str, Any]]:
        r = self._get(f"/beatmapsets/{beatmapset_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def get_beatmap_scores(
        self,
        beatmap_id: int,
        *,
        mode: str = "osu",
        mods: Optional[List[str]] = None,
        scope: str = "global",
        limit: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """GET /beatmaps/{bid}/scores — default returns top 50 global."""
        params: Dict[str, Any] = {"mode": mode, "type": scope, "limit": limit}
        if mods:
            # repeated param: osu! API wants mods[]=HD&mods[]=HR
            # requests handles list values correctly
            params["mods[]"] = mods
        r = self._get(f"/beatmaps/{beatmap_id}/scores", params=params)
        if r.status_code in (401, 403, 404):
            return None
        r.raise_for_status()
        return r.json()

    def search_beatmapsets(
        self,
        *,
        query: Optional[str] = None,
        mode: int = 0,  # 0 = osu!standard
        status: str = "ranked",
        sort: str = "plays_desc",
        cursor_string: Optional[str] = None,
    ) -> Dict[str, Any]:
        """GET /beatmapsets/search — returns one page of results.

        sort options include plays_desc, ranked_desc, rating_desc, favourites_desc.
        """
        params: Dict[str, Any] = {"m": mode, "s": status, "sort": sort}
        if query:
            params["q"] = query
        if cursor_string:
            params["cursor_string"] = cursor_string
        r = self._get("/beatmapsets/search", params=params)
        r.raise_for_status()
        return r.json()
