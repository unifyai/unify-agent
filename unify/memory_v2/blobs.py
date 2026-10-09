"""Content-addressed blob folder for bulk episode payloads (spec §5)."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

BLOB_ID = re.compile(r"^[0-9a-f]{64}\Z")


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha: str) -> Path:
        return self.root / sha[:2] / sha[2:]

    def put(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        p = self._path(sha)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, p)
        return sha

    def put_file(self, path: Path) -> str:
        """Store a file's bytes without reading it whole into memory; the same id :meth:`put` gives."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
        p = self._path(sha)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            shutil.copyfile(path, tmp)
            os.replace(tmp, p)
        return sha

    def get(self, sha: str) -> bytes:
        return self._path(sha).read_bytes()

    def has(self, sha: object) -> bool:
        """Whether *sha* is a well-formed blob id stored here (a malformed id is never looked up)."""
        return (
            isinstance(sha, str)
            and BLOB_ID.match(sha) is not None
            and self._path(sha).is_file()
        )

    def size(self, sha: str) -> int:
        return self._path(sha).stat().st_size

    def cap_text(self, text: str, limit_bytes: int) -> dict:
        raw = text.encode()
        if len(raw) <= limit_bytes:
            return {"text": text}
        excerpt = raw[:limit_bytes].decode("utf-8", errors="ignore")
        return {"excerpt": excerpt, "blob": self.put(raw), "bytes": len(raw)}
