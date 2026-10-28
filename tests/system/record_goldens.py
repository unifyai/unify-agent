"""Re-record the rendered goldens (guard J2) on a keyless tree: ``python -m tests.system.record_goldens``.

Writes ``tests/system/goldens/goldens.json``: today's render with every correction in :mod:`.goldens` that still
applies. Review the diff against ``claims.tsv`` before committing it.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.system.goldens import GOLDENS, golden_of
from tests.system.render import render_all

if __name__ == "__main__":
    golden, applied = golden_of(render_all())
    path = Path(__file__).parent / GOLDENS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"{len(golden)} texts, {len(applied)} corrections applied -> {path}")
