"""``python -m tests.system.goldens_apply <mode>``: the design-true golden of a recording of first requests
(:func:`tests.system.goldens.design_true`), written as ``goldens/first_requests-<mode>.json``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tests.system.goldens import FIRST_REQUESTS, design_true

if __name__ == "__main__":
    mode = sys.argv[1]
    path = Path(__file__).parent / FIRST_REQUESTS.format(mode=mode)
    recorded = json.loads(path.with_suffix(".recorded.json").read_text())
    path.write_text(json.dumps(design_true(recorded), indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(path)
