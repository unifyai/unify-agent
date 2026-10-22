"""The pin plugin of the library test kit (:mod:`unify.memory.pin`): a stepping clock from a fixed epoch and a seeded
``random``, installed only under its box name. The kit (``memlab``, the pin, the blobs a test names) is what the
writer's own tests import in its box; the gate runs no test (rev 2.1).
"""

import random
import shutil
import subprocess
import sys
from pathlib import Path

from unify.memory import qa as qa_mod


def test_the_pin_installs_only_under_its_box_name(tmp_path):
    import time

    before = time.time
    import unify.memory.pin  # noqa: F401 - imported on the host: changes nothing

    assert time.time is before
    box = tmp_path / "box"
    box.mkdir()
    shutil.copyfile(Path(qa_mod.__file__).with_name("pin.py"), box / "_memory_pin.py")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import _memory_pin, datetime, random, time; "
        "print(time.time_ns(), time.time_ns(), datetime.datetime.now().isoformat(), random.random())"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-I", "-c", code, str(box)],
            capture_output=True,
            text=True,
            env={"TZ": "UTC", "PATH": "/usr/bin:/bin", "MEMORY_PIN": variant},
            timeout=60,
            check=True,
        ).stdout
        for variant in ("0", "0", "1")
    ]
    assert (
        runs[0] == runs[1] != runs[2]
    )  # reproducible per variant, different across them
    a, b, now, rnd = runs[0].split()
    assert int(b) - int(a) == 1_000_000 and now.startswith("2001-09-09T01:46:40")
    assert float(rnd) == random.Random(0).random()
    a, b, now, rnd = runs[2].split()
    assert int(b) - int(a) == 7_000_000 and now.startswith("2033-05-18T03:33:20")
    assert float(rnd) == random.Random(1).random()
