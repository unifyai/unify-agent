import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from unify.memory_v2.sandbox_run import (
    ETC_ALLOW,
    PYTHON,
    RLIMIT_AS_BYTES,
    RLIMIT_FSIZE_BYTES,
    RLIMIT_NOFILE_COUNT,
    RLIMIT_NPROC_COUNT,
    SYSTEM_DIRS,
    _account_home,
    _python_roots,
    run_confined,
    run_pytest,
)

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)


def test_no_network_no_home_no_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "should-not-leak-123")
    code = (
        "import os, socket, pathlib\n"
        "print('KEY' in ''.join(os.environ))\n"
        "print(any(pathlib.Path.home().iterdir()) if pathlib.Path.home().exists() else False)\n"
        "s=socket.socket()\n"
        "try:\n s.connect(('1.1.1.1',80)); print('net')\nexcept OSError: print('nonet')\n"
    )
    r = run_confined([str(PYTHON), "-c", code], timeout_s=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "False", "nonet"]


def _home_files_outside_box() -> list[Path]:
    """Existing, readable files under the real home that the box must not expose (never read here).

    Under the repository's test sandbox only the checkout is visible beneath the home, so this test's
    own file is the probe there; outside it, ``~/.env`` and ``~/.bashrc`` are probed too.
    """
    home = _account_home()
    here = Path(__file__).resolve()
    candidates = [home / ".env", home / ".bashrc", here.parents[2] / ".env", here]
    roots = [r.resolve() for r in _python_roots()]
    found = []
    for p in candidates:
        try:
            if not p.is_file() or not os.access(p, os.R_OK):
                continue
        except OSError:
            continue
        rp = p.resolve()
        if p.is_relative_to(home) and not any(rp.is_relative_to(r) for r in roots):
            found.append(p)
    return found


def test_home_files_are_not_readable_inside():
    files = _home_files_outside_box()
    if not files:
        pytest.skip("no readable file under the home directory to probe")
    code = (
        "import sys\n"
        "for p in sys.argv[1:]:\n"
        "    try:\n"
        "        open(p, 'rb').close(); print('READABLE')\n"
        "    except OSError:\n"
        "        print('hidden')\n"
    )
    r = run_confined([str(PYTHON), "-c", code, *map(str, files)], timeout_s=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["hidden"] * len(files), files


def test_venv_site_packages_importable_inside():
    r = run_confined(
        [str(PYTHON), "-c", "import pytest, sys; print(sys.prefix != sys.base_prefix)"],
        timeout_s=60,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["True"]


def test_run_confined_timeout_kills_group():
    t0 = time.time()
    r = run_confined(
        [
            str(PYTHON),
            "-c",
            "import subprocess,time; subprocess.Popen(['sleep','300']); time.sleep(300)",
        ],
        timeout_s=2,
    )
    assert r.timed_out and time.time() - t0 < 30
    left = subprocess.run(
        ["pgrep", "-f", "^sleep 300$"],
        capture_output=True,
        text=True,
    )
    assert left.stdout.strip() == ""


def test_run_pytest_reports_pass_and_fail(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "test_x.py").write_text(
        "def test_ok():\n    assert 1\n\ndef test_bad():\n    assert 0\n",
    )
    out = run_pytest(
        "/box/t",
        python=PYTHON,
        ro={tmp_path / "t": "/box/t"},
        rw={},
        cwd="/box",
        timeout_s=120,
    )
    assert out.passed == {"test_x.py::test_ok"} and out.failed == {
        "test_x.py::test_bad",
    }
    assert out.valid and out.skipped == set()


def test_seccomp_denies_vsock_and_new_user_namespaces_allows_inet():
    # AF_VSOCK (40) reaches the Windows host under WSL2 whatever the network namespace.
    code = (
        "import ctypes, errno, os, socket\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "fd = libc.socket(40, 1, 0)\n"
        "print('vsock', fd, errno.errorcode.get(ctypes.get_errno()))\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "try:\n    s.connect(('1.1.1.1', 80)); print('inet connected')\n"
        "except OSError:\n    print('inet created-not-connected')\n"
        "a, b = socket.socketpair(); print('unix', a.family == socket.AF_UNIX)\n"
        "try:\n    os.unshare(os.CLONE_NEWUSER); print('newuser allowed')\n"
        "except OSError as e:\n    print('newuser', errno.errorcode[e.errno])\n"
    )
    r = run_confined([str(PYTHON), "-c", code], timeout_s=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "vsock -1 EAFNOSUPPORT",
        "inet created-not-connected",
        "unix True",
        "newuser EPERM",
    ]


def test_root_is_an_allowlist(tmp_path):
    (tmp_path / "x").mkdir()
    code = (
        "import json, os\n"
        "def ls(d):\n"
        "    try:\n        return sorted(os.listdir(d))\n"
        "    except OSError:\n        return None\n"
        "try:\n    open('/zz', 'w'); w = True\n"
        "except OSError:\n    w = False\n"
        "print(json.dumps({'root': ls('/'), 'etc': ls('/etc'), 'home': ls('/home'), 'root_writable': w,\n"
        "    'workspaces': os.path.exists('/workspaces'), 'docker': os.path.exists('/run/docker.sock'),\n"
        "    'var_docker': os.path.exists('/var/run/docker.sock'), 'mnt_c': os.path.exists('/mnt/c')}))\n"
    )
    r = run_confined(
        [str(PYTHON), "-c", code],
        ro={tmp_path / "x": "/box/x"},
        timeout_s=60,
    )
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    allowed = {d.strip("/") for d in SYSTEM_DIRS} | {
        "etc",
        "dev",
        "proc",
        "tmp",
        "home",
        "box",
    }
    allowed |= {root.parts[1] for root in _python_roots()}
    assert set(got["root"]) <= allowed, set(got["root"]) - allowed
    assert set(got["etc"]) <= {name.split("/")[0] for name in ETC_ALLOW}
    assert set(got["home"]) <= {
        root.parts[2] for root in _python_roots() if root.parts[1] == "home"
    }
    assert not got["root_writable"]
    assert not (got["workspaces"] or got["docker"] or got["var_docker"] or got["mnt_c"])


def test_rlimits_inside():
    code = (
        "import resource\n"
        "for n in ('RLIMIT_AS', 'RLIMIT_NPROC', 'RLIMIT_FSIZE', 'RLIMIT_NOFILE'):\n"
        "    print(resource.getrlimit(getattr(resource, n))[1])\n"
    )
    r = run_confined([str(PYTHON), "-c", code], timeout_s=60)
    assert r.returncode == 0, r.stderr
    assert [int(x) for x in r.stdout.split()] == [
        RLIMIT_AS_BYTES,
        RLIMIT_NPROC_COUNT,
        RLIMIT_FSIZE_BYTES,
        RLIMIT_NOFILE_COUNT,
    ]


def _pytest_dir(tmp_path, files: dict[str, str]) -> Path:
    root = tmp_path / "t"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def test_run_pytest_ids_for_classes_subdirs_and_skips(tmp_path):
    root = _pytest_dir(
        tmp_path,
        {
            "__init__.py": "",
            "test_x.py": (
                "import pytest\n"
                "class TestA:\n    def test_m(self):\n        assert 0\n"
                "def test_ok():\n    assert 1\n"
                "@pytest.mark.skip\ndef test_skipped():\n    assert 0\n"
                "@pytest.mark.xfail\ndef test_xf():\n    assert 0\n"
            ),
            "sub/__init__.py": "",
            "sub/test_x.py": "def test_ok():\n    assert 1\n",
        },
    )
    out = run_pytest(
        "/box/t",
        python=PYTHON,
        ro={root: "/box/t"},
        rw={},
        cwd="/box",
        timeout_s=120,
    )
    assert out.passed == {"test_x.py::test_ok", "sub/test_x.py::test_ok"}, out.output
    assert out.failed == {"test_x.py::TestA::test_m"}
    assert out.skipped == {"test_x.py::test_skipped", "test_x.py::test_xf"}
    assert out.valid and out.returncode == 1


def test_run_pytest_forged_junit_is_invalid(tmp_path):
    forged = (
        '<testsuites><testsuite><testcase classname="t.test_y" name="test_bad"/>'
        "</testsuite></testsuites>"
    )
    root = _pytest_dir(
        tmp_path,
        {
            "test_y.py": (
                "import atexit\n"
                f"atexit.register(lambda: open('/junit/r.xml', 'w').write({forged!r}))\n"
                "def test_bad():\n    assert 0\n"
            ),
        },
    )
    out = run_pytest(
        "/box/t",
        python=PYTHON,
        ro={root: "/box/t"},
        rw={},
        cwd="/box",
        timeout_s=120,
    )
    assert out.returncode == 1 and not out.valid


def test_run_pytest_malformed_junit_is_invalid(tmp_path):
    root = _pytest_dir(
        tmp_path,
        {
            "test_z.py": (
                "import atexit\n"
                "atexit.register(lambda: open('/junit/r.xml', 'w').write('<not xml'))\n"
                "def test_z():\n    assert 1\n"
            ),
        },
    )
    out = run_pytest(
        "/box/t",
        python=PYTHON,
        ro={root: "/box/t"},
        rw={},
        cwd="/box",
        timeout_s=120,
    )
    assert not out.valid and out.passed == set() and "passed" in out.output


@pytest.mark.parametrize("plant", ["symlink", "fifo", "dir"])
def test_run_pytest_never_follows_a_planted_junit(tmp_path, plant):
    host = tmp_path / "host.xml"
    host.write_text(
        '<testsuites><testsuite><testcase classname="t.test_v" name="host_only_case"/>'
        "</testsuite></testsuites>",
    )
    root = _planted_junit_dir(tmp_path, plant, host)
    t0 = time.time()
    out = run_pytest(
        "/box/t",
        python=PYTHON,
        ro={root: "/box/t"},
        rw={},
        cwd="/box",
        timeout_s=120,
    )
    assert out.returncode == 0 and not out.valid, out.output
    assert not any(
        "host_only_case" in name for name in out.passed | out.failed | out.skipped
    )
    assert time.time() - t0 < 60


def _planted_junit_dir(tmp_path, plant: str, host: Path) -> Path:
    make = {
        "symlink": f"os.symlink({str(host)!r}, '/junit/r.xml')",
        "fifo": "os.mkfifo('/junit/r.xml')",
        "dir": "os.mkdir('/junit/r.xml')",
    }[plant]
    return _pytest_dir(
        tmp_path,
        {
            "test_v.py": (
                "import atexit, os\n"
                "def _plant():\n"
                "    os.remove('/junit/r.xml')\n"
                f"    {make}\n"
                "atexit.register(_plant)\n"
                "def test_ok():\n    assert 1\n"
            ),
        },
    )


def test_run_pytest_planted_junit_leaks_no_descriptor(tmp_path):
    root = _planted_junit_dir(tmp_path, "dir", tmp_path / "unused.xml")
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(3):
        out = run_pytest(
            "/box/t",
            python=PYTHON,
            ro={root: "/box/t"},
            rw={},
            cwd="/box",
            timeout_s=120,
        )
        assert not out.valid
    assert len(os.listdir("/proc/self/fd")) == before


def test_output_capture_is_bounded_and_never_fails_to_decode():
    from unify.memory_v2.sandbox_run import CAPTURE_MAX_BYTES

    # floods stdout (also through the box's pid 1, which holds the host's pipes) and writes invalid UTF-8
    code = (
        "import os, sys\n"
        "chunk = b'x' * (1 << 20)\n"
        "for _ in range(6):\n"
        "    sys.stdout.buffer.write(chunk)\n"
        "sys.stdout.buffer.flush()\n"
        "with open('/proc/1/fd/1', 'wb') as f:\n"
        "    for _ in range(6):\n"
        "        f.write(chunk)\n"
        "    f.write(b'END')\n"
        "sys.stderr.buffer.write(b'bad \\xff\\xfe bytes')\n"
    )
    r = run_confined([str(PYTHON), "-c", code], timeout_s=60)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.endswith("END") and "earlier bytes dropped" in r.stdout
    assert len(r.stdout.encode()) <= CAPTURE_MAX_BYTES + 100
    assert r.stderr == "bad \ufffd\ufffd bytes"
