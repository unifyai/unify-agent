import json

from unify.memory_v2.episodes import Action
from unify.memory_v2.integration.adapters.shell import (
    MAX_ARG_CHARS,
    MAX_ARGV,
    MAX_TAIL_BYTES,
    ShellCellResult,
    action_from_audit,
    action_from_cell,
    bounded_tail,
    shell_actions,
    shell_fingerprint,
)
from unify.memory_v2.redact import Redactor

KEY = "sk-or-v1-" + "ab" * 32


def test_channel_is_basename_of_argv0():
    a = action_from_audit(
        {"event": "subprocess", "argv": ["/usr/bin/git", "status"], "cell": 3},
    )
    assert (a.kind, a.channel, a.method, a.cell) == ("shell", "git", "run", 3)
    assert a.args == ["/usr/bin/git", "status"] and a.kwargs == {}
    assert (
        action_from_audit({"argv": ["python3", "-V"], "cell": 0}).channel == "python3"
    )
    # no argv: fall back to the executable, then to "unknown"
    assert action_from_audit({"argv": [], "exe": "/bin/ls", "cell": 0}).channel == "ls"
    assert action_from_audit({"argv": [], "cell": 0}).channel == "unknown"
    # a command string (os.system) runs under sh -c
    s = action_from_audit({"event": "os.system", "argv": "ls | wc -l", "cell": 1})
    assert s.channel == "sh" and s.args == ["sh", "-c", "ls | wc -l"]
    # bytes / path argv entries are decoded
    assert action_from_audit({"argv": [b"/bin/echo", b"hi"], "cell": 0}).args == [
        "/bin/echo",
        "hi",
    ]


def test_bash_cell_channel_is_bash():
    a = action_from_cell(
        ShellCellResult(cell=2, command="cd x && ls", exit_code=0, stdout="a\n"),
    )
    assert (a.kind, a.channel, a.method, a.args, a.kwargs) == (
        "shell",
        "bash",
        "run",
        ["cd x && ls"],
        {},
    )


def test_argv_is_capped_with_markers():
    a = action_from_audit({"argv": ["echo"] + [str(i) for i in range(500)], "cell": 0})
    assert len(a.args) == MAX_ARGV + 1
    assert a.args[-1] == f"<truncated: {501 - MAX_ARGV} more args>"
    long = action_from_audit({"argv": ["echo", "z" * (MAX_ARG_CHARS + 10)], "cell": 0})
    assert long.args[1] == "z" * MAX_ARG_CHARS + "<truncated: 10 chars>"
    exact = action_from_audit({"argv": ["echo", "z" * MAX_ARG_CHARS], "cell": 0})
    assert exact.args == ["echo", "z" * MAX_ARG_CHARS]


SECRET = (
    "hunter2-secret-value-0123456789"  # pragma: allowlist secret (a fake test value)
)


def test_secret_straddling_the_argv_cap_leaves_no_fragment():
    red = Redactor({"API_KEY": SECRET})
    arg = "x" * (MAX_ARG_CHARS - 10) + SECRET
    a = action_from_audit({"argv": ["echo", arg], "cell": 0}, red)
    assert "hunter2" not in json.dumps(a.args)
    assert a.args[1].startswith("x" * (MAX_ARG_CHARS - 10) + "<secret:A")
    s = action_from_audit({"event": "os.system", "argv": arg, "cell": 0}, red)
    assert "hunter2" not in json.dumps(s.args)


def test_secret_straddling_the_command_cap_leaves_no_fragment():
    red = Redactor({"API_KEY": SECRET})
    cmd = "echo " + "x" * (MAX_ARG_CHARS - 15) + SECRET
    a = action_from_cell(ShellCellResult(0, cmd, 0), red)
    assert "hunter2" not in a.args[0]
    assert a.args[0].endswith("chars>")


def test_channel_is_redacted():
    red = Redactor({"API_KEY": SECRET})
    a = action_from_audit({"argv": [f"/tmp/{SECRET}", "x"], "cell": 0}, red)
    assert a.channel == "<secret:API_KEY>"
    # a secret containing a slash is redacted before the basename is taken
    red2 = Redactor({"TOKEN": "abc/defghijk"})
    b = action_from_audit({"argv": ["/opt/abc/defghijk"], "cell": 0}, red2)
    assert "defghijk" not in b.channel and "defghijk" not in json.dumps(b.args)
    k = action_from_audit({"argv": [], "exe": f"/bin/{KEY}", "cell": 0})
    assert KEY not in k.channel


def test_missing_or_bad_cell_is_minus_one():
    assert action_from_audit({"argv": ["true"], "cell": None}).cell == -1
    assert action_from_audit({"argv": ["true"]}).cell == -1
    assert action_from_audit({"argv": ["true"], "cell": "x", "exit_code": 0}).cell == -1
    assert action_from_cell(ShellCellResult(None, "ls", 0)).cell == -1


def test_zero_or_negative_tail_limit_is_empty():
    assert bounded_tail("abc", Redactor(), 0) == ""
    assert bounded_tail("abc", Redactor(), -5) == ""
    a = action_from_cell(ShellCellResult(0, "ls", 0, stdout="abc"), tail_bytes=0)
    assert a.response["stdout_tail"] == ""


def test_tail_is_capped_to_last_bytes():
    out = "".join(f"line {i}\n" for i in range(5000))
    a = action_from_cell(ShellCellResult(0, "seq", 0, stdout=out, stderr="e" * 20000))
    tail = a.response["stdout_tail"]
    assert len(tail.encode()) <= MAX_TAIL_BYTES
    assert tail.endswith("line 4999\n") and "line 0\n" not in tail
    assert len(a.response["stderr_tail"].encode()) == MAX_TAIL_BYTES
    # a custom cap, and a multi-byte character split at the cut is dropped, not mangled
    assert bounded_tail("é" * 10, Redactor(), 5) == "éé"
    small = action_from_cell(ShellCellResult(0, "x", 0, stdout="abcdef"), tail_bytes=3)
    assert small.response["stdout_tail"] == "def"


def test_redaction_of_argv_tails_and_command():
    red = Redactor({"API_KEY": "hunter2-secret-value"})
    a = action_from_audit(
        {
            "argv": ["curl", "-H", f"Authorization: Bearer {KEY}"],
            "cell": 0,
            "exit_code": 0,
            "stdout": "token=hunter2-secret-value ok",
            "stderr": f"warn {KEY}",
        },
        red,
    )
    blob = json.dumps([a.args, a.response])
    assert KEY not in blob and "hunter2-secret-value" not in blob
    assert "<redacted:key-shaped>" in a.args[2]
    assert a.response["stdout_tail"] == "token=<secret:API_KEY> ok"
    c = action_from_cell(
        ShellCellResult(0, f"export K={KEY}; echo $K", 0, stdout=KEY + "\n"),
        red,
    )
    assert KEY not in json.dumps([c.args, c.response])
    # a secret straddling the cut is redacted before the cut, so no fragment survives
    long = "x" * 100 + KEY + "y" * 10
    t = bounded_tail(long, Redactor(), 40)
    assert KEY[-20:] not in t and t.endswith("y" * 10)


def test_unrecorded_when_popen_has_no_captured_output():
    a = action_from_audit({"event": "subprocess", "argv": ["true"], "cell": 0})
    assert a.status == "unrecorded" and a.response is None and a.error is None


def test_ok_and_error_from_exit_code():
    ok = action_from_audit({"argv": ["true"], "cell": 0, "exit_code": 0, "stdout": ""})
    assert ok.status == "ok" and ok.response["exit_code"] == 0 and ok.error is None
    bad = action_from_audit({"argv": ["false"], "cell": 0, "exit_code": 1})
    assert bad.status == "error" and bad.error == "exit status 1"
    cell_bad = action_from_cell(ShellCellResult(1, "ls nope", 2, stdout="ls: nope\n"))
    assert cell_bad.status == "error" and cell_bad.response["exit_code"] == 2
    # the runner's error with no exit code (a timeout) is an error, not unrecorded
    timeout = action_from_cell(
        ShellCellResult(
            1,
            "sleep 999",
            None,
            error="The command timed out after 600s.",
        ),
    )
    assert timeout.status == "error" and "timed out" in timeout.error
    # no exit code and no error: unrecorded, though the output is kept
    unk = action_from_cell(ShellCellResult(1, "echo", None, stdout="x"))
    assert unk.status == "unrecorded" and unk.response["stdout_tail"] == "x"


def test_from_execute_result_reads_the_session_executor_dict():
    class Part:
        def __init__(self, text):
            self.text = text

    r = ShellCellResult.from_execute_result(
        4,
        "ls",
        {
            "stdout": [Part("a\n"), Part("b\n")],
            "stderr": [],
            "result": 0,
            "error": None,
        },
    )
    assert (r.cell, r.exit_code, r.stdout, r.stderr, r.error) == (
        4,
        0,
        "a\nb\n",
        "",
        None,
    )
    assert action_from_cell(r).status == "ok"


def test_shell_actions_orders_by_cell_and_skips_other_events():
    rows = shell_actions(
        [
            {"event": "open", "path": "/ws/a.csv", "cell": 0},
            {"event": "subprocess", "argv": ["git", "log"], "cell": 3},
            {"event": "subprocess", "argv": ["true"], "cell": 0},
        ],
        [ShellCellResult(1, "ls", 0, stdout="a\n")],
    )
    assert [(a.cell, a.channel) for a in rows] == [(0, "true"), (1, "bash"), (3, "git")]
    assert all(isinstance(a, Action) and a.kind == "shell" for a in rows)


def test_fingerprint_has_shapes_and_exit_codes_but_no_values():
    rows = [
        action_from_cell(
            ShellCellResult(0, "ls -l", 0, stdout="drwxr-xr-x 2 me\n-rw-r--r-- 1 me\n"),
        ),
        action_from_cell(
            ShellCellResult(1, "ls -l /nope", 2, stdout="ls: cannot access\n"),
        ),
        action_from_audit({"argv": ["git", "status"], "cell": 2}),
        Action(3, "venmo", "login", [], {}, {"token": "t"}, "ok"),  # not a shell row
    ]
    fp = shell_fingerprint(rows)
    assert set(fp) == {"bash", "git"}
    assert fp["bash"]["exit_codes"] == ["0", "2"]
    assert fp["bash"]["shapes"] == [
        "stdout:lines=1,first=a:|stderr:lines=0",
        "stdout:lines=2-9,first=a-a-a|stderr:lines=0",
    ]
    assert fp["git"] == {"exit_codes": ["unrecorded"], "shapes": []}
    text = json.dumps(fp)
    for value in ("drwx", "me", "cannot", "nope", "status"):
        assert value not in text
    # same structure, different values: same fingerprint
    again = shell_fingerprint(
        [
            action_from_cell(
                ShellCellResult(0, "ls -la", 0, stdout="lrwxr-xr-x 9 you\n-r 3 x\n"),
            ),
        ],
    )
    assert again["bash"]["shapes"] == ["stdout:lines=2-9,first=a-a-a|stderr:lines=0"]
