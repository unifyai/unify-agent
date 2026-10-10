"""Override detection: a function that replaces a value it computed from its input under a condition."""

import ast
import random
import textwrap
from pathlib import Path

from unify.memory_v2.analysis.overrides import MAX_NODES, find_overrides

# The stored function of office-v2 family F3 (case note f3-stored-procedure-error-case-20261008.md),
# reconstructed around its quoted lines: an over-cap claim is paid up to the limit (`pay = limit`), then every
# claim with a rule is paid nothing (`pay = Decimal("0.00")` under `if rule`), over-cap ones included.
F3 = '''
def process_expense_claims(apis, month):
    """Audit the month's expense claims and write the violations and reimbursement reports.

    Effect: write
    Input: env
    """
    D = Decimal
    policy = json.loads(apis.files.read("policy.json"))
    employees = {e["employee_id"]: e for e in csv.DictReader(io.StringIO(apis.files.read("employees.csv")))}
    rates = {r["currency"]: D(r["usd"]) for r in csv.DictReader(io.StringIO(apis.files.read("fx.csv")))}
    claims = list(csv.DictReader(io.StringIO(apis.files.read(f"claims/{month}.csv"))))
    violation_counts = Counter()
    violations = []
    totals = defaultdict(lambda: Decimal("0.00"))
    for claim in claims:
        employee = employees.get(claim["employee_id"])
        usd = (D(claim["amount"]) * rates[claim["currency"]]).quantize(D("0.01"))
        pay = usd
        rule = None
        if employee is None:
            rule = "unknown_employee"
        elif not claim.get("receipt") and usd > D(policy["receipt_threshold"]):
            rule = "missing_receipt"
        else:
            for category, section in policy["limits"].items():
                if category != claim["category"]:
                    continue
                action = section.get("action", "reject")
                limit = D(section.get(employee["grade"], "Infinity"))
                if action == "reject" and usd > limit:
                    rule = "over_limit"
                elif action == "cap" and usd > limit:
                    rule = "over_cap"
                    pay = limit
        if rule:
            violation_counts[rule] += 1
            violations.append({"claim_id": claim["claim_id"], "rule": rule})
            pay = Decimal("0.00")
        totals[claim["employee_id"]] += pay
    apis.files.write(f"reports/{month}-violations.json", json.dumps(violations))
    apis.files.write(f"reports/{month}-reimburse.csv", "".join(f"{k},{v}\\n" for k, v in sorted(totals.items())))
    return {"violations": dict(violation_counts), "totals": {k: str(v) for k, v in totals.items()}}
'''

# The ARC environment's stored parser (docs/design/memory-v2-end-to-end-example.md §4), verbatim.
DEMO_STATE = '''
def demo_state(observation):
    """Extract the request count and demonstration pairs from demo feedback.

    Effect: read
    """
    if (not isinstance(observation, dict) or observation.get("type") != "DemosFeedback"
            or type(observation.get("demo_requests_used")) is not int
            or not 1 <= observation["demo_requests_used"] <= 8
            or observation.get("refused") is not False
            or not isinstance(observation.get("pairs"), list)
            or len(observation["pairs"]) != 1):
        raise MemoryInputError("expected non-refused DemosFeedback with one pair and request count 1..8")
    for pair in observation["pairs"]:
        if (not isinstance(pair, dict) or set(pair) != {"input", "output", "note"}
                or pair["note"] is not None):
            raise MemoryInputError("expected a demonstration pair with input, output and null note")
        for name in ("input", "output"):
            grid = pair[name]
            if (not isinstance(grid, list) or not grid or len(grid) > 30
                    or not all(isinstance(row, list) and row and len(row) <= 30 for row in grid)
                    or len({len(row) for row in grid}) != 1
                    or not all(type(v) is int and 0 <= v <= 9 for row in grid for v in row)):
                raise MemoryInputError("expected rectangular ARC grid with colors 0..9")
    return observation["demo_requests_used"], observation["pairs"], observation["refused"]
'''

# A small flagged body for an `apis` function (the gate and index tests build on it): the user id read from
# the reply is replaced by a constant when the reply says the account is closed.
RULE_BODY = (
    "    reply = apis.venmo.me()\n"
    '    user = reply["user_id"]\n'
    '    if reply.get("closed"):\n'
    '        user = "closed"\n'
    "    return user\n"
)
RULE_FN = (
    '\ndef me_or_closed(apis):\n    """The logged-in user\'s id, or a marker for a closed account.\n\n'
    '    Effect: read\n    Input: env\n    """\n' + RULE_BODY
)


def _fn(src: str) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(src)).body[0]


def _lines(src: str) -> list[int]:
    return [o.line for o in find_overrides(_fn(src)).overrides]


def test_the_f3_expense_function_is_flagged_at_the_zeroing_line():
    rep = find_overrides(_fn(F3))
    assert rep.flagged and not rep.truncated
    zero = next(
        i for i, ln in enumerate(F3.splitlines(), 1) if 'pay = Decimal("0.00")' in ln
    )
    assert rep.line == zero
    assert {o.name for o in rep.overrides} == {"pay"}
    computed = {o.computed for o in rep.overrides}
    lines = F3.splitlines()
    assert {lines[c - 1].strip() for c in computed} == {"pay = usd", "pay = limit"}


def test_the_corrected_f3_function_is_flagged_too_since_capping_is_a_rule():
    # The fix zeroes pay only for rules other than over_cap; it still replaces a computed value under a
    # condition, so it is still a rule that needs evidence from two episodes.
    fixed = F3.replace(
        "        if rule:\n",
        '        if rule and rule != "over_cap":\n',
    )
    assert fixed != F3 and find_overrides(_fn(fixed)).flagged


def test_flagged_shapes():
    # a constant replaces a computed value under a condition, also when the condition reads the value
    assert _lines(
        "def f(claim):\n    pay = claim['usd']\n    if pay > 100:\n        pay = 0\n    return pay\n",
    ) == [4]
    # a value from another parameter replaces one computed from the first
    assert _lines(
        "def f(claim, policy):\n    pay = claim['usd']\n    if claim['over']:\n        pay = policy['cap']\n    return pay\n",
    ) == [4]
    # into a returned container
    assert _lines(
        "def f(rows):\n    out = []\n    for r in rows:\n        v = r['x']\n        if r['void']:\n            v = 0\n"
        "        out.append(v)\n    return out\n",
    ) == [6]
    # into the environment (a call on a parameter), with no return
    assert _lines(
        "def f(apis, rows):\n    for r in rows:\n        v = r['x']\n        if r['void']:\n            v = 0\n"
        "        apis.files.write(r['path'], v)\n",
    ) == [5]
    # in a match case and an except handler
    assert _lines(
        "def f(a):\n    v = a['x']\n    match a['kind']:\n        case 'void':\n            v = 0\n    return v\n",
    ) == [5]
    assert _lines(
        "def f(a):\n    v = a['x']\n    try:\n        w = int(a['y'])\n    except ValueError:\n            v = 0\n"
        "    return v\n",
    ) == [6]


def test_not_flagged_shapes():
    cases = [
        # input-dependent rewrites
        "def f(a):\n    x = g(a)\n    x = h(x)\n    return x\n",
        "def f(a):\n    x = g(a)\n    if c:\n        x = h(x)\n    return x\n",
        "def f(a):\n    x = g(a)\n    if c:\n        x += 1\n    return x\n",
        # exclusive branches: one value per path
        "def f(a):\n    if c:\n        x = g(a)\n    else:\n        x = 0\n    return x\n",
        # a reset after the value was used
        "def f(lines):\n    out = []\n    cur = ''\n    for l in lines:\n        cur = cur + l\n        if l == '':\n"
        "            out.append(cur)\n            cur = ''\n    return out\n",
        # an unconditional dead store, and a loop variable reused as a fresh counter
        "def f(a):\n    x = g(a)\n    x = 0\n    return x\n",
        "def f(a):\n    for i in range(len(a)):\n        for j in a:\n            use(i, j)\n    i = 0\n"
        "    while i < len(a):\n        i += 1\n    return i\n",
        # a default argument filled in (parameters are not computed values)
        "def f(rows, limit=None):\n    if limit is None:\n        limit = 10\n    return rows[:limit]\n",
        # a fallback when the computation raised (it bound nothing)
        "def f(s):\n    try:\n        v = int(s)\n    except ValueError:\n        v = 0\n    return v\n",
        # the replaced value never reaches an output
        "def f(a):\n    x = g(a)\n    if c:\n        x = 0\n    log(x)\n    return 1\n",
        # an early return is not a re-binding
        "def f(a):\n    x = g(a)\n    if c:\n        return 0\n    return x\n",
        # a value carried to the next iteration is not compared across iterations
        "def f(lines):\n    out = []\n    cur = None\n    for l in lines:\n        if l[0] == '#':\n"
        "            cur = l[1:]\n        elif not l:\n            cur = None\n        else:\n"
        "            out.append((cur, l))\n    return out\n",
    ]
    for src in cases:
        rep = find_overrides(_fn(src))
        assert not rep.flagged and not rep.truncated, src


def test_the_arc_parser_is_not_flagged():
    rep = find_overrides(_fn(DEMO_STATE))
    assert not rep.flagged and not rep.truncated


def _fixture_functions() -> list[tuple[str, ast.FunctionDef]]:
    """Every function defined in a source string of this suite's other test files (their stored libraries)."""
    here = Path(__file__).parent
    out = []
    for path in sorted(here.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "def " in node.value
            ):
                try:
                    tree = ast.parse(textwrap.dedent(node.value))
                except SyntaxError:
                    continue
                out += [
                    (f"{path.name}:{node.lineno}:{fn.name}", fn)
                    for fn in ast.walk(tree)
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
    return out


def test_no_library_function_in_the_suites_fixtures_is_flagged():
    fns = _fixture_functions()
    assert len(fns) >= 40  # the gate, held-out, kinds and checkout fixtures
    flagged = [name for name, fn in fns if find_overrides(fn).flagged]
    assert flagged == []


def test_bounded_on_a_huge_function():
    huge = (
        "def f(a):\n"
        + "    x = g(a)\n    if c:\n        x = 0\n" * (MAX_NODES // 8)
        + "    return x\n"
    )
    rep = find_overrides(_fn(huge))
    assert rep.truncated and rep.limit == "nodes" and not rep.flagged and rep.line == 0
    # a large function under the cap is analysed (and flagged)
    big = (
        "def f(a):\n"
        + "    x = g(a)\n    if c:\n        x = 0\n" * 300
        + "    return x\n"
    )
    rep = find_overrides(_fn(big))
    assert not rep.truncated and rep.flagged
    # nesting as deep as the tokenizer allows is analysed
    deep = "def f(a):\n    x = g(a)\n" + "".join(
        "    " * (i + 1) + "if c:\n" for i in range(90)
    )
    deep += "    " * 91 + "x = 0\n    return x\n"
    rep = find_overrides(_fn(deep))
    assert rep.flagged and not rep.truncated


# --- a seeded property: pure input-dependent rewrites are never flagged -----------------------------------


def _program(rng: random.Random, stmts: int = 14) -> tuple[str, list[str]]:
    """A random function of `a` and `b` in which every binding reads a value derived from `a`."""
    bound = ["a"]
    lines = ["def f(a, b):", "    out = []"]

    def reads(k: int = 2) -> list[str]:
        return [rng.choice(bound) for _ in range(rng.randint(1, k))]

    def value() -> str:
        xs = reads()
        form = rng.randrange(6)
        if form == 0:
            return f"g({', '.join(xs)})"
        if form == 1:
            return " + ".join(xs) + " + 1"
        if form == 2:
            return f"{xs[0]}[0]"
        if form == 3:
            return f"{xs[0]}.strip()"
        if form == 4:
            return f"({xs[0]} or 0)"
        return f"h({xs[0]}, b, 3)"

    def block(indent: int, n: int, depth: int) -> None:
        pad = "    " * indent
        for _ in range(n):
            kind = rng.randrange(9 if depth < 3 else 4)
            name = f"v{rng.randrange(5)}"
            if (
                kind == 0 and name not in bound
            ):  # a constant initialisation of a fresh name
                lines.append(f"{pad}{name} = 0")
            elif kind in (0, 1):
                lines.append(f"{pad}{name} = {value()}")
                bound.append(name)
            elif kind == 2 and len(bound) > 1:
                lines.append(f"{pad}{rng.choice(bound[1:])} += {value()}")
            elif kind == 3:
                lines.append(f"{pad}out.append({value()})")
            elif kind == 4:
                lines.append(f"{pad}if {rng.choice(bound)} > {rng.randrange(9)}:")
                block(indent + 1, rng.randint(1, 3), depth + 1)
                if rng.random() < 0.5:
                    lines.append(f"{pad}else:")
                    block(indent + 1, rng.randint(1, 3), depth + 1)
            elif kind == 5:
                lines.append(f"{pad}for {name} in {rng.choice(bound)}:")
                bound.append(name)
                block(indent + 1, rng.randint(1, 3), depth + 1)
            elif kind == 6:
                lines.append(f"{pad}while {rng.choice(bound)}:")
                block(indent + 1, rng.randint(1, 3), depth + 1)
                lines.append(f"{pad}    break")
            elif kind == 7:
                lines.append(f"{pad}try:")
                block(indent + 1, rng.randint(1, 2), depth + 1)
                lines.append(f"{pad}except ValueError:")
                block(indent + 1, rng.randint(1, 2), depth + 1)
            else:
                lines.append(f"{pad}{name} = {value()} if c else {value()}")
                bound.append(name)

    block(1, stmts, 0)
    returned = sorted(set(bound))
    return "\n".join(lines), returned


def test_property_pure_input_dependent_rewrites_are_never_flagged():
    for seed in range(300):
        rng = random.Random(seed)
        body, returned = _program(rng)
        src = body + f"\n    return out, {', '.join(returned)}\n"
        rep = find_overrides(_fn(src))
        assert not rep.flagged and not rep.truncated, (seed, src)
        # control: the same program with one conditional constant replacement is flagged at that line
        v = rng.choice(returned)
        n = len(body.splitlines())
        planted = (
            body
            + f"\n    {v} = g(a)\n    if c:\n        {v} = 0\n    return out, {', '.join(returned)}\n"
        )
        assert [o.line for o in find_overrides(_fn(planted)).overrides] == [n + 3], (
            seed,
            planted,
        )
