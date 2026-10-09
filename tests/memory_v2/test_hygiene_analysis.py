"""Memory hygiene, pure analysis (stage 4, build step 1): function-body anti-unification and the shared-subtree
finder.

The ARC and office libraries under ``hygiene_fixtures/`` are verbatim copies of the libraries two paid offline
replays grew (full-base-arc-225-150000 and full-base-office-v2-full-150000, 8 Oct 2026). They are kept as
``.pysrc`` so formatters never rewrite them; these tests read them with ``ast`` only (the differential runner
tests run the ARC one, confined).
"""

from __future__ import annotations

import ast
import random
import re
import time
import sys
from pathlib import Path

import pytest

from unify.memory_v2.analysis.fn_antiunify import (
    MAX_DEPTH,
    MAX_NODES,
    antiunify,
    antiunify_source,
    function_defs,
    inline_library_calls,
    normalise,
)
from unify.memory_v2.analysis.fn_dataflow import flows
from unify.memory_v2.analysis.shared_subtrees import (
    shared_subtrees,
    shared_subtrees_source,
)

FIXTURES = Path(__file__).with_name("hygiene_fixtures")
ARC = (FIXTURES / "arc_env.pysrc").read_text()
OFFICE = (FIXTURES / "office_worktree.pysrc").read_text()


# --- normalisation -----------------------------------------------------------------------------------------


def test_normalisation_strips_docstrings_and_renames_locals_in_first_use_order():
    a = function_defs(
        'def f(obs, prev=None):\n    """Doc."""\n    total = obs + 1\n    return total\n',
    )["f"]
    b = function_defs("def g(x, y=None):\n    z = x + 1\n    return z\n")["g"]
    na, nb = normalise(a), normalise(b)
    assert ast.dump(na.node) == ast.dump(nb.node)
    assert na.size == nb.size
    assert "Doc." not in ast.unparse(na.node)
    # the canonical names map back to each input's own names
    assert set(na.renames.values()) == {"obs", "prev", "total"}
    assert set(nb.renames.values()) == {"x", "y", "z"}


def test_free_names_globals_and_attributes_are_kept():
    fn = function_defs(
        "def f(x):\n    return isinstance(x, list) and helper(x).size\n",
    )["f"]
    text = ast.unparse(normalise(fn).node)
    assert "isinstance" in text and "helper" in text and ".size" in text
    assert "x" not in text.replace("isinstance", "")


def test_unparsable_source_gives_no_functions():
    assert function_defs("def f(:\n") == {}


# --- function-body anti-unification on the real ARC library ------------------------------------------------


def test_identical_bodies_up_to_renaming_keep_everything():
    g = antiunify_source(
        "def a(p):\n    q = p * 2\n    return q\n\ndef b(r):\n    s = r * 2\n    return s\n",
        ["a", "b"],
    )
    assert g.kept_share == 1.0 and g.holes == ()


def test_parse_submit_feedback_and_submit_state_generalise_with_holes_where_they_differ():
    # submit_state calls parse_submit_feedback first; inlining that sibling call shows the shared validation
    g = antiunify_source(ARC, ["parse_submit_feedback", "submit_state"])
    assert g is not None and not g.bounded
    assert g.kept_share >= 0.5
    assert 1 <= len(g.holes) <= 4
    # the five-field shape check survives whole in the generalisation
    assert "'attempts_used'" in g.source and "MemoryInputError" in g.source
    # each hole binds a subtree of each input (empty where one input has nothing there)
    for h in g.holes:
        assert len(h.bindings) == 2 and len(h.sizes) == 2
        assert f"__hole{h.index}__" in g.source
    # the extra consistency checks of submit_state are a hole bound to nothing in parse_submit_feedback
    assert any(h.sizes[0] == 0 and h.sizes[1] > 0 for h in g.holes)


def test_without_inlining_the_same_pair_keeps_less():
    inl = antiunify_source(ARC, ["parse_submit_feedback", "submit_state"])
    raw = antiunify_source(
        ARC,
        ["parse_submit_feedback", "submit_state"],
        inline=False,
    )
    assert raw.kept_share < inl.kept_share


def test_two_readers_of_the_demo_message_differ_in_one_hole():
    # both re-parse the demo message (through parse_demos_feedback) and differ only in what they return per grid
    g = antiunify_source(ARC, ["demo_dimensions", "demo_palette"])
    assert g.kept_share > 0.9 and len(g.holes) == 1
    (h,) = g.holes
    assert "len(" in h.bindings[0] and "sorted(" in h.bindings[1]


def test_a_consistent_renaming_is_kept_not_holed():
    # demo_request_count has an extra parameter, so its inlined locals are numbered differently; the renaming is
    # paired consistently instead of becoming holes
    g = antiunify_source(ARC, ["demo_request_count", "demo_availability"])
    assert g.kept_share > 0.7 and len(g.holes) <= 5
    assert "*__hole0__" in g.source  # the differing parameter lists


def test_unrelated_functions_across_libraries_score_low():
    defs = {**function_defs(ARC), **function_defs(OFFICE)}
    g = antiunify([defs["demo_change_mask"], defs["read_inventory"]])
    assert g.kept_share < 0.15


def test_three_way_generalisation_of_the_office_readers():
    defs = function_defs(OFFICE)
    g = antiunify([defs["read_ledger"], defs["read_retention"], defs["read_inventory"]])
    assert g is not None and len(g.sizes) == 3
    assert all(len(h.bindings) == 3 for h in g.holes)
    # the shared skeleton is kept: the path-type guard, the UTF-8 read and the re-raise as MemoryInputError
    assert "isinstance(_l0, (str, Path))" in g.source
    assert "read_text(encoding='utf-8')" in g.source and "from _m" in g.source
    assert 0.1 < g.kept_share < 0.5


def test_anti_unification_is_deterministic():
    runs = [
        antiunify_source(ARC, ["parse_submit_feedback", "submit_state"])
        for _ in range(3)
    ]
    assert len({r.source for r in runs}) == 1
    assert len({tuple((h.index, h.bindings) for h in r.holes) for r in runs}) == 1
    assert len({r.kept_share for r in runs}) == 1


def test_the_same_disagreement_reuses_one_hole():
    g = antiunify_source(
        "def a(x):\n    return f(x, 1) + f(x, 1)\n\ndef b(x):\n    return f(x, 2) + f(x, 2)\n",
        ["a", "b"],
    )
    assert len(g.holes) == 1 and g.source.count("__hole0__") == 2


def test_a_missing_function_name_gives_none():
    assert antiunify_source(ARC, ["parse_submit_feedback", "no_such_function"]) is None


def test_a_generalisation_is_valid_python_with_holes_as_names():
    g = antiunify_source(ARC, ["demo_dimensions", "demo_palette"])
    ast.parse(g.source)


# --- bounds -----------------------------------------------------------------------------------------------


def _wide(n: int, every: int, delta: int) -> str:
    lines = ["def f(x):"]
    for i in range(n):
        k = i + (delta if every and i % every == 0 else 0)
        lines.append(f"    y{i} = x + {k}")
    lines.append("    return x")
    return "\n".join(lines) + "\n"


def test_a_ten_thousand_node_function_does_not_blow_up():
    a = function_defs(_wide(2000, 0, 0))["f"]
    b = function_defs(_wide(2000, 10, 1))["f"]
    assert normalise(a).size >= 10_000
    t0 = time.monotonic()
    g = antiunify([a, b])
    assert time.monotonic() - t0 < 60
    assert g is not None and 0.5 < g.kept_share < 1.0
    rep = shared_subtrees({"a": a, "b": b}, min_size=4, min_functions=2)
    assert rep is not None


def test_a_tight_work_budget_marks_the_result_bounded():
    a = function_defs(_wide(300, 0, 0))["f"]
    b = function_defs(_wide(300, 3, 1))["f"]
    g = antiunify([a, b], budget=50)
    assert g is not None and g.bounded and g.kept_share < 1.0


def test_a_too_deep_function_is_refused_without_recursion_error():
    src = "def f(x):\n    return " + "-" * (MAX_DEPTH + 50) + "x\n"
    try:
        defs = function_defs(src)
    except RecursionError:  # pragma: no cover - function_defs must catch it
        pytest.fail("function_defs raised RecursionError")
    if defs:
        assert normalise(defs["f"]) is None
        assert antiunify([defs["f"], defs["f"]]) is None
        rep = shared_subtrees(defs, min_functions=1)
        assert rep.skipped == ("f",)


# --- shared subtrees --------------------------------------------------------------------------------------


def test_the_repeated_grid_validation_is_found_across_the_arc_library():
    rep = shared_subtrees_source(ARC, min_size=20, min_functions=2)
    grid = [
        c
        for c in rep.candidates
        if set(c.functions) == {"parse_demos_feedback", "validate_submit_action"}
        and c.node_type == "BoolOp"
    ]
    assert grid, [(c.functions, c.node_type, c.size) for c in rep.candidates]
    c = grid[0]
    # one occurrence per function, located by line, up to renaming (the grids are bound differently)
    assert len(c.occurrences) == 2
    assert all(o.lineno is not None for o in c.occurrences)
    assert c.size >= 40 and c.compression > 0
    assert "isinstance" in c.example and "len" in c.example
    # maximal: its own sub-expressions are not listed beside it
    inside = [
        d
        for d in rep.candidates
        if d is not c
        and set(d.functions) == set(c.functions)
        and all(
            any(
                o.function == p.function and p.lineno <= o.lineno <= p.end_lineno
                for p in c.occurrences
            )
            for o in d.occurrences
        )
    ]
    assert not inside


def test_the_office_path_guard_and_csv_read_are_shared_by_all_three_readers():
    rep = shared_subtrees_source(OFFICE, min_size=6, min_functions=3)
    examples = [c.example for c in rep.candidates]
    assert any("DictReader" in e and "StringIO" in e for e in examples), examples
    assert any("isinstance" in e and "Path" in e for e in examples), examples
    for c in rep.candidates:
        assert len(set(c.functions)) >= 3


def test_shared_subtrees_respect_alpha_renaming_but_not_different_structure():
    src = (
        "def a(p):\n    return [q * 2 + 1 for q in p if q > 0]\n\n"
        "def b(r):\n    return [s * 2 + 1 for s in r if s > 0]\n\n"
        "def c(t):\n    return [u * 3 + 1 for u in t if u > 0]\n"
    )
    rep = shared_subtrees_source(src, min_size=8, min_functions=2)
    assert any(set(c.functions) == {"a", "b"} for c in rep.candidates)
    assert not any("c" in c.functions and c.size >= 14 for c in rep.candidates)


def test_shared_subtrees_are_deterministic():
    runs = [shared_subtrees_source(ARC, min_size=8, min_functions=2) for _ in range(3)]
    keys = {
        tuple((c.key, c.functions, c.compression) for c in r.candidates) for r in runs
    }
    assert len(keys) == 1


# --- scope-aware renaming (review M1) -------------------------------------------------------------------------


def _au(src: str, names=("f", "g"), **kw):
    return antiunify_source(src, list(names), **kw)


@pytest.mark.parametrize(
    "src",
    [
        # a lambda parameter named like a builtin: f calls the builtin len, g calls the global z
        "def f(x):\n    h = lambda len: len\n    return len(x)\n\n"
        "def g(x):\n    h = lambda z: z\n    return z(x)\n",
        # a comprehension variable named like a builtin
        "def f(xs):\n    ys = [len for len in xs]\n    return len(ys)\n\n"
        "def g(xs):\n    ys = [n for n in xs]\n    return n(ys)\n",
        # a nested def's parameter named like a builtin
        "def f(x):\n    def h(len):\n        return len\n    return len(x)\n\n"
        "def g(x):\n    def h(z):\n        return z\n    return z(x)\n",
        # a global declaration is not a local
        "def f(x):\n    global y\n    y = x\n    return y\n\n"
        "def g(x):\n    z = x\n    return z\n",
        # class-body names are attributes, never renamed
        "def f(x):\n    class C:\n        a = 1\n    return C.a + x\n\n"
        "def g(x):\n    class C:\n        b = 1\n    return C.b + x\n",
    ],
)
def test_hostile_shadowing_pairs_never_reach_full_kept_share(src):
    g = _au(src)
    assert g is not None and g.kept_share < 1.0 and g.holes


@pytest.mark.parametrize(
    "src",
    [
        "def f(p):\n    q = [a * 2 for a in p]\n    h = lambda b: b + 1\n    return h(q)\n\n"
        "def g(r):\n    s = [c * 2 for c in r]\n    k = lambda d: d + 1\n    return k(s)\n",
        "def f(p):\n    t = 0\n    def inc():\n        nonlocal t\n        t += p\n    inc()\n    return t\n\n"
        "def g(r):\n    u = 0\n    def bump():\n        nonlocal u\n        u += r\n    bump()\n    return u\n",
        # the first iterable is read in the enclosing scope, so [x for x in x] is [z for z in y]
        "def f(x):\n    return [x for x in x]\n\ndef g(y):\n    return [z for z in y]\n",
        # a walrus in a comprehension binds in the function
        "def f(xs):\n    [(t := v) for v in xs]\n    return t\n\n"
        "def g(ys):\n    [(s := w) for w in ys]\n    return s\n",
    ],
)
def test_scope_respecting_renamings_keep_everything(src):
    g = _au(src)
    assert g.kept_share == 1.0 and g.holes == ()


def test_free_names_are_never_renamed_and_imports_keep_their_binding():
    fn = function_defs(
        "def f(x):\n    import os.path\n    import json\n    from m import k\n"
        "    return os.path.join(json.dumps(k), len(x))\n",
    )["f"]
    n = normalise(fn)
    text = ast.unparse(n.node)
    assert (
        "import os.path" in text and "os.path.join" in text
    )  # a dotted import's binding is kept
    assert re.search(r"import json as _l\d", text) and re.search(
        r"import k as _l\d",
        text,
    )
    assert "len(" in text and {"os", "len"} <= n.fixed


def test_keywords_at_a_call_of_a_nested_function_follow_its_renamed_parameters():
    fn = function_defs(
        "def f(x):\n    def inner(a, b=2):\n        return a - b\n    return inner(b=x, a=1)\n",
    )["f"]
    node = normalise(fn).node
    inner = node.body[0]
    call = node.body[1].value
    params = [p.arg for p in inner.args.args]
    assert sorted(k.arg for k in call.keywords) == sorted(params)
    compile(ast.fix_missing_locations(ast.Module([node], [])), "<norm>", "exec")


def test_annotations_of_assignments_are_dropped():
    a = function_defs("def f(x):\n    y: int = x\n    return y\n")["f"]
    b = function_defs("def f(x):\n    y: list[str] = x\n    return y\n")["f"]
    assert ast.dump(normalise(a).node) == ast.dump(normalise(b).node)


# --- random renamings: the property behind M1 -------------------------------------------------------------

_FREE_CALLS = ("len", "sorted", "abs")
_PLACEHOLDER = re.compile(r"\b[PVCAT]\d+\b")


def _random_function(rng: random.Random):
    """A random function template. Placeholders: P (parameters), V and T (function-scope locals), C and A
    (comprehension, lambda and nested-def binders, each read beside one outer variable). Returns the template,
    the function-scope placeholders and the (inner binder, outer variable it reads beside) pairs.
    """
    n = iter(range(10**6))
    params = [f"P{next(n)}" for _ in range(rng.randint(1, 3))]
    visible = list(params)
    inner: list[tuple[str, str]] = []
    lines: list[str] = []

    def expr(names, depth=0):
        r = rng.random()
        if depth > 2 or r < 0.35:
            return rng.choice(names) if rng.random() < 0.75 else str(rng.randint(0, 9))
        if r < 0.55:
            return f"{rng.choice(_FREE_CALLS)}({expr(names, depth + 1)})"
        return (
            f"({expr(names, depth + 1)} {rng.choice('+-*')} {expr(names, depth + 1)})"
        )

    for _ in range(rng.randint(2, 7)):
        kind = rng.choice(("assign", "comp", "lambda", "def", "for", "if"))
        new = f"V{next(n)}"
        if kind == "assign":
            lines.append(f"{new} = {expr(visible)}")
        elif kind == "comp":
            c, outer = f"C{next(n)}", rng.choice(visible)
            lines.append(
                f"{new} = [{c} * 2 + {outer} for {c} in {rng.choice(visible)}]",
            )
            inner.append((c, outer))
        elif kind == "lambda":
            # only ever called directly, positionally: its parameter name is not observable (review N4)
            a, outer = f"A{next(n)}", rng.choice(visible)
            lines.append(f"{new} = lambda {a}: {a} + {outer}")
            inner.append((a, outer))
            new, call = f"V{next(n)}", new
            lines.append(f"{new} = {call}({expr(visible)})")
        elif kind == "def":
            a, outer = f"A{next(n)}", rng.choice(visible)
            lines += [f"def {new}({a}):", f"    return {a} - {outer}"]
            inner.append((a, outer))
            new, call = f"V{next(n)}", new
            lines.append(f"{new} = {call}({expr(visible)})")
        elif kind == "for":
            t = f"T{next(n)}"
            lines += [
                f"for {t} in {rng.choice(visible)}:",
                f"    {new} = {expr(visible + [t])}",
            ]
            visible.append(t)
        else:
            lines += [f"if {expr(visible)}:", f"    raise ValueError({expr(visible)})"]
            continue
        visible.append(new)
    lines.append(
        f"return helper({', '.join(rng.sample(visible, min(2, len(visible))))})",
    )
    body = "".join(f"    {line}\n" for line in lines)
    return f"def fn({', '.join(params)}):\n{body}", visible, inner


def _named(template: str, mapping: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda m: mapping.get(m.group(0), m.group(0)), template)


def _fresh(rng: random.Random, template: str) -> dict[str, str]:
    names = sorted(set(_PLACEHOLDER.findall(template)))
    return {
        p: f"n{k}"
        for p, k in zip(names, rng.sample(range(10**6), len(names)), strict=True)
    }


def _kept(a: str, b: str) -> float:
    return antiunify([function_defs(a)["fn"], function_defs(b)["fn"]]).kept_share


@pytest.mark.parametrize("seed", range(40))
def test_random_renamings_keep_everything_and_captures_do_not(seed):
    rng = random.Random(seed)
    template, visible, inner = _random_function(rng)
    base = _named(template, _fresh(rng, template))
    # any injective renaming to fresh names
    assert _kept(base, _named(template, _fresh(rng, template))) == 1.0
    # every inner binder (lambda, comprehension, nested def) given the same name: still the same function
    reuse = _fresh(rng, template)
    for binder, _ in inner:
        reuse[binder] = "q"
    assert _kept(base, _named(template, reuse)) == 1.0
    # capture: a function-scope local renamed to the free name `helper` the return statement calls
    capture = _fresh(rng, template)
    capture[rng.choice(visible)] = "helper"
    assert _kept(base, _named(template, capture)) < 1.0
    # capture: an inner binder renamed to the outer variable read beside it
    if inner:
        binder, outer = rng.choice(inner)
        shadow = _fresh(rng, template)
        shadow[binder] = shadow[outer]
        assert _kept(base, _named(template, shadow)) < 1.0


@pytest.mark.parametrize("seed", range(40))
def test_random_escaping_functions_keep_their_parameter_names(seed):
    # review N4: once a nested function or lambda escapes (passed on, aliased, called with keywords or **), a
    # caller can pass its parameters by keyword, so renaming one must never look like the same function
    rng = random.Random(1000 + seed)
    kind = rng.choice(("def", "lambda"))
    use = rng.choice(
        (
            "return helper({h})",
            "k = {h}\n    return k(z=x)",
            "return [{h}]",
            "return {h}(**{{'z': x}})",
            "return {h}(z=x)",
        ),
    )
    if kind == "lambda" and use == "return {h}(z=x)":
        use = "return (lambda {p}: {p} + x)(z=x)"

    def src(param: str) -> str:
        if kind == "def":
            head = f"    def h({param}):\n        return {param} + x\n"
        else:
            head = f"    h = lambda {param}: {param} + x\n"
        return f"def fn(x):\n{head}    {use.format(h='h', p=param)}\n"

    same = antiunify([function_defs(src("z"))["fn"], function_defs(src("z"))["fn"]])
    renamed = antiunify(
        [function_defs(src("z"))["fn"], function_defs(src(f"w{seed}"))["fn"]],
    )
    assert same.kept_share == 1.0
    assert renamed.kept_share < 1.0 and renamed.holes


@pytest.mark.parametrize(
    "src",
    [
        # an alias of the nested function called by keyword: g raises TypeError, f does not
        "def f(o):\n    def inner(a):\n        return a\n    h = inner\n    return h(a=o)\n\n"
        "def g(o):\n    def inner(b):\n        return b\n    h = inner\n    return h(a=o)\n",
        # a lambda called by keyword
        "def f(o):\n    return (lambda a: a)(a=o)\n\ndef g(o):\n    return (lambda b: b)(a=o)\n",
        # keywords from a mapping
        "def f(o):\n    def inner(a):\n        return a\n    return inner(**{'a': o})\n\n"
        "def g(o):\n    def inner(b):\n        return b\n    return inner(**{'a': o})\n",
        # the frame's names are the value
        "def f(o):\n    a = o\n    return locals()\n\ndef g(o):\n    b = o\n    return locals()\n",
        # a nested class's name is its __name__
        "def f(o):\n    class A:\n        pass\n    return A.__name__\n\n"
        "def g(o):\n    class B:\n        pass\n    return B.__name__\n",
        # a nested function's name too, once it escapes
        "def f(o):\n    def a():\n        return o\n    return a.__name__\n\n"
        "def g(o):\n    def b():\n        return o\n    return b.__name__\n",
    ]
    + (
        [
            "def f(o):\n    type T = int\n    return T\n\ndef g(o):\n    type U = int\n    return U\n",
        ]
        if sys.version_info >= (3, 12)
        else []
    ),
)
def test_name_sensitive_pairs_never_reach_full_kept_share(src):
    # review N4: each pair normalised identically before, although the two functions behave differently
    g = _au(src)
    assert g is not None and g.kept_share < 1.0 and g.holes


def test_a_renamed_root_parameter_is_an_interface_difference():
    g = _au(
        "def f(obs, limit=3):\n    return obs[limit]\n\ndef g(o, n=3):\n    return o[n]\n",
    )
    assert g.kept_share == 1.0 and not g.same_signature
    assert g.signatures == (("obs", "limit"), ("o", "n"))
    g = _au(
        "def f(a, /, b, *c, d, **e):\n    return a\n\ndef g(a, /, b, *c, d, **e):\n    return a\n",
    )
    assert g.same_signature and g.signatures[0] == ("a", "/", "b", "*c", "d", "**e")


# --- inlining (review M2, M3, M7) -------------------------------------------------------------------------


def test_inlining_keeps_the_final_return_expression_and_its_calls():
    src = (
        "def parse(o):\n    check(o)\n    return decode(o)\n\n"
        "def a(o):\n    parse(o)\n    return 1\n\n"
        "def b(o):\n    check(o)\n    return 1\n"
    )
    g = antiunify_source(src, ["a", "b"])
    assert g.kept_share < 1.0
    assert any("decode(" in h.bindings[0] and h.sizes[1] == 0 for h in g.holes)
    # a bare name or constant returned is dropped (its evaluation does nothing)
    defs = function_defs(
        "def p(o):\n    check(o)\n    return o\n\ndef a(o):\n    p(o)\n    return 1\n",
    )
    body = inline_library_calls(defs["a"], defs).body
    assert [ast.unparse(s) for s in body] == ["check(o)", "return 1"]


def test_inlining_is_refused_where_a_name_would_change_meaning():
    def inlined(src: str, name: str) -> str:
        defs = function_defs(src)
        return ast.unparse(inline_library_calls(defs[name], defs))

    # the sibling reads the global `check`, which the caller binds
    src = "def helper(o):\n    check(o)\n\ndef c(o):\n    check = 1\n    helper(o)\n    return check\n"
    assert "helper(o)" in inlined(src, "c")
    # the sibling's lambda parameter would capture the caller's argument name
    src = "def helper(o):\n    f = lambda x: o + x\n    f(1)\n\ndef c(x):\n    helper(x)\n    return x\n"
    assert "helper(x)" in inlined(src, "c")
    # the sibling rebinds its parameter (in a loop)
    src = "def helper(o):\n    for o in range(3):\n        pass\n\ndef c(x):\n    helper(x)\n    return x\n"
    assert "helper(x)" in inlined(src, "c")
    # a caller local of the sibling's name is not the sibling
    src = "def helper(o):\n    check(o)\n\ndef c(x, helper):\n    helper(x)\n    return x\n"
    assert "helper(x)" in inlined(src, "c")


@pytest.mark.parametrize(
    "callee",
    [
        # review N5: a mutable default is shared across calls; inlined it would be a fresh list per call
        "def p(o, acc=[]):\n    acc.append(o)\n",
        "def p(o, acc={}):\n    acc[o] = 1\n",
        "def p(o, acc=make()):\n    acc.add(o)\n",
        # review N5: frame introspection would see the caller's frame
        "def p(o):\n    log(locals())\n",
        "def p(o):\n    log(vars())\n",
        "def p(o):\n    eval('o')\n",
        "def p(o):\n    exec('x = o')\n",
        "def p(o):\n    log(dir())\n",
        "def p(o):\n    super().check(o)\n",
    ],
)
def test_inlining_is_refused_where_it_would_change_behaviour(callee):
    defs = function_defs(callee + "\ndef c(o):\n    p(o)\n    return o\n")
    assert "p(o)" in ast.unparse(inline_library_calls(defs["c"], defs))


def test_constant_defaults_still_inline():
    defs = function_defs(
        "def p(o, k=3, t=(1, -2), n=None, f=FLAG):\n    check(o, k, t, n, f)\n\n"
        "def c(o):\n    p(o)\n    return o\n",
    )
    text = ast.unparse(inline_library_calls(defs["c"], defs))
    assert "p(o)" not in text and "check(o," in text


def test_inlining_stops_at_the_node_cap_fast_and_says_so():
    big = "def big(x):\n" + "".join(f"    y{i} = x + {i}\n" for i in range(9000))
    caller = "def caller(x):\n" + "    big(x)\n" * 1000 + "    return x\n"
    defs = function_defs(big + "\n" + caller)
    t0 = time.monotonic()
    n = normalise(defs["caller"], library=defs)
    assert time.monotonic() - t0 < 60
    assert n is not None and n.inline_capped and n.size <= MAX_NODES
    assert "big" in n.inlined


def test_mutual_recursion_is_never_re_entered():
    defs = function_defs(
        "def a(x):\n    b(x)\n    return 1\n\ndef b(x):\n    a(x)\n    return 2\n",
    )
    for depth in (1, 2, 4):
        n = normalise(defs["a"], library=defs, inline_depth=depth)
        assert n is not None
        assert (
            ast.unparse(n.node).count("a(") == 1
        )  # b's body inlined once, its call to a left as a call


def test_two_level_inlining_reaches_the_helpers_helper():
    defs = function_defs(
        "def inner(o):\n    check(o)\n\ndef outer(o):\n    inner(o)\n\n"
        "def a(o):\n    outer(o)\n    return 1\n\ndef b(o):\n    check(o)\n    return 1\n",
    )
    one = antiunify([defs["a"], defs["b"]], library=defs)
    two = antiunify([defs["a"], defs["b"]], library=defs, inline_depth=2)
    assert one.kept_share < 1.0 and two.kept_share == 1.0


def test_a_pair_that_only_shares_an_inlined_helper_is_marked_helper_driven():
    g = antiunify_source(ARC, ["demo_dimensions", "demo_palette"])
    assert g.shared_helpers == ("parse_demos_feedback",)
    assert g.helper_driven and g.helper_kept > g.kept / 2
    assert g.kept_share_uninlined is not None and g.kept_share_uninlined < g.kept_share
    assert g.owned_share < g.kept_share
    assert g.helper_kept + g.mixed_kept <= g.kept
    # submit_state calls parse_submit_feedback: what they keep is the callee's own code in one input and inlined
    # code in the other (mixed), not a shared helper
    h = antiunify_source(ARC, ["parse_submit_feedback", "submit_state"])
    assert not h.helper_driven and h.mixed_kept > 0 and h.shared_helpers == ()
    # without inlining nothing is split
    raw = antiunify_source(ARC, ["demo_dimensions", "demo_palette"], inline=False)
    assert raw.helper_kept == raw.mixed_kept == 0 and raw.kept_share_uninlined is None


def test_a_pair_where_one_calls_the_other_is_a_wrapper():
    # review N8: submit_state calls parse_submit_feedback; most of what they keep is that callee's code
    h = antiunify_source(ARC, ["parse_submit_feedback", "submit_state"])
    assert h.wrapper and h.calls == ((1, 0),) and not h.helper_driven
    assert 2 * h.mixed_kept >= h.kept
    for pair in (
        ["demo_dimensions", "demo_palette"],
        ["demo_change_mask", "submit_state"],
    ):
        assert not antiunify_source(ARC, pair).wrapper
    # a local of the same name is not the other input
    g = _au("def f(x):\n    return x\n\ndef g(x, f):\n    return f(x)\n")
    assert not g.wrapper


# --- literals and names (review M4, L3) -----------------------------------------------------------------------


def test_a_huge_integer_literal_is_keyed_without_repr():
    src = (
        "def f():\n    return 0x"
        + "f" * 4000
        + "\n\ndef g():\n    return 0x"
        + "e" * 4000
        + "\n"
    )
    g = _au(src)
    assert g is not None and len(g.holes) == 1 and g.kept_share < 1.0
    same = _au(src.replace("e" * 4000, "f" * 4000))
    assert same.kept_share == 1.0
    rep = shared_subtrees_source(src, min_size=1, min_functions=1)
    assert rep.skipped == ()


def test_long_strings_compare_by_digest_and_floats_by_bits():
    long_a, long_b = "x" * 500, "x" * 499 + "y"
    g = _au(f"def f():\n    return {long_a!r}\n\ndef g():\n    return {long_b!r}\n")
    assert len(g.holes) == 1
    g = _au("def f():\n    return 0.0\n\ndef g():\n    return -0.0\n")
    assert g.kept_share < 1.0


def test_paired_names_are_exposed_and_hole_names_avoid_real_names():
    # g's extra parameter shifts its canonical numbering, so a and b are paired under one new name
    g = _au(
        "def f(x):\n    a = x + 1\n    return a * 2\n\ndef g(y, z):\n    b = y + 1\n    return b * 2\n",
    )
    assert list(g.pairs.values()) == [("_l1", "_l2")]
    assert all(name in g.source for name in g.pairs)
    g = _au(
        "def f(x):\n    return __hole0__ + x\n\ndef g(x):\n    return __hole0__ - x\n",
    )
    (h,) = g.holes
    assert (
        h.name != "__hole0__" and h.name in g.source and "__hole0__ +" not in g.source
    )
    g = _au(
        "def f(x):\n    a = _m0(x)\n    return a\n\ndef g(y, z):\n    b = _m0(y)\n    return b\n",
    )
    assert g.pairs and "_m0" not in g.pairs and "_m0(" in g.source


# --- data flow per hole (design item 2) -----------------------------------------------------------------------


def test_the_flow_of_a_function_from_parameters_to_return_and_raise_sites():
    fn = normalise(
        function_defs(
            "def f(obs, limit, unused):\n"
            "    n = obs['n']\n"
            "    if n > limit:\n        raise ValueError('big')\n"
            "    out = []\n"
            "    for x in obs['xs']:\n        out.append(x * 2)\n"
            "    assert out\n"
            "    return out\n",
        )["f"],
    ).node
    s = flows(fn)
    assert (s.returns, s.raises) == (1, 2)
    # obs reaches the return and both raises; limit only the raise it guards; unused nothing
    assert s.param_sites == ((1, 2), (0, 1), (0, 0))
    guard, value = fn.body[1], fn.body[3].body[0].value.args[0]
    r = flows(fn, {0: [guard], 1: [value]}).regions
    assert (r[0].params, r[0].kind) == ((0, 1), "guard")
    assert (r[1].params, r[1].kind) == ((0,), "value")


def test_holes_are_typed_by_their_flow_on_the_real_arc_pair():
    g = antiunify_source(ARC, ["parse_submit_feedback", "submit_state"])
    kinds = {h.index: h.kind for h in g.holes}
    # submit_state's extra consistency checks guard a raise; the differing return value is a value
    extra = next(h for h in g.holes if h.sizes[0] == 0 and h.sizes[1] > 0)
    assert extra.flows[0] is None and extra.flows[1].raises >= 1
    assert "value" in kinds.values()
    for h in g.holes:
        for f, size in zip(h.flows, h.sizes, strict=True):
            assert (f is None) == (size == 0)
            if f is not None:
                assert f.params == (
                    0,
                )  # everything here depends on the one observation
    assert all(s is not None and s.params == ("_l0",) for s in g.flows)


def _region_flow(src: str, start: str):
    """The flow of the first statement of *src*'s function whose source starts with *start*."""
    fn = ast.parse(src).body[0]
    stmt = next(
        s
        for s in ast.walk(fn)
        if isinstance(s, ast.stmt) and s is not fn and ast.unparse(s).startswith(start)
    )
    s = flows(fn, {0: [stmt]})
    return s, s.regions[0]


@pytest.mark.parametrize(
    "src, start",
    [
        # review N2, each reported local with the parameter reaching no return before
        (
            "def f(p):\n    acc = []\n    view = acc\n    view.append(p)\n    return acc\n",
            "view.append",
        ),  # mutation through an alias
        (
            "def f(p):\n    acc = []\n    helper(acc, p)\n    return acc\n",
            "helper(",
        ),  # mutation by a call
        (
            "def f(p):\n    try:\n        check(p)\n    except ValueError:\n        return 0\n"
            "    return 1\n",
            "check(",
        ),  # an exception handler's context
        (
            "def f(p):\n    if g(p):\n        return 0\n    y = 5\n    return y\n",
            "y = 5",
        ),  # statements after an early return
        (
            "def f(p, xs):\n    t = 0\n    for x in xs:\n        t = t + x\n        if x == p:\n"
            "            break\n    return t\n",
            "t = t + x",
        ),  # a break decides how many times the loop body ran
    ],
)
def test_value_determining_code_is_never_called_local(src, start):
    s, r = _region_flow(src, start)
    assert r.kind in ("value", "nonlocal") and r.value_determining
    assert s.param_sites[0][0] >= 1  # the parameter decides what is returned


@pytest.mark.parametrize(
    "src, start",
    [
        ("def f(p):\n    global G\n    G = p\n    return 1\n", "G = p"),
        ("def f(p):\n    p.x = 1\n    return 1\n", "p.x"),
        ("def f(p):\n    del p[0]\n    return 1\n", "del"),
        ("def f(p):\n    notify(p)\n    return 1\n", "notify"),
        ("def f(p):\n    import os\n    return 1\n", "import"),
        ("def f(p):\n    with lock:\n        pass\n    return 1\n", "with"),
        ("def f(p):\n    x = sorted(p, key=k)\n    return 1\n", "x = sorted"),
    ],
)
def test_effects_the_graph_does_not_follow_are_nonlocal_never_local(src, start):
    _, r = _region_flow(src, start)
    assert r.untracked and r.kind == "nonlocal" and r.value_determining


def test_pure_local_code_and_guards_are_still_told_apart():
    _, r = _region_flow("def f(p):\n    t = len(p) + 1\n    return p\n", "t =")
    assert r.kind == "local" and not r.untracked
    s, r = _region_flow(
        "def f(p):\n    if p < 0:\n        raise ValueError('neg')\n    return p\n",
        "if p",
    )
    assert r.kind == "guard" and s.param_sites == ((1, 1),)


# A seeded soundness check by execution: delete one top-level statement; if that changes what the function
# returns on some input, the analysis must not have called the statement local (or, for raise-only changes,
# anything but local).
_HELPERS = (
    "def helper(xs, v):\n    xs.append(v)\n\n"
    "def check(v):\n    if v > 2:\n        raise ValueError(v)\n\n"
    "def g(v):\n    return v % 2 == 0\n"
)


def _random_program(rng: random.Random) -> list[str]:
    k = lambda: rng.randint(-2, 3)  # noqa: E731
    v = lambda: rng.choice(("p", "a", "b"))  # noqa: E731
    seq = lambda: rng.choice(("acc", "view"))  # noqa: E731
    kinds = [
        lambda: f"a = {v()} + {k()}",
        lambda: f"b = {v()} * 2",
        lambda: f"{seq()} = []",
        lambda: f"view = {seq()}",
        lambda: f"{seq()}.append({v()})",
        lambda: f"helper({seq()}, {v()})",
        lambda: f"if {v()} > {k()}:\n    return {v()}",
        lambda: f"if g({v()}):\n    return len({seq()})",
        lambda: f"if {v()} < {k()}:\n    raise ValueError('x')",
        lambda: f"try:\n    check({v()})\nexcept ValueError:\n    b = {k()}",
        lambda: f"check({v()})",
        lambda: f"for x in range({v()} % 3):\n    {seq()}.append(x)",
        lambda: f"for x in range(3):\n    a = a + x\n    if x == {v()}:\n        break",
        lambda: f"a = len({seq()})",
        lambda: f"b = {seq()}[0] if {seq()} else {k()}",
        lambda: f"t = {v()} - {k()}",
    ]
    return [rng.choice(kinds)() for _ in range(rng.randint(2, 7))]


def _program(stmts: list[str], ret: str) -> str:
    body = ["a = 0", "b = 0", "acc = []", "view = acc", *stmts, f"return {ret}"]
    lines = [f"    {line}" for s in body for line in s.split("\n")]
    return "def f(p):\n" + "\n".join(lines) + "\n"


def _outcomes(src: str) -> list[tuple]:
    ns: dict = {}
    exec(compile(_HELPERS + "\n" + src, "<random program>", "exec"), ns)
    out = []
    for p in range(-3, 6):
        try:
            value = ns["f"](p)
            out.append(("returned", repr(value)))
        except ValueError:
            out.append(("raised",))
    return out


@pytest.mark.parametrize("seed", range(40))
def test_random_programs_never_call_a_statement_that_matters_local(seed):
    rng = random.Random(seed)
    for _ in range(12):
        stmts = _random_program(rng)
        ret = rng.choice(("a", "b", "acc", "len(view)", "(a, len(acc))"))
        src = _program(stmts, ret)
        base = _outcomes(src)
        fn = ast.parse(src).body[0]
        for i in range(len(stmts)):
            region = fn.body[4 + i]
            r = flows(fn, {0: [region]}).regions[0]
            dropped = _outcomes(_program(stmts[:i] + ["pass"] + stmts[i + 1 :], ret))
            changed = [(x, y) for x, y in zip(base, dropped, strict=True) if x != y]
            if any(x[0] == y[0] == "returned" for x, y in changed):
                assert r.value_determining, (src, i, r)
            elif changed:
                assert r.kind != "local", (src, i, r)


def test_flow_analysis_is_bounded_and_says_so():
    from unify.memory_v2.analysis import fn_dataflow

    fn = normalise(function_defs(_wide(2000, 0, 0))["f"]).node
    assert not flows(fn).truncated
    every = tuple(range(len(fn.args.args)))
    for name, value, limit in (
        ("MAX_WORK", 100, "work"),
        ("MAX_STEPS", 1, "steps"),
        ("MAX_NODES", 50, "nodes"),
    ):
        old = getattr(fn_dataflow, name)
        setattr(fn_dataflow, name, value)
        try:
            s = flows(fn, {0: [fn.body[0]]})
        finally:
            setattr(fn_dataflow, name, old)
        # never silent: truncated, the limit named, and the conservative answer for every region
        assert s.truncated and s.limit == limit and s.param_sites is None
        assert s.regions[0].kind == "nonlocal" and s.regions[0].params == every
    s = flows(fn, {0: [fn.body[0]]}, budget_s=0.0)
    assert s.truncated and s.limit == "time"


def _wide_context(n: int) -> str:
    """Review N3's shape: one test reading n names governing n assignments."""
    return (
        "def f(p):\n    if "
        + " or ".join(f"a{i}" for i in range(n))
        + ":\n"
        + "".join(f"        b{i} = 1\n" for i in range(n))
        + "    return p\n"
    )


def test_a_wide_control_context_costs_linear_work():
    # each binder adds one edge to the context node, not one per name the test reads (quadratic before)
    fn = ast.parse(_wide_context(3000)).body[0]
    t0 = time.monotonic()
    s = flows(fn, {0: [fn.body[0]]})
    assert time.monotonic() - t0 < 1.0
    assert not s.truncated and s.regions[0].kind == "local"
    # review N3's 24k-node case (15 s and 650 MB before) and the 48k one: refused by the node cap, at once
    for n in (4000, 8000):
        fn = ast.parse(_wide_context(n)).body[0]
        t0 = time.monotonic()
        s = flows(fn, {0: [fn.body[0]]})
        assert time.monotonic() - t0 < 1.0
        assert s.truncated and s.limit == "nodes" and s.regions[0].kind == "nonlocal"


# --- shared subtrees at scale (review M9) ---------------------------------------------------------------------


def test_shared_subtree_greedy_is_near_linear_on_a_repetitive_library():
    src = ""
    for f in range(8):
        src += f"def f{f}(x, y):\n" + "".join(
            "    a = x + y * 2\n" if i % 2 else "    b = x - y * 3\n"
            for i in range(1500)
        )
        src += "    return a\n\n"
    t0 = time.monotonic()
    rep = shared_subtrees_source(src, min_size=4, min_functions=3)
    assert time.monotonic() - t0 < 60
    assert rep.candidates and rep.skipped == ()
    # picks never overlap within a function
    seen: dict[str, list[tuple[int, int]]] = {}
    for c in rep.candidates:
        for o in c.occurrences:
            for s, e in seen.get(o.function, ()):
                assert o.start + c.size <= s or e <= o.start
            seen.setdefault(o.function, []).append((o.start, o.start + c.size))


def test_the_greedy_work_bound_truncates():
    rep = shared_subtrees_source(OFFICE, min_size=6, min_functions=3, max_work=1)
    assert rep.truncated
