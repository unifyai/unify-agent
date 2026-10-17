"""Learned skills: library items whose explicit parameters are fitted to each request's own examples.

A learned skill is an ordinary library module (``memory/<package>/<module>.py``) that declares a parameter family
with literals, and one public function, the skeleton, that the item id names::

    LEARNED_FAMILY = "lookup"                    # "lookup", "affine" or "neighbourhood"
    LEARNED_OPTIONS = {"elementwise": True}      # the family's options (literals only)
    LEARNED_INITIAL = {"identity": 2.0}          # optional: the family's starting values (literals only)

    def recolour(x, model, config):              # the skeleton: the item; model is the fitted family
        return model(x)

    def _infer_config(examples, stored):         # optional: discrete settings derived from the examples
        return {}

    def _training_pairs(x, y, config):           # optional: the family's own (input, output) pairs of one example
        return [(x, y)]

The library module imports nothing from the harness; the families are this file's. In a cell::

    skill = memory.learned("memory.<package>.<module>:<name>")
    fitted = skill.fit(examples)     # examples: (input, output) pairs the request gave or the work produced
    fitted(x)                        # the skeleton with the fitted model and config

``fit`` builds a plain callable at run time from the stored state plus a bounded fit on *examples* (a list of
pairs or a dict of input to output; none: the item's own stored fixtures). It never changes the stored state:
only the harness's fixture-gated consolidation after the episode does (:func:`consolidated`).

Families (each convex, with hand-written gradients, fitted with Adam from the stored parameters, deterministic):

* ``lookup``: a soft table over arbitrary hashable categories (str, int, float, bool, None, tuples of these;
  a list is read as a tuple): logits per (input, output) pair, softmax and cross-entropy. ``elementwise``: the
  inputs and outputs are nested lists of one shape, mapped element by element. ``identity`` (initial): a fixed
  bonus for an output equal to its input, so an unseen input maps to itself.
* ``affine``: real vectors, ``y = x W + b``, mean squared error. Options ``inputs`` and ``outputs`` (sizes).
* ``neighbourhood``: a 2D array of categories to a 2D array of the same shape: each cell's
  (2 radius + 1)^2 neighbourhood, one-hot, through a linear softmax layer. Options ``values`` (the categories)
  or ``classes`` (0 .. classes-1) and ``radius`` (default 1); only for 2D data.

The stored state is a small JSON object (readable, diffable, versioned in the library's git):
``{"family", "options", "params", "config", "config_scores", "version"}``. Standard library and numpy only;
no network, no randomness, bounded steps and seconds.
"""

from __future__ import annotations

import ast
import json
import math
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np

FAMILIES = ("lookup", "affine", "neighbourhood")
STATE_VERSION = 1
#: Fit bounds: the most optimiser steps and wall-clock seconds one fit may take, and the most pairs it reads.
MAX_STEPS = 300
MAX_SECONDS = 5.0
MAX_PAIRS = 20_000
#: The most parameters a state may hold (a bound on what is stored and recorded).
MAX_PARAMS = 200_000
#: Cross-entropy fits stop once every training pair's own output has at least this probability.
CONFIDENCE = 0.6
#: The audit event a fit raises: (item id, the fitted state as JSON text, reproduced, steps). The harness keeps it,
#: harness-side only, for the consolidation after the episode; the text is "" when it is over MAX_STATE_CHARS.
FIT_EVENT = "memory.learned_fit"
MAX_STATE_CHARS = 200_000
#: The prefix of the confined check's one result line (:func:`main`).
CHECK_MARK = "LEARNED-CHECK "


class LearnedError(ValueError):
    """A learned skill cannot be built or fitted as asked; the message says what to change."""


# --- categories ----------------------------------------------------------------------------------------------------


def _tag(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return {"tuple": [_tag(v) for v in value]}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise LearnedError(
        f"a category must be a str, number, bool, None or a tuple of these, not {type(value).__name__}",
    )


def _untag(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"tuple"}:
        return tuple(_untag(v) for v in value["tuple"])
    return value


def category_key(value: Any) -> str:
    """The stored key of a category: canonical JSON (a list or tuple as a tagged tuple)."""
    return json.dumps(_tag(value), sort_keys=True, separators=(",", ":"))


def category_of(key: str) -> Any:
    return _untag(json.loads(key))


def _flat(value: Any) -> tuple[list, Any]:
    """A nested list's leaves in order, and its shape (a nested structure of lengths) to rebuild it."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        leaves: list = []
        shape: list = []
        for v in value:
            got, s = _flat(v)
            leaves.extend(got)
            shape.append(s)
        return leaves, shape
    return [value], None


def _rebuild(leaves: list, shape: Any, at: list[int]) -> Any:
    if shape is None:
        at[0] += 1
        return leaves[at[0] - 1]
    return [_rebuild(leaves, s, at) for s in shape]


# --- the optimiser -------------------------------------------------------------------------------------------------


def adam(
    vec: np.ndarray,
    loss_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    done: Callable[[np.ndarray], bool],
    *,
    lr: float,
    max_steps: int,
    max_seconds: float,
) -> tuple[np.ndarray, int, float]:
    """Adam from *vec* until *done* holds, *max_steps* steps or *max_seconds*: (vector, steps taken, last loss)."""
    m = np.zeros_like(vec)
    v = np.zeros_like(vec)
    b1, b2, eps = 0.9, 0.999, 1e-8
    start = time.monotonic()
    loss, _ = loss_and_grad(vec)
    steps = 0
    while steps < max_steps and not done(vec):
        if time.monotonic() - start > max_seconds:
            break
        loss, g = loss_and_grad(vec)
        steps += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1**steps)
        vh = v / (1 - b2**steps)
        vec = vec - lr * mh / (np.sqrt(vh) + eps)
    loss, _ = loss_and_grad(vec)
    return vec, steps, float(loss)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# --- families ------------------------------------------------------------------------------------------------------


class Family:
    """One parameter family: its options, its starting parameters, and fit/predict on its own pairs."""

    name = ""
    lr = 0.5

    def __init__(self, options: dict | None = None) -> None:
        self.options = dict(options or {})

    def initial(self, initial: dict | None) -> dict:
        raise NotImplementedError

    def fit(
        self, params: dict, pairs: list, max_steps: int, max_seconds: float,
    ) -> tuple[dict, dict]:
        """(task params, report {steps, loss, reproduced, pairs}) from *params* fitted on *pairs*."""
        raise NotImplementedError

    def predictor(self, params: dict) -> Callable[[Any], Any]:
        raise NotImplementedError

    def loss(self, params: dict, pairs: list) -> float:
        raise NotImplementedError

    def correct(self, params: dict, pairs: list) -> bool:
        predict = self.predictor(params)
        for x, y in pairs:
            try:
                if not self.same(predict(x), y):
                    return False
            except LearnedError:
                return False
        return True

    def same(self, got: Any, want: Any) -> bool:
        return category_key(got) == category_key(want)


class Lookup(Family):
    """A soft table: logits per (input, output) category pair; softmax and cross-entropy."""

    name = "lookup"
    lr = 0.5

    def initial(self, initial: dict | None) -> dict:
        init = dict(initial or {})
        table: dict[str, dict[str, float]] = {}
        strength = float(init.get("strength", 4.0))
        for x, y in init.get("table", []) or []:
            table.setdefault(category_key(x), {})[category_key(y)] = strength
        return {"logits": table, "identity": float(init.get("identity", 0.0))}

    def _elements(self, pairs: list) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for x, y in pairs:
            if self.options.get("elementwise"):
                xs, xshape = _flat(x)
                ys, yshape = _flat(y)
                if xshape != yshape:
                    raise LearnedError(
                        "elementwise lookup: an input and its output have different shapes",
                    )
                out.extend((category_key(a), category_key(b)) for a, b in zip(xs, ys))
            else:
                out.append((category_key(x), category_key(y)))
            if len(out) > MAX_PAIRS:
                raise LearnedError(f"more than {MAX_PAIRS} training pairs")
        return out

    def _dense(
        self, params: dict, ins: list[str], outs: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """The fitted logits (rows ins, columns outs) and the fixed identity bonus matrix."""
        table = params.get("logits", {})
        L = np.zeros((len(ins), len(outs)))
        col = {o: j for j, o in enumerate(outs)}
        for i, a in enumerate(ins):
            for o, v in table.get(a, {}).items():
                if o in col:
                    L[i, col[o]] = float(v)
        ident = float(params.get("identity", 0.0))
        B = np.zeros_like(L)
        if ident:
            for i, a in enumerate(ins):
                if a in col:
                    B[i, col[a]] = ident
        return L, B

    def _outputs(self, params: dict, extra: Iterable[str] = ()) -> list[str]:
        outs = set(extra)
        for row in params.get("logits", {}).values():
            outs.update(row)
        return sorted(outs)

    def fit(
        self, params: dict, pairs: list, max_steps: int, max_seconds: float,
    ) -> tuple[dict, dict]:
        el = self._elements(pairs)
        if not el:
            raise LearnedError("no training pairs")
        ins = sorted({a for a, _ in el})
        extra = {b for _, b in el} | (set(ins) if params.get("identity") else set())
        outs = self._outputs(params, extra)
        if len(ins) * len(outs) > MAX_PARAMS:
            raise LearnedError(f"lookup: more than {MAX_PARAMS} parameters")
        L0, B = self._dense(params, ins, outs)
        row = {a: i for i, a in enumerate(ins)}
        col = {o: j for j, o in enumerate(outs)}
        xi = np.array([row[a] for a, _ in el])
        yi = np.array([col[b] for _, b in el])
        n = len(el)
        shape = L0.shape

        def lg(vec: np.ndarray) -> tuple[float, np.ndarray]:
            L = vec.reshape(shape)
            P = _softmax(L[xi] + B[xi])
            loss = -np.log(P[np.arange(n), yi] + 1e-12).mean()
            G = P
            G[np.arange(n), yi] -= 1.0
            grad = np.zeros(shape)
            np.add.at(grad, xi, G / n)
            return float(loss), grad.ravel()

        def done(vec: np.ndarray) -> bool:
            P = _softmax(vec.reshape(shape)[xi] + B[xi])
            return bool((P[np.arange(n), yi] >= CONFIDENCE).all())

        vec, steps, loss = adam(
            L0.ravel().copy(),
            lg,
            done,
            lr=float(self.options.get("lr", self.lr)),
            max_steps=max_steps,
            max_seconds=max_seconds,
        )
        L = vec.reshape(shape)
        table = {a: dict(r) for a, r in params.get("logits", {}).items()}
        seen = set(el)  # an input's own outputs stay in its row (one known output gives a logit of 0)
        for i, a in enumerate(ins):
            r = table.setdefault(a, {})
            for j, o in enumerate(outs):
                if L[i, j] != 0.0 or o in r or (a, o) in seen:
                    r[o] = float(L[i, j])
        task = {"logits": table, "identity": float(params.get("identity", 0.0))}
        return task, {"steps": steps, "loss": loss, "reproduced": done(vec), "pairs": n}

    def _one(self, params: dict, x: Any) -> Any:
        a = category_key(x)
        row = params.get("logits", {}).get(a, {})
        ident = float(params.get("identity", 0.0))
        outs = self._outputs(params, [a] if ident else [])
        if not row and not ident:
            raise LearnedError(f"lookup: no stored output for the input {x!r}")
        scores = {o: float(row.get(o, 0.0)) + (ident if o == a else 0.0) for o in outs}
        best = max(sorted(scores), key=lambda o: scores[o])
        return category_of(best)

    def predictor(self, params: dict) -> Callable[[Any], Any]:
        def predict(x: Any) -> Any:
            if self.options.get("elementwise"):
                leaves, shape = _flat(x)
                got = [self._one(params, v) for v in leaves]
                return _rebuild(got, shape, [0]) if shape is not None else got[0]
            return self._one(params, x)

        return predict

    def loss(self, params: dict, pairs: list) -> float:
        el = self._elements(pairs)
        if not el:
            return 0.0
        ins = sorted({a for a, _ in el})
        extra = {b for _, b in el} | (set(ins) if params.get("identity") else set())
        outs = self._outputs(params, extra)
        L, B = self._dense(params, ins, outs)
        row = {a: i for i, a in enumerate(ins)}
        col = {o: j for j, o in enumerate(outs)}
        xi = np.array([row[a] for a, _ in el])
        yi = np.array([col[b] for _, b in el])
        P = _softmax(L[xi] + B[xi])
        return float(-np.log(P[np.arange(len(el)), yi] + 1e-12).mean())


class Affine(Family):
    """Real vectors: ``y = x W + b``; mean squared error (half the squared error, averaged over pairs)."""

    name = "affine"
    lr = 0.05

    def _sizes(self) -> tuple[int, int]:
        try:
            n, m = int(self.options["inputs"]), int(self.options["outputs"])
        except (KeyError, TypeError, ValueError):
            raise LearnedError(
                "affine: options need integer 'inputs' and 'outputs'",
            ) from None
        if n < 1 or m < 1 or n * m > MAX_PARAMS:
            raise LearnedError(
                "affine: 'inputs' and 'outputs' must be at least 1 and bounded",
            )
        return n, m

    def initial(self, initial: dict | None) -> dict:
        n, m = self._sizes()
        init = dict(initial or {})
        W = np.zeros((n, m))
        if init.get("identity"):
            W[np.arange(min(n, m)), np.arange(min(n, m))] = 1.0
        if "weights" in init:
            W = np.asarray(init["weights"], dtype=float).reshape(n, m)
        b = np.asarray(init.get("bias", [0.0] * m), dtype=float).reshape(m)
        return {"weights": W.tolist(), "bias": b.tolist()}

    def _vec(self, v: Any, size: int, what: str) -> np.ndarray:
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size != size:
            raise LearnedError(f"affine: an {what} has {a.size} values, not {size}")
        if not np.isfinite(a).all():
            raise LearnedError(f"affine: an {what} is not finite")
        return a

    def _data(self, pairs: list) -> tuple[np.ndarray, np.ndarray]:
        n, m = self._sizes()
        if not pairs:
            raise LearnedError("no training pairs")
        if len(pairs) > MAX_PAIRS:
            raise LearnedError(f"more than {MAX_PAIRS} training pairs")
        X = np.stack([self._vec(x, n, "input") for x, _ in pairs])
        Y = np.stack([self._vec(y, m, "output") for _, y in pairs])
        return X, Y

    def _tolerance(self, Y: np.ndarray) -> np.ndarray:
        return float(self.options.get("tolerance", 0.01)) * np.maximum(1.0, np.abs(Y))

    def fit(
        self, params: dict, pairs: list, max_steps: int, max_seconds: float,
    ) -> tuple[dict, dict]:
        n, m = self._sizes()
        X, Y = self._data(pairs)
        N = len(X)
        W0 = np.asarray(params["weights"], dtype=float).reshape(n, m)
        b0 = np.asarray(params["bias"], dtype=float).reshape(m)
        tol = self._tolerance(Y)

        def split(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return vec[: n * m].reshape(n, m), vec[n * m :]

        def lg(vec: np.ndarray) -> tuple[float, np.ndarray]:
            W, b = split(vec)
            E = X @ W + b - Y
            loss = 0.5 * (E * E).sum() / N
            return float(loss), np.concatenate(
                [(X.T @ E / N).ravel(), E.sum(axis=0) / N],
            )

        def done(vec: np.ndarray) -> bool:
            W, b = split(vec)
            return bool((np.abs(X @ W + b - Y) <= tol).all())

        vec, steps, loss = adam(
            np.concatenate([W0.ravel(), b0]),
            lg,
            done,
            lr=float(self.options.get("lr", self.lr)),
            max_steps=max_steps,
            max_seconds=max_seconds,
        )
        W, b = split(vec)
        return {"weights": W.tolist(), "bias": b.tolist()}, {
            "steps": steps,
            "loss": loss,
            "reproduced": done(vec),
            "pairs": N,
        }

    def predictor(self, params: dict) -> Callable[[Any], Any]:
        n, m = self._sizes()
        W = np.asarray(params["weights"], dtype=float).reshape(n, m)
        b = np.asarray(params["bias"], dtype=float).reshape(m)

        def predict(x: Any) -> Any:
            y = self._vec(x, n, "input") @ W + b
            return (
                float(y[0])
                if m == 1 and not isinstance(x, (list, tuple, np.ndarray))
                else y.tolist()
            )

        return predict

    def same(self, got: Any, want: Any) -> bool:
        g = np.asarray(got, dtype=float).reshape(-1)
        w = np.asarray(want, dtype=float).reshape(-1)
        return g.shape == w.shape and bool((np.abs(g - w) <= self._tolerance(w)).all())

    def loss(self, params: dict, pairs: list) -> float:
        n, m = self._sizes()
        X, Y = self._data(pairs)
        W = np.asarray(params["weights"], dtype=float).reshape(n, m)
        b = np.asarray(params["bias"], dtype=float).reshape(m)
        E = X @ W + b - Y
        return float(0.5 * (E * E).sum() / len(X))


class Neighbourhood(Family):
    """A 2D array of categories to one of the same shape: each cell's neighbourhood, one-hot, through a linear
    softmax layer."""

    name = "neighbourhood"
    lr = 0.3

    def _values(self) -> list[str]:
        if "values" in self.options:
            vals = [category_key(v) for v in self.options["values"]]
        elif "classes" in self.options:
            vals = [category_key(i) for i in range(int(self.options["classes"]))]
        else:
            raise LearnedError(
                "neighbourhood: options need 'values' (the categories) or 'classes'",
            )
        if not vals or len(set(vals)) != len(vals):
            raise LearnedError(
                "neighbourhood: the categories must be distinct and at least one",
            )
        return vals

    def _dims(self) -> tuple[int, int, int]:
        r = int(self.options.get("radius", 1))
        if r < 0 or r > 3:
            raise LearnedError("neighbourhood: 'radius' must be 0 to 3")
        V = len(self._values())
        D = (2 * r + 1) ** 2 * (V + 1)
        if D * V > MAX_PARAMS:
            raise LearnedError("neighbourhood: too many parameters for these options")
        return r, V, D

    def initial(self, initial: dict | None) -> dict:
        r, V, D = self._dims()
        init = dict(initial or {})
        W = np.zeros((D, V))
        ident = float(init.get("identity", 0.0))
        if ident:
            k = 2 * r + 1
            centre = (r * k + r) * (V + 1)
            for c in range(V):
                W[centre + c, c] = ident
        return {"weights": W.tolist(), "bias": [0.0] * V}

    def _grid(self, g: Any) -> np.ndarray:
        """A 2D array of category indices (V for none)."""
        if isinstance(g, np.ndarray):
            g = g.tolist()
        if (
            not isinstance(g, (list, tuple))
            or not g
            or not all(isinstance(r, (list, tuple)) for r in g)
        ):
            raise LearnedError("neighbourhood: the data must be a non-empty 2D array")
        w = len(g[0])
        if w == 0 or any(len(r) != w for r in g):
            raise LearnedError(
                "neighbourhood: the rows of a 2D array must have one length",
            )
        idx = {v: i for i, v in enumerate(self._values())}
        try:
            return np.array([[idx[category_key(c)] for c in r] for r in g], dtype=int)
        except KeyError as exc:
            raise LearnedError(
                f"neighbourhood: a value is not one of the categories: {exc}",
            ) from None

    def _features(self, a: np.ndarray) -> np.ndarray:
        r, V, D = self._dims()
        k = 2 * r + 1
        h, w = a.shape
        pad = np.full((h + 2 * r, w + 2 * r), V, dtype=int)
        pad[r : r + h, r : r + w] = a
        F = np.zeros((h * w, D))
        rows = np.arange(h * w)
        for dy in range(k):
            for dx in range(k):
                sym = pad[dy : dy + h, dx : dx + w].reshape(-1)
                F[rows, (dy * k + dx) * (V + 1) + sym] = 1.0
        return F

    def _data(self, pairs: list) -> tuple[np.ndarray, np.ndarray]:
        Fs, ys = [], []
        for x, y in pairs:
            a, b = self._grid(x), self._grid(y)
            if a.shape != b.shape:
                raise LearnedError(
                    "neighbourhood: an input and its output have different shapes",
                )
            Fs.append(self._features(a))
            ys.append(b.reshape(-1))
        if not Fs:
            raise LearnedError("no training pairs")
        F, y = np.concatenate(Fs), np.concatenate(ys)
        if len(y) > MAX_PAIRS:
            raise LearnedError(f"more than {MAX_PAIRS} training cells")
        return F, y

    def fit(
        self, params: dict, pairs: list, max_steps: int, max_seconds: float,
    ) -> tuple[dict, dict]:
        r, V, D = self._dims()
        F, y = self._data(pairs)
        n = len(y)
        W0 = np.asarray(params["weights"], dtype=float).reshape(D, V)
        b0 = np.asarray(params["bias"], dtype=float).reshape(V)

        def split(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return vec[: D * V].reshape(D, V), vec[D * V :]

        def lg(vec: np.ndarray) -> tuple[float, np.ndarray]:
            W, b = split(vec)
            P = _softmax(F @ W + b)
            loss = -np.log(P[np.arange(n), y] + 1e-12).mean()
            P[np.arange(n), y] -= 1.0
            return float(loss), np.concatenate(
                [(F.T @ P / n).ravel(), P.sum(axis=0) / n],
            )

        def done(vec: np.ndarray) -> bool:
            W, b = split(vec)
            P = _softmax(F @ W + b)
            return bool((P[np.arange(n), y] >= CONFIDENCE).all())

        vec, steps, loss = adam(
            np.concatenate([W0.ravel(), b0]),
            lg,
            done,
            lr=float(self.options.get("lr", self.lr)),
            max_steps=max_steps,
            max_seconds=max_seconds,
        )
        W, b = split(vec)
        return {"weights": W.tolist(), "bias": b.tolist()}, {
            "steps": steps,
            "loss": loss,
            "reproduced": done(vec),
            "pairs": n,
        }

    def predictor(self, params: dict) -> Callable[[Any], Any]:
        r, V, D = self._dims()
        W = np.asarray(params["weights"], dtype=float).reshape(D, V)
        b = np.asarray(params["bias"], dtype=float).reshape(V)
        vals = [category_of(v) for v in self._values()]

        def predict(x: Any) -> Any:
            a = self._grid(x)
            z = (self._features(a) @ W + b).argmax(axis=1).reshape(a.shape)
            return [[vals[int(c)] for c in row] for row in z]

        return predict

    def loss(self, params: dict, pairs: list) -> float:
        r, V, D = self._dims()
        F, y = self._data(pairs)
        W = np.asarray(params["weights"], dtype=float).reshape(D, V)
        b = np.asarray(params["bias"], dtype=float).reshape(V)
        P = _softmax(F @ W + b)
        return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())


_FAMILY = {"lookup": Lookup, "affine": Affine, "neighbourhood": Neighbourhood}


def family(name: str, options: dict | None = None) -> Family:
    if name not in _FAMILY:
        raise LearnedError(
            f"unknown family {name!r}; the families are {', '.join(FAMILIES)}",
        )
    return _FAMILY[name](options)


# --- the stored state ----------------------------------------------------------------------------------------------


def initial_state(family_name: str, options: dict | None, initial: dict | None) -> dict:
    fam = family(family_name, options)
    return {
        "version": STATE_VERSION,
        "family": family_name,
        "options": dict(options or {}),
        "params": fam.initial(initial),
        "config": {},
        "config_scores": [],
    }


def state_text(state: dict) -> str:
    """The stored form: canonical, readable JSON (one key per line), ending with a newline."""
    return json.dumps(state, sort_keys=True, indent=1, allow_nan=False) + "\n"


def compatible(state: Any, family_name: str, options: dict | None) -> bool:
    """Whether a stored *state* belongs to the module's current declaration (same family and options)."""
    return (
        isinstance(state, dict)
        and state.get("version") == STATE_VERSION
        and state.get("family") == family_name
        and state.get("options") == dict(options or {})
        and isinstance(state.get("params"), dict)
    )


def save_npz(state: dict, path: str) -> None:
    """The state as a ``.npz`` (its JSON text as one string array; no pickling)."""
    np.savez(path, state=np.array(state_text(state)))


def load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["state"]))


def _param_vector(params: Any) -> list[tuple[tuple, float]]:
    """Every number of a params tree with its path (dict keys and list indices)."""
    out: list[tuple[tuple, float]] = []

    def walk(v: Any, path: tuple) -> None:
        if isinstance(v, dict):
            for k in sorted(v):
                walk(v[k], path + (k,))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, path + (i,))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append((path, float(v)))

    walk(params, ())
    return out


def _merge(base: Any, task: Any, step: float) -> Any:
    """``base + step * (task - base)`` over two params trees; a key in only one side counts as 0 on the other."""
    if isinstance(base, dict) or isinstance(task, dict):
        b = base if isinstance(base, dict) else {}
        t = task if isinstance(task, dict) else {}
        return {
            k: _merge(b.get(k, 0.0), t.get(k, 0.0), step)
            for k in sorted(set(b) | set(t))
        }
    if isinstance(base, list) and isinstance(task, list) and len(base) == len(task):
        return [_merge(x, y, step) for x, y in zip(base, task)]
    if isinstance(base, (int, float)) and isinstance(task, (int, float)):
        return float(base) + step * (float(task) - float(base))
    return base


def consolidated(state: dict, task: dict, weight: float, rate: float) -> dict:
    """The Reptile-style step: ``p + rate * weight * (p_task - p)`` for the parameters, and for the discrete config
    the outcome-weighted record (*task*'s config gains *weight*; the stored config becomes the best-scored, the
    current one kept on a tie). *task* is a fitted state (:attr:`Fitted.state`) of the same family and options.
    """
    if not compatible(task, state.get("family", ""), state.get("options")):
        raise LearnedError(
            "the task state does not belong to this skill's family and options",
        )
    step = max(0.0, min(1.0, float(rate) * float(weight)))
    out = json.loads(json.dumps(state))
    out["params"] = _merge(state.get("params", {}), task.get("params", {}), step)
    scores = [dict(r) for r in state.get("config_scores", []) if isinstance(r, dict)]
    cfg = task.get("config", {}) or {}
    for r in scores:
        if r.get("config") == cfg:
            r["score"] = float(r.get("score", 0.0)) + float(weight)
            break
    else:
        if cfg:
            scores.append({"config": cfg, "score": float(weight)})
    out["config_scores"] = scores
    current = state.get("config", {}) or {}
    best = max((r["score"] for r in scores), default=None)
    cur_score = next((r["score"] for r in scores if r.get("config") == current), None)
    if best is not None and (cur_score is None or best > cur_score):
        out["config"] = next(r["config"] for r in scores if r["score"] == best)
    return out


# --- the skill and its fitted callable -----------------------------------------------------------------------------


def declaration(source: str) -> dict | None:
    """A module's learned-skill declaration read from its source without running it: ``{"family", "options",
    "initial", "skeleton"}`` (the one public function's name, or None), or None when the module declares none.
    Raises :class:`LearnedError` for a declaration that is not literals."""
    tree = ast.parse(source)
    found: dict[str, Any] = {}
    public: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in ("LEARNED_FAMILY", "LEARNED_OPTIONS", "LEARNED_INITIAL"):
                try:
                    found[name] = ast.literal_eval(node.value)
                except ValueError:
                    raise LearnedError(f"{name} must be a literal") from None
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef),
        ) and not node.name.startswith("_"):
            public.append(node.name)
    if "LEARNED_FAMILY" not in found:
        return None
    fam = found["LEARNED_FAMILY"]
    options = found.get("LEARNED_OPTIONS", {})
    initial = found.get("LEARNED_INITIAL", {})
    if fam not in FAMILIES:
        raise LearnedError(f"LEARNED_FAMILY must be one of {', '.join(FAMILIES)}")
    if not isinstance(options, dict) or not isinstance(initial, dict):
        raise LearnedError("LEARNED_OPTIONS and LEARNED_INITIAL must be dict literals")
    family(fam, options).initial(initial)  # the declaration must build
    return {
        "family": fam,
        "options": options,
        "initial": initial,
        "skeleton": public[0] if len(public) == 1 else None,
    }


def as_pairs(examples: Any) -> list[tuple[Any, Any]]:
    """(input, output) pairs from a list of pairs or a dict of input to output."""
    if isinstance(examples, dict):
        return [(k, v) for k, v in examples.items()]
    if not isinstance(examples, (list, tuple)):
        raise LearnedError(
            "examples must be a list of (input, output) pairs or a dict of input to output",
        )
    out = []
    for e in examples:
        if isinstance(e, dict) and set(e) >= {"input", "output"}:
            out.append((e["input"], e["output"]))
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            out.append((e[0], e[1]))
        else:
            raise LearnedError(
                "each example must be an (input, output) pair or {'input': ..., 'output': ...}",
            )
    return out


class LearnedSkill:
    """A learned skill: its declaration, its stored state, its hooks and its stored fixtures. :meth:`fit` returns a
    plain callable; nothing here changes the stored state."""

    def __init__(
        self,
        item_id: str,
        state: dict,
        *,
        skeleton: Callable | None = None,
        infer_config: Callable | None = None,
        training_pairs: Callable | None = None,
        fixtures: list | None = None,
        audit: bool = True,
    ) -> None:
        self.item_id = item_id
        self.state = state
        self.family = family(state["family"], state.get("options"))
        self._skeleton = skeleton
        self._infer = infer_config
        self._pairs = training_pairs
        self.fixtures = list(fixtures or [])
        self._audit = audit

    def __repr__(self) -> str:
        return (
            f"<learned skill {self.item_id}: family {self.state['family']}, {len(self.fixtures)} stored fixture "
            "case(s); fitted = skill.fit(examples); fitted(x)>"
        )

    def stored_examples(self) -> list[tuple[Any, Any]]:
        """The (input, output) pairs of the item's stored fixture cases."""
        out: list[tuple[Any, Any]] = []
        for case in self.fixtures:
            out.extend(as_pairs(case.get("examples", [])))
        return out

    def config_for(self, pairs: list) -> dict:
        stored = dict(self.state.get("config", {}) or {})
        if self._infer is None:
            return stored
        got = self._infer(pairs, stored)
        if got is None:
            return stored
        if not isinstance(got, dict):
            raise LearnedError("_infer_config must return a dict")
        json.dumps(got)  # the config must be plain data
        return got

    def family_pairs(self, pairs: list, config: dict) -> list:
        if self._pairs is None:
            return list(pairs)
        out: list = []
        for x, y in pairs:
            out.extend(as_pairs(list(self._pairs(x, y, config))))
        return out

    def fit(
        self,
        examples: Any = None,
        *,
        max_steps: int = MAX_STEPS,
        max_seconds: float = MAX_SECONDS,
    ) -> Callable[[Any], Any]:
        """A plain callable fitted to *examples* ((input, output) pairs; None: the stored fixtures' pairs), built
        from the stored state; the stored state is not changed. The callable has ``reproduced`` (every example's
        output given back), ``steps``, ``loss``, ``config`` and ``state`` (the fitted state) as attributes.
        """
        pairs = as_pairs(examples) if examples is not None else self.stored_examples()
        if not pairs:
            raise LearnedError(
                "no examples: pass (input, output) pairs, or store fixture cases for the item",
            )
        steps = max(0, min(int(max_steps), 10 * MAX_STEPS))
        seconds = max(0.0, min(float(max_seconds), 10 * MAX_SECONDS))
        config = self.config_for(pairs)
        fpairs = self.family_pairs(pairs, config)
        params, report = self.family.fit(self.state["params"], fpairs, steps, seconds)
        model = self.family.predictor(params)
        skeleton = self._skeleton

        def fitted(x: Any) -> Any:
            return skeleton(x, model, config) if skeleton is not None else model(x)

        task = dict(self.state)
        task.update(params=params, config=config)
        reproduced = bool(report["reproduced"])
        if reproduced and skeleton is not None:
            reproduced = all(self._same(fitted, x, y) for x, y in pairs)
        fitted.reproduced = reproduced  # type: ignore[attr-defined]
        fitted.steps = report["steps"]  # type: ignore[attr-defined]
        fitted.loss = report["loss"]  # type: ignore[attr-defined]
        fitted.config = config  # type: ignore[attr-defined]
        fitted.state = task  # type: ignore[attr-defined]
        fitted.model = model  # type: ignore[attr-defined]
        if self._audit:
            text = state_text(task)
            try:
                sys.audit(
                    FIT_EVENT,
                    self.item_id,
                    text if len(text) <= MAX_STATE_CHARS else "",
                    reproduced,
                    int(report["steps"]),
                )
            except Exception:  # noqa: BLE001 - an audit hook never changes the call
                pass
        return fitted

    def _same(self, fitted: Callable, x: Any, y: Any) -> bool:
        try:
            got = fitted(x)
        except Exception:  # noqa: BLE001 - a skeleton that raises does not reproduce
            return False
        try:
            return self.family.same(got, y)
        except Exception:  # noqa: BLE001
            return (
                category_key(got) == category_key(y)
                if _plain(got) and _plain(y)
                else got == y
            )


def _plain(v: Any) -> bool:
    try:
        _tag(v)
        return True
    except LearnedError:
        return False


def load_skill(
    item_id: str,
    module: Any,
    source: str,
    state_json: str | None = None,
    fixtures_json: str | None = None,
    audit: bool = True,
) -> LearnedSkill:
    """The skill of an imported library *module* (its *source* gives the declaration), from a stored state when it
    is compatible with the declaration, else from the declaration's initial values."""
    decl = declaration(source)
    if decl is None:
        raise LearnedError(
            f"{item_id} is not a learned skill (its module declares no LEARNED_FAMILY)",
        )
    state = None
    if state_json:
        try:
            state = json.loads(state_json)
        except ValueError:
            state = None
    if not compatible(state, decl["family"], decl["options"]):
        state = initial_state(decl["family"], decl["options"], decl["initial"])
    name = item_id.split(":", 1)[1] if ":" in item_id else decl["skeleton"]
    if ":" not in item_id and name:
        item_id = f"{item_id}:{name}"
    skeleton = getattr(module, name, None) if name else None
    fixtures = []
    if fixtures_json:
        try:
            fixtures = list(json.loads(fixtures_json).get("cases", []))
        except (ValueError, AttributeError):
            fixtures = []
    return LearnedSkill(
        item_id,
        state,
        skeleton=skeleton,
        infer_config=getattr(module, "_infer_config", None),
        training_pairs=getattr(module, "_training_pairs", None),
        fixtures=fixtures,
        audit=audit,
    )


# --- the fixture check (run confined by the harness) ---------------------------------------------------------------


def check_cases(
    skill: LearnedSkill, cases: list, max_steps: int = MAX_STEPS,
) -> list[dict]:
    """Per fixture case: fit on its examples, then whether the examples and the queries are reproduced, the loss on
    the queries (the family's, from the fitted parameters; the examples' when there are no queries) and the steps.
    """
    out = []
    for case in cases:
        row: dict[str, Any] = {
            "passed": False,
            "loss": None,
            "steps": None,
            "error": None,
        }
        try:
            ex = as_pairs(case.get("examples", []))
            qs = as_pairs(case.get("queries", []))
            fitted = skill.fit(ex, max_steps=max_steps)
            ok = bool(fitted.reproduced) and all(
                skill._same(fitted, x, y) for x, y in qs
            )
            fp = skill.family_pairs(qs or ex, fitted.config)
            loss = skill.family.loss(fitted.state["params"], fp)
            row.update(
                passed=ok,
                loss=loss if math.isfinite(loss) else None,
                steps=int(fitted.steps),
            )
        except Exception as exc:  # noqa: BLE001 - a case that raises fails
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        out.append(row)
    return out


def gate_states(old: list[dict], new: list[dict]) -> tuple[bool, str]:
    """The consolidation gate: every case that passed under the old state passes under the new one, and every case
    that did not has a loss no higher than before. No cases: refused (nothing to protect the update with).
    """
    if not old:
        return False, "no_fixtures"
    if len(old) != len(new):
        return False, "case_count"
    for o, n in zip(old, new):
        if o["passed"] and not n["passed"]:
            return False, "fixture_regressed"
        if not o["passed"]:
            if n["loss"] is None or (
                o["loss"] is not None and n["loss"] > o["loss"] + 1e-9
            ):
                return False, "loss_rose"
    return True, "ok"


def main(argv: list[str]) -> int:
    """``python learned.py check <job.json>``, run confined with the library at ``/memory`` on the path: imports the
    module, checks each listed state on the fixture cases (:func:`check_cases`) and prints one line,
    ``LEARNED-CHECK {"<label>": [rows], ...}``. A job: ``{"module", "item", "source_path", "fixtures_path",
    "states": {label: state or null (initial)}}``."""
    import importlib
    import io
    import os

    if len(argv) != 3 or argv[1] != "check":
        print("usage: learned.py check <job.json>", file=sys.stderr)
        return 2
    with open(argv[2], encoding="utf-8") as f:
        job = json.load(f)
    real_out = sys.stdout
    result: dict[str, Any] = {}
    try:
        sys.stdout = (
            io.StringIO()
        )  # what library code prints never reaches the result line
        with open(job["source_path"], encoding="utf-8") as f:
            source = f.read()
        fixtures = ""
        if job.get("fixtures_path") and os.path.exists(job["fixtures_path"]):
            with open(job["fixtures_path"], encoding="utf-8") as f:
                fixtures = f.read()
        module = importlib.import_module(job["module"])
        for label, state in sorted(job["states"].items()):
            skill = load_skill(
                job["item"],
                module,
                source,
                json.dumps(state) if state is not None else None,
                fixtures,
                audit=False,
            )
            result[label] = check_cases(skill, skill.fixtures)
    except Exception as exc:  # noqa: BLE001 - reported, never raised past the line
        result = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    finally:
        sys.stdout = real_out
    print(CHECK_MARK + json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
