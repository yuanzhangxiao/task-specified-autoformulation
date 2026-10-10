"""Run one of LLM-SR's selected programs in a separate, confined process.

LLM-SR's own search executes every program it scores. Outside that search,
this is the one place a program it selected is executed, as Phase C's plan
declares (``llm_sr_programs`` states the exception to the rule against it).
The parent has already checked the program's syntax tree against an
allowlist; this process is the second guard:

- it can import only numpy, math and the listed scipy modules, all imported
  before the program is read;
- the program's namespace holds what LLM-SR's specification module defines
  (``np``, ``MAX_NPARAMS``, ``params``) and a short list of builtins;
- it cannot write a file, and on Linux its memory is limited;
- the parent limits its time and kills it past the limit.

The request comes on standard input and the reply goes to standard output,
both as JSON, so neither side reads anything that can run code. This script
imports nothing from its package.
"""

from __future__ import annotations

import builtins
import json
import math
import os
import resource
import sys
import time

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import scipy.special
from scipy.optimize import minimize

#: Their specification template fixes the parameter vector length.
MAX_NPARAMS = 10

#: What an import inside a program may bind, by module name.
MODULES = {
    "numpy": np,
    "math": math,
    "scipy": scipy,
    "scipy.integrate": scipy.integrate,
    "scipy.interpolate": scipy.interpolate,
    "scipy.ndimage": scipy.ndimage,
    "scipy.signal": scipy.signal,
    "scipy.special": scipy.special,
}

#: Builtins a program may name: arithmetic, containers, iteration and the
#: exceptions their guards raise and catch. Nothing that reads attributes by
#: name, compiles text, or reaches files, the process or the interpreter.
BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "complex", "dict", "divmod", "enumerate",
    "filter", "float", "frozenset", "globals", "hasattr", "int", "isinstance",
    "iter", "len", "list", "locals", "map", "max", "min", "next", "pow", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "zip",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception",
    "FloatingPointError", "ImportError", "IndexError", "KeyError",
    "LookupError", "NameError", "NotImplementedError", "OverflowError",
    "RecursionError", "RuntimeError", "RuntimeWarning", "StopIteration",
    "TypeError", "UnboundLocalError", "UserWarning", "ValueError", "Warning",
    "ZeroDivisionError",
)

#: The memory a program may use, in bytes, where the system enforces it.
MEMORY_LIMIT = 4 * 1024**3


#: Packages whose own parts may be imported while a program runs.
PACKAGES = frozenset({"numpy", "scipy"})

_REAL_IMPORT = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    """The import statement inside a program: listed modules only.

    numpy's compiled code imports its own private parts lazily, through the
    builtins of whichever frame called it, so ``x.mean()`` in a program
    imports ``numpy._core._methods`` through this function. Parts of numpy
    and scipy are therefore passed to the real import; a program cannot
    name one, because the allowlist refuses such an import.
    """
    root = name.split(".")[0]
    if level == 0 and name in MODULES:
        # `import scipy.signal` binds the top-level package, as Python's does.
        return MODULES[name] if fromlist else MODULES[root]
    if level == 0 and root in PACKAGES:
        return _REAL_IMPORT(name, globals, locals, fromlist, level)
    raise ImportError(f"import of {name!r} is not allowed")


def _builtins() -> dict:
    allowed = {name: getattr(builtins, name) for name in BUILTIN_NAMES}
    allowed["__import__"] = _import
    return allowed


def _confine() -> bool:
    """Forbid writing files and limit memory; whether memory was limited."""
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    except (ValueError, OSError):  # macOS does not enforce an address-space limit
        return False
    return True


def _load(source: str, name: str):
    """Define the program as their specification module would, and return it."""
    namespace = {
        "__builtins__": _builtins(),
        "__name__": "llm_sr_program",
        "np": np,
        "MAX_NPARAMS": MAX_NPARAMS,
        "params": [1.0] * MAX_NPARAMS,
    }
    exec(compile(source, "<llm-sr program>", "exec"), namespace)
    return namespace[name]


def _refit(function, request: dict) -> dict:
    """Their evaluate's fit, call for call: BFGS from all ones over the table.

    As in their specification, the program receives the columns of one input
    array, so a program that changes its inputs in place changes them for
    the next trial vector too, as it did in their search.
    """
    inputs = np.asarray(request["inputs"], dtype=float)
    outputs = np.asarray(request["outputs"], dtype=float)
    columns = [inputs[:, index] for index in range(inputs.shape[1])]

    def loss(params):
        y_pred = function(*columns, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)  # noqa: E731
    result = minimize(loss_partial, [1.0] * MAX_NPARAMS, method="BFGS")
    return {"x": [float(value) for value in result.x], "fun": float(result.fun)}


def _rows(function, inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
    """The program's output over fresh copies of some rows, one per row."""
    fresh = inputs.copy()
    value = np.asarray(
        function(*[fresh[:, index] for index in range(fresh.shape[1])],
                 params.copy()),
        dtype=float,
    )
    if value.shape not in {(), (1,), (len(fresh),)}:
        raise ValueError(
            f"the program returned shape {value.shape} for {len(fresh)} rows"
        )
    return np.broadcast_to(value, (len(fresh),))


def _evaluate(function, request: dict) -> dict:
    """Outputs on the whole table, and how much removing later rows changes them.

    Each cut keeps the rows before it. A program that reads only earlier rows
    gives the same output on them whether or not later rows are present. A
    cut the program cannot run on is recorded with its error.
    """
    inputs = np.asarray(request["inputs"], dtype=float)
    params = np.asarray(request["parameters"], dtype=float)
    whole = np.array(_rows(function, inputs, params))
    changes: list[float | None] = []
    errors: list[str | None] = []
    for cut in request["cuts"]:
        try:
            prefix = _rows(function, inputs[:cut], params)
        except Exception as exc:
            changes.append(None)
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        changes.append(float(np.max(np.abs(prefix - whole[:cut]))))
        errors.append(None)
    return {"outputs": whole.tolist(), "changes": changes, "errors": errors}


def _rollout(functions, request: dict) -> dict:
    """Step each trajectory along its grid with the trapezoid (Heun) rule.

    The programs see only the trajectory's rows so far: the simulated targets
    and the supplied channels. The output for the last row is the derivative
    there. A predictor step adds the next row, the derivative is taken again
    on it, and the two derivatives are averaged.
    """
    targets = list(request["target_columns"])
    params = [np.asarray(item, dtype=float) for item in request["parameters"]]
    step = float(request["step"])
    seconds = float(request["seconds_per_trajectory"])
    results = []
    for values in request["trajectories"]:
        history = np.array(values, dtype=float)
        deadline = time.monotonic() + seconds

        def slope(rows: int, history=history, deadline=deadline) -> np.ndarray:
            if time.monotonic() > deadline:
                raise TimeoutError(f"the rollout ran past {seconds:g} s")
            derivative = np.array([
                _rows(function, history[:rows], vector)[-1]
                for function, vector in zip(functions, params, strict=True)
            ])
            if not np.isfinite(derivative).all():
                raise FloatingPointError("the derivative is not finite")
            return derivative

        row = 0
        try:
            state = history[0, targets].copy()
            current = slope(1)
            for row in range(1, len(history)):
                history[row, targets] = state + step * current
                predicted = slope(row + 1)
                state = state + 0.5 * step * (current + predicted)
                if not np.isfinite(state).all():
                    raise FloatingPointError("the state is not finite")
                history[row, targets] = state
                current = slope(row + 1)
        except Exception as exc:
            results.append(
                {"ok": False, "row": row, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        results.append({"ok": True, "predictions": history[:, targets].tolist()})
    return {"trajectories": results}


def main() -> None:
    """Read one request, run it, write one reply; nothing else is printed."""
    reply_stream = sys.stdout
    request = json.loads(sys.stdin.read())
    memory_limited = _confine()
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - what a program prints
    started = time.monotonic()
    try:
        functions = [
            _load(source, name)
            for source, name in zip(
                request["sources"], request["names"], strict=True
            )
        ]
        mode = request["mode"]
        if mode == "refit":
            (function,) = functions
            reply = _refit(function, request)
        elif mode == "evaluate":
            (function,) = functions
            reply = _evaluate(function, request)
        elif mode == "rollout":
            reply = _rollout(functions, request)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    except BaseException as exc:
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    reply["memory_limited"] = memory_limited
    reply["seconds"] = round(time.monotonic() - started, 3)
    reply_stream.write(json.dumps(reply))
    reply_stream.flush()


if __name__ == "__main__":
    main()
