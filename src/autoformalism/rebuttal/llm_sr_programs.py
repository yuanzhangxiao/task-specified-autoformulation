"""Run LLM-SR's selected program where no equation of the current values can.

LLM-SR evolves Python functions. Given its rows in time order, the programs it
ranks highest read along them: in the budget pilot, a filter of the meal
series that stands in for the unobserved gut. Such a function of the history
has no counterpart in the restricted grammar the shared evaluator reads, so a
Phase C plan may declare (``program_rollout``) that LLM-SR's pick is run as
the program it is:

- Its syntax tree is checked against an allowlist before anything runs:
  arithmetic, control flow, helper functions, and listed numpy, math and scipy
  names. Anything else is refused with a named reason.
- It then runs in a separate process (``llm_sr_program_worker``) that can
  import nothing else, write no file, and is killed past its time limit.
- Its coefficients are refitted with their evaluator's own call, BFGS from all
  ones on the same table, which their evaluator ran and discarded.
- Each trajectory is rolled out along its observation grid with the trapezoid
  (Heun) rule. The program is given only that trajectory's rows so far: the
  simulated targets and the supplied channels. A rollout cannot see later rows
  or another trajectory, whatever the program does.
- Whether the program's output depends on later rows of its table is measured
  and recorded. It chooses nothing.

Executing proposer-written text is otherwise forbidden here (AGENTS.md). This
is the declared exception for this baseline, agreed on 2026-10-09: LLM-SR's own
search executes every program it scores, the selected program has run there
already, and the allowlist and the process are the guards.
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from autoformalism.data import DatasetSplit, SplitName, TrainingScaler, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.llm_sr_upstream import MAX_NPARAMS, InexpressibleProgram

#: The script that runs a program, launched by path with the same interpreter.
WORKER = Path(__file__).with_name("llm_sr_program_worker.py")

#: How the sealed selection names this execution, for whoever scores it.
EXECUTION = "llm_sr_program_grid_rollout"

#: Removing later rows may change earlier outputs by rounding, as an FFT of a
#: different length does; past this fraction of the largest output it is read.
LOOK_AHEAD_TOLERANCE = 1e-9

#: Bounds on a program's text, checked before it is parsed further.
MAX_SOURCE_CHARACTERS = 100_000
MAX_NODES = 20_000

#: Constructs a program may use: statements, expressions and operators.
#: Classes, generators' ``yield``, ``async``, ``match`` and decorators are not.
ALLOWED_NODES: frozenset[type[ast.AST]] = frozenset({
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.While, ast.If,
    ast.Break, ast.Continue, ast.Pass, ast.Expr, ast.Assert, ast.Delete,
    ast.Try, ast.ExceptHandler, ast.Raise, ast.With, ast.withitem, ast.Import,
    ast.ImportFrom, ast.alias, ast.Global, ast.Nonlocal,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
    ast.keyword, ast.Attribute, ast.Subscript, ast.Slice, ast.Starred,
    ast.Name, ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.comprehension, ast.Lambda, ast.NamedExpr, ast.JoinedStr,
    ast.FormattedValue, ast.Load, ast.Store, ast.Del,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.MatMult, ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.UAdd, ast.USub, ast.Not, ast.Invert, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn,
})

#: Modules a program may import; the worker binds these and nothing else.
IMPORTABLE = frozenset({
    "numpy", "math", "scipy", "scipy.integrate", "scipy.interpolate",
    "scipy.ndimage", "scipy.signal", "scipy.special",
})

#: numpy's computational names: constants, types, array creation and shaping,
#: elementwise math, reductions, sorting, interpolation, convolution, FFT and
#: linear algebra. File input and output, random numbers, ctypes and numpy's
#: internals are not listed.
_NUMPY = """
pi e inf nan newaxis euler_gamma float64 float32 float16 double single
longdouble int64 int32 int16 int8 uint8 intp int_ bool_ complex128 ndarray
dtype generic number floating integer finfo iinfo errstate seterr geterr
array asarray asanyarray ascontiguousarray asfortranarray atleast_1d
atleast_2d atleast_3d copy zeros ones empty full zeros_like ones_like
empty_like full_like eye identity arange linspace logspace geomspace meshgrid
indices fromiter concatenate stack hstack vstack dstack column_stack row_stack
append insert delete pad repeat tile roll flip fliplr flipud reshape ravel
squeeze expand_dims transpose swapaxes moveaxis broadcast_to broadcast_arrays
split array_split hsplit vsplit take take_along_axis put put_along_axis place
putmask resize trim_zeros tril triu diag diagonal diagflat trace rot90
abs absolute fabs add subtract multiply divide true_divide floor_divide
negative positive power float_power mod fmod remainder divmod reciprocal sign
sqrt cbrt square exp exp2 expm1 log log2 log10 log1p logaddexp logaddexp2 sin
cos tan arcsin arccos arctan arctan2 asin acos atan atan2 hypot sinh cosh
tanh arcsinh arccosh arctanh asinh acosh atanh deg2rad rad2deg degrees
radians floor ceil trunc rint round around fix maximum minimum fmax fmin clip
heaviside sinc i0 nan_to_num real imag conj conjugate angle isnan isinf
isfinite isneginf isposinf isreal iscomplex signbit copysign nextafter
spacing ldexp frexp modf gcd lcm logical_and logical_or logical_not
logical_xor greater greater_equal less less_equal equal not_equal isclose
allclose array_equal array_equiv where select piecewise choose any all sum
nansum prod nanprod cumsum cumprod nancumsum nancumprod mean nanmean average
median nanmedian std nanstd var nanvar min max amin amax nanmin nanmax ptp
percentile nanpercentile quantile nanquantile argmin argmax nanargmin
nanargmax argsort sort searchsorted digitize histogram histogram_bin_edges
bincount count_nonzero nonzero flatnonzero argwhere unique diff ediff1d
gradient convolve correlate interp trapz trapezoid cross dot vdot inner outer
matmul tensordot einsum kron polyval polyfit polyder polyint roots vectorize
apply_along_axis isscalar ndim shape size result_type cov corrcoef unwrap
fft linalg
"""
_NUMPY_FFT = """
ifft rfft irfft fftfreq rfftfreq fftshift ifftshift hfft ihfft fft2 ifft2
fftn ifftn rfftn irfftn
"""
_NUMPY_LINALG = """
solve lstsq norm inv pinv det slogdet eig eigh eigvals eigvalsh cholesky qr
svd matrix_power matrix_rank cond multi_dot
"""
#: Attributes and methods of arrays, of ``finfo``, of Python numbers,
#: containers and exceptions, and of an ODE solution.
_OBJECTS = """
T flat itemsize nbytes astype flatten tolist item fill eps tiny resolution
smallest_normal epsneg precision bits get items keys values update setdefault
extend pop count index clear remove reverse bit_length is_integer args y t
success message status
"""
_MATH = """
acos acosh asin asinh atan atan2 atanh ceil comb copysign cos cosh degrees
dist erf erfc exp expm1 fabs factorial floor fmod frexp fsum gamma gcd hypot
isclose isfinite isinf isnan isqrt lcm ldexp lgamma log log10 log1p log2 modf
nextafter perm pow prod radians remainder sin sinh sqrt tan tanh tau trunc ulp
"""
_SCIPY = """
integrate interpolate ndimage signal special
lfilter lfilter_zi lfiltic filtfilt sosfilt sosfilt_zi sosfiltfilt butter
cheby1 cheby2 bessel ellip fftconvolve oaconvolve savgol_filter savgol_coeffs
medfilt detrend lsim dlsim cont2discrete bilinear tf2sos zpk2sos tf2zpk
zpk2tf get_window windows gaussian exponential hann hamming resample decimate
upfirdn firwin freqz impulse step
expit logit erfinv erfcinv gammaln gammainc gammaincc beta betaln betainc
digamma softmax log_softmax xlogy xlog1py logsumexp exp1 expi expn lambertw
binom exp10 i1 iv kv jv yv hyp1f1 hyp2f1 zeta boxcox inv_boxcox ndtr ndtri
log_ndtr dawsn huber pseudo_huber rel_entr entr kl_div
odeint solve_ivp cumulative_trapezoid cumtrapz simpson simps quad romb
interp1d CubicSpline PchipInterpolator Akima1DInterpolator UnivariateSpline
InterpolatedUnivariateSpline make_interp_spline BSpline splrep splev
pchip_interpolate lagrange derivative antiderivative
uniform_filter1d gaussian_filter1d convolve1d correlate1d shift uniform_filter
gaussian_filter median_filter maximum_filter1d minimum_filter1d
"""
#: Every attribute name a program may read or call, on any object. Names are
#: listed rather than refused, so that no path through an allowed object
#: reaches a frame, a module's globals or the interpreter.
ALLOWED_ATTRIBUTES: frozenset[str] = frozenset(
    " ".join((_NUMPY, _NUMPY_FFT, _NUMPY_LINALG, _OBJECTS, _MATH, _SCIPY)).split()
)

#: Builtins refused by name, though never bound: they read attributes by
#: name, compile text or open files.
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "getattr", "setattr", "delattr", "vars",
    "breakpoint", "open",
})


class ProgramRefused(InexpressibleProgram):
    """A program outside the allowlist, or one that failed to run."""


@dataclass(frozen=True)
class CheckedProgram:
    """A selected program that passed the allowlist, ready to run."""

    source: str
    name: str
    variables: tuple[str, ...]


def _refuse(reason: str, node: ast.AST | None = None) -> ProgramRefused:
    line = getattr(node, "lineno", None)
    return ProgramRefused(reason, f"line {line}" if line else "")


def check_program(source: str, variables: Sequence[str]) -> CheckedProgram:
    """Check one recorded program against the allowlist; nothing is executed.

    ``source`` is the function as LLM-SR's profiler recorded it, and
    ``variables`` the argument names its specification gave, in order; the
    function must take exactly those and ``params``.
    """
    if len(source) > MAX_SOURCE_CHARACTERS:
        raise ProgramRefused("the program is too long", f"{len(source)} characters")
    try:
        module = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise ProgramRefused("the program does not parse", str(exc)) from exc
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise ProgramRefused("the record is not one function definition")
    function = module.body[0]
    if function.decorator_list:
        raise _refuse("a decorator", function)
    expected = [*variables, "params"]
    arguments = function.args
    if (
        [item.arg for item in arguments.args] != expected
        or arguments.posonlyargs
        or arguments.kwonlyargs
        or arguments.vararg
        or arguments.kwarg
    ):
        raise ProgramRefused(
            "the signature differs from the specification's",
            ", ".join(item.arg for item in arguments.args),
        )
    nodes = list(ast.walk(module))
    if len(nodes) > MAX_NODES:
        raise ProgramRefused("the program is too large", f"{len(nodes)} nodes")
    for node in nodes:
        _check_node(node)
    return CheckedProgram(
        source=textwrap.dedent(source), name=function.name, variables=tuple(variables)
    )


def _check_node(node: ast.AST) -> None:
    """Refuse one construct outside the allowlist, naming it."""
    if type(node) not in ALLOWED_NODES:
        raise _refuse(f"{type(node).__name__} is not allowed", node)
    for name in _bound_names(node):
        if name.startswith("__"):
            raise _refuse(f"the name {name} is not allowed", node)
    if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
        raise _refuse(f"{node.id} is not allowed", node)
    if isinstance(node, ast.Attribute) and (
        node.attr.startswith("_") or node.attr not in ALLOWED_ATTRIBUTES
    ):
        raise _refuse(f"the attribute {node.attr} is not allowed", node)
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name not in IMPORTABLE:
                raise _refuse(f"import {alias.name} is not allowed", node)
    if isinstance(node, ast.ImportFrom):
        if node.level or node.module not in IMPORTABLE:
            raise _refuse(f"import from {node.module} is not allowed", node)
        for alias in node.names:
            if alias.name not in ALLOWED_ATTRIBUTES and (
                f"{node.module}.{alias.name}" not in IMPORTABLE
            ):
                raise _refuse(
                    f"import {alias.name} from {node.module} is not allowed", node
                )
    if isinstance(node, ast.With):
        for item in node.items:
            context = item.context_expr
            if not (
                isinstance(context, ast.Call)
                and isinstance(context.func, ast.Attribute)
                and context.func.attr == "errstate"
            ):
                raise _refuse("with is allowed only for numpy.errstate", node)


def _bound_names(node: ast.AST) -> tuple[str, ...]:
    """Every identifier a node names or binds."""
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.FunctionDef, ast.alias)):
        return tuple(
            item for item in (node.name, getattr(node, "asname", None)) if item
        )
    if isinstance(node, ast.arg):
        return (node.arg,)
    if isinstance(node, ast.keyword):
        return (node.arg,) if node.arg else ()
    if isinstance(node, ast.ExceptHandler):
        return (node.name,) if node.name else ()
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return tuple(node.names)
    return ()


# --- running a program in its own process ----------------------------------


def _environment() -> dict[str, str]:
    """Only what the worker needs to import numpy and scipy, and no secrets."""
    import scipy

    sites = dict.fromkeys(
        str(Path(module.__file__).resolve().parents[1]) for module in (np, scipy)
    )
    threads = dict.fromkeys(
        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
         "VECLIB_MAXIMUM_THREADS"),
        "1",
    )
    return {
        "PATH": os.defpath,
        "PYTHONPATH": os.pathsep.join(sites),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        **threads,
    }


def run_worker(request: dict, *, seconds: float) -> dict:
    """Send one request to a fresh worker process and read its reply.

    The process is killed once ``seconds`` have passed. A reply is plain JSON;
    an error the program raised comes back as ``ProgramRefused``.
    """
    with tempfile.TemporaryDirectory(prefix="llm-sr-program-") as directory:
        try:
            completed = subprocess.run(
                # No user site, no bytecode, and not the script's own
                # directory on the path: only numpy and scipy's sites.
                [sys.executable, "-s", "-B", "-P", str(WORKER)],
                input=json.dumps(request).encode("utf-8"),
                capture_output=True,
                timeout=seconds,
                cwd=directory,
                env=_environment(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ProgramRefused(
                "the program ran past its time limit", f"{seconds:g} s"
            ) from None
    try:
        reply = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        reply = None
    if not isinstance(reply, dict):
        lines = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ProgramRefused(
            "the program's process ended without a reply",
            f"exit status {completed.returncode}"
            + (f"; {lines[-1][:300]}" if lines else ""),
        )
    if "error" in reply:
        raise ProgramRefused("the program raised", str(reply["error"])[:500])
    return reply


@dataclass(frozen=True)
class Refit:
    """The coefficients their evaluator's fit finds, and its loss."""

    parameters: tuple[float, ...]
    loss: float


def refit(
    program: CheckedProgram,
    values: NDArray[np.float64],
    label: NDArray[np.float64],
    *,
    seconds: float = 600.0,
) -> Refit:
    """Repeat the fit their specification runs and discards, by running it.

    BFGS from all ones over all ``MAX_NPARAMS`` coefficients on the same rows,
    in the same call, so the result is the one behind LLM-SR's score. Their
    limit is 30 seconds; this one is wider, as the machine may be busier.
    """
    reply = run_worker(
        {
            "mode": "refit",
            "sources": [program.source],
            "names": [program.name],
            "inputs": np.asarray(values, dtype=float).tolist(),
            "outputs": np.asarray(label, dtype=float).tolist(),
        },
        seconds=seconds,
    )
    parameters = tuple(float(value) for value in reply["x"])
    loss = float(reply["fun"])
    if len(parameters) != MAX_NPARAMS or not math.isfinite(loss):
        raise ProgramRefused("the coefficient refit did not converge", f"loss {loss}")
    return Refit(parameters, loss)


def look_ahead_cuts(rows_per_trajectory: Sequence[int]) -> tuple[int, ...]:
    """Where to cut the table: mid-way through each trajectory and after it.

    The table holds the trajectories one after another, so a cut after a
    trajectory removes the next ones, and a cut mid-way removes its own later
    rows as well.
    """
    total = sum(rows_per_trajectory)
    cuts: set[int] = set()
    start = 0
    for rows in rows_per_trajectory:
        cuts.update({start + rows // 2, start + rows})
        start += rows
    return tuple(sorted(cut for cut in cuts if 0 < cut < total))


def training_outputs(
    program: CheckedProgram,
    parameters: Sequence[float],
    values: NDArray[np.float64],
    label: NDArray[np.float64],
    rows_per_trajectory: Sequence[int],
    *,
    seconds: float = 300.0,
) -> tuple[NDArray[np.float64], float, dict]:
    """The refitted program's outputs and error, and whether it reads later rows.

    The look-ahead record states, over every cut, the largest change in the
    output on the rows before the cut, as a fraction of the largest output.
    It is recorded, not acted on.
    """
    if sum(rows_per_trajectory) != len(values):
        raise ValueError("the trajectories' rows do not add up to the table")
    cuts = look_ahead_cuts(rows_per_trajectory)
    reply = run_worker(
        {
            "mode": "evaluate",
            "sources": [program.source],
            "names": [program.name],
            "parameters": list(parameters),
            "inputs": np.asarray(values, dtype=float).tolist(),
            "cuts": list(cuts),
        },
        seconds=seconds,
    )
    outputs = np.asarray(reply["outputs"], dtype=float)
    with np.errstate(all="ignore"):
        error = float(np.mean((outputs - np.asarray(label, dtype=float)) ** 2))
    largest = float(np.max(np.abs(outputs))) if outputs.size else 0.0
    scale = largest if math.isfinite(largest) and largest > 0.0 else 1.0
    changes = [
        None if change is None else change / scale for change in reply["changes"]
    ]
    measured = [(change, cut) for change, cut in zip(changes, cuts, strict=True)
                if change is not None]
    worst, at = max(measured, default=(0.0, None))
    failed = [
        {"cut": cut, "error": message}
        for cut, message in zip(cuts, reply["errors"], strict=True)
        if message is not None
    ]
    return outputs, error, {
        "cuts": len(cuts),
        "largest_relative_change": worst,
        "at_cut": at,
        "reads_later_rows": bool(worst > LOOK_AHEAD_TOLERANCE),
        "tolerance": LOOK_AHEAD_TOLERANCE,
        "cuts_that_failed": failed[:5],
        "cuts_that_failed_count": len(failed),
    }


# --- rolling out along the observation grid --------------------------------


def observation_step(split: DatasetSplit) -> float:
    """The one sampling interval shared by every trajectory of a split.

    A program written for rows means the same thing on new rows only if they
    are as far apart, so a split whose grid is not uniform is refused.
    """
    steps: list[float] = []
    for trajectory in split.trajectories:
        differences = np.diff(np.asarray(trajectory.time, dtype=float))
        if differences.size == 0 or not np.allclose(
            differences, differences[0], rtol=1e-6, atol=0.0
        ):
            raise ValueError(
                f"trajectory {trajectory.trajectory_id} is not on a uniform grid"
            )
        steps.append(float(np.mean(differences)))
    if not steps or not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        raise ValueError(f"the {split.name.value} trajectories have different steps")
    return steps[0]


def same_step(first: float, second: float) -> bool:
    """Whether two grids are equally spaced, to rounding."""
    return math.isclose(first, second, rel_tol=1e-6, abs_tol=0.0)


def trajectory_rows(
    trajectory: Trajectory, channels: Sequence[str], targets: Sequence[str]
) -> NDArray[np.float64]:
    """One trajectory's rows over the channels, as the training table has them.

    Targets are known only at the first row; later ones are left undefined,
    so that a rollout that read one by mistake would fail rather than leak.
    """
    rows = len(trajectory.time)
    columns = []
    for name in channels:
        for mapping in (
            trajectory.targets, trajectory.auxiliaries, trajectory.external_inputs
        ):
            if name in mapping:
                columns.append(np.asarray(mapping[name], dtype=float))
                break
        else:
            if name not in trajectory.fixed_covariates:
                raise ValueError(f"trajectory is missing channel {name}")
            columns.append(np.full(rows, float(trajectory.fixed_covariates[name])))
    table = np.column_stack(columns)
    table[1:, [list(channels).index(target) for target in targets]] = np.nan
    return table


@dataclass(frozen=True)
class ProgramRolloutScore:
    """The normalized rollout error, or why there is none."""

    error: float | None
    failure: str | None = None


def program_rollout_error(
    programs: Mapping[str, tuple[CheckedProgram, Sequence[float]]],
    channels: Sequence[str],
    context: ValidationContext,
    train: DatasetSplit,
    evaluate: DatasetSplit,
    *,
    step: float,
    seconds: float = 60.0,
) -> ProgramRolloutScore:
    """The shared evaluator's rollout error, for programs on the grid.

    The same quantity as ``llm_ode_driver.development_rollout_error``: per
    trajectory, the mean over targets of the mean squared rollout error scaled
    by the training standard deviation, averaged over trajectories. Only TRAIN
    and VALIDATION are accepted. ``seconds`` bounds each trajectory, as there.
    """
    if train.name is not SplitName.TRAIN or evaluate.name not in {
        SplitName.TRAIN,
        SplitName.VALIDATION,
    }:
        raise ValueError("selection requires TRAIN and VALIDATION, never TEST")
    if set(programs) != set(context.targets):
        raise ValueError("one program per target is required")
    found = observation_step(evaluate)
    if not same_step(found, step):
        raise ValueError(
            f"the {evaluate.name.value} grid steps {found:g}, "
            f"the training grid {step:g}"
        )
    scaling = TrainingScaler().fit(train).scales
    scales = {
        target: float(scaling[f"target:{target}"].standard_deviation)
        for target in context.targets
    }
    ordered = [programs[target] for target in context.targets]
    request = {
        "mode": "rollout",
        "sources": [program.source for program, _ in ordered],
        "names": [program.name for program, _ in ordered],
        "parameters": [list(parameters) for _, parameters in ordered],
        "trajectories": [
            trajectory_rows(trajectory, channels, context.targets).tolist()
            for trajectory in evaluate.trajectories
        ],
        "target_columns": [list(channels).index(target) for target in context.targets],
        "step": step,
        "seconds_per_trajectory": seconds,
    }
    try:
        reply = run_worker(
            request, seconds=seconds * len(evaluate.trajectories) + 60.0
        )
    except ProgramRefused as exc:
        return ProgramRolloutScore(None, str(exc))
    squared: list[float] = []
    for trajectory, result in zip(
        evaluate.trajectories, reply["trajectories"], strict=True
    ):
        if not result["ok"]:
            return ProgramRolloutScore(
                None,
                f"{trajectory.trajectory_id}, row {result['row']}: {result['reason']}",
            )
        predictions = np.asarray(result["predictions"], dtype=float)
        errors = [
            np.square(
                (predictions[:, column] - trajectory.targets[target]) / scales[target]
            )
            for column, target in enumerate(context.targets)
        ]
        if not all(np.isfinite(value).all() for value in errors):
            return ProgramRolloutScore(
                None, f"{trajectory.trajectory_id}: the rollout error is not finite"
            )
        squared.append(float(np.mean([np.mean(value) for value in errors])))
    return ProgramRolloutScore(float(np.mean(squared)) if squared else None)
