"""Opt-in private coefficient aggregation and multi-output affine profiling.

The restricted polynomial certificate preserves the original model and domains.
Projection is conditional on the state matrix, not a global-optimum certificate.
All observations, channel weights and scoped initializers come from training.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from time import monotonic

import casadi as ca
import numpy as np
from scipy.optimize import lsq_linear

from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_gain_reduction as gains
from autoformalism.fitting import prefix_structural_reduction as structural
from autoformalism.fitting.affine_precision import TransitionEvaluator
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.fitting.simulation import trajectory_forcing

PROTOCOL = "prefix-private-multioutput-profile-1"
RANK_TOLERANCE = 1e-12
KKT_TOLERANCE = 1e-10
FACE_TOLERANCE = 1e-9


def policy_receipt() -> dict:
    """Freeze the deliberately restricted proof and numerical projection policy."""
    return {
        "protocol": PROTOCOL,
        "aggregation": "private-Laurent-coefficient-with-domain-surjective-section",
        "profiling": "joint-affine-forcing-and-initials-fixed-state-matrix",
        "readouts": "multiple-parameter-free-identity-states",
        "bounds_preserved": True,
        "rank_tolerance": RANK_TOLERANCE,
        "kkt_tolerance": KKT_TOLERANCE,
        "active_face_tolerance": FACE_TOLERANCE,
        "linear_solve": "column-normalized-BVLS-tol1e-12-maxiter100",
        "derivative": "exact-fixed-strict-active-face; unavailable-at-ambiguity",
        "training_mask": "all-supplied-training-samples-and-targets",
        "reference_coefficients_used": False,
        "global_minimum_certified": False,
    }


class ProfileUnavailable(NumericalInvalidTrial):
    """A numerical projection cannot supply the requested qualified derivative."""

    def __init__(self, message: str, audit: dict):
        super().__init__(message)
        self.audit = deepcopy(audit)


def _check(deadline: float) -> None:
    if not np.isfinite(deadline):
        raise ValueError("finite profile deadline required")
    if monotonic() >= deadline:
        raise TimeoutError("multi-output profile deadline reached")


def _finite(value, description):
    value = np.asarray(value, float)
    if not np.isfinite(value).all():
        raise NumericalInvalidTrial(f"nonfinite profile {description}")
    return value


def _symbols(polynomial):
    return {name for term in polynomial for name, _ in term}


def _value(polynomial, values):
    result = 0
    for powers, coefficient in polynomial.items():
        term = float(coefficient)
        for name, degree in powers:
            term *= values[name] ** degree
        result += term
    return result


def _serialized(polynomial):
    return [
        {"powers": dict(powers), "coefficient": str(coefficient)}
        for powers, coefficient in sorted(polynomial.items())
    ]


def _base_symbolic_lift(base, vector):
    """Compose the previously certified gain, inactive and private-decay charts."""
    active = base.base
    values = dict(zip(base.names, ca.vertsplit(vector), strict=True))
    values.update({n: float(base.anchor[active.names.index(n)]) for n in base.inactive})
    for decay in base.decays:
        rate = values[decay.coordinate]
        product = decay.tau_lower * rate
        values[decay.tau] = decay.tau_lower + 1 / rate
        values[decay.divisor] = 1.0
        values[decay.gain] = rate * product / (1 + product)
    for i, name in enumerate(active.gain_reduction.names):
        if name not in active.names:
            values[name] = float(active.anchor[i])
    values.update({p.gain: 1.0 for p in active.gain_reduction.products})
    return ca.vertcat(*[values[n] for n in active.gain_reduction.original_names])


def _coefficient_slots(original, base):
    """Expand restricted definitions, then apply existing exact chart identities."""
    trees = gains._expressions(original)
    processes = {
        key[8:]: value for key, value in trees.items() if key.startswith("process:")
    }
    candidate = original.system.model.validated.candidate
    if candidate.constraints or original.problem.base_candidate.constraints:
        raise ValueError("explicit constraints are unsupported by private profiling")
    if any(p.bounds is not None for p in candidate.parameters):
        raise ValueError(
            "explicit parameter bounds are unsupported by private profiling"
        )
    positive = {p.name for p in candidate.parameters if p.domain.value == "positive"}
    if not all(
        structural._safe_definition(tree, processes, positive, set(), {})
        for tree in trees.values()
    ):
        raise ValueError("retained definition domain is not certified")
    substitutions = {p.gain: Fraction(1) for p in base.base.gain_reduction.products}
    substitutions.update(
        {n: Fraction(str(base.anchor[base.base.names.index(n)])) for n in base.inactive}
    )
    features = set(original.system.model.state_names) | set(original.system.inputs)
    if "constant" in features:
        raise ValueError("reserved constant feature name is unsupported by profiling")
    slots = {}
    for equation in candidate.state_equations:
        polynomial = structural._polynomial(
            trees[f"rhs:{equation.state}"], processes, set(original.system.names)
        )
        substituted = {}
        for powers, coefficient in polynomial.items():
            remaining = []
            for name, degree in powers:
                if name in substitutions:
                    coefficient *= substitutions[name] ** degree
                else:
                    remaining.append((name, degree))
            substituted = structural._sum(substituted, {tuple(remaining): coefficient})
        for decay in base.decays:
            if decay.state != equation.state:
                continue
            terms = (
                tuple(sorted(((decay.state, 1), (decay.tau, -1)))),
                tuple(sorted(((decay.state, 1), (decay.gain, 1), (decay.divisor, -1)))),
            )
            if any(substituted.get(term) != -1 for term in terms):
                raise ValueError(
                    "private-decay certificate no longer matches equations"
                )
            for term in terms:
                del substituted[term]
            substituted[tuple(sorted(((decay.state, 1), (decay.coordinate, 1))))] = (
                Fraction(-1)
            )
        for powers, coefficient in substituted.items():
            dynamic = [(name, degree) for name, degree in powers if name in features]
            if len(dynamic) > 1 or (dynamic and dynamic[0][1] != 1):
                raise ValueError(
                    "profiling requires jointly affine state/input dynamics"
                )
            feature = dynamic[0][0] if dynamic else "constant"
            parameter_powers = tuple((n, d) for n, d in powers if n not in features)
            if set(dict(parameter_powers)) - set(base.names):
                raise ValueError("unsupported time or unbound coefficient dependency")
            key = equation.state, feature
            slots[key] = structural._sum(
                slots.get(key, {}), {parameter_powers: coefficient}
            )
    return slots


@dataclass(frozen=True)
class PrivateCoefficient:
    """One coefficient with a constant affine section in the previous chart."""

    coordinate: str
    state: str
    feature: str
    polynomial: dict
    pivot: str
    slope: float
    intercept: float
    frozen: dict[str, float]
    lower: float
    upper: float


class AggregatedReduction:
    """A domain-preserving private coefficient quotient, composed with old lifts."""

    def __init__(self, oracle, retained):
        self.base = oracle.reduction
        original = oracle.original
        self.anchor = self.base.reduce(original.vector(retained))
        self.slots = _coefficient_slots(original, self.base)
        symbolic = ca.SX.sym("base_coordinates", len(self.base.names))
        lifted = _base_symbolic_lift(self.base, symbolic)
        initial = ca.vertcat(
            *[
                original.system.initial_symbolic(
                    row, lifted[original.indices[row.trajectory_id].tolist()]
                )
                for row in original.rows
            ]
        )
        system = original.system
        observation = system.observe(
            ca.SX.sym("private_time"),
            ca.SX.sym("private_state", system.state_count),
            lifted[original.indices[original.rows[0].trajectory_id].tolist()],
            ca.SX.sym("private_input", len(system.inputs)),
        )
        open_rates = {decay.coordinate for decay in self.base.decays}
        consumers = {name: [] for name in self.base.names}
        for key, polynomial in self.slots.items():
            for name in _symbols(polynomial):
                consumers[name].append(key)
        self.blocks = []
        for (state, feature), polynomial in sorted(self.slots.items()):
            members = sorted(_symbols(polynomial))
            if not members:
                continue
            if any(len(consumers[n]) != 1 for n in members):
                raise ValueError("shared coefficient consumer is unsupported")
            if any(
                ca.depends_on(initial, symbolic[self.base.names.index(n)])
                for n in members
            ):
                raise ValueError("dynamic coefficient also used by an initializer")
            if any(
                ca.depends_on(observation, symbolic[self.base.names.index(n)])
                for n in members
            ):
                raise ValueError("dynamic coefficient also used by an observation")
            if len(members) == 1 and len(polynomial) == 1:
                continue
            bounds = {
                n: (
                    self.base.lower[self.base.names.index(n)],
                    self.base.upper[self.base.names.index(n)],
                )
                for n in members
            }
            # A private, unrestricted, standalone real term spans the entire real line.
            pivots = [
                n
                for n in members
                if bounds[n] == (-np.inf, np.inf)
                and ((n, 1),) in polynomial
                and sum(n in dict(term) for term in polynomial) == 1
            ]
            frozen = {
                n: (
                    0.0
                    if bounds[n][0] <= 0 <= bounds[n][1] and n not in open_rates
                    else 1.0
                )
                for n in members
            }
            if pivots:
                pivot = pivots[0]
                lower, upper = -np.inf, np.inf
            elif (
                not set(members) & open_rates
                and all(bounds[n] == (0, np.inf) for n in members)
                and all(
                    not term or (len(term) == 1 and term[0][1] == 1)
                    for term in polynomial
                )
            ):
                signs = {np.sign(float(polynomial[((n, 1),)])) for n in members}
                if len(signs) != 1:
                    raise ValueError(
                        "mixed signed bounded coefficient sum is unsupported"
                    )
                pivot = members[0]
                offset = float(polynomial.get((), 0))
                lower, upper = (
                    (offset, np.inf) if next(iter(signs)) > 0 else (-np.inf, offset)
                )
            else:
                raise ValueError(
                    "private coefficient has no certified full-domain section"
                )
            if any(
                not bounds[n][0] <= value <= bounds[n][1] for n, value in frozen.items()
            ):
                raise ValueError(
                    "canonical coefficient section violates original bounds"
                )
            slope = float(polynomial[((pivot, 1),)])
            intercept = float(_value(polynomial, frozen))
            del frozen[pivot]
            self.blocks.append(
                PrivateCoefficient(
                    f"coefficient:{state}:{feature}",
                    state,
                    feature,
                    polynomial,
                    pivot,
                    slope,
                    intercept,
                    frozen,
                    lower,
                    upper,
                )
            )
        replaced = {block.pivot: block.coordinate for block in self.blocks}
        removed = {name for block in self.blocks for name in block.frozen}
        self.names = tuple(
            replaced.get(n, n) for n in self.base.names if n not in removed
        )
        blocks = {block.coordinate: block for block in self.blocks}
        self.lower = np.asarray(
            [
                blocks[n].lower
                if n in blocks
                else self.base.lower[self.base.names.index(n)]
                for n in self.names
            ]
        )
        self.upper = np.asarray(
            [
                blocks[n].upper
                if n in blocks
                else self.base.upper[self.base.names.index(n)]
                for n in self.names
            ]
        )
        self._section_jacobian = np.zeros((len(self.base.names), len(self.names)))
        for i, name in enumerate(self.names):
            pivot, slope = (
                (blocks[name].pivot, blocks[name].slope)
                if name in blocks
                else (name, 1)
            )
            self._section_jacobian[self.base.names.index(pivot), i] = 1 / slope
        self.source_sha256 = gains._source_identity(original)

    @property
    def receipt(self):
        record = {
            "protocol": PROTOCOL,
            "source_sha256": self.source_sha256,
            "base_reduction": self.base.receipt,
            "effective_names": list(self.names),
            "private_coefficients": [
                {
                    "coordinate": b.coordinate,
                    "state": b.state,
                    "feature": b.feature,
                    "polynomial": _serialized(b.polynomial),
                    "pivot": b.pivot,
                    "slope": b.slope,
                    "intercept": b.intercept,
                    "frozen": b.frozen,
                    "domain": [gains._bound(b.lower), gains._bound(b.upper)],
                }
                for b in self.blocks
            ],
            "proof": (
                "exact-private-coefficient-surjection; "
                "unchanged-initializers-and-readouts"
            ),
            "original_model_unchanged": True,
            "original_parameters_identified": False,
        }
        return record | {"sha256": content_sha256(record)}

    def _section(self, vector):
        values = dict(zip(self.names, vector, strict=True))
        for block in self.blocks:
            values.update(block.frozen)
            values[block.pivot] = (
                values[block.coordinate] - block.intercept
            ) / block.slope
        return [values[name] for name in self.base.names]

    def vector(self, parameters):
        vector = gains._checked_vector(
            gains._mapping_vector(parameters, self.names),
            self.names,
            self.lower,
            self.upper,
        )
        # Hashable admissible vectors need not have a representable floating lift.
        for decay in self.base.decays:
            if (
                decay.coordinate in self.names
                and vector[self.names.index(decay.coordinate)] <= 0
            ):
                raise ValueError("effective decay requires strictly positive rate")
        return vector

    def _numeric_section(self, vector):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            section = _finite(self._section(vector), "coefficient section")
        return self.base._checked(section)

    def reduce(self, original_vector):
        values = dict(
            zip(self.base.names, self.base.reduce(original_vector), strict=True)
        )
        values.update(
            {b.coordinate: float(_value(b.polynomial, values)) for b in self.blocks}
        )
        return self.vector({name: values[name] for name in self.names})

    def lift(self, vector):
        checked = self.vector(dict(zip(self.names, vector, strict=True)))
        return self.base.lift(self._numeric_section(checked))

    def lift_jacobian(self, vector):
        checked = self.vector(dict(zip(self.names, vector, strict=True)))
        return (
            self.base.lift_jacobian(self._numeric_section(checked))
            @ self._section_jacobian
        )

    def symbolic_lift(self, vector):
        return _base_symbolic_lift(
            self.base, ca.vertcat(*self._section(ca.vertsplit(vector)))
        )

    def symbolic_rhs(self, vector, states, inputs):
        values = dict(
            zip(self.base.names, self._section(ca.vertsplit(vector)), strict=True)
        )
        features = states | inputs | {"constant": 1.0}
        blocks = {
            (b.state, b.feature): vector[self.names.index(b.coordinate)]
            for b in self.blocks
        }
        return ca.vertcat(
            *[
                sum(
                    (blocks[key] if key in blocks else _value(poly, values))
                    * features[key[1]]
                    for key, poly in self.slots.items()
                    if key[0] == state
                )
                for state in states
            ]
        )


class AggregatedTrainingOracle:
    """Evaluate unchanged original equations and compose their exact lift Jacobian."""

    def __init__(self, oracle, retained):
        self.base_oracle, self.original = oracle, oracle.original
        self.precision_mode = oracle.precision_mode
        self.reduction = AggregatedReduction(oracle, retained)
        self.names, self.lower, self.upper = (
            self.reduction.names,
            self.reduction.lower,
            self.reduction.upper,
        )
        self.start = self.parameters(
            self.reduction.reduce(self.original.vector(retained))
        )
        self.units = np.asarray([max(abs(value), 1.0) for value in self.start.values()])
        self.sensitive_names = tuple(
            n for n in oracle.sensitive_names if n in self.names
        )
        self._local_probe_floors: dict[str, float] | None = None

    @property
    def recovery_audit(self):
        audit = self.base_oracle.recovery_audit | {
            "multioutput_profile_policy": policy_receipt(),
            "coordinate_reduction": self.reduction.receipt,
            "optimizer_coordinates": len(self.names),
            "sensitive_names": list(self.sensitive_names),
            "optimizer_units": "max(abs(retained-effective-coordinate),1); frozen",
        }
        if self._local_probe_floors is not None:
            audit["local_probe_policy"] = {
                "protocol": "current-coordinate-local-probes-1",
                "floors": dict(self._local_probe_floors),
                "scale": "max(abs(current_coordinate), declared_floor)",
                "reciprocal_coordinates": "also capped by nonzero magnitude",
                "optimizer_units_changed": False,
                "parameter_domains_changed": False,
            }
        return audit

    def configure_local_probes(self, floors: dict[str, float]) -> None:
        """Freeze opt-in physical probe scales independently of optimizer units.

        Floors are declared before fitting, e.g. inverse observation span for a
        signed rate near zero. They set a probe radius, never a parameter bound.
        Historical callers retain their original scaling and audit identities.
        """
        if not floors or set(floors) - set(self.names):
            raise ValueError("local probe floors require known coordinates")
        if any(
            isinstance(value, bool) or not np.isfinite(value) or value <= 0
            for value in floors.values()
        ):
            raise ValueError("local probe floors must be finite and positive")
        frozen = {name: float(value) for name, value in sorted(floors.items())}
        if self._local_probe_floors not in (None, frozen):
            raise ValueError("local probe policy is already frozen")
        self._local_probe_floors = frozen

    def vector(self, parameters):
        return self.reduction.vector(parameters)

    def parameters(self, vector):
        values = dict(zip(self.names, np.asarray(vector, float).tolist(), strict=True))
        self.vector(values)
        return values

    def local_coordinate_scales(self, vector):
        values = self.vector(self.parameters(vector))
        units = np.maximum(abs(values), self.units)
        for name, floor in (self._local_probe_floors or {}).items():
            i = self.names.index(name)
            units[i] = max(abs(values[i]), floor)
        for name in self.sensitive_names:
            i = self.names.index(name)
            if values[i] != 0:
                units[i] = min(abs(values[i]), units[i])
                if name not in (self._local_probe_floors or {}):
                    units[i] = min(units[i], self.units[i])
        return units

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        residual, derivative = self.original.evaluate(
            self.reduction.lift(vector), deadline, jacobian=jacobian, method=method
        )
        return residual, derivative @ self.reduction.lift_jacobian(
            vector
        ) if jacobian else None


class EndpointChartOracle:
    """Freeze an opt-in chart at one endpoint without changing its model family.

    Floors must cover every coordinate and be declared by the experiment from
    public training scales. They are numerical units, not bounds or priors. No
    Jacobian, residual or optimization outcome determines the frozen units.
    """

    def __init__(self, oracle, parameters: dict[str, float], floors: dict[str, float]):
        if set(floors) != set(oracle.names):
            raise ValueError("endpoint chart floors must cover every coordinate")
        if any(
            isinstance(value, bool) or not np.isfinite(value) or value <= 0
            for value in floors.values()
        ):
            raise ValueError("endpoint chart floors must be finite and positive")
        vector = oracle.vector(parameters)
        self.base_oracle = oracle
        self.names = oracle.names
        self.original = oracle.original
        self.source_coordinate_units = np.array(oracle.units, float, copy=True)
        self.units = np.maximum(abs(vector), [floors[name] for name in self.names])
        self.units.setflags(write=False)
        self.source_coordinate_units.setflags(write=False)
        self._chart_receipt = {
            "protocol": "frozen-endpoint-coordinate-chart-1",
            "coordinate_names": list(self.names),
            "anchor_parameters_sha256": content_sha256(parameters),
            "anchor_vector": vector.tolist(),
            "floors": {name: float(floors[name]) for name in self.names},
            "coordinate_units": self.units.tolist(),
            "source_coordinate_units": self.source_coordinate_units.tolist(),
            "rule": "max(abs(frozen_endpoint_coordinate),declared_training_floor)",
            "jacobian_based_scaling": False,
            "parameter_domains_changed": False,
            "original_equations_changed": False,
        }
        self._chart_receipt["sha256"] = content_sha256(self._chart_receipt)

    def __getattr__(self, name):
        return getattr(self.base_oracle, name)

    @property
    def recovery_audit(self):
        """Expose both old units and the new frozen chart for auditability."""
        return self.base_oracle.recovery_audit | {
            "optimizer_units": "frozen-endpoint-coordinate-chart-1; see chart receipt",
            "endpoint_coordinate_chart": deepcopy(self._chart_receipt)
        }

    def local_coordinate_scales(self, vector):
        """Use the same frozen chart, with local reciprocal singularity caps."""
        values = self.vector(self.parameters(vector))
        scales = np.maximum(abs(values), self.units)
        for name in self.sensitive_names:
            i = self.names.index(name)
            if values[i] != 0:
                scales[i] = min(abs(values[i]), scales[i])
        return scales


def _project(design, offset, d_design, d_offset, lower, upper, deadline):
    """Solve bounded LS; differentiate only a resolved full-rank strict face."""
    _check(deadline)
    a, b = _finite(design, "design"), _finite(offset, "offset")
    units = _finite(np.linalg.norm(a, axis=0), "design column norms")
    audit = {"protocol": PROTOCOL, "derivative_available": False}
    zero_columns = units == 0
    units = np.where(zero_columns, 1.0, units)
    h = a / units
    singular = np.linalg.svd(h, compute_uv=False)
    audit.update(
        column_normalized_singular_values=singular.tolist(), column_units=units.tolist()
    )
    rank = int(np.sum(singular > RANK_TOLERANCE * singular[0])) if len(singular) else 0
    audit.update(
        design_rank=rank,
        design_columns=a.shape[1],
        zero_columns=np.flatnonzero(zero_columns).tolist(),
    )
    _check(deadline)
    solved = lsq_linear(
        h,
        -b,
        bounds=(lower * units, upper * units),
        method="bvls",
        tol=1e-12,
        max_iter=100,
    )
    _check(deadline)
    if not solved.success or not np.isfinite(solved.x).all():
        raise ProfileUnavailable(
            "bounded linear solve failed", audit | {"status": "solve_failed"}
        )
    fitted = np.clip(solved.x / units, lower, upper)
    residual = _finite(a @ fitted + b, "projected residual")
    gradient = _finite(h.T @ residual, "normalized KKT gradient")
    active = solved.active_mask.astype(int)
    free = active == 0
    violation = np.where(
        free,
        abs(gradient),
        np.where(active < 0, np.maximum(-gradient, 0), np.maximum(gradient, 0)),
    )
    tolerance = KKT_TOLERANCE * max(1.0, float(np.linalg.norm(residual)))
    audit.update(
        status="conditional_optimum",
        active_mask=active.tolist(),
        normalized_kkt_residual=float(np.max(violation)),
        kkt_tolerance=tolerance,
        normalized_gradient=gradient.tolist(),
        optimality=float(solved.optimality),
        training_nmse=float(np.mean(residual * residual)),
        iterations=int(solved.nit),
        bounds_satisfied=True,
    )
    if np.max(violation) > tolerance:
        raise ProfileUnavailable(
            "projection KKT residual unresolved", audit | {"status": "kkt_unavailable"}
        )
    if d_design is None:
        return fitted, residual, None, audit
    if rank < a.shape[1]:
        raise ProfileUnavailable(
            "rank-deficient projection design", audit | {"status": "rank_unavailable"}
        )
    margin = FACE_TOLERANCE * np.maximum(1.0, abs(solved.x))
    distance = np.minimum((fitted - lower) * units, (upper - fitted) * units)
    ambiguous = (free & (distance <= margin)) | (~free & (abs(gradient) <= tolerance))
    if np.any(ambiguous):
        raise ProfileUnavailable(
            "projection active face is ambiguous",
            audit
            | {
                "status": "active_face_unavailable",
                "ambiguous_indices": np.flatnonzero(ambiguous).tolist(),
            },
        )
    jacobian = np.einsum("nkp,k->np", d_design, fitted) + d_offset
    if np.any(free):
        u, s, vt = np.linalg.svd(h[:, free], full_matrices=False)
        term = np.einsum(
            "nkp,n->kp", d_design[:, free] / units[free][None, :, None], residual
        )
        derivative = -vt.T @ (
            (u.T @ jacobian) / s[:, None] + (vt @ term) / s[:, None] ** 2
        )
        jacobian += h[:, free] @ derivative
    _check(deadline)
    audit.update(
        derivative_available=True,
        derivative="fixed-strict-active-face exact residual derivative",
    )
    return fitted, residual, _finite(jacobian, "projected Jacobian"), audit


class MultiOutputProfile:
    """Propagate jointly affine source/initial basis columns for all target states."""

    def __init__(self, oracle, *, checkpoint: Callable[[dict], None] | None = None):
        self.oracle, self.original = oracle, oracle.original
        self.checkpoint = checkpoint
        self.calls_started = self.calls_completed = self.calls_failed = 0
        system, reduction = self.original.system, oracle.reduction
        p, n, m = len(oracle.names), system.state_count, len(system.inputs)
        theta, x, u = (
            ca.SX.sym("profile_theta", p),
            ca.SX.sym("profile_state", n),
            ca.SX.sym("profile_input", m),
        )
        rhs = reduction.symbolic_rhs(
            theta,
            dict(zip(system.model.state_names, ca.vertsplit(x), strict=True)),
            dict(zip(system.inputs, ca.vertsplit(u), strict=True)),
        )
        a = ca.jacobian(rhs, x)
        forcing = ca.substitute(rhs, x, ca.SX.zeros(n))
        lifted = reduction.symbolic_lift(theta)
        local = lifted[
            self.original.indices[self.original.rows[0].trajectory_id].tolist()
        ]
        readout = system.observe(ca.SX.sym("profile_time"), x, local, u)
        outputs = []
        for channel in range(len(system.channels)):
            identity = [
                i for i in range(n) if ca.simplify(readout[channel] - x[i]).is_zero()
            ]
            if len(identity) != 1:
                raise ValueError("profiling requires parameter-free identity outputs")
            outputs.append(identity[0])
        if len(outputs) < 2 or len(set(outputs)) != len(outputs):
            raise ValueError("profiling requires distinct multiple identity outputs")
        initials = [
            system.initial_symbolic(
                row, lifted[self.original.indices[row.trajectory_id].tolist()]
            )
            for row in self.original.rows
        ]
        combined = ca.vertcat(forcing, *initials)
        self.inner = tuple(
            i
            for i in range(p)
            if not ca.depends_on(a, theta[i]) and ca.depends_on(combined, theta[i])
        )
        self.outer = tuple(i for i in range(p) if i not in self.inner)
        gamma = theta[list(self.inner)]
        if (
            not self.inner
            or not self.outer
            or ca.depends_on(ca.jacobian(combined, gamma), gamma)
        ):
            raise ValueError("source and initialization block is not jointly affine")
        if ca.depends_on(a, ca.vertcat(x, u)) or ca.depends_on(
            ca.jacobian(forcing, u), u
        ):
            raise ValueError("autonomous affine forcing required")
        k, nb = len(self.inner) + 1, len(self.outer)
        self.n, self.k, self.nb, self.outputs = n, k, nb, outputs
        generator = ca.SX.zeros(3 * n, 3 * n)
        generator[:n, :n], generator[:n, n : 2 * n], generator[n : 2 * n, 2 * n :] = (
            a,
            ca.SX.eye(n),
            ca.SX.eye(n),
        )
        self.generator = ca.Function(
            "profile_generator",
            [theta],
            [generator, ca.jacobian(ca.vec(generator), theta)[:, list(self.outer)]],
        )

        def basis(expression, name, arguments):
            columns = ca.horzcat(
                ca.substitute(expression, gamma, ca.SX.zeros(len(self.inner))),
                ca.jacobian(expression, gamma),
            )
            return ca.Function(
                name,
                arguments,
                [columns, ca.jacobian(ca.vec(columns), theta)[:, list(self.outer)]],
            )

        self.sources = basis(forcing, "profile_sources", [theta, u])
        self.initializers = [
            basis(initial, f"profile_initial_{i}", [theta])
            for i, initial in enumerate(initials)
        ]
        self.transitions = TransitionEvaluator(
            recovery.precision_policy(oracle.precision_mode)
        )
        self.audit = {
            **policy_receipt(),
            "reduction_sha256": reduction.receipt["sha256"],
            "training_sha256": content_sha256(self.original.training),
            "inner_names": [oracle.names[i] for i in self.inner],
            "outer_names": [oracle.names[i] for i in self.outer],
            "channel_order": list(system.channels),
            "target_scales": self.original.scales,
            "training_rows": [
                {
                    "trajectory_id": row.trajectory_id,
                    "sample_count": len(row.time),
                    "time_sha256": content_sha256(row.time.tolist()),
                    "mask": "all",
                }
                for row in self.original.rows
            ],
            "state_matrix_independent_of_inner": True,
            "forcing_and_initials_jointly_affine": True,
            "initialization_scopes_preserved": True,
            "transition_precision": self.transitions.audit,
        }

    def _notify(self, stage, deadline):
        _check(deadline)
        if self.checkpoint is not None:
            self.checkpoint(
                {
                    "stage": stage,
                    "calls_started": self.calls_started,
                    "calls_completed": self.calls_completed,
                    "calls_failed": self.calls_failed,
                }
            )
        _check(deadline)

    def design(self, vector, deadline, *, jacobian=True):
        """Return weighted basis arrays in the original row/time/channel ordering."""
        _check(deadline)
        self.oracle.vector(self.oracle.parameters(vector))
        n, k, nb = self.n, self.k, self.nb
        generator, derivative = self.generator(vector)
        generator = _finite(generator, "generator")
        derivative = _finite(derivative, "generator derivative").reshape(
            3 * n, 3 * n, nb, order="F"
        )
        scales = np.asarray(
            [self.original.scales[c] for c in self.original.system.channels]
        )
        cache, designs, offsets, derivatives, offset_derivatives = {}, [], [], [], []

        def unpack(function, *arguments):
            value, local = function(*arguments)
            return _finite(value, "basis"), _finite(local, "basis derivative").reshape(
                n, k, nb, order="F"
            )

        for row, initializer in zip(self.original.rows, self.initializers, strict=True):
            self._notify("basis_trajectory", deadline)
            state, ds = unpack(initializer, vector)
            forcing = trajectory_forcing(self.original.system.model, row)
            inputs = np.asarray(
                [
                    [forcing.value(name, t) for name in self.original.system.inputs]
                    for t in row.time
                ]
            )
            zero, dzero = unpack(
                self.sources, vector, np.zeros(len(self.original.system.inputs))
            )
            outputs, gradients = [], []
            for i, time in enumerate(row.time):
                _check(deadline)
                if i:
                    dt = float(time - row.time[i - 1])
                    if dt not in cache:
                        cache[dt] = self.transitions(
                            generator * dt,
                            [derivative[:, :, j] * dt for j in range(nb)]
                            if jacobian
                            else [],
                            deadline,
                        )
                    matrix, dm = cache[dt]
                    source, dsource = unpack(self.sources, vector, inputs[i - 1])
                    slope, dslope = unpack(
                        self.sources, vector, (inputs[i] - inputs[i - 1]) / dt
                    )
                    extended = np.vstack([state, source, slope - zero])
                    if jacobian:
                        dextended = np.concatenate(
                            [ds, dsource, dslope - dzero], axis=0
                        )
                        ds = np.einsum("ab,bkj->akj", matrix[:n], dextended) + np.stack(
                            [(d @ extended)[:n] for d in dm], axis=-1
                        )
                    state = _finite((matrix @ extended)[:n], "propagated basis")
                outputs.append(state[self.outputs].copy())
                if jacobian:
                    gradients.append(ds[self.outputs].copy())
            observed = np.column_stack(
                [row.targets[c] for c in self.original.system.channels]
            )
            values = np.asarray(outputs)
            designs.append(
                (values[:, :, 1:] / scales[None, :, None]).reshape(-1, k - 1)
            )
            offsets.append(((values[:, :, 0] - observed) / scales).ravel())
            if jacobian:
                local = np.asarray(gradients) / scales[None, :, None, None]
                derivatives.append(local[:, :, 1:, :].reshape(-1, k - 1, nb))
                offset_derivatives.append(local[:, :, 0, :].reshape(-1, nb))
        _check(deadline)
        return {
            "design": _finite(np.vstack(designs), "design"),
            "offset": _finite(np.concatenate(offsets), "offset"),
            "d_design": _finite(np.vstack(derivatives), "design derivative")
            if jacobian
            else None,
            "d_offset": _finite(np.vstack(offset_derivatives), "offset derivative")
            if jacobian
            else None,
        }

    def evaluate(self, vector, deadline, *, jacobian=True):
        """Return a conditional optimum and, when qualified, its outer derivative."""
        self.calls_started += 1
        try:
            self._notify("profile_started", deadline)
            arrays = self.design(vector, deadline, jacobian=jacobian)
            self._notify("bounded_linear_solve", deadline)
            fitted, residual, derivative, audit = _project(
                **arrays,
                lower=self.oracle.lower[list(self.inner)],
                upper=self.oracle.upper[list(self.inner)],
                deadline=deadline,
            )
            full = np.asarray(vector, float).copy()
            full[list(self.inner)] = fitted
            self.oracle.vector(self.oracle.parameters(full))
            self.calls_completed += 1
            self._notify("profile_completed", deadline)
            return full, residual, derivative, audit | {"certificate": self.audit}
        except BaseException:
            self.calls_failed += 1
            raise


def make_oracle(
    problem, training, numerical, retained, *, precision_mode="guarded", checkpoint=None
):
    """Build the unchanged original, aggregate joint oracle and conditional profile."""
    original, base = recovery.make_oracle(
        problem,
        training,
        numerical,
        retained,
        precision_mode=precision_mode,
        structural_reduction=True,
    )
    oracle = AggregatedTrainingOracle(base, retained)
    return original, oracle, MultiOutputProfile(oracle, checkpoint=checkpoint)
