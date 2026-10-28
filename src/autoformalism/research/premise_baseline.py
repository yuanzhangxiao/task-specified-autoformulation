"""Bounded, established interval baselines for Lyra's profiled amplitude family.

All certificate endpoints use outward-rounded Decimal intervals. Floating-point
values only order subdivision candidates; they never exclude a cell or certify
a scientific conclusion. No optimizer failure is interpreted as exclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from typing import Any

from autoformalism.research.premise_intervals import Interval, kernel

PROTOCOL = "vega-premise-profiled-baseline-v1"
BUDGETS = (64, 256, 1024, 4096)
METHODS = ("profiled", "direct")
P = Interval.point
A_PRIOR = Interval(Decimal("0.5"), Decimal("2"))
S_PRIOR = Interval(Decimal("0"), Decimal("0.6"))
TIMES = tuple(Decimal(t) for t in (1, 2, 3, 5))
SIGMA = P("0.02")
Q = Decimal(10)
T = Decimal(7)
TAU = Decimal(3)
DELTA = Decimal(4)
RHO = Interval(Decimal("0.5"), Decimal("2"))
TOLERANCES = ("0.1", "0.25", "0.5")
EXTREMA_GAP = Decimal("0.002")


@dataclass(frozen=True)
class Problem:
    """Public observations and preparation contract; no generating parameters."""

    case_id: str
    observations: tuple[str, ...]
    contract: str

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("case_id must be a nonempty string")
        if type(self.observations) is not tuple or len(self.observations) != 4:
            raise ValueError("four exact decimal observation strings are required")
        for value in self.observations:
            if type(value) is not str:
                raise ValueError("observations must be exact decimal strings")
            P(value)
        if type(self.contract) is not str or self.contract not in (
            "coupled",
            "independent",
        ):
            raise ValueError("unknown preparation contract")


def problems() -> tuple[Problem, ...]:
    """Return prespecified numerical cases, not a random calibration campaign."""
    records = (
        (
            "decaying",
            (
                "0.569564139883174",
                "0.660730953498783",
                "0.627780709657703",
                "0.446426867715446",
            ),
        ),
        (
            "near-constant",
            (
                "0.624120558828558",
                "0.870664716763387",
                "0.946212931632136",
                "0.998262053000915",
            ),
        ),
    )
    return tuple(
        Problem(case_id, observations, contract)
        for case_id, observations in records
        for contract in ("coupled", "independent")
    )


def coverage() -> Interval:
    """Exact chi-square(4) threshold q=10: coverage is 1-6 exp(-5)."""
    return P(1) - P(6) * P(-5).exp()


def _kernel(s: Interval, time: Decimal, cost: dict | None) -> Interval:
    if cost is not None:
        cost["kernels"] += 1
    return kernel(s, time)


def _features(s: Interval, cost: dict | None) -> list[Interval]:
    return [_kernel(s, t, cost) for t in TIMES]


def _amplitude_domain(amplitude: Interval) -> None:
    if not isinstance(amplitude, Interval) or (
        amplitude.lo < A_PRIOR.lo or amplitude.hi > A_PRIOR.hi
    ):
        raise ValueError("amplitude must lie within the declared [0.5,2] prior")


def _quadratic(
    problem: Problem, features: list[Interval]
) -> tuple[Interval, Interval, Interval]:
    scaled = [f / SIGMA for f in features]
    data = [P(y) / SIGMA for y in problem.observations]
    a = sum((f.square() for f in scaled), P(0))
    h = sum((f * y for f, y in zip(scaled, data, strict=True)), P(0))
    if a.lo <= 0:
        raise ValueError("positive conditional amplitude curvature not established")
    estimate = h / a
    gram = (
        sum(
            (
                (scaled[i] * data[j] - scaled[j] * data[i]).square()
                for i, j in combinations(range(4), 2)
            ),
            P(0),
        )
        / a
    )
    residual = sum(
        ((y - estimate * f).square() for y, f in zip(data, scaled, strict=True)),
        P(0),
    )
    lo, hi = max(Decimal(0), gram.lo, residual.lo), min(gram.hi, residual.hi)
    if lo > hi:
        raise ValueError("independent Q_min enclosures disagree")
    return a, estimate, Interval(lo, hi)


def _profile(
    problem: Problem, s: Interval, domain: Interval, cost: dict | None
) -> tuple[Interval | None, dict[str, Any]]:
    _amplitude_domain(domain)
    a, estimate, minimum = _quadratic(problem, _features(s, cost))
    return _amplitude_enclosure(a, estimate, minimum, domain)


def _amplitude_enclosure(
    a: Interval, estimate: Interval, minimum: Interval, domain: Interval
) -> tuple[Interval | None, dict[str, Any]]:
    proof: dict[str, Any] = {"q_min": minimum.to_json()}
    if minimum.lo > Q:
        return None, proof
    radius = ((P(Q) - P(minimum.lo)) / P(a.lo)).sqrt().hi
    lower = max(domain.lo, (P(estimate.lo) - P(radius)).lo)
    upper = min(domain.hi, (P(estimate.hi) + P(radius)).hi)
    proof["amplitude_outer"] = [str(lower), str(upper)]
    return (None if lower > upper else Interval(lower, upper)), proof


def profile_amplitude(
    problem: Problem, s: Interval, domain: Interval = A_PRIOR
) -> Interval | None:
    """Enclose the union of conditional feasible amplitudes across a rate cell."""
    return _profile(problem, s, domain, None)[0]


def _q_from_features(
    problem: Problem, amplitude: Interval, features: list[Interval]
) -> Interval:
    return sum(
        (
            ((P(y) - amplitude * f) / SIGMA).square()
            for y, f in zip(problem.observations, features, strict=True)
        ),
        P(0),
    )


def q_value(problem: Problem, amplitude: Interval, s: Interval) -> Interval:
    """Outward evaluation of the declared observation quadratic."""
    _amplitude_domain(amplitude)
    return _q_from_features(problem, amplitude, _features(s, None))


def _task_bounds(
    problem: Problem, amplitude: Interval, s: Interval, cost: dict | None
) -> Interval:
    _amplitude_domain(amplitude)
    base = amplitude * _kernel(s, T, cost)
    pulse = _kernel(s, DELTA, cost)
    attenuation = (-(s * P(TAU))).exp()
    if problem.contract == "independent":
        lower = base + P(amplitude.lo) * P(RHO.lo) * attenuation * pulse
        upper = base + P(amplitude.hi) * P(RHO.hi) * pulse
    else:
        # A*rho=1 is preserved algebraically, never replaced by marginal boxes.
        lower = base + attenuation * pulse
        upper = base + pulse
    return Interval(lower.lo, upper.hi)


def task_bounds(problem: Problem, amplitude: Interval, s: Interval) -> Interval:
    """Bound all future responses, eliminating gamma and dose analytically."""
    return _task_bounds(problem, amplitude, s, None)


def _points_at(
    problem: Problem, s: Decimal, domain: Interval, cost: dict
) -> list[dict[str, Any]]:
    """Verify center and near-boundary amplitudes, retaining their uncertainty."""
    features = _features(P(s), cost)
    a, estimate, minimum = _quadratic(problem, features)
    outer, _ = _amplitude_enclosure(a, estimate, minimum, domain)
    if outer is None:
        return []
    center = min(domain.hi, max(domain.lo, estimate.midpoint()))
    amplitudes = [center]
    for endpoint in (outer.lo, outer.hi):
        # A proposal, not a certificate: each point must pass Q_upper<=q below.
        proposed = (
            P(center) * P("0.0000000001") + P(endpoint) * P("0.9999999999")
        ).midpoint()
        if proposed not in amplitudes:
            amplitudes.append(proposed)
    witnesses = []
    for amplitude in amplitudes:
        cost["point_tests"] += 1
        q = _q_from_features(problem, P(amplitude), features)
        if q.hi <= Q:
            task = _task_bounds(problem, P(amplitude), P(s), cost)
            witnesses.append(
                {
                    "A": str(amplitude),
                    "s": str(s),
                    "q": q.to_json(),
                    "task": task.to_json(),
                }
            )
    return witnesses


def _update_witness(state: dict, witness: dict | None) -> None:
    if witness is None:
        return
    # task.lo/hi enclose the two attainable pulse extremes at this feasible point.
    # Reevaluate each extreme separately before using an inner extremum bound.
    problem = Problem(
        **{**state["problem"], "observations": tuple(state["problem"]["observations"])}
    )
    a, s = P(witness["A"]), P(witness["s"])
    f = _kernel(s, T, state["cost"])
    pulse = _kernel(s, DELTA, state["cost"])
    if problem.contract == "coupled":
        low = a * f + (-(s * P(TAU))).exp() * pulse
        high = a * f + pulse
    else:
        low = a * f + a * P(RHO.lo) * (-(s * P(TAU))).exp() * pulse
        high = a * f + a * P(RHO.hi) * pulse
    if state["L_hi"] is None or low.hi < Decimal(state["L_hi"]):
        state["L_hi"] = str(low.hi)
        state["lower_witness"] = witness
    if state["U_lo"] is None or high.lo > Decimal(state["U_lo"]):
        state["U_lo"] = str(high.lo)
        state["upper_witness"] = witness


def _zero_face(problem: Problem, cost: dict) -> dict[str, Any]:
    """Use the convex amplitude minimum at s=0, common to both methods."""
    features = _features(P(0), cost)
    _, estimate, _ = _quadratic(problem, features)
    clipped = Interval(
        min(A_PRIOR.hi, max(A_PRIOR.lo, estimate.lo)),
        min(A_PRIOR.hi, max(A_PRIOR.lo, estimate.hi)),
    )
    minimum = _q_from_features(problem, clipped, features)
    points = _points_at(problem, Decimal(0), A_PRIOR, cost)
    witness = points[0] if points else None
    status = (
        "excluded"
        if minimum.lo > Q
        else ("witnessed" if witness is not None else "unresolved")
    )
    return {"status": status, "minimum_q": minimum.to_json(), "witness": witness}


def _models(problem: Problem, witness: dict | None, cost: dict) -> list[dict]:
    if witness is None:
        return []
    a, s = P(witness["A"]), P(witness["s"])
    base, pulse = a * _kernel(s, T, cost), _kernel(s, DELTA, cost)
    coefficient = P(1) if problem.contract == "coupled" else a * P("1.25")
    result = []
    for name, fraction in (
        ("constant_capacity", "0"),
        ("balanced_rates", "0.5"),
        ("constant_ligand", "1"),
    ):
        gamma = (s * P(fraction)).midpoint()
        prediction = base + coefficient * (-(P(gamma) * P(TAU))).exp() * pulse
        result.append(
            {
                "name": name,
                "A": witness["A"],
                "s": witness["s"],
                "gamma": str(gamma),
                "q": witness["q"],
                "rho": "1/A" if problem.contract == "coupled" else "1.25",
                "prediction": prediction.to_json(),
            }
        )
    return result


def _evaluate(
    problem: Problem, state: dict, s: Interval, domain: Interval, parent: int | None
) -> None:
    state["cost"]["boxes"] += 1
    if state["method"] == "profiled":
        amplitude, proof = _profile(problem, s, domain, state["cost"])
    else:
        q = _q_from_features(problem, domain, _features(s, state["cost"]))
        amplitude = None if q.lo > Q else domain
        proof = {"q": q.to_json()}
    node = {
        "id": len(state["nodes"]),
        "parent": parent,
        "s": s.to_json(),
        "A_domain": domain.to_json(),
        "proof": proof,
        "A_outer": None if amplitude is None else amplitude.to_json(),
        "status": "excluded" if amplitude is None else "retained",
        "task": None,
    }
    state["nodes"].append(node)
    if amplitude is None:
        return
    node["task"] = _task_bounds(problem, amplitude, s, state["cost"]).to_json()
    state["frontier"].append(node["id"])
    for witness in _points_at(problem, s.midpoint(), amplitude, state["cost"]):
        _update_witness(state, witness)


def initialize(problem: Problem, method: str) -> dict:
    """Freeze common anchor candidates before either global-search implementation."""
    if method not in METHODS:
        raise ValueError("unknown baseline method")
    state: dict[str, Any] = {
        "protocol": PROTOCOL,
        "method": method,
        "problem": {
            "case_id": problem.case_id,
            "observations": list(problem.observations),
            "contract": problem.contract,
        },
        "cost": {"boxes": 0, "kernels": 0, "point_tests": 0},
        "nodes": [],
        "frontier": [],
        "L_hi": None,
        "U_lo": None,
        "lower_witness": None,
        "upper_witness": None,
    }
    anchors = []
    for index in range(33):
        s = (P("0.6") * P(index) / P(32)).midpoint()
        for witness in _points_at(problem, s, A_PRIOR, state["cost"]):
            anchors.append(witness)
            _update_witness(state, witness)
    state["anchors"] = anchors
    anchor = (
        min(anchors, key=lambda w: (Decimal(w["q"][1]), Decimal(w["s"])))
        if anchors
        else None
    )
    state["candidates"] = _models(problem, anchor, state["cost"])
    state["zero_face"] = _zero_face(problem, state["cost"])
    _update_witness(state, state["zero_face"]["witness"])
    state["preparation_cost"] = dict(state["cost"])
    _evaluate(problem, state, S_PRIOR, A_PRIOR, None)
    return state


def _bounds(state: dict) -> dict[str, list[str]] | None:
    if not state["frontier"]:
        if state["L_hi"] is not None:
            raise ValueError("exclusion contradicts a certified feasible witness")
        return None
    if state["L_hi"] is None:
        return None
    tasks = [Interval.from_json(state["nodes"][i]["task"]) for i in state["frontier"]]
    low, high = min(t.lo for t in tasks), max(t.hi for t in tasks)
    if low > Decimal(state["L_hi"]) or Decimal(state["U_lo"]) > high:
        raise ValueError("outer extrema contradict certified witnesses")
    return {"L": [str(low), state["L_hi"]], "U": [state["U_lo"], str(high)]}


def _resolved_precision(state: dict) -> bool:
    bounds = _bounds(state)
    if bounds is None:
        return not state["frontier"]
    return all(
        (P(values[1]) - P(values[0])).hi <= EXTREMA_GAP for values in bounds.values()
    )


def advance(problem: Problem, state: dict, budget: int) -> dict:
    """Continue deterministic subdivision without dropping unfinished cells."""
    _identity(problem, state)
    if type(budget) is not int or budget < state["cost"]["boxes"]:
        raise ValueError("budget must cover all prior box evaluations")
    while state["cost"]["boxes"] + 2 <= budget and not _resolved_precision(state):

        def priority(index: int) -> tuple[float, int]:
            node = state["nodes"][index]
            if state["L_hi"] is None:
                width = float(node["s"][1]) - float(node["s"][0])
                return width, -index
            low, high = map(float, node["task"])
            return max(float(state["L_hi"]) - low, high - float(state["U_lo"])), -index

        index = max(state["frontier"], key=priority)
        node = state["nodes"][index]
        s, amplitude = (
            Interval.from_json(node["s"]),
            Interval.from_json(node["A_outer"]),
        )
        split_s = state["method"] == "profiled" or (
            float((P(s.hi) - P(s.lo)).hi) / 0.6
            >= float((P(amplitude.hi) - P(amplitude.lo)).hi) / 1.5
        )
        axis = s if split_s else amplitude
        middle = axis.midpoint()
        if middle in (axis.lo, axis.hi):
            state["precision_exhausted"] = True
            break
        state["frontier"].remove(index)
        node["status"] = "split"
        for half in (Interval(axis.lo, middle), Interval(middle, axis.hi)):
            _evaluate(
                problem,
                state,
                half if split_s else s,
                amplitude if split_s else half,
                index,
            )
    return state


def _decision(risk: Interval) -> dict[str, str]:
    return {
        tolerance: "adequate"
        if risk.hi <= Decimal(tolerance)
        else ("inadequate" if risk.lo > Decimal(tolerance) else "unresolved")
        for tolerance in TOLERANCES
    }


def _identity(problem: Problem, state: dict) -> None:
    expected = {
        "case_id": problem.case_id,
        "observations": list(problem.observations),
        "contract": problem.contract,
    }
    if (
        state["problem"] != expected
        or state["protocol"] != PROTOCOL
        or state["method"] not in METHODS
    ):
        raise ValueError("state problem, method or protocol differs")


def summarize(problem: Problem, state: dict) -> dict:
    """Report outer/inner extrema and fixed-model risks without new evaluations."""
    _identity(problem, state)
    bounds = _bounds(state)
    witnessed = state["L_hi"] is not None
    face = state["zero_face"]["status"]
    result: dict[str, Any] = {
        "case_id": problem.case_id,
        "contract": problem.contract,
        "method": state["method"],
        "cost": dict(state["cost"]),
        "preparation_cost": state["preparation_cost"],
        "bounds": bounds,
        "forecast_risk": None,
        "forecast_decisions": {},
        "candidate_reports": [],
        "witnessed_nonempty": witnessed,
        "incompatible": not state["frontier"] and not witnessed,
        "frontier_cells": len(state["frontier"]),
        "excluded_cells": sum(n["status"] == "excluded" for n in state["nodes"]),
        "extrema_precision_resolved": _resolved_precision(state),
        "joint_conflict": "established"
        if witnessed and face == "excluded"
        else ("not_established" if face == "witnessed" else "unresolved"),
        "individual_rejection": {"constant_ligand": False, "constant_capacity": False}
        if witnessed
        else None,
        "zero_face": state["zero_face"],
    }
    if bounds is None:
        return result
    llo, lhi = map(Decimal, bounds["L"])
    ulo, uhi = map(Decimal, bounds["U"])
    risk_lo = max(Decimal(0), ((P(ulo) - P(lhi)) / P(2)).lo)
    risk_hi = ((P(uhi) - P(llo)) / P(2)).hi
    risk = Interval(risk_lo, risk_hi)
    result["forecast_risk"] = risk.to_json()
    result["forecast_decisions"] = _decision(risk)
    for model in state["candidates"]:
        prediction = Interval.from_json(model["prediction"])
        lower = max(
            Decimal(0), (P(ulo) - P(prediction.hi)).lo, (P(prediction.lo) - P(lhi)).lo
        )
        upper = max((P(uhi) - P(prediction.lo)).hi, (P(prediction.hi) - P(llo)).hi)
        risk = Interval(lower, upper)
        result["candidate_reports"].append(
            {
                **model,
                "risk": risk.to_json(),
                "decisions": _decision(risk),
            }
        )
    return result
