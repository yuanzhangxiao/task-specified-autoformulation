"""Find equal-argument contradictions for T2 disposal on fitting events only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research.hidden_start_contrasts import _initial_tokens
from autoformalism.research.trajectory_function_fitting import _rows
from autoformalism.search.topology_compatibility import _closure
from scripts.audit_hidden_shape_t2 import FROZEN
from scripts.matched_equation_experiment import DEFAULT as SOURCE
from scripts.matched_equation_experiment import _registered

DEFAULT = SOURCE.parent / "balanced-disposal-20261126/dependencies"
SLACK = 1e-10
LINEAR = "linear_complete_knots"
STEP = "previous_complete_knots"


def _subsets(names):
    return [s for size in range(len(names) + 1) for s in combinations(names, size)]


def _schedules(graph, extra, rows, interpolation):
    """Give equal keys only to latent values the candidate must keep equal."""
    action = "I_act"
    prospective = graph["sources"] | {"U": ["Uii", action, *extra]}
    edited_graph = graph | {"sources": prospective}
    upstream = _closure(action, prospective)
    states = sorted(
        k for k in upstream if graph["variable_types"].get(k) == "differential"
    )
    external = upstream - set(prospective) - {"t"}
    fixed = sorted(external & rows[0].covariates.keys())
    forcing = sorted(external - set(fixed))
    known = all(interpolation.get(k) in {LINEAR, STEP} for k in forcing)
    registry = {}

    def intern(key):
        return registry.setdefault(key, len(registry))

    hidden = []
    for row in rows:
        initial = _initial_tokens(edited_graph, row, "t")
        values = row.inputs | row.auxiliaries
        # At t0, equality of the hidden initializer alone suffices. Later
        # the initial values of the entire autonomous upstream closure matter.
        keys = [intern(("initial", row.time[0], initial[action]))]
        prefix = intern(
            (
                "start",
                row.time[0],
                tuple(initial[s] for s in states),
                tuple(row.covariates[s] for s in fixed),
            )
        )
        for j in range(1, len(row.time)):
            if known:
                samples = tuple(
                    (values[s][j - 1], values[s][j])
                    if interpolation[s] == LINEAR
                    else (values[s][j - 1],)
                    for s in forcing
                )
                prefix = intern(
                    ("interval", prefix, row.time[j - 1], row.time[j], samples)
                )
            else:
                prefix = intern(("unknown_forcing", row.trajectory_id, j))
            keys.append(prefix)
        hidden.append(keys)
    return hidden, {
        "upstream_differential_states": states,
        "external_forcings": forcing,
        "complete_forcing_interpolation": known,
    }


def _compare(points, arguments, hidden=None, initial_only=False):
    """Half the largest spread is a necessary uniform absolute output error."""
    groups = defaultdict(list)
    for k, points_for_event in enumerate(points):
        for j, (values, response, event, time) in enumerate(points_for_event):
            if initial_only and j:
                break
            key = tuple(values[s] for s in arguments)
            if hidden is not None:
                key = (hidden[k][j], *key)
            groups[key].append((response, event, time))
    count, repeated_points, contradicted = 0, 0, 0
    strongest, witness = Fraction(0), None
    for key, values in groups.items():
        if len(values) < 2:
            continue
        count += 1
        repeated_points += len(values)
        a, b = min(values), max(values)
        radius = (b[0] - a[0]) / 2
        contradicted += radius > SLACK
        if radius > strongest:
            strongest = radius
            witness = {
                "measured_arguments": dict(
                    zip(arguments, key[1:] if hidden is not None else key, strict=True)
                ),
                "events": [a[1], b[1]],
                "times": [a[2], b[2]],
                "D": [float(a[0]), float(b[0])],
            }
    return {
        "eligible": contradicted == 0,
        "minimum_uniform_error": float(strongest),
        "repeated_argument_groups": count,
        "observations_in_repeated_groups": repeated_points,
        "conflicting_groups": contradicted,
        "witness": witness,
    }


def screen_dependencies(graph: dict, train_rows, *, interpolation=None) -> dict:
    """Screen candidate readouts assuming U=Uii+D and shared deterministic laws."""
    rows = _rows(train_rows, "train")
    if (
        graph["variable_types"].get("I_act") != "differential"
        or graph["variable_types"].get("U") != "algebraic"
    ):
        raise ValueError("Expected algebraic U and differential I_act")
    measured = ("I", "Gp", "Gt")
    points = []
    for row in rows:
        data = row.targets | row.auxiliaries
        points.append(
            [
                (
                    {name: data[name][j] for name in measured},
                    Fraction(data["U"][j]) - Fraction(data["Uii"][j]),
                    row.trajectory_id,
                    time,
                )
                for j, time in enumerate(row.time)
            ]
        )
    hypotheses = {
        "+".join(names) or "constant": {
            **_compare(points, names),
            "initial": _compare(points, names, initial_only=True),
        }
        for names in _subsets(measured)
    }
    candidates = {}
    for extra in _subsets(("Gp", "Gt")):
        keys, context = _schedules(graph, extra, rows, interpolation or {})
        candidates["+".join(("I_act", *extra))] = {
            **_compare(points, extra, keys),
            "initial": _compare(points, extra, keys, initial_only=True),
            "equality_contract": context,
        }
    return {
        "readout": "U=Uii+D",
        "fitting_events": [r.trajectory_id for r in rows],
        "sample_count": sum(map(len, points)),
        "absolute_error_slack": SLACK,
        "bound_assumptions": [
            "Compared measured arguments are exact and reproduced by the model.",
            "Each candidate is one shared function of its listed arguments.",
            "Equal hidden keys require the declared initializer or matched initial "
            "states and identical complete forcing histories in the candidate graph.",
            "Separate measurement errors in U and Uii add in the error bound for D.",
        ],
        "measured_candidates": hypotheses,
        "hidden_candidates": candidates,
    }


def run(source: Path, output: Path) -> dict:
    """Verify the frozen training manifest and seal four panel-level audits."""
    frozen = _registered(source / "plan.json", FROZEN)
    results = {}
    with _lock(output):
        for panel in frozen["panels"]:
            key = panel["panel_id"]
            if key.split("-")[0] not in {"graph02", "graph03"}:
                continue
            packet = _registered(source / "packets" / f"{key}.json", panel["packet"])
            report = screen_dependencies(
                packet["graph"],
                packet["train_rows"],
                interpolation=packet["interpolation"],
            )
            result = sealed_write(
                output / f"{key}.json", {"packet": panel["packet"], **report}
            )
            results[key] = result["artifact_sha256"]
        return sealed_write(
            output / "summary.json",
            {"protocol": "effective-disposal-source-screen-1", "panels": results},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(run(args.source, args.output)["artifact_sha256"])
