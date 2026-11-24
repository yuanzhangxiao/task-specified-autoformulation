"""Test a single additional measured initial value for T2 insulin action."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import fit_t2_initial_disposal_order as order
from scripts.construction_functions import file_hash

GAIN = "init_tissue_gain"
CONTROL = "6e0e84e3c19fc54d9b548993f2afe6689a6d2402088ea035358a8f2887d0458e"
OUT = Path("output/t2-initial-order-20261123/initializer-probe")


def tissue_map(parent: dict, rows: list[dict]) -> tuple[dict, dict]:
    """Add a centered public Gt(0) reading exclusively to the action initializer."""
    reference = [r for r in rows if r["trajectory_id"] == "train_000"]
    if len(reference) != 1 or any(
        not r["trajectory_id"].startswith("train_") or r["time"][0] != 0 for r in rows
    ):
        raise ValueError("Expected public training initials and a fasting reference")
    ref = reference[0]
    if any(ref["inputs"]["meal_rate_g_per_min"]):
        raise ValueError("Expected the fasting training reference")
    center = ref["auxiliaries"]["Gt"][0]
    child = deepcopy(parent)
    rule = child["graph"]["initializers"]["I_act"]
    f = child["formula"]
    if rule != {"mode": "map", "public_initials": [], "state_initials": ["I"]} or (
        f["initializers"]["I_act"] != "afcoef15+afcoef16*state0_I"
        or GAIN in f["parameters"]
    ):
        raise ValueError("Expected the original insulin-only action initializer")
    rule["public_initials"] = ["Gt"]
    f["initializers"]["I_act"] += f"+{GAIN}*((Gt-({center!r}))/100)"
    f["parameters"][GAIN] = 0.0
    gp = Fraction(ref["targets"]["Gp"][0])
    gt = Fraction(center)
    return child, {
        "reference_event": ref["trajectory_id"],
        "reference_initial_Gt": center,
        "training_events": len(rows),
        "centered_Gp_equals_negative_centered_Gt": all(
            Fraction(r["targets"]["Gp"][0]) - gp
            == gt - Fraction(r["auxiliaries"]["Gt"][0])
            for r in rows
        ),
    }


def active_start(model, data, start):
    """Compute starting values in the original bounded parameters."""
    start_values = order.coupled.orion.optimizer_start(
        model, model["formula"]["parameters"], "fixed_insulin", start
    )
    active = set(data.active)
    return {
        name: max(start_values[name], data.lo[i]) if i in active else data.reference[i]
        for i, name in enumerate(data.names)
    }


class TissueTraining(order.base._Training):
    """Extend only the authenticated parent's active coordinates and preserve bounds."""

    def __init__(self, parent, child, rows):
        if tissue_map(parent, rows)[0] != child:
            raise ValueError(
                "Only the registered tissue initialization change is qualified"
            )
        super().__init__(parent, rows)
        old_residual = self.engine.residual(self.reference)
        bounds = dict(zip(self.names, self.lo, strict=True))
        active = {self.names[i] for i in self.active} | {GAIN}
        starts = {s: active_start(parent, self, s) for s in order.coupled.orion.STARTS}
        self.engine = order.base.FitProblem(
            child, [order.base._midpoints(r) for r in rows], substeps=2
        )
        self.names = self.engine.names
        self.reference = self.engine._vector(child["formula"]["parameters"])
        self.lo = np.array([bounds[n] if n != GAIN else -np.inf for n in self.names])
        self.active = [i for i, n in enumerate(self.names) if n in active]
        difference = self.engine.residual(self.reference) - old_residual
        self.zero_gain_max_scaled_difference = float(np.max(np.abs(difference)))
        if self.zero_gain_max_scaled_difference > 1e-10:
            raise ValueError(
                "Zero initializer gain failed to reproduce parent trajectories"
            )
        self.start_differences = {}
        for start, expected in starts.items():
            actual = active_start(child, self, start)
            delta = max(abs(actual[n] - v) for n, v in expected.items())
            if delta != 0 or actual[GAIN] != 0:
                raise ValueError("Extended model changed the common optimizer starts")
            self.start_differences[start] = float(delta)


def run(root: Path) -> dict:
    """Fit the nested map against the frozen two-start half-gap control."""
    control = sealed_read(order.OUT / "summary.json")
    control_plan = sealed_read(order.OUT / "plan.json")
    source = sealed_read(order.SOURCE)
    training = order.coupled.INPUTS / "train.json"
    parent = source["models"]["delayed_refit"]
    if (
        control["artifact_sha256"] != CONTROL
        or control["plan"] != control_plan["artifact_sha256"]
        or source["artifact_sha256"] != order.FREEZE
        or source["validation_opened_at_freeze"]
        or control_plan["source"] != order.FREEZE
        or control_plan["model"] != content_hash(parent)
        or control_plan["training"] != file_hash(training)
        or control_plan["budget_per_arm_per_start"] != 80
        or control_plan["starts"] != list(order.coupled.orion.STARTS)
        or control_plan["required_fractions_of_each_nonzero_observed_gap"]["half_gap"]
        != 0.5
        or control["validation_opened"]
        or control["test_opened"]
    ):
        raise ValueError("Expected the matched half-gap training comparison")
    for name, sha in control_plan["code"].items():
        code = Path(order.__file__).with_name(name)
        if not code.is_file() and name == Path(order.initial_order.__file__).name:
            code = Path(order.initial_order.__file__)
        if file_hash(code) != sha:
            raise ValueError(f"Matched solver or evaluator changed: {name}")
    intake = sealed_read(training)
    cases = [
        c
        for c in intake["cases"].values()
        if "U" in c["evidence"]["context"]["targets"]
    ]
    if not cases or not all(
        "Gt" in c["initials"] and "Gt" in c["evidence"]["context"]["auxiliaries"]
        for c in cases
    ):
        raise ValueError("Tissue glucose is unavailable as an observed initial reading")
    rows = order.coupled.data.load_training(training)
    child, evidence = tissue_map(parent, rows)
    with _lock(root / "lock"):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "t2-initial-tissue-map-1",
                "control": CONTROL,
                "control_plan": control["plan"],
                "model": content_hash(child),
                "code": file_hash(Path(__file__)),
                "training": control_plan["training"],
                "source": order.FREEZE,
                "evidence": evidence,
                "fit_arm": "half_gap",
                "budget_per_start": 80,
                "starts": list(order.coupled.orion.STARTS),
            },
        )
        identity = plan["artifact_sha256"]
        data = TissueTraining(parent, child, rows)
        qualified = []
        for start in order.coupled.orion.STARTS:
            path = root / "fits" / f"{start}.json"
            if not path.exists():
                result = order.optimize(child, data, "half_gap", 80, start)
                sealed_write(path, {"plan": identity, "start": start, **result})
            fitted = sealed_read(path)
            if fitted["plan"] != identity or fitted["start"] != start:
                raise ValueError("Initializer probe checkpoint differs")
            qualified.append(
                order.qualify(child, rows, "half_gap", fitted, root, identity)
            )
        eligible = [r for r in qualified if r["eligible"]]
        selected = {"eligible": False}
        if eligible:
            minimum = min(r["pooled"] for r in eligible)
            selected = next(r for r in eligible if r["pooled"] <= minimum + 1e-10)
            saved_model = order.coupled.insulin.with_parameters(
                child, selected["parameters"]
            )
            if order.coupled.data.parameter_violations(saved_model):
                raise ValueError(
                    "Selected vector lost the frozen insulin law or domain"
                )
            sealed_write(
                root / "selected-model.json",
                {
                    "plan": identity,
                    "training_file_sha256": control_plan["training"],
                    "source_training_freeze": order.FREEZE,
                    "model_sha256": content_hash(saved_model),
                    "model": saved_model,
                },
            )
        return sealed_write(
            root / "summary.json",
            {
                "plan": identity,
                "selected": selected,
                "attempts": qualified,
                "control": control["arms"]["half_gap"],
                "evidence": evidence,
                "equation": child["formula"]["initializers"]["I_act"],
                "zero_gain_max_scaled_difference": data.zero_gain_max_scaled_difference,
                "common_start_max_difference": data.start_differences,
                "validation_opened": False,
                "test_opened": False,
            },
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=OUT)
    result = run(cli.parse_args().root)
    print(
        json.dumps(
            {
                k: {f: result[k].get(f) for f in ("eligible", "pooled", "per_target")}
                for k in ("control", "selected")
            },
            indent=2,
        )
    )
