"""M13: short optimization trials precede allocation of the remaining budget."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import mesh_refinement_process as process
from autoformalism.fitting import mesh_refinement_worker as mesh
from autoformalism.fitting import portfolio_starts as starts
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_fit, recovery_mesh


class Portfolio:
    """One coordinator, one wall-clock/call allowance, durable subprocess journals."""

    def __init__(self, base: dict, policy: dict, folder: Path):
        self.base, self.policy, self.folder = base, policy, folder
        self.begun = monotonic()
        self.deadline = self.begun + policy["seconds"]
        self.fit_end = self.deadline - policy["certificate_seconds"]
        self.bounds = starts.domain(base)
        self.halted = False
        self.charge = 0
        self.result = {
            "selected": None,
            "levels": [],
            "certificates": [],
            "portfolio": [],
            "budget_restarted": False,
            "reference_values_used": False,
            "validation_used_for_fitting": False,
            "rollout_call_accounting": [],
        }

    def save(self) -> None:
        public._write(self.folder / "progress.json", self.result)

    def operate(
        self, mode: str, payload: dict, folder: Path, cap: float, *, certificate=False
    ) -> dict | None:
        until = self.deadline if certificate else self.fit_end
        seconds = min(cap, until - monotonic())
        if self.halted or seconds < 1:
            return None
        payload = {**payload, "seconds": seconds}
        outcome = process.invoke(mode, payload, folder, seconds)
        self.halted |= not outcome["termination_confirmed"]
        value = (
            public._read(folder / "result.json")
            if (not self.halted and (folder / "result.json").exists())
            else None
        )
        if value and value.get("payload_sha256") != public.content_sha256(payload):
            raise ValueError("portfolio worker output identity differs")
        return {"process": outcome, "value": value}

    def retain(self, value: dict | None, origin: str) -> dict | None:
        """Only a finite, complete training evaluation may become an incumbent."""
        if not value or not value.get("parameters"):
            return None
        score = value.get("training_nmse")
        if not isinstance(score, (int, float)) or not np.isfinite(score) or score < 0:
            return None
        if not starts.valid(value["parameters"], self.bounds):
            raise ValueError("portfolio result parameter domain differs")
        point = {
            "parameters": value["parameters"],
            "training_nmse": score,
            "origin": origin,
        }
        incumbent = self.result["selected"]
        if incumbent is None or score < incumbent["training_nmse"]:
            self.result["selected"] = point
        return point

    def certificate(self, *, final=False) -> bool:
        selected = self.result["selected"]
        if not selected or self.halted:
            return False
        if not final and selected["training_nmse"] > self.policy["training_nmse"]:
            return False
        digest = public.content_sha256(selected["parameters"])
        for old in self.result["certificates"]:
            if old["parameters_sha256"] == digest:
                return old["passed"]
        out = self.operate(
            "recovery_check",
            {k: self.base[k] for k in ("request", "training")}
            | {"parameters": selected["parameters"]}
            | {
                k: self.policy[k]
                for k in ("training_nmse", "trajectory_nmse", "solver_agreement")
            },
            self.folder / "checks" / digest,
            self.policy["certificate_seconds"],
            certificate=True,
        )
        if not out:
            return False
        value = out["value"] or {}
        if value and value.get("parameters") != selected["parameters"]:
            raise ValueError("portfolio certificate vector differs")
        passed = bool(out["process"]["status"] == "complete" and value.get("passed"))
        self.result["certificates"].append(
            {"parameters_sha256": digest, **out, "passed": passed}
        )
        self.save()
        return passed

    def collocation(self) -> dict | None:
        """One bounded medium solve supplies a candidate, never replaces the start."""
        grid, audit = recovery_mesh.grids(
            self.base,
            self.policy["medium_target"],
            self.policy["minimum_intervals"],
            self.policy["observation_anchors"],
        )
        payload = mesh.native_payload(
            self.base,
            self.base["start"],
            grid,
            recovery_mesh.initial_guesses(self.base, grid),
        ) | {"strict_parameter_bounds": True}
        path = self.folder / "level-0"
        out = self.operate(
            "native", payload, path / "native", self.policy["native_seconds"]
        )
        if not out:
            return None
        record = {"level": 0, "mesh": audit, "process": out["process"]}
        status = path / "native/final_checkpoint_status.json"
        record["final_checkpoint"] = (
            public._read(status)
            if status.exists()
            else {"status": "not_published_before_worker_exit", "published": False}
        )
        self.result["levels"].append(record)
        self.save()
        if self.halted:
            return None
        candidates = []
        for i, point in enumerate(recovery_fit.checkpoint_points(path / "native")):
            if not starts.valid(point["parameters"], self.bounds):
                continue
            screened = self.operate(
                "point",
                {k: self.base[k] for k in ("request", "training")}
                | {
                    "parameters": point["parameters"],
                    "screening": {
                        "method": "Radau",
                        "point_seconds": self.policy["point_seconds"],
                    },
                },
                path / f"screen-{i}",
                self.policy["point_seconds"],
            )
            if screened and screened["process"]["status"] == "complete":
                value = screened["value"]
                if value and value.get("status") == "complete":
                    if value["parameters"] != point["parameters"]:
                        raise ValueError("portfolio screen vector differs")
                    retained = self.retain(value, f"collocation_checkpoint_{i}")
                    if retained:
                        candidates.append(retained)
            self.save()
        return min(candidates, key=lambda p: p["training_nmse"]) if candidates else None

    def trial(self, candidate: dict, *, initial: bool) -> bool:
        """One bounded warm restart; a failed derivative does not kill the portfolio."""
        p = self.policy
        cap = p["probe_seconds"] if initial else p["continuation_seconds"]
        call_limit = min(
            p["probe_calls"] if initial else p["continuation_calls"],
            p["maximum_rollout_calls"] - self.charge,
        )
        if self.halted or call_limit < 5 or self.fit_end - monotonic() < 5:
            return False
        index = sum(len(c["trials"]) for c in self.result["portfolio"])
        folder = self.folder / "trials" / f"{index:03d}"
        before = candidate.get("best")
        parameters = before["parameters"] if before else candidate["parameters"]
        out = self.operate(
            "recovery_rollout",
            {k: self.base[k] for k in ("request", "training", "coordinates")}
            | {
                "points": [{"parameters": parameters, "source": candidate["source"]}],
                "maximum_calls": call_limit,
                "training_nmse": p["training_nmse"],
                "trajectory_nmse": p["trajectory_nmse"],
            },
            folder,
            cap,
        )
        if not out:
            return False
        value = out["value"]
        # A worker killed before publishing its final count cannot get free calls.
        observed = value.get("actual_residual_calls") if value else None
        if observed is not None and (
            type(observed) is not int or not 0 <= observed <= call_limit
        ):
            raise ValueError("invalid worker evaluation accounting")
        charge = observed if observed is not None else call_limit
        self.charge += charge
        self.result["rollout_call_accounting"].append(
            {
                "trial": index,
                "observed_calls": observed,
                "budget_charge": charge,
                "unknown_calls_charged_at_cap": observed is None,
            }
        )
        best = folder / "refinement/best.json"
        if not self.halted and best.exists():
            partial = public._read(best)
            if (
                not value
                or value.get("training_nmse") is None
                or (partial["training_nmse"] < value["training_nmse"])
            ):
                value = {**(value or {}), **partial}
        point = (
            self.retain(value, f"trial_{index}:{candidate['source']}")
            if not self.halted
            else None
        )
        if point and (
            before is None or point["training_nmse"] < before["training_nmse"]
        ):
            candidate["best"] = point
        stages_path = folder / "refinement/stages.json"
        stages = public._read(stages_path) if stages_path.exists() else []
        reason = (
            stages[-1]["stop_reason"]
            if stages
            else ((out["value"] or {}).get("stop_reason", out["process"]["status"]))
        )
        elapsed = out["process"].get("elapsed_seconds", 0)
        improvement = (
            max(0, before["training_nmse"] - point["training_nmse"])
            if before and point
            else 0
        )
        relative = improvement / max(before["training_nmse"], 1e-30) if before else None
        candidate["progress_per_second"] = improvement / max(elapsed, 1e-9)
        candidate["trials"].append(
            {
                "index": index,
                "initial": initial,
                "process": out["process"],
                "stop_reason": reason,
                "relative_improvement": relative,
                "training_nmse": point["training_nmse"] if point else None,
            }
        )
        if reason in {
            "point_allowance_exhausted",
            "numerical_failure",
            "no_feasible_training_point",
            "optimizer_terminated",
            "stalled",
        } or not candidate.get("best"):
            candidate["retired"] = reason
        if not initial and relative is not None:
            candidate["stagnant_visits"] = (
                candidate.get("stagnant_visits", 0) + 1
                if relative < p["progress_floor"]
                else 0
            )
            if candidate["stagnant_visits"] >= p["stagnant_visits"]:
                candidate["retired"] = "insufficient_progress_across_trials"
        self.save()
        return True

    def run(self, arm: str) -> dict:
        candidates = starts.generate(self.base, self.policy["perturbation_spread"])
        if arm == "checkpoint_portfolio":
            point = self.collocation()
            if point and point["parameters"] != self.base["start"]:
                candidates[1] = {
                    "source": point["origin"],
                    "parameters": point["parameters"],
                }
        unique = {}
        for candidate in candidates:
            unique.setdefault(public.content_sha256(candidate["parameters"]), candidate)
        candidates = list(unique.values())
        self.result["portfolio"] = [
            {**c, "trials": [], "retired": None} for c in candidates
        ]
        self.save()
        certified = self.certificate()
        for candidate in self.result["portfolio"]:
            if certified or not self.trial(candidate, initial=True):
                break
            certified = self.certificate()
            self.reject_uncertified_stop(candidate, certified)
        while not certified:
            candidate = starts.choose(self.result["portfolio"])
            if not candidate or not self.trial(candidate, initial=False):
                break
            certified = self.certificate()
            self.reject_uncertified_stop(candidate, certified)
        if not certified:
            certified = self.certificate(final=True)
        self.result.update(
            stop_reason="training_prediction_certified"
            if certified
            else "cleanup_unconfirmed"
            if self.halted
            else "portfolio_or_budget_stop",
            fit_seconds=monotonic() - self.begun,
            fitting_allowance_seconds=self.policy["seconds"],
            rollout_calls_charged=self.charge,
        )
        self.save()
        return self.result

    def reject_uncertified_stop(self, candidate: dict, certified: bool) -> None:
        """Do not repeatedly stop at the same point after independent rejection."""
        if not certified and candidate["trials"][-1]["stop_reason"] == (
            "training_accuracy_reached_pending_replay"
        ):
            candidate["retired"] = "training_stop_not_certified"
            self.save()


def fit(base: dict, arm: str, policy: dict, folder: Path) -> dict:
    """Preserve historical rollout control and never restart interrupted budgets."""
    if arm == "rollout_only":
        return recovery_fit.fit(base, arm, policy, folder)
    if arm not in {"rollout_portfolio", "checkpoint_portfolio"}:
        raise ValueError("unknown portfolio arm")
    identity = public.content_sha256({"base": base, "arm": arm, "policy": policy})
    path = folder / "fit-started.json"
    if path.exists():
        if read_seal(path)["identity"] != identity:
            raise ValueError("portfolio resume identity differs")
        progress = folder / "progress.json"
        prior = (
            public._read(progress)
            if progress.exists()
            else {"selected": None, "levels": []}
        )
        return {
            **prior,
            "stop_reason": "interrupted_no_fit_restart",
            "budget_restarted": False,
        }
    runner = Portfolio(base, policy, folder)
    seal(path, {"identity": identity})
    return runner.run(arm)
