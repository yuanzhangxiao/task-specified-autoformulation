"""Saved-endpoint continuation and a labelled absolute-insulin domain control."""

from __future__ import annotations

from itertools import pairwise
from time import monotonic, process_time

import numpy as np

from autoformalism.fitting import concentration_profile as positive
from autoformalism.fitting import insulin_revision_fit as previous
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "insulin-positive-continuation-kernel-1"
DENSE_SUBDIVISIONS = 8
DENSE_TOLERANCE = 1e-8


def verification_grid(training: PublicSplit) -> PublicSplit:
    """Resample only the existing public linear schedules, never add fitting data.

    Interpolated targets are bookkeeping for recovering predictions from the
    rollout residual. Their errors are never used as a fitting/selection loss.
    """
    payload = training.model_dump(mode="json")
    payload["fingerprint"] += ":dense-domain-verification-only"
    for row in payload["rows"]:
        old = np.asarray(row["time"])
        times = np.concatenate(
            [
                np.linspace(a, b, DENSE_SUBDIVISIONS, endpoint=False)
                for a, b in pairwise(old)
            ]
            + [old[-1:]]
        )
        for group in ("targets", "auxiliaries", "external_inputs"):
            row[group] = {
                name: np.interp(times, old, values).tolist()
                for name, values in row[group].items()
            }
        row["time"] = times.tolist()
    return PublicSplit.model_validate(payload)


class ContinuationFit(previous.InsulinRevisionFit):
    """Keep the same family/scopes; recenter the chart at a verified saved fit."""

    def __init__(
        self,
        problem,
        parent_problem,
        training,
        numerical,
        warm_start_parameters,
        *,
        endpoint,
        enforce_positive,
    ):
        if type(enforce_positive) is not bool:
            raise ValueError("explicit boolean domain-control arm required")
        super().__init__(
            problem, parent_problem, training, numerical, warm_start_parameters
        )
        self.enforce_positive = enforce_positive
        self.source_endpoint = endpoint
        canonical = previous.canonical_parameters(self.original, endpoint["parameters"])
        self.original, self.oracle = previous.polish.make_oracle(
            problem, training, numerical, canonical
        )
        self.start = dict(self.oracle.start)
        if enforce_positive:
            self.profiler_factory = positive.InsulinConcentrationProfile
        profiler = self.profiler_factory(self.oracle)
        self.profile_audit = profiler.audit
        self.chart = previous.CenteredMealChart(
            self.oracle, self.start, profiler, self.feedback_unit
        )
        self.dense_training = verification_grid(training)
        self.dense = previous.kernel.TrainingOracle(
            problem, self.dense_training, numerical
        )

    def setup_audit(self):
        return super().setup_audit() | {
            "protocol": PROTOCOL,
            "source_endpoint_sha256": previous.public.content_sha256(
                self.source_endpoint
            ),
            "enforce_positive": self.enforce_positive,
            "constraint_channel": "I" if self.enforce_positive else None,
            "dense_verification_subdivisions": DENSE_SUBDIVISIONS,
            "dense_scaled_tolerance": DENSE_TOLERANCE,
            "continuous_positivity_certified": False,
            "source_search_budget_renewed": False,
            "state_equations_and_parameter_domains_unchanged": True,
        }

    def concentration_audit(self, parameters, seconds):
        """Independently replay originals on a finer grid in both matched arms."""
        wall, cpu = monotonic(), process_time()
        residual, _ = self.dense.evaluate(
            self.dense.vector(parameters),
            monotonic() + seconds,
            jacobian=False,
            method="Radau",
        )
        channels = self.dense.system.channels
        residual = residual.reshape(-1, len(channels))
        outputs = {}
        for j, name in enumerate(channels):
            observed = np.concatenate(
                [np.asarray(row.targets[name]) for row in self.dense_training.rows]
            )
            prediction = observed + self.dense.scales[name] * residual[:, j]
            scale = self.original.scales[name]
            outputs[name] = {
                "minimum": float(np.min(prediction)),
                "negative_samples": int(np.sum(prediction < 0)),
                "samples_below_tolerance": int(
                    np.sum(prediction < -DENSE_TOLERANCE * scale)
                ),
                "samples": len(prediction),
                "physical_tolerance": DENSE_TOLERANCE * scale,
            }
        return {
            "passed": outputs["I"]["samples_below_tolerance"] == 0,
            "enforced_channel": "I",
            "outputs": outputs,
            "method": "independent_original_Radau",
            "subdivisions": DENSE_SUBDIVISIONS,
            "scope": "all training horizons on finer grid, not continuous certificate",
            "interpolated_targets_used_for_fitting": False,
            "continuous_positivity_certified": False,
            "wall_seconds": monotonic() - wall,
            "cpu_seconds": process_time() - cpu,
            "whole_training_rollouts": 1,
        }

    def verify(self, parameters, expected_nmse, seconds, save):
        started = monotonic()
        if set(parameters) == set(self.original.names):
            parameters = previous.continuation.to_effective(
                self.original,
                self.oracle,
                previous.canonical_parameters(self.original, parameters),
            )
        result = super().verify(parameters, expected_nmse, seconds, save)
        if result.get("original_equations_verified"):
            for field in ("verification_calls_attempted", "verification_calls_started"):
                result[field] = result.get(field, 0) + 1
            save(result)
            try:
                audit = self.concentration_audit(
                    result["parameters"], max(0.001, seconds - (monotonic() - started))
                )
            except BaseException:
                result["verification_calls_failed"] = (
                    result.get("verification_calls_failed", 0) + 1
                )
                save(result)
                raise
            result["verification_calls_completed"] = (
                result.get("verification_calls_completed", 0) + 1
            )
            result["timing"]["numerical_active_wall_seconds"] += audit["wall_seconds"]
            result["timing"]["elapsed_wall_seconds"] += audit["wall_seconds"]
            result["timing"]["elapsed_cpu_seconds"] += audit["cpu_seconds"]
            result["concentration_audit"] = audit
            result["domain_admissible"] = audit["passed"]
            result["eligible_for_arm"] = not self.enforce_positive or audit["passed"]
            # Replay correctness and domain admissibility are distinct fields.
            save(result)
        return result

    def verify_source(self, seconds, save):
        """Recheck source without calling a negative fit physically admissible."""
        return self.verify(
            self.source_endpoint["parameters"],
            self.source_endpoint["training_nmse"],
            seconds,
            save,
        )

    def search(self, policy, save, *, profiled_start):
        """Seed retention with the qualified arm-feasible profile, not raw source.

        In particular, a lower-loss negative source must not prevent feasible
        positive-arm checkpoints from being retained by the common search code.
        """
        self.oracle.vector(profiled_start)
        return previous.roots.root_profile_search(
            self.oracle,
            profiled_start,
            policy.search_seconds,
            policy.maximum_calls,
            save,
            fixed=self.fixed,
            profiler_factory=self.profiler_factory,
            coordinate_chart=self.chart,
            failure_diagnostics=True,
            checkpoint_policy=previous.SearchCheckpointPolicy(
                interval_seconds=policy.checkpoint_interval_seconds
            ),
            overall_wall_seconds=policy.search_seconds
            + policy.persistence_headroom_seconds,
        )

    def local_audit(self, endpoint, seconds, save):
        """Test small feasible directions with re-solved inner variables.

        This is a sampled descent check, not a derivative or convergence proof.
        Every useful sampled point is checkpointed for independent verification.
        """
        deadline = monotonic() + seconds
        profiler = self.profiler_factory(self.oracle)
        parameters = previous.continuation.to_effective(
            self.original, self.oracle, endpoint["parameters"]
        )
        chart = previous.CenteredMealChart(
            self.oracle, parameters, profiler, self.feedback_unit
        )
        center, units = chart.encode(parameters), chart.units(parameters)
        baseline = profiler.profile(parameters, deadline)
        result = {
            "status": "partial",
            "probes": [],
            "baseline": baseline,
            "audited_endpoint_sha256": previous.public.content_sha256(endpoint),
            "audited_parameters_sha256": previous.public.content_sha256(
                endpoint["parameters"]
            ),
            "calls_started": 1,
            "calls_completed": 1,
            "best": baseline,
            "convergence_certified": False,
            "resolved_descent": False,
            "objective_improvement": 0.0,
            "interpretation": "finite direction probes, not derivative checks",
        }
        save(result)
        for step in (0.01, 0.001):
            for index in profiler.outer:
                name = self.oracle.names[index]
                if name in self.fixed:
                    continue
                for sign in (-1, 1):
                    previous.roots._check(deadline)
                    point = center.copy()
                    point[index] += sign * step * units[index]
                    if not chart.lower[index] <= point[index] <= chart.upper[index]:
                        continue
                    result["calls_started"] += 1
                    probe = {"coordinate": name, "relative_step": sign * step}
                    try:
                        trial = profiler.profile(
                            self.oracle.parameters(chart.decode(point)), deadline
                        )
                        result["calls_completed"] += 1
                        probe.update(
                            status="evaluated", training_nmse=trial["training_nmse"]
                        )
                        if trial["training_nmse"] < result["best"]["training_nmse"]:
                            result["best"] = trial
                        improvement = (
                            baseline["training_nmse"] - result["best"]["training_nmse"]
                        )
                        result.update(
                            objective_improvement=improvement,
                            resolved_descent=improvement
                            > max(1e-9, 1e-7 * baseline["training_nmse"]),
                        )
                    except (ValueError, previous.roots.NumericalInvalidTrial) as error:
                        probe.update(status="unavailable", error=str(error)[:300])
                    result["probes"].append(probe)
                    save(result)
        improvement = baseline["training_nmse"] - result["best"]["training_nmse"]
        result.update(
            status="complete",
            resolved_descent=improvement > max(1e-9, 1e-7 * baseline["training_nmse"]),
            objective_improvement=improvement,
            all_probes_available=all(
                p["status"] == "evaluated" for p in result["probes"]
            ),
        )
        save(result)
        return result
