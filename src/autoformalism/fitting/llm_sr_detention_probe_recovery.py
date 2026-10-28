"""Durable exact-sample recovery for detention search derivative qualification.

Only completed residuals and authenticated reference columns are reused. The
campaign owns nonrenewable operation allowances; this cache grants no new budget,
optimizer convergence, parameter confidence, or permission to use held-out data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from time import monotonic, process_time

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_recovery as previous
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_local_derivatives as local
from autoformalism.fitting import public_fitting as public

PROTOCOL = "llm-sr-detention-probe-recovery-1"


@lru_cache(maxsize=1)
def _implementation_identity() -> str:
    """Pin all numerical/compiler dependencies once in a frozen worker process."""
    return public._source_identity()


def _array(raw, *, length: int | None = None) -> np.ndarray:
    """Validate a finite complete residual/column before any cache use."""
    value = np.asarray(raw, dtype=float)
    if (
        value.ndim != 1
        or not len(value)
        or not np.isfinite(value).all()
        or (length is not None and len(value) != length)
    ):
        raise ValueError("invalid cached residual or derivative column")
    return value


class SampleCache:
    """An endpoint/algorithm-bound cache of complete numerical samples only.

    Files are atomically sealed. In-memory failures prevent a repeated failed
    request from receiving another try in the same coordinate allowance. A new
    invocation may retry missing samples only under the caller's new allowance.
    """

    def __init__(self, root: Path, identity: dict):
        self.root = Path(root)
        self.identity = identity
        self.identity_sha256 = public.content_sha256(identity)
        self.residual_length = identity.get("residual_length")
        self.hits = self.misses = self.writes = self.seeded = 0
        self.io_seconds = self.integration_seconds = self.integration_cpu_seconds = 0.0
        self.memory: dict[str, np.ndarray] = {}
        self.failed: dict[str, Exception] = {}
        self.requests: list[dict] = []
        start = monotonic()
        with public._lock(self.root):
            seal(self.root / "identity.json", identity)
        self.io_seconds += monotonic() - start

    def request(self, point, kind="residual", *, index=None, method="Radau"):
        """Build an exact binary64 vector key, not rounded parameter matching."""
        vector = _array(point)
        return {
            "cache_identity_sha256": self.identity_sha256,
            "kind": kind,
            "vector": vector.tolist(),
            "vector_sha256": adequacy._digest(vector),
            "coordinate_index": index,
            "method": method,
        }

    def _path(self, request):
        return self.root / "samples" / (public.content_sha256(request) + ".json")

    def load(self, request):
        """Authenticate a matching sample or report it absent, never synthesize it."""
        key = public.content_sha256(request)
        if key in self.memory:
            self.hits += 1
            return self.memory[key].copy()
        path = self._path(request)
        began = monotonic()
        if not path.exists():
            self.io_seconds += monotonic() - began
            return None
        record = read_seal(path)
        self.io_seconds += monotonic() - began
        if record.get("request") != request or record.get("status") != "complete":
            raise ValueError("cached sample request differs")
        value = _array(record["values"], length=self.residual_length)
        if adequacy._digest(value) != record.get("values_sha256"):
            raise ValueError("cached sample array digest differs")
        self.memory[key] = value.copy()
        self.hits += 1
        return value

    def save(self, request, raw, *, provenance):
        """Publish a completed sample; conflicting repeated results are errors."""
        value = _array(raw, length=self.residual_length)
        previous_value = self.load(request)
        if previous_value is not None:
            if not np.array_equal(previous_value, value):
                raise ValueError("conflicting residual or derivative cache sample")
            return
        record = {
            "request": request,
            "values": value.tolist(),
            "values_sha256": adequacy._digest(value),
            "provenance": provenance,
            "status": "complete",
        }
        began = monotonic()
        with public._lock(self.root):
            path = self._path(request)
            if path.exists():
                concurrent = read_seal(path)
                if (
                    concurrent["request"] != request
                    or concurrent.get("status") != "complete"
                    or concurrent.get("values_sha256") != adequacy._digest(value)
                    or not np.array_equal(_array(concurrent["values"]), value)
                ):
                    raise ValueError("conflicting concurrent numerical sample")
            else:
                seal(path, record)
                self.writes += 1
        self.io_seconds += monotonic() - began
        self.memory[public.content_sha256(request)] = value.copy()

    def residual(self, oracle, point, deadline, method="Radau"):
        """Evaluate only missing exact samples within the caller's deadline."""
        if monotonic() >= deadline:
            raise TimeoutError("sample recovery deadline exhausted")
        request = self.request(point, method=method)
        key = public.content_sha256(request)
        found = self.load(request)
        if found is not None:
            self.requests.append({"key": key, "status": "reused"})
            return found
        if key in self.failed:
            raise self.failed[key]
        self.misses += 1
        began, cpu = monotonic(), process_time()
        try:
            value = _array(oracle(point, deadline, method))
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as exc:
            self.failed[key] = exc
            self.requests.append({"key": key, "status": "failed", "error": str(exc)})
            raise
        finally:
            self.integration_seconds += monotonic() - began
            self.integration_cpu_seconds += process_time() - cpu
        self.save(request, value, provenance={"kind": "new_complete_rollout"})
        self.requests.append({"key": key, "status": "computed"})
        return value


def _seed(cache, oracle, point, identity, parameters, records, policy):
    """Import raw base/FD arrays only after the existing coordinate proof checks."""
    indexed = set()
    chart_point = dict(zip(oracle.names, point.tolist(), strict=True))
    provenance, contradictions = [], {}
    for record in records:
        if record.get("original_parameters") != parameters:
            raise ValueError("seed original endpoint differs")
        i, check = previous._validate_row(
            record, identity, chart_point, oracle.chart.domain, policy, "tight"
        )
        if i in indexed:
            raise ValueError("duplicate seed coordinate")
        indexed.add(i)
        # An interruption before the reference column finished contributes no
        # seed; its missing later audit probes must not be reconstructed from
        # scalar error summaries. Validate whatever raw pair is actually present.
        if "base_residual" not in record or "jacobian_column" not in record:
            continue
        residual = _array(record["base_residual"], length=cache.residual_length)
        column = _array(record["jacobian_column"], length=len(residual))
        if (
            adequacy._digest(residual) != check.base_residual_sha256
            or adequacy._digest(column[:, None]) != check.base_jacobian_sha256
        ):
            raise ValueError("seed raw arrays differ from audited digests")
        if check.failed_indices:
            # Other coordinates may still reuse trustworthy references. This
            # coordinate instead receives a fresh reference and must recheck at
            # least every previously attempted scale before a passing stop.
            contradictions[i] = max(
                (probe.scale_index + 1 for probe in check.local_probe_attempts),
                default=policy.numerical.derivative.local_max_scales,
            )
            continue
        origin = {
            "kind": "historical_raw_reference",
            "evidence_sha256": record["evidence_sha256"],
        }
        cache.save(cache.request(point), residual, provenance=origin)
        cache.save(
            cache.request(point, "reference_column", index=i), column, provenance=origin
        )
        cache.seeded += 1
        provenance.append(record["evidence_sha256"])
    return provenance, contradictions


def direction(
    item,
    context,
    training,
    parameters: Mapping[str, float],
    index: int,
    *,
    mode,
    cache_root: Path,
    policy=None,
    seconds: float = 240,
    seed_directions: Sequence[dict] = (),
    checkpoint: Callable[[dict], None] | None = None,
) -> dict:
    """Recover one search audit with shared exact samples and unchanged thresholds.

    The caller must seal a fresh nonrenewable allowance before calling. This
    routine never reopens a consumed campaign operation or fits parameters.
    Historical evidence is provenance, not an automatic passing search decision.
    """
    began, cpu, deadline = original._clock(seconds)
    policy = chart.effective_policy(policy, mode=mode)
    identity = chart._identity(
        item, context, training, parameters, mode, "search", "tight", policy
    )
    oracle = chart.ChartOracle(
        item, context, training, parameters, mode, policy, "tight", checkpoint
    )
    oracle.native.role = "verification"
    point = oracle.vector(parameters)
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(point)
    ):
        raise ValueError("direction index outside chart roster")
    cache = SampleCache(
        cache_root,
        {
            "protocol": PROTOCOL,
            "coordinate_identity": identity,
            "implementation_sha256": _implementation_identity(),
            "runtime": public._runtime(),
            "integration_settings": oracle.settings.model_dump(mode="json"),
            "reference_fd_relative_step": oracle.policy.finite_difference_relative_step,
            "residual_length": sum(
                len(row.time) for row in oracle.native.split.trajectories
            ),
        },
    )
    sources, contradictions = _seed(
        cache, oracle, point, identity, parameters, seed_directions, policy
    )
    native_residual = oracle.residual
    oracle.residual = lambda value, stop, method="Radau": cache.residual(
        native_residual, value, stop, method
    )
    state = {}

    def evaluate(value, stop):
        full = point.copy()
        full[index] = value[0]
        request = cache.request(full, "reference_column", index=index)
        column = cache.load(request)
        if column is None:
            residual, column = previous._column(oracle, full, index, stop)
            cache.save(
                request, column, provenance={"kind": "new_complete_reference_column"}
            )
        else:
            residual = oracle.residual(full, stop)
        state.update(base_residual=residual, jacobian_column=column)
        return residual, column[:, None]

    def residual_only(value, stop):
        full = point.copy()
        full[index] = value[0]
        return oracle.residual(full, stop)

    check = local.local_derivative_check(
        evaluate,
        point[index : index + 1],
        oracle.lower[index : index + 1],
        oracle.upper[index : index + 1],
        deadline,
        coordinate_units=oracle.units[index : index + 1],
        policy=policy.numerical.derivative,
        residual_only=residual_only,
        local_coordinate_scales=oracle.local_coordinate_scales(point)[
            index : index + 1
        ],
        search_error_limits=None
        if index in contradictions
        else (
            policy.search_gradient_absolute_error,
            policy.search_gradient_relative_error,
        ),
        minimum_probe_scales=contradictions.get(index, 0),
    )
    outcome = (
        "timed_out"
        if check.error and "TimeoutError" in check.error
        else (
            "unavailable"
            if check.error or check.checked_indices != (0,)
            else "failed"
            if check.failed_indices
            else "checked"
        )
    )
    result = {
        "identity": identity,
        "index": index,
        "name": oracle.names[index],
        "purpose": "search",
        "precision": "tight",
        "parameters": dict(zip(oracle.names, point.tolist(), strict=True)),
        "original_parameters": dict(parameters),
        "coordinate_names": list(oracle.names),
        "coordinate_units": oracle.units.tolist(),
        "lower_bounds": oracle.lower.tolist(),
        "upper_bounds": oracle.upper.tolist(),
        "integration_settings": oracle.settings.model_dump(mode="json"),
        "optimizer_fd_relative_step": oracle.policy.finite_difference_relative_step,
        "derivative": public._jsonable(check),
        "status": "complete"
        if check.checked_indices == (0,) and not check.error
        else "unavailable",
        "diagnostic_outcome": outcome,
        **{key: value.tolist() for key, value in state.items()},
        "parameter_information_qualified": False,
        "full_model_convergence_claimed": False,
        "validation_or_test_access": False,
        "imported_evidence": None,
        "probe_recovery": {
            "protocol": PROTOCOL,
            "cache_identity_sha256": cache.identity_sha256,
            "seed_evidence_sha256": sources,
            "seeded_reference_columns": cache.seeded,
            "historical_failed_coordinates": sorted(contradictions),
            "required_historical_probe_scales": contradictions.get(index, 0),
            "cache_hits": cache.hits,
            "missing_residual_requests": cache.misses,
            "new_sample_records": cache.writes,
            "sample_requests": cache.requests,
            "search_error_limits": [
                policy.search_gradient_absolute_error,
                policy.search_gradient_relative_error,
            ],
            "old_allowances_renewed": False,
        },
    }
    result |= original._accounting(oracle, began, cpu, "verification")
    result["timing"] |= {
        "cache_io_wall_seconds": cache.io_seconds,
        "residual_compute_wall_seconds": cache.integration_seconds,
        "residual_compute_cpu_seconds": cache.integration_cpu_seconds,
        "callback_wall_seconds": None,
        "numerical_active_wall_seconds": None,
        "scope": (
            "Elapsed covers cache IO and numerical work; residual compute includes "
            "native progress callbacks; absent exclusive categories are unknown."
        ),
    }
    result |= {
        "numerical_cost": dict(oracle.cost),
        "wall_seconds": monotonic() - began,
        "budget_exhausted": monotonic() >= deadline,
    }
    result["evidence_sha256"] = public.content_sha256(result)
    if checkpoint:
        checkpoint(result)
    return result
