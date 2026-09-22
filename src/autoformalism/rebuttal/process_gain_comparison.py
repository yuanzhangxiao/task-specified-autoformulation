"""Historical paired gain API and basin-specific evaluator diagnostics."""

from autoformalism.search.process_gains import (  # noqa: F401
    _outer_gain,
    _replace,
    compile_bundle,
)
from autoformalism.search.signed_processes import conversion_value

POLICIES = ("explicit_conversion", "independent_gains")


def conservation_diagnostic(bundle, parameters, training):
    """Training-only cancellation of declared transfers in named physical depths.

    This checks the isolated declared term, not the model's total water balance,
    latent-state meaning, or missing/duplicated physical pathways.
    """
    results = []
    for process in bundle.get("gain_compilation", {}).get("processes", []):
        if process["kind"] != "transfer":
            continue
        uses = process["uses"]
        row = {
            "process": process["process"],
            "scope": "declared_term_in_named_depth_coordinates",
            "scientific_conservation_certified": False,
        }
        dynamic = {s["name"] for s in bundle["candidate"]["states"]}
        if (
            {u["target"] for u in uses} != {"h_up", "h_down"}
            or not {"h_up", "h_down"} <= dynamic
            or parameters is None
        ):
            results.append({**row, "status": "unavailable", "relative_imbalance": None})
            continue
        imbalances = []
        for trajectory in training["rows"]:
            cov = trajectory["fixed_covariates"]
            values = [
                (-1 if u["sign"] == "negative" else 1)
                * parameters[u["gain"]]
                * conversion_value(u["conversion"], cov)
                * cov["area_up" if u["target"] == "h_up" else "area_down"]
                for u in uses
            ]
            total = sum(abs(v) for v in values)
            imbalances.append(abs(sum(values)) / total if total else None)
        finite = [v for v in imbalances if v is not None]
        value = max(finite, default=None)
        results.append(
            {
                **row,
                "status": "evaluated"
                if value is not None
                else "zero_transfer_uninformative",
                "relative_imbalance": value,
                "cancellation_pass": value <= 1e-8 if value is not None else None,
            }
        )
    return results
