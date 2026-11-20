"""Aggregate training diagnostics across fitted endpoints of one frozen parent."""

from __future__ import annotations

from collections.abc import Mapping

RULES = {
    "minimum_starts": 2,
    "material_rmse_fraction": 0.05,
    "accurate_training_nmse": 1e-6,
    "replication": "same trajectory, target, time window and sign of change error",
    "routing": "structural investigation is provisional and requires free, "
    "unexhausted fits plus a repeatable verified substitution",
}


def _window_key(row: dict) -> tuple:
    w = row["window"]
    return (
        w["trajectory_id"],
        w["target"],
        w["start"],
        w["end"],
        w["predicted_change"] > w["observed_change"],
    )


def _direction(outcome: dict) -> str:
    """Require window RMSE and absolute terminal error to move together."""
    margin = max(1e-12, RULES["material_rmse_fraction"] * outcome["free_rmse"])
    change = outcome["rmse_reduction"]
    end = outcome["terminal_error_reduction"]
    if change > margin and end > 0:
        return "improves"
    if change < -margin and end < 0:
        return "worsens"
    return "mixed_or_small"


def _effects(endpoints: Mapping[str, dict]) -> dict:
    effects = {}
    for start, row in endpoints.items():
        for probe in row["probes"]:
            if probe["status"] != "ready":
                continue
            for target, outcome in sorted(probe["outcomes"].items()):
                key = (probe["driver"], target)
                effect = effects.setdefault(
                    key,
                    {
                        "driver": key[0],
                        "target": key[1],
                        "path": probe["paths"][target],
                        "starts": {},
                    },
                )
                effect["starts"][start] = {**outcome, "direction": _direction(outcome)}
    return effects


def _balance_summary(rows: list[dict]) -> str | None:
    """Describe signed fitted contributions only when every balance closed."""
    balances = [r.get("balance") for r in rows]
    if not all(b and b["status"] == "ready" for b in balances):
        return None
    labels = [(term["expression"], term["sign"]) for term in balances[0]["terms"]]
    if any(
        [(term["expression"], term["sign"]) for term in b["terms"]] != labels
        for b in balances[1:]
    ):
        return None
    values = [
        [b["terms"][j]["signed_integral"] for b in balances] for j in range(len(labels))
    ]
    largest = sorted(range(len(labels)), key=lambda j: -max(map(abs, values[j])))[:3]
    terms = [
        f"{labels[j][0]} ({min(values[j]):+.4g} to {max(values[j]):+.4g})"
        for j in largest
    ]
    return (
        f"The largest integrated contributions to {balances[0]['state']} "
        f"in the fitted balance are {', '.join(terms)} (balance-1)."
    )


def summarize(endpoints: Mapping[str, dict], search: Mapping[str, dict]) -> dict:
    """Build the deterministic feedback shared by both arms."""
    if not endpoints or endpoints.keys() != search.keys():
        raise ValueError("search status and endpoint roster must match")
    evidence = {
        "fit-1": {
            "starts": {
                k: {
                    "search": search[k],
                    "training_nmse": r["training_nmse"],
                    "probe_status": r["status"],
                    "reason": r["reason"],
                    "trajectories": r["trajectories"],
                    "windows": r["compared_windows"],
                }
                for k, r in endpoints.items()
            },
            "scope": "Saved training initial values. Search evidence is local.",
        },
        "window-1": {"starts": {k: r["window"] for k, r in endpoints.items()}},
    }
    advice = []
    ready = "insufficient_evidence"
    rows = list(endpoints.values())
    replicated = (
        len(rows) >= RULES["minimum_starts"]
        and all(r["window"] is not None for r in rows)
        and len({_window_key(r) for r in rows}) == 1
    )
    # Each start's balance refers to its own selected window.
    evidence["balance-1"] = {
        "starts": {k: r.get("balance") for k, r in endpoints.items()},
        "scope": "Signed fitted amounts during each start's selected window.",
    }
    if replicated:
        w = rows[0]["window"]
        predictions = [r["window"]["predicted_change"] for r in rows]
        advice.append(
            f"Across {len(rows)} endpoints, {w['target']} on {w['trajectory_id']} "
            f"from {w['start']:g} to {w['end']:g} has the largest standardized "
            f"change error. Recorded change {w['observed_change']:.4g}, fitted "
            f"range {min(predictions):.4g} to {max(predictions):.4g}."
        )
        balance = _balance_summary(rows)
        if balance:
            advice.append(balance)
    repeatable = []
    for i, ((driver, target), entry) in enumerate(
        sorted(_effects(endpoints).items()), 1
    ):
        key = f"probe-{i}"
        evidence[key] = entry
        directions = {v["direction"] for v in entry["starts"].values()}
        if (
            replicated
            and entry["starts"].keys() == endpoints.keys()
            and len(directions) == 1
        ):
            direction = directions.pop()
            if direction != "mixed_or_small":
                repeatable.append(key)
                advice.append(
                    f"Substituting recorded {driver} {direction} both the {target} "
                    f"window RMSE and terminal absolute error at all tested endpoints "
                    f"({key}). Inspect the path " + " → ".join(entry["path"]) + "."
                )
    if len(rows) < RULES["minimum_starts"] or any(
        r["training_nmse"] is None for r in rows
    ):
        advice.append(
            "Obtain independent verified training replays before equation review."
        )
    elif all(
        r["training_nmse"] <= RULES["accurate_training_nmse"] for r in rows
    ) and all(
        r["window"]
        and r["reason"] == "window change errors below presentation threshold"
        for r in rows
    ):
        ready = "retain"
        advice.append(
            "Retain the accurate endpoints and verify public mechanism obligations."
        )
    elif not replicated or any(
        s["search_scope"] != "local_free_fit" or s["budget_exhausted"]
        for s in search.values()
    ):
        ready = "continue_fitting"
        advice.append(
            "Check the mismatch after unrestricted fitting from distinct starts. "
            "The available starts disagree on its location or include a restricted "
            "or budget-limited fit."
        )
    elif repeatable:
        ready = "conditional_equation_review"
        advice.append(
            "Compare changes to the implicated equations after refitting. Check the "
            "targeted response and every other output by free rollout."
        )
    else:
        advice.append(
            "The current probes provide insufficient evidence for an equation revision."
        )
    return {"evidence": evidence, "deterministic_feedback": advice, "readiness": ready}
