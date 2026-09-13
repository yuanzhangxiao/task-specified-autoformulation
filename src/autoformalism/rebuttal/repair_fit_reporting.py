"""Read fitter stage evidence without changing any optimizer or its acceptance."""


def _any_known(values):
    """Unknown is not false; one affirmative observation is sufficient for any."""
    values = list(values)
    if True in values:
        return True
    return False if values and all(v is False for v in values) else None


def refinement_report(refinement: dict) -> dict:
    """Distinguish stage convergence, selected point provenance, and causal replay."""
    recovery = refinement.get("method") == "feasibility_then_refinement"
    records = (
        refinement.get("stages", [])
        if recovery
        else ([{"mode": "refinement", "result": refinement}] if refinement else [])
    )
    stages = []
    selected_converged = False
    for record in records:
        result = record.get("result") or {}
        native = result.get("optimizer_native_success")
        verified = result.get("optimizer_success")
        point = result.get("native_optimizer_parameters", result.get("parameters"))
        matches = isinstance(point, dict) and point == refinement.get("parameters")
        selected_converged |= native is True and verified is True and matches
        stages.append(
            {
                "mode": record.get("mode"),
                "source": record.get("source"),
                "native_success": native,
                "verified_success": verified,
                "selected_point_matches": matches,
                "message": result.get("message", record.get("error")),
                "finite_evaluations": result.get("valid_residual_evaluations"),
            }
        )
    production = refinement.get("production_training_rollout_verified")
    return {
        "optimizer_stages": stages,
        "any_stage_native_success": _any_known(s["native_success"] for s in stages),
        "any_stage_verified_success": _any_known(s["verified_success"] for s in stages),
        "selected_optimizer_converged": selected_converged if stages else None,
        "selected_training_rollout_verified": production,
        "selection": refinement.get("selection"),
        "finite_residual_evaluations": refinement.get("valid_residual_evaluations"),
    }


def fit_outcomes(fit: dict) -> dict:
    """A completed finite fit is not itself a convergence or scientific verdict."""
    report = refinement_report(fit.get("refinement") or {})
    return {
        **report,
        "finite_candidate": fit.get("status") == "complete",
        "native_optimizer_success": report["any_stage_native_success"],
        "verified_optimizer_success": (
            report["selected_optimizer_converged"]
            and report["selected_training_rollout_verified"] is True
        )
        if report["selected_optimizer_converged"] is not None
        else None,
    }
