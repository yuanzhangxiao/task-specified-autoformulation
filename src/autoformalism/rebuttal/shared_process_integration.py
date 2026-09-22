"""Small general integration pilot; report execution separately from discovery."""

from collections import Counter
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.search.shared_model_revision import relationships


def tasks(config) -> list[dict]:
    """One construction and one revision per seed/prompt, without extra fit arms."""
    result = []
    for ci, cell in enumerate(config.public_cells):
        for seed in config.seeds:
            arms = ["full", "brief_only"]
            if (ci + seed) % 2:
                arms.reverse()
            for arm in arms:
                result.append(
                    {
                        "index": len(result),
                        "cell": cell,
                        "seed": seed,
                        "arm": arm,
                        "process_guidance": "on",
                        "task_id": f"cell{ci:02d}_seed{seed}_{arm}",
                        "shared_round_zero": None,
                    }
                )
    return result


def write_report(root: Path, summary: dict) -> None:
    """Preserve missing/failed cases and both trial and retained-model evidence."""
    rows = []
    equations = [
        "# General shared-process integration equations",
        "",
        "Trial and retained models are distinct; only public development data.",
        "",
    ]
    for row in summary["rows"]:
        directory = (
            root / "results" / row["task_id"] / f"round_{row['phase_round']:02d}"
        )
        proposal = (
            sealed_read(directory / "proposal.json")
            if (directory / "proposal.json").exists()
            else {}
        )
        result = (
            sealed_read(directory / "result.json")
            if (directory / "result.json").exists()
            else {}
        )
        selected = result.get("selected")
        trial = result.get("trial")
        record = {
            **row,
            "construction_fallback": proposal.get("fallback_used"),
            "construction_attempts": proposal.get("attempts", [])
            if row["round"] == 0
            else None,
            "retained_shared_relationships": relationships(
                selected["bundle"]["candidate"]
            )
            if selected
            else None,
            "trial_shared_relationships": relationships(trial["bundle"]["candidate"])
            if trial
            else None,
            "revision_propagation": (proposal.get("decision") or {})
            .get("provenance", {})
            .get("shared_law_revision"),
            "scientific_correctness_certified": False,
            "retained_public_predicates": selected.get("certificate")
            if selected
            else None,
            "trial_public_predicates": trial.get("certificate") if trial else None,
            "retained_complexity": _complexity(selected),
            "trial_complexity": _complexity(trial),
        }
        rows.append(record)
        for label, subject in (("trial", trial), ("retained", selected)):
            if subject is None:
                continue
            c = subject["bundle"]["candidate"]
            equations += [
                f"## {row['task_id']} / round {row['round']} / {label}",
                "",
                "```text",
            ]
            equations += [f"{p['name']} = {p['expression']}" for p in c["processes"]]
            equations += [f"{e['state']}' = {e['rhs']}" for e in c["state_equations"]]
            equations += [
                f"output {o['channel']} = {o['expression']}"
                for o in c["observation_mappings"]
            ]
            equations += ["```", ""]
    value = {
        "protocol": "shared-process-integration-1",
        "plan_sha256": summary["plan_sha256"],
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "planned_round_rows": len(rows),
        "benchmark_specific_checks": False,
        "integration_only": True,
        "test_data_opened": False,
        "automatic_followup": False,
        "independent_solver_replay": "not_performed",
        "scope": "Existing public predicates and training replay; not full "
        "science certification.",
    }
    public._write(root / "integration_summary.json", value)
    (root / "EQUATIONS.md").write_text("\n".join(equations) + "\n")
    lines = [
        "# General shared-process integration",
        "",
        "Fresh variables, optional processes, one revision; frozen fitter. No "
        "domain-specific probes.",
        "Training mismatch is conditional on fitted parameters. Validation "
        "selects the incumbent.",
        "Shared syntax and passing predicates do not certify conservation or "
        "scientific correctness.",
        "No independent BDF/Radau comparison, automatic follow-up or test access.",
        "",
        "| Task | Round | Status | Proposal | Fallback | Trial val NMSE | "
        "Retained val NMSE | Calls cumulative | Tokens cumulative |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(r.get(k))
                for k in (
                    "task_id",
                    "round",
                    "status",
                    "proposal_status",
                    "construction_fallback",
                    "trial_validation_nmse",
                    "retained_validation_nmse",
                    "cumulative_requests",
                    "cumulative_tokens",
                )
            )
            + " |"
        )
    (root / "INTEGRATION.md").write_text("\n".join(lines) + "\n")


def _complexity(subject: dict | None) -> dict | None:
    """Report explicit model sizes without treating missing models as zeros."""
    if subject is None:
        return None
    candidate = subject["bundle"]["candidate"]
    return {
        field: len(candidate[field])
        for field in ("states", "processes", "parameters", "state_equations")
    }
