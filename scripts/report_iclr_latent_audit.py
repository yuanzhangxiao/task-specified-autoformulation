"""Build all-run and per-benchmark tables from authenticated latent histories."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.collect_iclr_latent_histories import read_sealed
from scripts.iclr_latent_revision_audit import (
    summarize,
    summarize_mechanisms,
    write_table,
)


def histogram(values: list[int]) -> dict:
    """Include the denominator and percentages with every empirical histogram."""
    counts = dict(sorted(Counter(values).items()))
    return {
        "n": len(values),
        "counts": counts,
        "percentages": {k: 100 * n / len(values) for k, n in counts.items()},
        "with_any": sum(v > 0 for v in values),
        "states": sum(values),
    }


def semantic_role(state: dict, review: dict | None, requirements: list[dict]) -> dict:
    """Distinguish absent graph paths from evidence of a different intended role."""
    if state["required_mechanisms"]:
        return {
            "classification": "on_required_path",
            "required_roles": state["required_mechanisms"],
        }
    if state["birth"].get("lineage_mechanisms"):
        return {
            "classification": "required_role_path_lost",
            "required_roles": state["birth"]["lineage_mechanisms"],
        }
    if not requirements:
        return {"classification": "no_mechanistic_requirements", "required_roles": []}
    if review is None:
        raise ValueError("unreviewed state outside specified paths")
    return {k: review[k] for k in ("classification", "required_roles", "reason")}


def label(cell: str, metadata: dict) -> str:
    """Keep original and stress variants individually identifiable."""
    item = metadata.get(cell, {})
    if cell.startswith("acquisition_"):
        return f"T2-hard; noise {100 * item['noise']:g}%; {item['stride']} min"
    if cell.startswith("sensor_"):
        return "CSTR-hard; observed " + ",".join(["T", *item["mask"]])
    if cell.startswith("device_"):
        return (
            f"Device {item['dimension']} states; {item['variant']}; "
            f"generator {item['generator_seed']}"
        )
    names = {
        "phase_b_dalla_man_t1_canonical_named_easy": "Original T1-easy",
        "phase_b_anonymous_system_t1_canonical_obfuscated_easy": (
            "Original T1-easy obfuscated"
        ),
        "phase_b_dalla_man_t1_perturbed_named_easy": "Original T1-perturbed",
        "phase_b_anonymous_system_t1_perturbed_obfuscated_easy": (
            "Original T1-perturbed obfuscated"
        ),
        "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy": (
            "Original CSTR-easy"
        ),
        "phase_b_alien_device_unknown_device_mechanism_canonical_functional_easy": (
            "Original alien device"
        ),
        "phase_b_dalla_man_t1_canonical_named_hard": "Original T1-hard",
        "phase_b_dalla_man_t2_canonical_named_easy": "Original T2-easy",
        "phase_b_dalla_man_t2_canonical_named_hard": "Original T2-hard",
        "phase_c_dalla_man_t1_perturbed_named_easy_rates_v1": "Current T1-perturbed",
        "phase_c_dalla_man_t2_canonical_named_easy_rates_v1": (
            "T2-easy guided revisions"
        ),
        "phase_c_cstr_controlled_reactor_mechanism_perturbed_named_hard_reset_v1": (
            "CSTR-perturbed targeted"
        ),
        "cstr_easy_reference_events_v2": "CSTR-easy reference-input study",
    }
    return names.get(cell, cell)


def augment(
    audit: dict, reviews: list[dict], metadata: dict
) -> tuple[list, list, list]:
    """Add conservative semantic labels without changing structural audit results."""
    review_map = {(r["run_id"], r["model_id"], r["state"]): r for r in reviews}
    runs = [dict(r) for r in audit["runs"] if r["snapshot"] == "latest_available"]
    states = [dict(s) for s in audit["states"] if s["snapshot"] == "latest_available"]
    mechanisms = [
        dict(m) for m in audit["mechanisms"] if m["snapshot"] == "latest_available"
    ]
    grouped_states, grouped_mechanisms = defaultdict(list), defaultdict(list)
    for m in mechanisms:
        grouped_mechanisms[m["run_id"]].append(m)
    for s in states:
        key = s["run_id"], s["model_id"], s["state"]
        s["role_review"] = semantic_role(
            s, review_map.get(key), grouped_mechanisms[s["run_id"]]
        )
        grouped_states[s["run_id"]].append(s)
    for r in runs:
        r["benchmark_label"] = label(r["benchmark"], metadata)
        local = grouped_states[r["run_id"]]
        r["required_mechanism_count"] = len(grouped_mechanisms[r["run_id"]])
        for status in [
            "outside_required_role",
            "ambiguous_required_relation",
            "no_mechanistic_requirements",
        ]:
            r[status + "_count"] = sum(
                s["role_review"]["classification"] == status for s in local
            )
        r["role_assigned_latent_count"] = sum(
            bool(s["role_review"]["required_roles"]) for s in local
        )
        r["guided_feedback_addition_count"] = sum(
            s["retained_pure_augmentation"]
            and s["birth"]["feedback"].get("numerical_feedback_supplied", False)
            for s in local
        )
        for m in grouped_mechanisms[r["run_id"]]:
            m["role_latent_count"] = sum(
                m["id"] in s["role_review"]["required_roles"] for s in local
            )
            m["benchmark_label"] = r["benchmark_label"]
    return runs, states, mechanisms


def tables(runs: list[dict], mechanisms: list[dict]) -> dict:
    """Report every arm, plus explicit Full-only and all-arm views."""
    out = {}
    for keys in [
        ("run_class", "proposer"),
        ("run_class", "proposer", "arm"),
        ("cohort", "benchmark_label", "arm"),
        ("cohort", "benchmark_label"),
    ]:
        groups = defaultdict(list)
        for r in runs:
            groups[tuple(r[k] for k in keys)].append(r)
        result = []
        for key, rows in sorted(groups.items()):
            ready = [r for r in rows if r["available"]]
            entry = {**dict(zip(keys, key, strict=True)), **summarize(rows)}
            entry["endpoint_stages"] = dict(Counter(r["endpoint_stage"] for r in ready))
            for field in [
                "outside_required_role_count",
                "ambiguous_required_relation_count",
                "no_mechanistic_requirements_count",
                "role_assigned_latent_count",
                "guided_feedback_addition_count",
            ]:
                applicable = (
                    [r for r in ready if r["required_mechanism_count"] > 0]
                    if field in {
                        "outside_required_role_count",
                        "ambiguous_required_relation_count",
                    }
                    else ready
                )
                entry[field] = histogram([r[field] for r in applicable])
            entry["available_with_requirements"] = sum(
                r["required_mechanism_count"] > 0 for r in ready
            )
            result.append(entry)
        out["/".join(keys)] = result
    mm = defaultdict(list)
    for m in mechanisms:
        mm[(m["cohort"], m["benchmark_label"], m["arm"], m["id"])].append(m)
    out["mechanisms"] = [
        {
            "cohort": k[0],
            "benchmark": k[1],
            "arm": k[2],
            "requirement": k[3],
            **summarize_mechanisms(rows),
            "role_latent_histogram": histogram([r["role_latent_count"] for r in rows]),
        }
        for k, rows in sorted(mm.items())
    ]
    return out


def htext(h: dict) -> str:
    """Compact state count : number of models notation."""
    return "; ".join(f"{k}:{v}" for k, v in h.items()) or "unavailable"


def report(root: Path, analysis: Path, destination: Path) -> None:
    """Write portable tables and an evidence-bounded interpretation."""
    audit = read_sealed(analysis / "audit.json")
    reviews = json.loads((root / "semantic_reviews.json").read_text())["rows"]
    metadata = json.loads((root / "benchmark_metadata.json").read_text())["cells"]
    runs, states, mechanisms = augment(audit, reviews, metadata)
    result = tables(runs, mechanisms)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "tables.json").write_text(json.dumps(result, indent=2))
    write_table(destination / "runs.csv", runs)
    write_table(destination / "states.csv", states)
    write_table(destination / "mechanisms.csv", mechanisms)
    write_table(
        destination / "by_benchmark_all_arms.csv", result["cohort/benchmark_label/arm"]
    )
    write_table(
        destination / "by_benchmark_pooled_arms.csv", result["cohort/benchmark_label"]
    )
    write_table(destination / "by_mechanism.csv", result["mechanisms"])
    lines = [
        "ALL-RUN RETAINED LATENT-STATE AUDIT — 17 NOVEMBER 2026",
        "",
        (
            "Coverage: 339 same-pipeline run assignments (333 20B, 6 120B), "
            "plus 15 separately reported guided alternatives."
        ),
        (
            "A run is a campaign plan + task/seed/arm. Copies and resumes of "
            "the same plan are not new runs."
        ),
        (
            "There are 319 available selected endpoints: 313 20B and 6 120B. "
            "Of these, 254 have endpoint selection/pruning records; 65 "
            "original-campaign endpoints remain pre-pruning."
        ),
        (
            "One original post-pruning endpoint still lacks its final critic "
            "receipt. The ledger retains this status."
        ),
        (
            "The six later targeted runs are recorded failures without a "
            "retained fit, not zero-state successes."
        ),
        (
            "Eight additional 120B cases (24 tasks) were prepared but have no"
            " recorded visits; they are not executed-run denominators."
        ),
        (
            "The two-visit CPU pilot is included and separately identifiable;"
            " all other standard assignments have 15 recorded search visits."
        ),
        (
            "The original campaign contains 160 search assignments across 9 "
            "benchmark cases, not 192 independent runs. The latter number "
            "also includes alternate endpoint policies."
        ),
        (
            "The older six-visit submission snapshots are earlier checkpoints"
            " of these original runs and are not counted again."
        ),
        "",
        "DEFINITIONS",
        "",
        (
            "Active latent: a declared hidden differential coordinate with a "
            "nonzero symbolic fitted dependency path to an output. Exact-zero"
            " fitted terms are removed. This is structural activity, not a "
            "sensitivity threshold or proof of mechanistic correctness."
        ),
        (
            "Modeled supplied auxiliaries (the Candidate-15 Q/Uii alias) are "
            "recorded separately and excluded from hidden-state counts."
        ),
        (
            "Retained addition: a new coordinate introduced after a fitted "
            "incumbent, surviving in the selected endpoint. Strict "
            "augmentation additionally requires an increase in latent "
            "dimension with no simultaneous state removals. "
            "Renames/replacements are reported separately."
        ),
        (
            "Numerically cited: the accepted reply references records in its "
            "exact matched training-evidence request. It establishes recorded"
            " numerical motivation; it does not prove that numerical feedback"
            " alone caused the change."
        ),
        (
            "Supplemental: a retained strict augmentation whose final "
            "required pathway already had at least one active latent "
            "immediately before introduction. This is a conservative "
            "graph-based count; intended roles with incomplete paths may add "
            "further cases."
        ),
        (
            "Required-path membership: union of simple paths from each "
            "declared driver to its declared target. States may belong to "
            "multiple requirements. For broad clauses without driver bindings"
            " (CSTR), all upstream latents of the named balance are included "
            "and explicitly labeled. Such a clause cannot identify separate "
            "reaction/transport/jacket counts."
        ),
        (
            "Zero path-latents is split into a represented path without a "
            "hidden coordinate and a missing driver-to-target path. Neither "
            "is automatically mechanism compliance; an explicit memory "
            "requirement may still be violated."
        ),
        (
            "Semantic role review: graph-unassigned states are checked "
            "against retained history, descriptions and hypotheses. A "
            "lost/wrong input connection is not counted as a new mechanism. "
            "Ambiguities remain explicit. No-spec arms have no required "
            "mechanisms by design and are separated from unrequested roles "
            "under a nonempty specification."
        ),
        (
            "Histogram notation k:n means n available models have k such "
            "states. Missing endpoints do not enter histograms. All-run rates"
            " also retain the planned/executed denominator in tables.json."
        ),
        "",
        (
            "LATEST AVAILABLE ENDPOINTS (final where available; otherwise "
            "labeled pre-pruning)"
        ),
        "",
        (
            "Group | Available/planned | numerical-addition models | "
            "supplemental models | active-latent histogram | "
            "numerical-addition histogram"
        ),
    ]
    groups = [r for r in result["run_class/proposer"] if r["run_class"] == "search"] + [
        r for r in result["run_class/proposer/arm"] if r["arm"] == "full"
    ]
    for r in groups:
        name = (
            f"{r['run_class']} {r['proposer'].split('-')[-1]} "
            f"{r.get('arm', 'all arms')}"
        )
        n, s = (
            r["numerically_cited_addition_count"],
            r["supplements_existing_mechanism_latent_count"],
        )
        lines.append(
            f"{name} | {r['available']}/{r['planned']} | "
            f"{n['models_with_any']} ({n['percentage_of_available']:.1f}%) | "
            f"{s['models_with_any']} ({s['percentage_of_available']:.1f}%) | "
            f"{htext(r['active_latent_count']['histogram'])} | "
            f"{htext(n['histogram'])}"
        )
    lines += [
        "",
        "FULL AFTER ENDPOINT SELECTION ONLY",
        "",
        "Proposer | Available | Numerical-addition models | Supplemental models",
    ]
    for proposer in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
        subset = [
            r
            for r in audit["runs"]
            if r["snapshot"] == "final"
            and r["run_class"] == "search"
            and r["arm"] == "full"
            and r["proposer"] == proposer
        ]
        summary = summarize(subset)
        n = summary["numerically_cited_addition_count"]
        s = summary["supplements_existing_mechanism_latent_count"]
        lines.append(
            f"{proposer} | {summary['available']} | "
            f"{n['models_with_any']} ({n['percentage_of_available']:.1f}%) | "
            f"{s['models_with_any']} ({s['percentage_of_available']:.1f}%)"
        )
    lines += [
        "",
        "STATES OUTSIDE REQUIRED MECHANISM ROLES",
        "",
        "These are conservative intended-role judgments, "
        "distinct from graph-path absence.",
        "Group | Models with requirements | "
        "Clear outside-role histogram | Ambiguous histogram",
    ]
    for r in groups:
        lines.append(
            f"{r['proposer']} {r.get('arm', 'all arms')} | "
            f"{r['available_with_requirements']} | "
            f"{htext(r['outside_required_role_count']['counts'])} | "
            f"{htext(r['ambiguous_required_relation_count']['counts'])}"
        )
    lines += [
        "These two role histograms include available models with nonempty mechanism "
        "requirements. No-spec latents are separately listed in "
        "no_mechanistic_requirements_count; an empty specification is not evidence "
        "of novelty under a mechanistic specification.",
        "The two clear outside roles occur in original-campaign ablations: delayed "
        "EGP under a meal-only requirement, and a v05-driven state under a u01-to-v01 "
        "requirement. Three feedback/auxiliary-memory assignments remain ambiguous.",
        "No Full endpoint has a clear outside role under this broad requirement "
        "mapping. This does not rule out new substructure within a required pathway.",
        "",
        "FULL REQUIRED-PATH STATE COUNTS, LATEST AVAILABLE",
        "",
        "Proposer / clause | Models | k:n histogram | Path present with 0 latents | "
        "Missing path | Broad balance presence unassessed",
    ]
    mechanism_groups = defaultdict(list)
    for m in mechanisms:
        if m["run_class"] == "search" and m["arm"] == "full":
            mechanism_groups[(m["proposer"], m["id"])].append(m)
    for (proposer, clause), items in sorted(mechanism_groups.items()):
        s = summarize_mechanisms(items)
        lines.append(
            f"{proposer} / {clause} | {s['requirement_model_pairs']} | "
            f"{htext(s['latent_count_histogram'])} | "
            f"{s['represented_without_latents']} | {s['path_absent']} | "
            f"{s['broad_balance_presence_unassessed']}"
        )
    lines.append(
        "Meal clauses from original T1 (meal_response), T2/current T1 "
        "(meal_pathway), and obfuscated cases (input_response) remain separate. "
        "Counts may overlap across clauses; they must not be summed as unique states."
    )
    lines += [
        "",
        "GUIDED ALTERNATIVES (training guidance, not the standard citation protocol)",
        "Arm | Fitted/attempted | Retained guided additions | active-latent histogram",
    ]
    for r in result["run_class/proposer/arm"]:
        if r["run_class"] == "guided_alternative":
            lines.append(
                f"{r['arm']} | {r['available']}/{r['planned']} | "
                f"{htext(r['guided_feedback_addition_count']['counts'])} | "
                f"{htext(r['active_latent_count']['histogram'])}"
            )
    lines += [
        "",
        "FULL BY BENCHMARK (same histogram notation)",
        "",
        (
            "Cohort / benchmark | Available/planned | post-fit numerical "
            "addition | after existing same-path latent | active states | "
            "additional states"
        ),
    ]
    for r in result["cohort/benchmark_label/arm"]:
        if r["arm"] != "full":
            continue
        n, s = (
            r["numerically_cited_addition_count"],
            r["supplements_existing_mechanism_latent_count"],
        )
        rate = (
            f"{n['models_with_any']} ({n['percentage_of_available']:.1f}%)"
            if r["available"]
            else "unavailable"
        )
        lines.append(
            f"{r['cohort']} / {r['benchmark_label']} | "
            f"{r['available']}/{r['planned']} | {rate} | "
            f"{s['models_with_any']} | "
            f"{htext(r['active_latent_count']['histogram'])} | "
            f"{htext(n['histogram'])}"
        )
    lines += [
        "",
        "INTERPRETATION AND LIMITS",
        "",
        (
            "Do not pool pre-pruning and final endpoints under a 'final "
            "models' label. analysis/ contains final-only, end-search-only "
            "and latest-available snapshots for sensitivity checks."
        ),
        (
            "Do not interpret the pooled all-arm percentage as Full's success"
            " rate: fixed-state, no-latent, no-spec, incorrect-spec and "
            "component ablations have different permissions and objectives."
        ),
        (
            "The 120B cohort contains only CSTR-hard and T2-hard, three seeds"
            " each. It is not a matched all-benchmark comparison with 20B."
        ),
        (
            "Guided alternatives: 4/5 Candidate-15 alternatives and 1/5 "
            "Candidate-21 alternatives have fitted exports; 0/5 fresh "
            "alternatives do. Candidate-15 c15-3 adds active M after the "
            "parent's meal-path latent A. Its request explicitly supplied "
            "training timing/boundary diagnostics. Thus 1/4 fitted "
            "Candidate-15 alternatives (1/5 attempted) adds retained memory. "
            "The generated alternatives share a parent/request; they are not "
            "independent search seeds. The standard R001 citation definition "
            "does not apply to this different protocol."
        ),
        (
            "The guided deterministic zero-basal control is not counted as a "
            "proposal. Three earlier single-reply Candidate-15 experiments "
            "are inventoried separately in guided_earlier_attempts.json (one "
            "admitted for refit, two rejected); no final fitted export is "
            "established for those records."
        ),
        (
            "No new model fitting, proposal generation, selection or test "
            "evaluation was performed. Forecast improvements and causal "
            "usefulness of particular added states require their separate "
            "held-out/intervention evidence."
        ),
        (
            "Initial-construction latents are not called numerical-feedback "
            "additions merely because they are present in an endpoint. "
            "Histories with missing predecessor fits cannot establish a "
            "strict post-fit addition."
        ),
        (
            "This audit covers identified benchmark-search campaigns using "
            "the final-component/controlled-search pipeline and the guided "
            "Candidate-15/21 alternatives. Local development, mocked smoke "
            "tests, prefit construction portfolios, solver restarts, external"
            " baselines and duplicate exports are inventoried by protocol "
            "rather than silently pooled as independent runs."
        ),
        (
            "All per-run, per-state and per-mechanism records, including "
            "exact hypothesis and request identities, accompany this report. "
            "Counts are descriptive empirical frequencies, not independent "
            "Bernoulli trials or claims of physical mechanism identification."
        ),
    ]
    (destination / "REPORT.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.analysis, args.output)


if __name__ == "__main__":
    main()
