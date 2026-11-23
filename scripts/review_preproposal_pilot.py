"""Render the measured screen, downstream replay controls, and both proposer prompts."""

import argparse
import json
from html import escape
from pathlib import Path

from autoformalism.research import preproposal_cases as cases
from autoformalism.research import preproposal_selection as selection
from autoformalism.research import preproposal_topology as topology
from autoformalism.search.preproposal_screen import assess_options
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from scripts.audit_preproposal_transfer import all_time_conflicts


def pre(value):
    content = (
        value
        if isinstance(value, str)
        else json.dumps(value, indent=2, ensure_ascii=False)
    )
    return "<pre>" + escape(content) + "</pre>"


def table(headings, rows):
    head = "".join("<th>" + escape(h) + "</th>" for h in headings)
    body = "".join(
        "<tr>" + "".join("<td>" + escape(str(v)) + "</td>" for v in row) + "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table>"
    )


def page(title, sections):
    return """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<style>body{font:17px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1.2rem;
color:#172830}h1,h2{line-height:1.2}h2{margin-top:2rem}table{border-collapse:collapse;
display:block;overflow-x:auto}
td,th{text-align:left;padding:.48rem .75rem;border-bottom:1px solid #cbd8dc}
th{background:#edf4f4}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f6;
padding:1rem}summary{cursor:pointer;color:#165970}a{color:#005f71}</style>""" + (
        f"<title>{escape(title)}</title><h1>{escape(title)}</h1>" + "\n".join(sections)
    )


def replay(frozen, case, project):
    # Load the archived graph only after freezing the screen and its candidate IDs.
    receipt = json.loads((project / frozen["provenance"]["receipt_path"]).read_text())
    saved = receipt["value"]
    original = {n: x["sources"] for n, x in saved["targets"].items()}
    original.update({x["name"]: x["sources"] for x in saved["additional_components"]})
    examples = [
        ("Original insulin sources", "history:EGP", [], [], []),
        ("Add Gt to glucose only", "history:Gt", [], [], ["Gt"]),
        ("Direct EGP route", "history:EGP", ["EGP"], [], []),
        ("EGP reaches insulin via generated Gp", "history:EGP", ["Gp"], [], []),
        ("Direct Gt route", "history:Gt", ["Gt"], [], []),
    ]
    if frozen.get("minimum_target_gap", 0):
        for row in case.candidates:
            bases = [s[8:] for s in row.sources if s.startswith("initial:")]
            if bases:
                initial = bases[0]
                examples.append(
                    (
                        f"Meal, administered insulin and baseline {initial}",
                        f"initial:{initial}",
                        ["meal_rate_g_per_min", "Uid"],
                        bases,
                        [],
                    )
                )
    packet = TrainingCompatibilityEvidence.model_validate(frozen["evidence"])
    rows = []
    for label, option, insulin, baseline, glucose in examples:
        candidate = next(x for x in case.candidates if option in x.sources)
        choice = {
            "candidate_id": candidate.candidate_id,
            "rationale": "Offline wiring control.",
        }
        graph = {n: list(v) for n, v in original.items()}
        graph["I"].extend(insulin)
        graph["Gp"].extend(glucose)
        rule = (
            {"mode": "map", "public_initials": baseline}
            if baseline
            else {"mode": "value", "scope": "shared_training"}
        )
        init = [
            {
                "state": "Uid",
                "rule": rule,
                "justification": "Controlled time-zero choice.",
            }
        ]
        reply = {"hypothesis": label, "sources": graph, "latent_initializers": init}
        result = topology.assess_topology(case, choice, reply, frozen)
        rows.append(
            {
                "label": label,
                "chosen_options": list(candidate.sources),
                "reply": reply,
                "assessment": result,
                "equal_prefix_audit": all_time_conflicts(
                    packet,
                    case.target,
                    result["training_checks"][case.target]["actual_options"],
                ),
            }
        )
    return rows


def witness_figure(frozen, controls, out):
    """Show observed differences up to the strongest matched-prefix witness."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    original = controls[0]
    mixed = next(r for r in controls if r["label"].endswith("baseline Gp"))
    trajectories = {r["trajectory_id"]: r for r in frozen["evidence"]["trajectories"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, control, title in zip(
        axes,
        [original, mixed],
        ["Original insulin sources", "Both inputs and baseline glucose"],
        strict=True,
    ):
        witness = control["equal_prefix_audit"]["witness"]
        left, right = (trajectories[n] for n in witness["trajectories"])
        end = left["time"].index(witness["time"]) + 1
        delta = [
            abs(a - b)
            for a, b in zip(
                left["targets"]["I"][:end], right["targets"]["I"][:end], strict=True
            )
        ]
        ax.plot(left["time"][:end], delta, color="#19576a", linewidth=2)
        ax.scatter(witness["time"], delta[-1], color="#ab4923", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Minutes since start")
        ax.set_ylabel("Absolute insulin difference (pmol/L)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.23)
        ax.text(
            0.03,
            0.96,
            " and ".join(witness["trajectories"]),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    fig.savefig(out / "insulin-witnesses.png", dpi=170)
    plt.close(fig)


def render(project, folder, transfer, out):
    out.mkdir(parents=True, exist_ok=True)
    frozen = cases.load_frozen_cases(folder / "frozen-cases.json")
    strict = cases.load_strict_t2(project)
    totals, control_rows, details, prompt_sections, records = [], [], [], [], {}
    for identifier, raw in {**frozen, strict["case_id"]: strict}.items():
        screened = cases.screen_case(raw)
        case = selection.SelectionCase.model_validate(
            cases.selection_case(raw, screened)
        )
        base = assess_options(screened, ())
        totals.append(
            (
                identifier,
                screened.pair_count,
                base.conflicting_pairs,
                screened.subsets_examined,
                screened.subset_counts["contradicted"],
                screened.subset_counts["screened"],
                len(screened.candidates),
            )
        )
        details.append(
            "<details><summary>"
            + escape(identifier)
            + " · minimal source sets and a training witness</summary>"
            + pre([c.sources for c in case.candidates])
            + pre(case.evidence_summary)
            + "</details>"
        )
        if not selection.needs_selection(case) or identifier not in frozen:
            continue
        selection_prompt = selection.messages(case)
        row = case.candidates[0]
        choice = {
            "candidate_id": row.candidate_id,
            "rationale": "Illustrative selection for this preview.",
        }
        topology_prompt = topology.topology_messages(case, choice, raw)
        prompt_sections += [
            f"<h2>{escape(identifier)}</h2>",
            "<h3>Scientific selection</h3>",
            pre(selection_prompt[0]),
            pre(json.loads(selection_prompt[1])),
            "<h3>Topology after an illustrative selection</h3>",
            pre(topology_prompt[0]),
            pre(json.loads(topology_prompt[1])),
            "<details><summary>Structured output schemas</summary>"
            + pre(selection.selection_schema(case).model_json_schema())
            + pre(topology.topology_schema(case).model_json_schema())
            + "</details>",
        ]
        records[identifier] = replay(raw, case, project)
        if raw["minimum_target_gap"] > 0:
            witness_figure(raw, records[identifier], out)
        for r in records[identifier]:
            assessment = r["assessment"]
            control_rows.append(
                (
                    screened.minimum_target_gap,
                    r["label"],
                    assessment["status"],
                    assessment["training_checks"]["I"]["conflicting_pairs"],
                    assessment["exact_selected_target_check"]["conflicting_pairs"],
                    f"{r['equal_prefix_audit']['max_equal_prefix_gap']:.3g}",
                )
            )
    transfer_rows = []
    for r in json.loads(transfer.read_text())["rows"]:
        for sensitivity in r["sensitivities"]:
            inventory = ", ".join(n["name"] for n in r["inventory"])
            transfer_rows.append(
                (
                    r["target"],
                    inventory,
                    sensitivity["minimum_target_gap"],
                    sensitivity["pair_count"],
                    sensitivity["subsets_examined"],
                    "; ".join(
                        " + ".join(c) or "no addition"
                        for c in sensitivity["minimal_options"]
                    )
                    or "none",
                )
            )
    content = [
        "<p>The screen runs after variable declaration. It tests which permitted "
        "information could distinguish training trajectories in time. A "
        "choice may reach "
        "the target through several states. Its scientific role, "
        "functions and achievable "
        "NMSE remain open.</p><h2>Search space</h2>",
        table(
            (
                "Case",
                "Pairs",
                "Conflicts before additions",
                "Source subsets",
                "Contradicted",
                "Survived",
                "Minimal",
            ),
            totals,
        ),
        *details,
        "<p>The seed-102 inventory was projected from a joint "
        "declaration. Its old edges "
        "were withheld from both proposer prompts. The second insulin "
        "inventory and the "
        "basin inventory came from genuine variable-stage exits. Both "
        "insulin inventories "
        "use the same 16 training records. Every basin trajectory has a "
        "different observed "
        "initial downstream depth, so the screen gives no additional "
        "dependency there.</p>",
        "<h2>Realized topology checks</h2><p>Seed 102 proposed "
        "dI/dt = F(I, insulin infusion). For this test I(0) is observed. The controls "
        "hold the other original edges fixed except for the changes "
        "named below. They set "
        "the latent Uid initializer to a shared constant unless the "
        "example specifies a "
        "baseline map. The original I subsystem has no path from Uid. The controls are "
        "deterministic examples. None was fitted or generated by the new 20B run.</p>",
        table(
            (
                "I gap (pmol/L)",
                "Wiring control",
                "Outcome",
                "Conflicting insulin pairs",
                "Conflicting at zero gap",
                "Largest matched-prefix gap (pmol/L)",
            ),
            control_rows,
        ),
        "<p>The indirect example adds generated Gp to the insulin "
        "derivative. Gp already "
        "receives EGP, so no measured future glucose values are used. "
        "Reaching a source "
        "establishes an information path. A fitted law may still ignore "
        "that source.</p>",
        '<figure><img style="max-width:100%" src="insulin-witnesses.png" '
        'alt="Largest insulin difference between matched training trajectories, '
        'by source set"><figcaption>Each panel follows the pair with the largest '
        "observed difference while the realized information still matches. "
        "The vertical axes have different scales.</figcaption></figure>",
        "<h2>Transfer and sensitivity</h2><p>Thresholds are additional "
        "absolute gaps in "
        "target units. The input-equality assumptions stay fixed. In "
        "CSTR the inventory "
        "determines whether C is supplied or must be generated. For the alien device "
        "roundoff-scale differences change the exact answer.</p>",
        table(
            (
                "Target",
                "Declared variables",
                "Target gap",
                "Pairs",
                "Source subsets",
                "Inclusion-minimal additions",
            ),
            transfer_rows,
        ),
        "<h2>What the live pilot measures</h2><p>Five draws rank up to "
        "three scientific "
        "choices per insulin menu. Each selected choice gets a separate full topology. "
        "Score the fraction of draws yielding a graph that realizes the "
        "choice, passes the "
        "public path checks and survives the training comparisons. Read "
        "the rationale against "
        "the equations. No optimization or held-out scoring occurs in "
        "this experiment.</p>",
        '<p><a href="PROMPTS.html">Inspect selection and topology prompts</a> · '
        '<a href="replay.json">Inspect each control graph and assessment</a></p>',
    ]
    (out / "REPORT.html").write_text(
        page("Pre-topology screening and scientific selection", content)
    )
    (out / "PROMPTS.html").write_text(
        page("20B scientific selection and topology prompts", prompt_sections)
    )
    (out / "replay.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.project, args.cases, args.transfer, args.out)
