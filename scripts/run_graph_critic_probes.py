"""Freeze the graph critic on the development parent, then evaluate fresh parents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import graph_critic_advice as advice
from autoformalism.fitting import graph_critic_probes as probes
from autoformalism.fitting.public_fitting import _lock, content_sha256
from scripts import compare_graph_critic_feedback as critic
from scripts import graph_critic_parents as parents


def _plan() -> dict:
    """Identify the specific rules, public inputs, and endpoint roster before replay."""
    code = [Path(m.__file__) for m in (advice, probes, parents, critic)] + [
        Path(__file__)
    ]
    return {
        "protocol": "graph-critic-evaluation-1",
        "development_parent": parents.ROSTER[0],
        "fresh_parents": list(parents.ROSTER[1:]),
        "transfer_controls": [parents.TRANSFER_CONTROL],
        "selection": "All non-15 parents from the November 3 targeted roster. "
        "Previously studied meal-memory sibling evaluated separately.",
        "terminal_sha256": parents.TERMINALS,
        "training_sha256": parents.TRAINING,
        "rules": probes.RULES,
        "routing": advice.RULES,
        "critic_policy_sha256": content_sha256(critic.policy()),
        "implementation": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code
        },
        "comparison": "One hosted critic call per parent on identical evidence. "
        "Compare delivery, grounding, proposed check and respect for fitting limits.",
    }


def _equations(model) -> dict[str, str]:
    v = model.validated
    return {
        f"{kind}:{key}": expression.source
        for kind, values in (
            ("state", v.equation_expressions),
            ("process", v.process_expressions),
            ("target", v.observation_expressions),
        )
        for key, expression in values.items()
    }


def _packet(root: Path, parent_id: str, cohort: str, identity: str, repo: Path) -> dict:
    data = parents.load_parent(parent_id, repo)
    endpoints = {}
    for label, point in data.starts.items():
        path = root / "replays" / parent_id / f"{label}.json"
        binding = {
            "rules_identity": identity,
            "terminal_sha256": parents.TERMINALS[parent_id][label],
            "problem_sha256": data.provenance["problem_sha256"],
            "training_sha256": data.provenance["training_sha256"],
        }
        if not path.exists():
            print(f"Replaying {cohort} {parent_id}, start {label}", flush=True)
            result = probes.probe_endpoint(
                data.model,
                data.training,
                point.parameters,
                point.initial_states,
                expected_nmse=point.training_nmse,
            )
            seal(path, {"binding": binding, "result": result})
        saved = read_seal(path)
        if saved["binding"] != binding:
            raise ValueError(
                f"saved graph replay belongs to a different fit or policy: {path}"
            )
        endpoints[label] = saved["result"]
    search = {name: point.search_status for name, point in data.starts.items()}
    return {
        "protocol": "graph-critic-feedback-1",
        "parent_id": parent_id,
        "cohort": cohort,
        "freeze_sha256": identity,
        "public_task": data.public_prompt,
        "equations": _equations(data.model),
        **advice.summarize(endpoints, search),
    }


def _calibrated(packet: dict) -> bool:
    """Reproduce the two directional effects on the development case only."""
    evidence = packet["evidence"]
    windows = evidence["window-1"]["starts"]
    if packet["readiness"] != "conditional_equation_review" or not all(
        (w["target"], w["trajectory_id"], w["start"], w["end"])
        == ("Gp", "train_005", 200, 300)
        for w in windows.values()
    ):
        return False
    effects = {
        (e["driver"], e["target"]): (
            set(e["starts"]) == set(windows),
            {v["direction"] for v in e["starts"].values()},
        )
        for key, e in evidence.items()
        if key.startswith("probe-")
    }
    return effects.get(("Gp", "I")) == (True, {"improves"}) and effects.get(
        ("I", "Gp")
    ) == (True, {"worsens"})


def campaign(stage: str, root: Path, repo: Path) -> dict:
    plan = _plan()
    with _lock(root):
        if stage == "freeze":
            packet = _packet(
                root,
                plan["development_parent"],
                "development",
                content_sha256(plan),
                repo,
            )
            if not _calibrated(packet):
                raise ValueError(
                    "development replay failed to reproduce the calibrated effects"
                )
            seal(root / "development.json", packet)
            frozen = {**plan, "development_packet_sha256": content_sha256(packet)}
            seal(root / "freeze.json", frozen)
            return {
                "freeze_sha256": content_sha256(frozen),
                "fresh_parents": plan["fresh_parents"],
            }
        if stage != "run":
            raise ValueError("use freeze or run")
        freeze = read_seal(root / "freeze.json")
        dev = read_seal(root / "development.json")
        if freeze != {**plan, "development_packet_sha256": content_sha256(dev)} or (
            dev["freeze_sha256"] != content_sha256(plan) or not _calibrated(dev)
        ):
            raise ValueError("development record, frozen rules or roster differ")
        digest = content_sha256(freeze)
        summary = {"freeze_sha256": digest}
        for cohort, parents_key in (
            ("fresh", "fresh_parents"),
            ("transfer_control", "transfer_controls"),
        ):
            rows = []
            for name in freeze[parents_key]:
                packet = _packet(root, name, cohort, digest, repo)
                seal(root / "feedback" / f"{name}.json", packet)
                rows.append(
                    {
                        "parent": name,
                        "readiness": packet["readiness"],
                        "packet_sha256": content_sha256(packet),
                    }
                )
            summary[parents_key] = rows
        seal(root / "deterministic.json", summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    print(json.dumps(campaign(args.stage, args.root, args.repo), indent=2))


if __name__ == "__main__":
    main()
