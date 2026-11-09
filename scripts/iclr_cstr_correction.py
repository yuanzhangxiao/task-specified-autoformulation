"""Stage a new CSTR-only component rerun with the submitted public interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.iclr_submission_evidence import PLAN, digest

CSTR = "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy"
NAMES = ("manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv")
EXPECTED = {
    "train.csv": "112799ce2be91a95486b0efa2090782264a34f960fec94a6f69fc1175ca3d979",
    "validation.csv": (
        "b98d9950047068d952c3af17435b0fa8d3d6c44ff36779c15afba983345f3ef9"
    ),
    "proposer_prompt.txt": (
        "0a3b4c055402ed26352363c4d367b791c41460ea21e8ca42043c495ea68eb99c"
    ),
}


def stage(original: Path, corrected: Path, output: Path) -> dict:
    """Prefer original files everywhere except the corrected reactor data/manifest."""
    if any(
        a.resolve().is_relative_to(b.resolve())
        for a, b in (
            (output, original),
            (original, output),
            (output, corrected),
            (corrected, output),
        )
    ):
        raise ValueError("use a new output outside the source directories")
    old = sealed_read(original / "plan.json")
    if old["artifact_sha256"] != PLAN or old["config"]["public_cells"][4] != CSTR:
        raise ValueError("require the original nine-case campaign receipt")
    manifest = json.loads((corrected / "manifest.json").read_text())
    if (
        manifest["benchmark_id"] != CSTR
        or manifest["reference_generation_protocol"] != "reference-events-2"
        or manifest["private_reference_available_to_methods"] is not False
    ):
        raise ValueError("invalid corrected reactor release")
    for name, sha in EXPECTED.items():
        if digest(corrected / name) != sha:
            raise ValueError("reactor prompt or corrected public CSV identity differs")
    chosen = []
    for cell, cfg in old["cells"].items():
        for name in NAMES:
            path = original / "public" / "phase_b_v1" / cell / name
            if digest(path) != cfg["assets"][name]:
                raise ValueError("original campaign assets differ")
            source = (
                corrected / name
                if cell == CSTR and name != "proposer_prompt.txt"
                else path
            )
            if (
                cell == CSTR
                and name == "proposer_prompt.txt"
                and digest(path) != EXPECTED[name]
            ):
                raise ValueError("submitted reactor prompt differs")
            rel = Path("phase_b_v1") / cell / name
            chosen.append((source, rel, digest(source)))
    with public._lock(output):
        for _, rel, sha in chosen:
            dest = output / rel
            if dest.exists() and digest(dest) != sha:
                raise ValueError("conflicting corrected rerun output")
        for source, rel, _ in chosen:
            dest = output / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(source.read_bytes())
        return sealed_write(
            output / "correction.json",
            {
                "protocol": "iclr-submitted-cstr-correction-1",
                "plan_sha256": PLAN,
                "only_replaced_cell": CSTR,
                "blocks": [16, 17, 18, 19],
                "search_arms": 4,
                "planned_search_lineages": 16,
                "public_files": {str(rel): sha for _, rel, sha in chosen},
                "test_opened": False,
                "note": (
                    "Fresh search on corrected inputs. Report at six visits "
                    "and separately after the final freeze."
                ),
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--corrected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(stage(a.original, a.corrected, a.output)["artifact_sha256"])
