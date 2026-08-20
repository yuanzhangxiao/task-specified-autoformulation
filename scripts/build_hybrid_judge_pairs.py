"""Augment frozen judge pairs with production-relevant retained disconnections."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel


def _unique_name(payload: dict[str, object], stem: str) -> str:
    occupied = {
        *(
            item["name"]
            for collection in ("states", "processes", "parameters")
            for item in payload[collection]
        )
    }
    if stem not in occupied:
        return stem
    suffix = 2
    while f"{stem}_{suffix}" in occupied:
        suffix += 1
    return f"{stem}_{suffix}"


def _retained_disconnection(pair: AdversarialPair) -> AdversarialPair:
    baseline_processes = {item.name for item in pair.valid_candidate.processes}
    source_process = next(
        item
        for item in pair.adversarial_candidate.processes
        if item.name not in baseline_processes and item.mechanisms
    )
    payload = pair.valid_candidate.model_dump(mode="json")
    process_name = _unique_name(payload, "claimed_pathway")
    state_name = _unique_name(payload, "claimed_pathway_memory")
    payload["candidate_id"] = f"{pair.valid_candidate.candidate_id}_retained_claim"
    payload["parent_candidate_id"] = None
    payload["change_summary"] = "Candidate submitted for scientific assessment."
    payload["processes"].append(
        {
            "name": process_name,
            "expression": source_process.expression,
            "mechanisms": list(source_process.mechanisms),
            "description": "Additional candidate mechanism pathway.",
            "unit": "unspecified",
        }
    )
    payload["states"].append(
        {
            "name": state_name,
            "kind": "latent",
            "mechanisms": list(source_process.mechanisms),
            "description": "Additional candidate mechanism memory.",
            "unit": "unspecified",
        }
    )
    payload["state_equations"].append(
        {"state": state_name, "rhs": f"{process_name} - {state_name}"}
    )
    payload["initial_conditions"].append(
        {"state": state_name, "scope": "global", "fixed_value": 0.0}
    )
    candidate = CandidateModel.model_validate(payload)
    digest = hashlib.sha256(f"{pair.pair_id}:retained".encode()).hexdigest()[:16]
    return AdversarialPair(
        pair_id=f"hybrid_{digest}",
        benchmark_id=pair.benchmark_id,
        tier=pair.tier,
        mutation_type="retained_disconnected_claimed_mechanism",
        valid_candidate=pair.valid_candidate,
        adversarial_candidate=candidate,
    )


def augment_pairs(
    pairs: tuple[AdversarialPair, ...],
) -> tuple[AdversarialPair, ...]:
    """Retain original controls and add one reachable-to-runtime disconnection."""
    output = list(pairs)
    output.extend(
        _retained_disconnection(pair)
        for pair in pairs
        if pair.mutation_type == "disconnected_claimed_mechanism"
    )
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in args.input_pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    augmented = augment_pairs(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in augmented),
        encoding="utf-8",
    )
    print(
        f"wrote {len(augmented)} hybrid pairs to {args.output}; "
        f"added_retained_disconnections={len(augmented) - len(pairs)}"
    )


if __name__ == "__main__":
    main()
