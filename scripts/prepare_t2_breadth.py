"""Project the reviewed T2 training receipts into a portable construction intake."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.construction_functions import file_hash

FITTED = "insulin-feedback-repair-result-20261108/returned/inputs.json"
LAST = "insulin-feedback-repair-result-20261108/returned/result.json"
PORTFOLIO = "construction-remaining-portfolio-delivery-20261105/source/inputs.json"
WINDOWS = "construction-portfolio-balance-review-20261105/measured-windows.json"
ENDPOINT = (
    "insulin-positive-continuation-analysis-9d821bb2/returned/results/"
    "nested-zero-positive/result.json"
)
INPUT_FILES = {
    FITTED: "36f8e4d8d71f410a553e492fc94d08d12202839e19ae1a4228122c48c22a2e6d",
    LAST: "71fb6ec66cb6a4dc54d55fa0e76c61bbf932dec637098b2ff7734691e43308d0",
    PORTFOLIO: "2755e0b96b9182b0ef538d90811953da5cc5e846ecc442baf7318b31bbb80c3d",
    WINDOWS: "0fdf45c2f773eb78bb03d83d7ab3e0b015fdfc56dd7b93616199a7d47d286b4e",
    ENDPOINT: "0a6dc2ddae314da02c30e98c5e65660e7945afd660ba24d6a15adc8307c80086",
}


def prepare(reviews: Path, destination: Path) -> dict:
    """Cross-check both anchors against the same public training release and prefix."""
    for name, digest in INPUT_FILES.items():
        if file_hash(reviews / name) != digest:
            raise ValueError(f"Review source differs: {name}")
    old, failure = sealed_read(reviews / FITTED), sealed_read(reviews / LAST)
    data, endpoint = read_seal(reviews / PORTFOLIO), read_seal(reviews / ENDPOINT)
    (row,) = [r for r in data["records"] if r["candidate_id"] == "candidate-21"]
    anchor = row["anchor"]
    training = data["training"]
    if (
        not anchor["original_equations_verified"]
        or not endpoint["retained"]["original_equations_verified"]
        or content_sha256(endpoint) != old["result_identity"]["fit_terminal_sha256"]
        or any(
            endpoint["retained"]["parameters"][n] != v
            for n, v in old["problem"]["parameter_guesses"].items()
        )
        or data["test_data_opened"]
        or data["validation_exported"]
        or training["name"] != "train"
        or old["evidence"]["split"] != "train"
        or content_sha256(training) != old["fitting_contract"]["training_sha256"]
        or data["prefix"] != old["fitting_contract"]["prefix_policy"]
        or old["problem"]["context"] != row["problem"]["context"]
        or content_sha256(row["problem"]) != row["problem_sha256"]
        or failure["details"]["unknown"] != ["direct_ingestion"]
    ):
        raise ValueError("Fitted parent, split, prefix or last failure differs")
    problem = deepcopy(row["problem"])
    problem["parameter_guesses"] = {
        k: anchor["parameters"][k] for k in problem["parameter_guesses"]
    }
    return sealed_write(
        destination,
        {
            "protocol": "t2-three-origin-training-inputs-1",
            "sources": INPUT_FILES,
            "training": training,
            "fitting_contract": old["fitting_contract"],
            "evidence": old["evidence"],
            "brief": old["brief"],
            "aliases": old["aliases"],
            "c15": {
                k: old[k]
                for k in (
                    "problem",
                    "repair_history",
                    "revision_protocol",
                    "boundary_witness",
                    "boundary_feedback",
                    "result_identity",
                )
            },
            "c15_training_parameters": endpoint["retained"]["parameters"],
            "last_failure": {
                "error": failure["error"],
                "details": failure["details"],
                "receipt_sha256": failure["artifact_sha256"],
            },
            "c21": {
                "problem": problem,
                "source_problem_sha256": row["problem_sha256"],
                "anchor_sha256": content_sha256(anchor),
                "training_nmse": anchor["training_nmse"],
                "parameters_including_training_roots": anchor["parameters"],
            },
            "glucose_windows": json.loads((reviews / WINDOWS).read_text())["train_003"],
            "test_data_opened": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.reviews, args.out)
    print(
        json.dumps({"intake_sha256": result["artifact_sha256"], "file": str(args.out)})
    )
