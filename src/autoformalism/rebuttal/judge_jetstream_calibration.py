"""Portable revalidation of the current judge on the frozen consensus cases."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from autoformalism.expressions import ValidationContext, repair_protected_declarations
from autoformalism.judging import HybridScoringConfig, extract_public_requirements
from autoformalism.llm import LLMConfig, jetstream
from autoformalism.llm.config import OllamaResponseMode
from autoformalism.llm.exceptions import LLMError
from autoformalism.rebuttal import judge_sign_recheck as original
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.rebuttal.hybrid_labels import HybridCalibrationLabels
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_scientific_judge import judge_protocol
from autoformalism.search.hybrid_pair import PairedHybridJudge
from scripts.run_hybrid_judge import (
    _atomic_system_prompt,
    _system_prompt,
    _task_context,
)

PROTOCOL = "jetstream-known-case-revalidation-1"
CONFIG = original.REPO / "configs/hybrid_judge_consensus_validation_v1.json"
MODEL = "vllm:openai/gpt-oss-120b"
SIGN_POLICY = "outer-factor-sign-2"
ORDERS = ("baseline_a", "baseline_b")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_bundle(bundle: dict) -> None:
    """Require the entire predeclared suite, never a score-selected subset."""
    config = json.loads(CONFIG.read_text())
    if bundle["protocol"] != PROTOCOL or bundle["test_data_opened"] is not False:
        raise ValueError("requires the public calibration export")
    if bundle["source_config"] != config:
        raise ValueError("source consensus protocol differs from frozen config")
    pairs = [AdversarialPair.model_validate(p) for p in bundle["pairs"]]
    labels = [HybridCalibrationLabels.model_validate(p) for p in bundle["labels"]]
    ids = {p.pair_id for p in pairs}
    types = config["pair_construction"]
    expected_types = (
        types["known_tie_types"]
        + types["known_dominance_types"]
        + types["unlabeled_stability_types"]
    )
    if (
        len(pairs) != 14
        or len(ids) != 14
        or Counter(p.mutation_type for p in pairs) != dict.fromkeys(expected_types, 2)
    ):
        raise ValueError("expected all 14 frozen consensus pairs")
    if len(labels) != 14 or {p.pair_id for p in labels} != ids:
        raise ValueError("labels must cover every pair exactly once")
    expected_preferences = {
        **dict.fromkeys(types["known_tie_types"], "tie"),
        **dict.fromkeys(types["known_dominance_types"], "baseline"),
        **dict.fromkeys(types["unlabeled_stability_types"], "unlabeled"),
    }
    lookup = {label.pair_id: label for label in labels}
    for pair in pairs:
        if (
            lookup[pair.pair_id].overall_preference.value
            != expected_preferences[pair.mutation_type]
        ):
            raise ValueError("overall labels differ from the frozen mutation contract")
        entry = bundle["contexts"][f"{pair.benchmark_id}/{pair.tier}"]
        ValidationContext.model_validate(entry["context"])
        if not entry["public_prompt"].strip():
            raise ValueError("missing public task prompt")
    outcomes = bundle["historical_rows"] + bundle["historical_failures"]
    keys = [(r["pair_id"], int(r["repetition"]), r["order"]) for r in outcomes]
    expected = {(p, r, order) for p in ids for r in range(5) for order in ORDERS}
    if len(keys) != 140 or set(keys) != expected:
        raise ValueError("historical outcomes must cover all 140 orientations once")
    if any(r["judge_model"] != MODEL for r in outcomes):
        raise ValueError("unexpected historical judge model")


def export(source: Path, data_root: Path, output: Path) -> dict:
    """Package existing cases and labels; context uses the original public loader.

    The loader opens train/validation to recover declared numeric channel bounds.
    No trajectories, fitted scores, private equations or test split are exported.
    """
    files = {
        "pairs": source / "pairs.jsonl",
        "labels": source / "hybrid_labels.jsonl",
        "source_config": source / "protocol_config.json",
        "historical_rows": source / "gpt-oss-120b/hybrid_judge_scores.csv",
        "historical_failures": source / "gpt-oss-120b/hybrid_judge_failures.jsonl",
        "pair_manifest": source / "consensus_validation_pairs_manifest.json",
        "frozen_inputs": source / "frozen_inputs.sha256",
    }
    # Read only the declared inputs, never arbitrary paths from a checksum file.
    hashes = {}
    for line in files["frozen_inputs"].read_text().splitlines():
        match = re.fullmatch(r"([a-f0-9]{64}) [ *](.+)", line)
        if not match:
            raise ValueError("invalid historical checksum record")
        name = Path(match[2]).name
        if name in hashes:
            raise ValueError("duplicate historical checksum name")
        hashes[name] = match[1]
    for key in ("pairs", "labels", "source_config", "pair_manifest"):
        if hashes.get(files[key].name) != _hash(files[key]):
            raise ValueError(f"historically frozen input differs: {key}")
    pairs = _jsonl(files["pairs"])
    manifest = json.loads(files["pair_manifest"].read_text())
    if (
        manifest["selected_baseline_count"] != 2
        or manifest["pair_count"] != 14
        or manifest["selected_pair_ids"] != [p["pair_id"] for p in pairs]
    ):
        raise ValueError("historical pair manifest differs")
    contexts = {}
    for raw in pairs:
        pair = AdversarialPair.model_validate(raw)
        key = f"{pair.benchmark_id}/{pair.tier}"
        if key not in contexts:
            prompt, context = _task_context(data_root, pair)
            contexts[key] = {
                "public_prompt": prompt,
                "context": context.model_dump(mode="json"),
            }
    with files["historical_rows"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    value = {
        "protocol": PROTOCOL,
        "pairs": pairs,
        "labels": _jsonl(files["labels"]),
        "source_config": json.loads(files["source_config"].read_text()),
        "pair_manifest": manifest,
        "historical_rows": rows,
        "historical_failures": _jsonl(files["historical_failures"]),
        "contexts": contexts,
        "source_files": {
            key: {"path": str(path.resolve()), "sha256": _hash(path)}
            for key, path in files.items()
        },
        "test_data_opened": False,
    }
    _validate_bundle(value)
    return sealed_write(output, value)


def _plan(bundle: dict) -> dict:
    _validate_bundle(bundle)
    scientific_protocol = judge_protocol(sign_policy=SIGN_POLICY)
    # The live critic may try a second seed after a failed pair; calibration
    # instead evaluates every predeclared seed, with no adaptive seed retries.
    scientific_protocol.pop("distinct_seed_attempts")
    scripts = (
        "scripts/judge_jetstream_calibration.py",
        "scripts/run_hybrid_judge.py",
        "scripts/analyze_hybrid_symmetric_aggregation.py",
        "scripts/analyze_hybrid_consensus_validation.py",
    )
    return {
        "protocol": PROTOCOL,
        "bundle_sha256": bundle["artifact_sha256"],
        "source_sha256": original.public._source_identity(),
        "runtime": original.public._runtime(),
        "scripts": {name: _hash(original.REPO / name) for name in scripts},
        "scientific_protocol": scientific_protocol,
        "seed_schedule": "all_five_seeds_both_orientations_no_adaptive_retries",
        "transport_policy": jetstream.POLICY,
        "output_contract": jetstream.OUTPUT_CONTRACT,
        "endpoint": jetstream.ENDPOINT,
        "served_model_revision": None,
        "settings_forwarding": "requested_not_independently_verified",
        "pairs": 14,
        "repetitions": 5,
        "seed_base": 10000,
        "planned_orientations": 140,
        "planned_llm_stages": 280,
        "maximum_physical_requests": 2800,
        "automatic_followup": False,
        "model_promotion": False,
        "test_data_opened": False,
    }


def freeze(source: Path, root: Path) -> dict:
    """Bind source cases, evaluator-only labels, current code and settings."""
    if source.resolve().is_relative_to(root.resolve()):
        raise ValueError("source export must be outside the output root")
    bundle = sealed_read(source)
    plan = _plan(bundle)
    with original.public._lock(root):
        sealed_write(
            root / "inputs.json",
            {k: v for k, v in bundle.items() if k != "artifact_sha256"},
        )
        return sealed_write(root / "plan.json", plan)


def verify(root: Path) -> tuple[dict, dict]:
    """Check both the sealed inputs and current executable identity."""
    plan, bundle = sealed_read(root / "plan.json"), sealed_read(root / "inputs.json")
    if {k: v for k, v in plan.items() if k != "artifact_sha256"} != _plan(bundle):
        raise ValueError("frozen Jetstream calibration identity differs")
    return plan, bundle


def units(bundle: dict):
    """Yield a deterministic complete schedule, independent of verdicts."""
    for index, raw in enumerate(bundle["pairs"]):
        for repetition in range(5):
            for order in ORDERS:
                yield (
                    f"pair-{index:02d}/seed-{repetition}/{order}",
                    raw,
                    repetition,
                    order,
                )


def _assess(
    raw: dict, entry: dict, repetition: int, order: str, work: Path, transport
) -> dict:
    """Reuse the production scientific stages; labels never enter this function."""
    pair = AdversarialPair.model_validate(raw)
    context = ValidationContext.model_validate(entry["context"])
    client = jetstream.client(
        LLMConfig(
            provider="vllm",
            model="openai/gpt-oss-120b",
            cache_directory=work / "cache",
            log_path=work / "events.jsonl",
            max_attempts=10,
            initial_backoff_seconds=1,
            max_backoff_seconds=30,
            jitter_fraction=0,
            vllm_reasoning_effort="low",
            vllm_temperature=0.2,
            vllm_seed=10000 + repetition,
            timeout_seconds=900,
            max_output_tokens=6144,
        ),
        transport,
    )
    requirements = extract_public_requirements(entry["public_prompt"])
    judge = PairedHybridJudge(
        seeded_clients=((10000 + repetition, client),),
        requirements=requirements,
        task_inputs=tuple(context.external_inputs),
        system_prompt=_system_prompt(
            entry["public_prompt"],
            context,
            MODEL,
            OllamaResponseMode.JSON_SCHEMA,
            atomic_mode=True,
        ),
        atomic_system_prompt=_atomic_system_prompt(
            entry["public_prompt"], context, MODEL
        ),
        scoring=HybridScoringConfig(**judge_protocol()["scoring"]),
        identity=PROTOCOL,
        sign_policy=SIGN_POLICY,
    )
    # Match the original calibration runner's protected-symbol canonicalization.
    a, repairs_a = repair_protected_declarations(pair.valid_candidate, context)
    b, repairs_b = repair_protected_declarations(pair.adversarial_candidate, context)
    if order == "baseline_b":
        a, b = b, a
    result = judge._orientation(client, a, b)
    decision = result.decision_value_for_a
    return {
        "pair_id": pair.pair_id,
        "judge_model": MODEL,
        "mutation_type": pair.mutation_type,
        "repetition": str(repetition),
        "order": order,
        "baseline_position": "A" if order == "baseline_a" else "B",
        "baseline_decision_value": ""
        if decision is None
        else str(decision if order == "baseline_a" else -decision),
        "requirements": requirements.model_dump_json(),
        "absolute_assessments": json.dumps(
            [v.model_dump(mode="json") for v in result.result.absolute_assessments]
        ),
        "comparative_assessments": json.dumps(
            [v.model_dump(mode="json") for v in result.result.comparative_assessments]
        ),
        "deterministic_assessments": json.dumps(
            [v.model_dump(mode="json") for v in result.deterministic]
        ),
        "baseline_repairs": json.dumps(repairs_a),
        "mutated_repairs": json.dumps(repairs_b),
    }


def run(
    root: Path, key_supplier: Callable[[], str], progress: Callable[[str], None] = print
) -> None:
    """Resume unstarted orientations; interrupted units never get a fresh budget."""
    with original.public._lock(root):
        plan, bundle = verify(root)
        key = None

        def credential():
            nonlocal key
            if key is None:
                key = key_supplier().strip()
                if not key or any(c in key for c in "\r\n"):
                    raise ValueError("a nonempty single-line API key is required")
            return key

        for number, (name, raw, repetition, order) in enumerate(units(bundle), 1):
            work = root / "orientations" / name
            identity = {"plan": plan["artifact_sha256"], "unit": name}
            if (work / "result.json").exists():
                if sealed_read(work / "result.json")["identity"] != identity:
                    raise ValueError("orientation identity differs")
                continue
            if (work / "started.json").exists():
                if sealed_read(work / "started.json")["identity"] != identity:
                    raise ValueError("started orientation identity differs")
                sealed_write(
                    work / "result.json",
                    {"identity": identity, "status": "interrupted"},
                )
                continue
            credential()  # Do not spend a unit on a cancelled credential prompt.
            progress(f"Orientation {number}/140: {name}")
            sealed_write(work / "started.json", {"identity": identity})
            transport = jetstream.JetstreamTransport(
                work / "calls", credential, progress
            )
            try:
                entry = bundle["contexts"][f"{raw['benchmark_id']}/{raw['tier']}"]
                row = _assess(raw, entry, repetition, order, work, transport)
                result = {"identity": identity, "status": "reviewed", "row": row}
            except LLMError as error:
                result = {
                    "identity": identity,
                    "status": "unavailable",
                    "error_type": type(error).__name__,
                }
            except KeyboardInterrupt:
                sealed_write(
                    work / "result.json",
                    {"identity": identity, "status": "interrupted"},
                )
                raise
            sealed_write(work / "result.json", result)
            progress(f"Orientation {number}/140: {result['status']}")
            # Stop provider outages/authentication failures before consuming the suite.
            # The failed unit remains terminal; rerunning resumes only unstarted units.
            if result["status"] != "reviewed":
                progress(
                    "Stopped after unavailable orientation; "
                    "inspect logs before resuming."
                )
                break
