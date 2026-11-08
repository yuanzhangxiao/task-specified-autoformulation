"""Import verified original T2 call slots once, including interrupted generations."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import strict_provider_schema, visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts.construction_functions import file_hash

V1_PLAN = "ef0f1fe4abc2d7ce85b2de57dc64109dc9b592c1a25d9782768b46ac2db4585b"
BRANCHES = [f"{g}-{i}" for g in ("c15", "c21") for i in range(5)]
FRESH = [f"fresh-{i}" for i in range(100, 105)]
SLOTS = (
    {"c15-five", "c21-five"}
    | {f"{b}-patch_correction" for b in BRANCHES}
    | {
        f"{b}-{stage}"
        for b in FRESH
        for stage in ("topology", "topology_correction", "laws", "complete_correction")
    }
)


def bind_source(root: Path, previous: Path | None, intake: str) -> None:
    """Freeze a predecessor location separately from portable scientific identities."""
    binding_path = root / "predecessor.json"
    if binding_path.exists():
        bound = sealed_read(binding_path)
        if bound["intake_sha256"] != intake:
            raise ValueError("Predecessor intake differs")
        if previous and previous.exists() and bound["path"] != str(previous.resolve()):
            raise ValueError("The frozen predecessor cannot be replaced")
        return
    if previous is None or not previous.exists():
        sealed_write(
            binding_path, {"path": None, "plan_sha256": None, "intake_sha256": intake}
        )
        return
    previous = previous.resolve()
    if root.resolve().is_relative_to(previous) or previous.is_relative_to(
        root.resolve()
    ):
        raise ValueError("Use separate, non-nested result roots for the two protocols")
    plan = sealed_read(previous / "plan.json")
    if (
        plan["artifact_sha256"] != V1_PLAN
        or sealed_read(previous / "inputs.json")["artifact_sha256"] != intake
    ):
        raise ValueError(
            "Previous root does not contain the reviewed 27-call plan and intake"
        )
    binding = {"path": str(previous), "plan_sha256": V1_PLAN, "intake_sha256": intake}
    if (root / "legacy").exists() and not (root / "predecessor.json").exists():
        raise ValueError("A new predecessor cannot be added after generation started")
    sealed_write(root / "predecessor.json", binding)


def snapshot(root: Path) -> dict:
    """Copy durable original slots into one atomic snapshot."""
    dest = root / "legacy"
    if (dest / "index.json").exists():
        return sealed_read(dest / "index.json")
    with tempfile.TemporaryDirectory(dir=root, prefix=".prefix-") as temporary:
        stage = Path(temporary) / "snapshot"
        index = {"plan_sha256": None, "slots": {}, "physical_requests": 0}
        binding = root / "predecessor.json"
        bound = sealed_read(binding)
        if bound["path"] is not None:
            source = Path(bound["path"])
            with _lock(source / "worker-lock"):
                if sealed_read(source / "plan.json")["artifact_sha256"] != V1_PLAN:
                    raise ValueError(
                        "The previous plan changed before its calls were copied"
                    )
                index["plan_sha256"] = V1_PLAN
                shutil.copy2(source / "plan.json", _parent(stage / "plan.json"))
                for slot in sorted((source / "steps").glob("*")):
                    if slot.is_symlink() or not slot.is_dir() or slot.name not in SLOTS:
                        raise ValueError(f"Unexpected original generation slot: {slot}")
                    calls = sorted((slot / "calls").glob("*.json"))
                    if len(calls) > 1:
                        raise ValueError(
                            f"Multiple physical calls in original slot: {slot}"
                        )
                    files = calls + (
                        [slot / "reply.json"] if (slot / "reply.json").exists() else []
                    )
                    if not files:
                        continue  # Only a tokenizer preflight could have occurred.
                    hashes = {}
                    for file in files:
                        if file.is_symlink() or not file.is_file():
                            raise ValueError(f"Expected a regular call receipt: {file}")
                        name = str(file.relative_to(slot))
                        copied = _parent(stage / "steps" / slot.name / name)
                        shutil.copy2(file, copied)
                        hashes[name] = file_hash(copied)
                    index["slots"][slot.name] = hashes
                    index["physical_requests"] += len(calls)
                for slot in index["slots"]:
                    if slot in {"c15-five", "c21-five"} or slot.endswith("-topology"):
                        continue
                    origin, i, stage_name = slot.split("-", 2)
                    first_slot = (
                        f"fresh-{i}-topology" if origin == "fresh" else f"{origin}-five"
                    )
                    if first_slot not in index["slots"] or (
                        stage_name == "complete_correction"
                        and f"fresh-{i}-laws" not in index["slots"]
                    ):
                        raise ValueError(f"Original call {slot} lost its prerequisite")
                # A purged call file is not an unused budget slot.
                for receipt in (source / "first").glob("*.json"):
                    row = sealed_read(receipt)
                    slot = (
                        row["name"] + "-topology"
                        if row["group"] == "fresh"
                        else row["group"] + "-five"
                    )
                    if slot not in index["slots"]:
                        raise ValueError(
                            f"Original first-response checkpoint lacks its call: {slot}"
                        )
                for receipt in (source / "results").glob("*.json"):
                    row = sealed_read(receipt)
                    for event in row["history"]:
                        slot = (
                            row["group"] + "-five"
                            if event["stage"] == "patch"
                            else row["name"] + "-" + event["stage"]
                        )
                        if slot not in index["slots"]:
                            raise ValueError(
                                f"Original branch checkpoint lacks its call: {slot}"
                            )
                summary = source / "summary.json"
                if (
                    summary.exists()
                    and json.loads(summary.read_text()).get("physical_requests")
                    != index["physical_requests"]
                ):
                    raise ValueError(
                        "Original physical call count differs from its report"
                    )
        if index["physical_requests"] > 27:
            raise ValueError("Original campaign exceeded its 27-call limit")
        result = sealed_write(stage / "index.json", index)
        stage.rename(dest)
        return result


def _parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def visible(record: dict) -> dict:
    """Recover the same visible response or terminal failure without network access."""
    if record["status"] == "inflight":
        return {
            "status": "uncertain",
            "error": "Original call was interrupted after reservation",
        }
    result = {"status": record["status"]}
    if record["status"] == "responded":
        try:
            value = json.loads(
                json.dumps(visible_response(record), sort_keys=True, allow_nan=False)
            )
            result.update(status="visible", value=value)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            result.update(status="incomplete_reply", error=str(exc))
    return result


def prior_reply(root, index, slot, seed, request, settings):
    """Authenticate the complete original prompt, schema, sampling seed and response."""
    files = index["slots"].get(slot)
    if files is None:
        return None
    source = root / "legacy/steps" / slot
    for name, digest in files.items():
        if file_hash(source / name) != digest:
            raise ValueError(f"Original call data changed in {slot}")
    identity = content_hash([index["plan_sha256"], slot])
    schema = request["response_model"].model_json_schema()
    key = content_hash([identity, seed, request["system"], request["user"], schema])
    saved = sealed_read(source / "reply.json") if "reply.json" in files else None
    if saved and saved["key"] != key:
        raise ValueError(f"Original prompt differs in {slot}")
    calls = [name for name in files if name.startswith("calls/")]
    if not calls:
        if (
            not saved
            or saved.get("status") != "preflight_blocked"
            or "request_hash" in saved
        ):
            raise ValueError(f"Original generation receipt is missing in {slot}")
        return {
            "status": saved["status"],
            "error": saved["error"],
            "source_plan": index["plan_sha256"],
        }
    path = source / calls[0]
    record = json.loads(path.read_text())
    expected_body = {
        "model": settings.model,
        "messages": [{"role": k, "content": request[k]} for k in ("system", "user")],
        "stream": False,
        "reasoning_effort": settings.reasoning_effort,
        "temperature": settings.temperature,
        "max_tokens": settings.max_output_tokens,
        "seed": int(content_hash([seed, slot, 0])[:8], 16) % (2**31),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request["response_model"].__name__,
                "strict": True,
                "schema": strict_provider_schema(schema),
            },
        },
    }
    physical = record["request"]
    if (
        physical["namespace"] != identity
        or physical["protocol"] != "scientific-staged-topology-1"
        or physical.get("budget_reservation_policy") != "serving-tokens-plus-output-1"
        or physical["body"] != expected_body
        or physical["settings"] != settings.model_dump(mode="json")
        or record["step"] != slot
        or record["attempt"] != 0
        or record["request_hash"] != path.stem
        or content_hash(physical) != path.stem
    ):
        raise ValueError(f"Original physical request differs in {slot}")
    result = {
        **visible(record),
        "request_hash": path.stem,
        "physical_receipt_sha256": files[calls[0]],
    }
    if (
        saved
        and {k: v for k, v in saved.items() if k not in {"key", "artifact_sha256"}}
        != result
    ):
        raise ValueError(
            f"Original visible reply differs from its physical record in {slot}"
        )
    return {**result, "source_plan": index["plan_sha256"]}
