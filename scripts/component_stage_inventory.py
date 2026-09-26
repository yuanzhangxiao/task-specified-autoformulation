"""Project saved stage events into a compact, payload-free mechanical inventory."""

from __future__ import annotations

import re

if __package__:
    from .component_audit_io import Snapshot, diagnostic, digest, label
else:
    from component_audit_io import Snapshot, diagnostic, digest, label


def stage_for(step: str, default: str) -> str:
    """Group documented step prefixes without inferring scientific correctness."""
    if step.startswith(("variable", "inventory", "memory")):
        return "variables"
    if step.startswith(("process_review", "optional_process_review")):
        return "process_proposal"
    if step.startswith(("process_uses", "topology", "equation_topology")):
        return "topology"
    if step.startswith(("initial", "causal")):
        return "initialization"
    return default


def events(
    value: dict,
    *,
    task: str,
    index: int,
    route: str,
    stage: str,
    source: str,
    key: str = "events",
) -> list[dict]:
    """Retain every saved event, including unknown outcomes and rejection hashes."""
    result = []
    for ordinal, event in enumerate(value.get(key) or []):
        if not isinstance(event, dict):
            continue
        step = label(event.get("step")) or stage
        accepted = event.get("accepted")
        result.append(
            {
                "task": task,
                "round": index,
                "route": route,
                "stage": stage_for(step, stage),
                "step": step,
                "source": source,
                "ordinal": ordinal,
                "request_sha256": event.get("request_hash")
                if re.fullmatch(r"[0-9a-f]{64}", str(event.get("request_hash")))
                else None,
                "attempt": event.get("attempt")
                if type(event.get("attempt")) is int
                else None,
                "outcome": "accepted"
                if accepted is True
                else "rejected"
                if accepted is False
                else "unrecorded",
                "partial_acceptance": event.get("partial_acceptance") is True,
                **diagnostic(event.get("error") or event.get("feedback")),
            }
        )
    return result


def inventory(
    reader: Snapshot, base: str, task: str, index: int, proposal: dict | None
) -> tuple[list[dict], list[dict]]:
    """Read known stage paths only; never scan call caches, prompts, or data files.

    Final results take precedence over progress snapshots so events are not
    counted twice. A sealed wrapper takes precedence over its inner stage file.
    These are saved validator events, not independent LLM-call counts.
    """
    stages, attempts = [], []
    routes = {"direct": base, "variables": f"{base}/variables"}
    routes.update(
        {r: f"{base}/{r}" for r in ("process", "ordinary", "ordinary_fallback")}
    )
    for route, directory in routes.items():
        kinds = ("variables",) if route == "variables" else ("topology", "functions")
        for kind in kinds:
            inner = directory if kind == "variables" else f"{directory}/{kind}"
            wrapper = (
                f"{base}/variables.json"
                if kind == "variables"
                else f"{directory}/topology_stage.json"
            )
            # The function-stage wrapper is singular; the working directory is plural.
            if kind == "functions":
                wrapper = f"{directory}/function_stage.json"
            candidates = (
                (wrapper, True),
                (f"{inner}/result.json", False),
                (f"{inner}/progress.json", False),
            )
            for source, sealed in candidates:
                if not reader.exists(source):
                    continue
                value = reader.read(source, sealed=sealed)
                if value is not None and sealed:
                    value = value.get("result")
                if not isinstance(value, dict):
                    reader.issue(source, "invalid_stage_shape")
                    break
                stages.append(
                    {
                        "task": task,
                        "round": index,
                        "route": route,
                        "stage": kind,
                        "source": source,
                        "status": label(value.get("status")) or "unrecorded",
                        "progress_only": source.endswith("progress.json"),
                        **diagnostic(value.get("error")),
                    }
                )
                attempts.extend(
                    events(
                        value,
                        task=task,
                        index=index,
                        route=route,
                        stage=kind,
                        source=source,
                    )
                )
                break
        if route == "variables":
            continue
        source = f"{directory}/topology/process_review.json"
        if reader.exists(source):
            value = reader.read(source)
            if value:
                if value.get("sha256") != digest(
                    {k: v for k, v in value.items() if k != "sha256"}
                ):
                    reader.issue(source, "process_review_seal_differs")
                stages.append(
                    {
                        "task": task,
                        "round": index,
                        "route": route,
                        "stage": "process_proposal",
                        "source": source,
                        "status": label(value.get("status")) or "unrecorded",
                        "suggestions": len(value.get("suggestions") or []),
                        **diagnostic(value.get("error")),
                    }
                )
        source = f"{directory}/functions/initialization/state.json"
        if reader.exists(source):
            value = reader.read(source)
            if value:
                if value.get("state_sha256") != digest(
                    {k: v for k, v in value.items() if k != "state_sha256"}
                ):
                    reader.issue(source, "initialization_state_seal_differs")
                stages.append(
                    {
                        "task": task,
                        "round": index,
                        "route": route,
                        "stage": "initialization",
                        "source": source,
                        "status": label(value.get("status")) or "unrecorded",
                    }
                )
                attempts.extend(
                    events(
                        value,
                        task=task,
                        index=index,
                        route=route,
                        stage="initialization",
                        source=source,
                        key="attempts",
                    )
                )
    if proposal:
        source = f"{base}/proposal.json"
        # Route summaries are not attempts with accepted/rejected outcomes.
        revisions = [
            a
            for a in proposal.get("attempts", [])
            if isinstance(a, dict) and "accepted" in a
        ]
        attempts.extend(
            events(
                {"events": revisions},
                task=task,
                index=index,
                route="revision",
                stage="revision",
                source=source,
            )
        )
    return stages, attempts
