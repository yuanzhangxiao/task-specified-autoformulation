"""Response-oriented model revision, separate from delivery/request failures."""

import json

from autoformalism.expressions import ModelValidationError
from autoformalism.llm.response_revision import PromptPreflightError
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    RevisionContractError,
)
from autoformalism.search import review_revision_multi as edits
from autoformalism.search.response_evidence import presentation
from autoformalism.search.review_multi_construction import public_contract
from autoformalism.staged_topology import content_hash

POLICY = "response-oriented-revision-1"
SYSTEM_PROMPT = (
    edits.SYSTEM_PROMPT.replace("e.g. E001", "e.g. R001")
    + """

training_evidence is a compact response comparison, measured on complete training
rollouts. Compare observed and predicted amplitudes, peaks, return toward initial
levels, bias and early/middle/late error. Use every target's overview, including
poorly fitted targets. Null timing means unidentifiable under the stated summary
rule, not zero delay. Multiple inputs/repeated events do not isolate causal effects.
Optional diagnostic samples are a small selection; the fitting objective still
uses all training samples. Cite only R... identifiers actually in evidence_catalog.
Do not treat a response summary as proof of the faulty term or of convergence.
"""
)


def payload(bundle, packet, parameters, response, retry=None):
    """Replace verbose residual tables; do not truncate equations or obligations."""
    value = edits.payload(bundle, packet, parameters, retry)
    evidence, refs = presentation(response, packet)
    value.update(
        protocol=POLICY,
        training_evidence=evidence,
        evidence_catalog=[{"ref": ref, "record": name} for ref, name in refs.items()],
    )
    if value.get("retry_feedback"):
        value["retry_feedback"].pop("available_evidence_refs", None)
    return value


def propose(plan, task, parent, client, response):
    """Three equation attempts; deterministic delivery failures stop immediately."""
    from autoformalism.rebuttal import review_deadline_pipeline as pipeline

    selected = parent["selected"]
    bundle, packet = selected["bundle"], selected["packet"]
    if response is None:
        return {"status": "residual_evidence_unavailable"}
    attempts, feedback = [], None
    for attempt in range(3):
        user = payload(
            bundle, packet, selected["fit"]["parameters"], response, feedback
        )
        user["public_target_contract"] = public_contract(
            plan["cells"][task["cell"]], task
        )
        user["public_requirement_findings"] = pipeline._certificate_feedback(
            selected["certificate"], task
        )
        record, raw = None, None
        try:
            record = client.call(
                system=SYSTEM_PROMPT,
                user=json.dumps(user, sort_keys=True, separators=(",", ":")),
                response_model=edits.ScientificRevision,
                step="review_response_content",
                attempt=attempt,
            )
            if record["status"] != "responded":
                attempts.append(
                    {
                        "request_hash": record["request_hash"],
                        "accepted": False,
                        "code": "PROVIDER_REQUEST_FAILED",
                        "error": record.get("error"),
                        "raw": None,
                    }
                )
                return {
                    "status": "provider_request_failed",
                    "attempts": attempts,
                    "revision_policy": POLICY,
                }
            shown = json.loads(record["request"]["body"]["messages"][1]["content"])
            refs = {r["ref"]: r["record"] for r in shown["evidence_catalog"]}
            raw = visible_response(record)
            decision = edits.apply_edits(bundle, packet, raw, reference_catalog=refs)
            certificate = None
            if decision["bundle"] is not None:
                certificate = pipeline.certificates(
                    decision["bundle"], plan["cells"][task["cell"]], task
                )
                if not certificate["eligible_for_development_selection"]:
                    raise RevisionContractError(
                        "PUBLIC_MODEL_REQUIREMENTS",
                        "Correct failed public predicates.",
                        **pipeline._certificate_feedback(certificate, task),
                    )
            attempts.append(
                {"request_hash": record["request_hash"], "accepted": True, "raw": raw}
            )
            return {
                "status": decision["outcome"],
                "bundle": decision["bundle"],
                "certificate": certificate,
                "decision": decision,
                "attempts": attempts,
                "revision_policy": POLICY,
            }
        except PromptPreflightError as error:
            return {
                "status": "request_preflight_failed",
                "error": str(error),
                "attempts": attempts,
                "revision_policy": POLICY,
            }
        except RepairBudgetExceeded as error:
            return {
                "status": "revision_failed",
                "error": str(error),
                "attempts": attempts,
            }
        except (ValueError, KeyError, TypeError, ModelValidationError) as error:
            if record is None:
                raise
            feedback = edits.feedback(
                bundle, packet, selected["fit"]["parameters"], raw, error
            )
            feedback.pop("available_evidence_refs", None)
            if isinstance(error, ModelValidationError):
                feedback["details"] = [
                    {"code": d.code, "location": d.location, "message": d.message}
                    for d in error.diagnostics
                ]
            attempts.append(
                {
                    "request_hash": record["request_hash"],
                    "record_sha256": content_hash(record),
                    "accepted": False,
                    "raw": raw,
                    "feedback": feedback,
                }
            )
    return {
        "status": "revision_failed",
        "attempts": attempts,
        "revision_policy": POLICY,
    }
