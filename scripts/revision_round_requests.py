"""Scientific generation and blinded selection from a common training packet."""

from __future__ import annotations

import json
import random

from scripts.revision_round_provider import MODEL, digest

GENERATE = (
    "Revise the given model in one round. Return exactly three different alternative "
    "edits in the permitted format. Use scientific knowledge and the training "
    "evidence. Prefer plausible mechanisms and distinguishable alternatives. "
    "The runtime fits shared coefficients. Include each edit only once."
)
RANK = (
    "Rank all model options for scientific plausibility and consistency with the "
    "public training evidence. Return every ID exactly once, best first. "
    "An unchanged parent is a legitimate choice. The same deterministic fit and "
    "admission rules apply to every option. Consider mechanisms, parsimony and "
    "whether the modification addresses the discrepancy."
)


def body(instruction: str, user: dict, schema: dict, seed: int) -> dict:
    return {
        "model": MODEL,
        "seed": seed,
        "temperature": 0.7,
        "max_tokens": 1536,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(user, sort_keys=True)},
        ],
        "reasoning_effort": "low",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "RevisionRoundReply",
                "strict": True,
                "schema": schema,
            },
        },
    }


def public(packet: dict) -> dict:
    """Reveal the allowed edit grammar and shared evidence, hiding the generator."""
    return {key: packet[key] for key in ("edit_contract", "evidence", "parent")}


def generation(packet: dict, seed: int) -> dict:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["alternatives"],
        "properties": {
            "alternatives": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": packet["edit_schema"],
            }
        },
    }
    return body(GENERATE, public(packet), schema, seed)


def ranking(packet: dict, choices: dict, case: str, seed: int) -> tuple[dict, dict]:
    """Permute each judge's pooled menu independently of provenance and fit score."""
    ordered = sorted(choices)
    random.Random(digest([case, seed])).shuffle(ordered)
    labels = {f"M{i + 1}": key for i, key in enumerate(ordered)}
    menu = {label: choices[key] for label, key in labels.items()}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["rank"],
        "properties": {
            "rank": {
                "type": "array",
                "minItems": len(menu),
                "maxItems": len(menu),
                "items": {"type": "string", "enum": list(menu)},
            }
        },
    }
    return body(RANK, {**public(packet), "options": menu}, schema, seed), labels
