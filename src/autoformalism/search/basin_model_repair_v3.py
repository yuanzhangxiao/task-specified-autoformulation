"""Normalize redundant binding delivery without changing scientific equations."""

from __future__ import annotations

from pydantic import FiniteFloat, field_validator, model_validator

from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.search import basin_model_repair_v2 as previous

POLICY = "basin-assembled-model-repair-3"
old = previous.old
assessment = previous.assessment


class ParameterBinding(StrictSchema):
    """An empty entry is a no-op; two conflicting operations remain an error."""

    parameter: Identifier
    value: FiniteFloat | None = None
    same_as: Identifier | None = None

    @field_validator("value", mode="before")
    @classmethod
    def numeric_constant(cls, value):
        return old.ParameterBinding.numeric_constant(value)

    @model_validator(mode="after")
    def at_most_one(self):
        if self.value is not None and self.same_as is not None:
            raise ValueError("provide value OR same_as, not both")
        return self


class ModelRepair(previous.ModelRepair):
    """The existing repair contract with narrowly normalized binding entries."""

    parameter_bindings: tuple[ParameterBinding, ...] = ()


def normalize(
    bundle: dict, raw: dict, aliases: dict | None = None
) -> tuple[dict, dict]:
    """Remove empty entries and one redundant edge per parameter equality cycle."""
    reply = ModelRepair.model_validate(raw)
    aliases = bundle.get("repair_parameter_aliases", {}) if aliases is None else aliases
    existing = {
        p["name"] for p in bundle["initialization"]["base_candidate"]["parameters"]
    }
    available = existing | set(aliases) | {p.name for p in reply.new_parameters}
    names = [b.parameter for b in reply.parameter_bindings]
    if len(names) != len(set(names)):
        raise ValueError("one binding per parameter")
    missing = set(names) - available
    if missing:
        raise ValueError(f"binding names unavailable parameters: {sorted(missing)}")
    empty = [
        b.parameter
        for b in reply.parameter_bindings
        if b.value is None and b.same_as is None
    ]
    cleaned = reply.model_dump(mode="json")
    cleaned["parameter_bindings"] = [
        b.model_dump(mode="json")
        for b in reply.parameter_bindings
        if b.parameter not in empty
    ]
    compatible = previous.ModelRepair.model_validate(cleaned)
    bindings, repeated = previous._aliases(compatible, aliases)
    requested = {b.parameter: b for b in bindings}
    cycles, remove, done = [], set(), set()
    for start in sorted(requested):
        path = []
        name = start
        while name in requested and name not in done:
            if name in path:
                cycle = sorted(path[path.index(name) :])
                # Preserve an existing declaration when possible; acyclic aliases
                # keep their originally requested direction and warm-start identity.
                representative = min(cycle, key=lambda n: (n not in existing, n))
                remove.add(representative)
                cycles.append({"members": cycle, "representative": representative})
                break
            path.append(name)
            name = requested[name].same_as
        done.update(path)
    cleaned["parameter_bindings"] = [
        b.model_dump(mode="json") for b in bindings if b.parameter not in remove
    ]
    # Every non-representative still passes the existing role/domain/bounds/scope
    # checks against the retained representative in previous.apply.
    return cleaned, {
        "empty_bindings_removed": empty,
        "equality_cycles": cycles,
        "repeated_alias_bindings": repeated,
    }


def apply(bundle, raw, case, *, aliases=None):
    """Reuse the v2 compiler; retain original input and normalization evidence."""
    cleaned, evidence = normalize(bundle, raw, aliases)
    tx = previous.apply(bundle, cleaned, case, aliases=aliases)
    return {
        **tx,
        "policy": POLICY,
        "reply": raw,
        "normalized_reply": cleaned,
        "binding_normalizations": evidence,
    }


def transition(current, parent, raw, case, surveys, *, aliases=None):
    """Preserve the v2 static gate and immediate handoff for explicit patches."""
    reply = ModelRepair.model_validate(raw)
    if not reply.has_edits() and not reply.accept_displayed:
        raise ValueError(
            "empty reply: supply an explicit patch or accept the current model"
        )
    tx = apply(current, raw, case, aliases=aliases) if reply.has_edits() else None
    child = tx["bundle"] if tx else current
    static = assessment(child, case, surveys)
    status = (
        "static_repair_required"
        if any(c["status"] == "fail" for c in static["checks"])
        else "retained"
        if child["candidate"] == parent["candidate"]
        else "eligible_for_fit"
    )
    return {
        "bundle": child,
        "transaction": tx,
        "status": status,
        "static_assessment": static,
    }
