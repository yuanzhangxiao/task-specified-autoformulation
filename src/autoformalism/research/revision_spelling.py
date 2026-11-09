"""Resolve conventional Greek identifier spellings in a scoped revision patch."""

from __future__ import annotations

import io
import tokenize
import unicodedata
from collections import defaultdict
from copy import deepcopy

from pydantic import ValidationError

from autoformalism.research.construction_revision_edits import RevisionValidationError
from autoformalism.schemas.prefix_initialization import PrefixFitProblem

_LETTERS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
]
_GREEK = str.maketrans(
    {
        unicodedata.lookup(
            f"GREEK {case} LETTER {name.upper().replace('LAMBDA', 'LAMDA')}"
        ): name if case == "SMALL" else name.capitalize()
        for name in _LETTERS
        for case in ("SMALL", "CAPITAL")
    }
)


def _namespace(names: list[str]) -> set[str]:
    spellings = defaultdict(set)
    for name in names:
        name = name.strip()
        spellings[name.translate(_GREEK)].add(name)
    for name, choices in spellings.items():
        if len(choices) > 1:
            raise RevisionValidationError(
                "AMBIGUOUS_IDENTIFIER_SPELLING",
                "distinct declarations in this scope have the same transliteration",
                canonical=name,
                declarations=sorted(choices),
            )
    return set(spellings)


def normalize(parent: PrefixFitProblem, raw: dict, schema) -> tuple[dict, list[dict]]:
    """Canonicalize whole identifier tokens only against declarations in their scope."""
    audit = []
    try:
        parsed = schema.model_validate(raw)
    except ValidationError as exc:
        # Structured identifier fields can fail the ASCII schema before formula parsing.
        # Retry only its identifier pattern. All other schema checks still apply.
        edits = []
        for error in exc.errors():
            old = error["input"]
            if (
                error["type"] != "string_pattern_mismatch"
                or error.get("ctx", {}).get("pattern") != "^[A-Za-z][A-Za-z0-9_]*$"
                or not isinstance(old, str)
                or old.translate(_GREEK) == old
            ):
                raise
            steps, node = [], raw
            for step in error["loc"]:
                # Pydantic inserts the discriminator tag in union error locations.
                if (
                    isinstance(node, dict)
                    and step not in node
                    and step == node.get("mode")
                ):
                    continue
                steps.append(step)
                node = node[step]
            edits.append((steps, old, old.translate(_GREEK)))
        fixed = deepcopy(raw)
        for steps, old, new in edits:
            node = fixed
            for step in steps[:-1]:
                node = node[step]
            node[steps[-1]] = new
            audit.append({"field": steps, "from": old, "to": new})
        parsed = schema.model_validate(fixed)

    patch = parsed.model_dump(mode="json")
    context = parent.context
    global_names = _namespace(
        [
            context.time_symbol,
            *context.targets,
            *context.external_inputs,
            *context.auxiliaries,
            *context.fixed_covariates,
            *context.lagged_targets,
            *(s.name for s in parent.base_candidate.states),
            *(p.name for p in parent.base_candidate.processes),
            *(p.name for p in parent.base_candidate.parameters),
            *(
                r["name"]
                for field in ("new_states", "new_processes", "new_parameters")
                for r in raw.get(field, ())
            ),
        ]
    )
    locals_by_state = {}
    for field in ("new_states", "initialization_edits"):
        for i, state in enumerate(raw.get(field, ())):
            initial = state["initial"]
            if initial["mode"] == "map":
                locals_by_state[field, i] = _namespace(
                    [
                        *(p["name"] for p in initial.get("parameters", ())),
                        *(
                            r["alias"]
                            for refs in ("state_initials", "public_initials")
                            for r in initial.get(refs, ())
                        ),
                    ]
                )
    for entry in audit:
        steps = entry["field"]
        local = "initial" in steps and steps[-1] in {"name", "alias"}
        scope = locals_by_state[steps[0], steps[1]] if local else global_names
        if entry["to"] not in scope:
            raise RevisionValidationError(
                "UNKNOWN_IDENTIFIER_SPELLING",
                "the spelling has no declaration in this scope",
                **entry,
            )

    def formula(row, key, path, scope):
        source = row[key]
        if source.isascii():
            return
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except (tokenize.TokenError, IndentationError):
            return  # The restricted expression parser reports malformed source.
        changes = {}
        for i, token in enumerate(tokens):
            new = token.string.translate(_GREEK)
            if token.type == tokenize.NAME and token.string != new and new in scope:
                changes[token.string] = new
                tokens[i] = token._replace(string=new)
        if changes:
            row[key] = tokenize.untokenize(tokens)
            audit.extend(
                {"field": [*path, key], "from": old, "to": new}
                for old, new in changes.items()
            )

    if "edit" in patch:
        formula(patch["edit"], "expression", ["edit"], global_names)
    for i, edit in enumerate(patch.get("edits", ())):
        formula(edit, "expression", ["edits", i], global_names)
    for field, key in (("new_processes", "expression"), ("new_states", "rhs")):
        for i, row in enumerate(patch[field]):
            formula(row, key, [field, i], global_names)
    for (field, i), local_names in locals_by_state.items():
        formula(
            patch[field][i]["initial"],
            "expression",
            [field, i, "initial"],
            local_names,
        )
    return patch, audit
