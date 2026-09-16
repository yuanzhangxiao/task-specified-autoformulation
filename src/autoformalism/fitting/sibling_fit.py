"""A sealed compatible warm start for one child, without changing fit profiles."""

from __future__ import annotations

import math
from pathlib import Path
from time import time

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.collocation_sensitivity import _role_start
from autoformalism.fitting.fitter import _training_variables
from autoformalism.fitting.models import FitConfig
from autoformalism.schemas.public_fitting import (
    PublicFitRequest,
    PublicFitResult,
    PublicSplit,
)

POLICY = "identical-parameter-declaration-warm-start-1"
EDIT_POLICY = "model-content-boundary-warm-start-1"


def compatible_seed(
    parent: PublicFitRequest,
    child: PublicFitRequest,
    parameters: dict[str, float],
    training: PublicSplit,
    *,
    allow_initialization_changes: bool = False,
) -> dict:
    """Retain values only for identical declarations, including latent boundaries."""
    if (
        parent.initialization_plan != child.initialization_plan
        and not allow_initialization_changes
    ):
        raise ValueError("child must preserve the scientific initialization plan")
    if parent.context != child.context or parent.profile != child.profile:
        raise ValueError("child must preserve public context and numerical profile")
    old, _, _ = public._lower(parent)
    new, guesses, new_initials = public._lower(child)
    old_specs = {p.name: p for p in old.validated.candidate.parameters}
    new_specs = {p.name: p for p in new.validated.candidate.parameters}
    if set(parameters) != set(old_specs) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in parameters.values()
    ):
        raise ValueError("parent requires a complete finite parameter vector")
    train = public.unpack_split(training)
    for variable in _training_variables(old, train, FitConfig()):
        if (
            not variable.lower
            <= parameters[variable.name.removeprefix("parameter:")]
            <= variable.upper
        ):
            raise ValueError("parent parameter outside its declared domain")
    changed_boundaries = {
        state
        for state, rule in child.initialization_plan.rules.items()
        if parent.initialization_plan.rules.get(state) != rule
    }
    reset_initials = {
        name
        for state in changed_boundaries
        for name in new_initials["bindings"][state]["parameters"]
    }
    retained = sorted(
        n
        for n, p in new_specs.items()
        if old_specs.get(n) == p and n not in reset_initials
    )
    full = {**_role_start(new.validated.candidate, train), **guesses}
    full.update({name: parameters[name] for name in retained})
    init_names = set(new.parameter_names) - {
        p.name for p in child.base_candidate.parameters
    }
    if not init_names <= set(retained) and not allow_initialization_changes:
        raise ValueError(
            "all initializer parameter declarations must survive unchanged"
        )
    if set(full) != set(new_specs):
        raise ValueError("child warm start is not a complete parameter vector")
    for variable in _training_variables(new, train, FitConfig()):
        if (
            not variable.lower
            <= full[variable.name.removeprefix("parameter:")]
            <= variable.upper
        ):
            raise ValueError("child warm start outside its declared domain")
    return {
        "policy": EDIT_POLICY if allow_initialization_changes else POLICY,
        "parameters": full,
        "parameter_sha256": public.content_sha256(full),
        "retained_parameters": retained,
        "retained_initializer_parameters": sorted(init_names & set(retained)),
        "fresh_parameters": sorted(set(new_specs) - set(retained)),
        "removed_parameters": sorted(set(old_specs) - set(new_specs)),
        "changed_declarations": sorted(
            n
            for n in old_specs.keys() & new_specs.keys()
            if old_specs[n] != new_specs[n]
        ),
        "initialization_plan_unchanged": (
            parent.initialization_plan == child.initialization_plan
        ),
        **(
            {
                "changed_boundaries": sorted(changed_boundaries),
                "reset_initializer_parameters": sorted(reset_initials & set(old_specs)),
            }
            if allow_initialization_changes
            else {}
        ),
    }


def _freeze(
    parent,
    child,
    parameters,
    training,
    validation,
    lineage,
    allow_initialization_changes=False,
):
    body = {
        "protocol": "numerical-sibling-fit-1",
        "public_fit": public._bundle(child, training, validation),
        "parent_request": public._jsonable(parent),
        "parent_parameters": parameters,
        "seed": compatible_seed(
            parent,
            child,
            parameters,
            training,
            allow_initialization_changes=allow_initialization_changes,
        ),
        "lineage": lineage,
        "runtime": public._runtime(),
        **({"warm_start_policy": EDIT_POLICY} if allow_initialization_changes else {}),
    }
    return {**body, "identity": public.content_sha256(body)}


def prepare_child_fit(
    parent: PublicFitRequest,
    child: PublicFitRequest,
    parameters: dict[str, float],
    training: PublicSplit,
    validation: PublicSplit,
    directory: Path,
    *,
    lineage: dict,
    allow_initialization_changes: bool = False,
) -> dict:
    """Bind the full vector separately from the unchanged scientific request schema."""
    frozen = _freeze(
        parent,
        child,
        parameters,
        training,
        validation,
        lineage,
        allow_initialization_changes=allow_initialization_changes,
    )
    with public._lock(directory):
        path = directory / "freeze.json"
        if path.exists():
            if public._read(path) != frozen:
                raise ValueError("child fit freeze differs")
        else:
            if any(p.name != ".lock" for p in directory.iterdir()):
                raise ValueError("child fit directory is not empty")
            public._write(path, frozen)
    return frozen


def _load(directory):
    frozen = public._read(directory / "freeze.json")
    bundle = frozen["public_fit"]
    parent = PublicFitRequest.model_validate(frozen["parent_request"])
    child = PublicFitRequest.model_validate(bundle["request"])
    train = PublicSplit.model_validate(bundle["training"])
    val = PublicSplit.model_validate(bundle["validation"])
    expected = _freeze(
        parent,
        child,
        frozen["parent_parameters"],
        train,
        val,
        frozen["lineage"],
        allow_initialization_changes=frozen.get("warm_start_policy") == EDIT_POLICY,
    )
    if expected != frozen:
        raise ValueError("child fit source, seed, data or profile differs")
    return frozen, child, train, val


def _read_result(directory, frozen, request):
    path = directory / "result.json"
    if not path.exists():
        return None
    envelope = public._read(path)
    if public.content_sha256(envelope["result"]) != envelope["sha256"]:
        raise ValueError("child result digest differs")
    result = PublicFitResult.model_validate(envelope["result"])
    base = public._result_base(
        {**frozen["public_fit"], "identity": frozen["identity"]}, request
    )
    if any(
        public._jsonable(getattr(result, k)) != public._jsonable(v)
        for k, v in base.items()
    ):
        raise ValueError("child result lineage differs")
    if result.backend_result_sha256 is not None:
        raw = public._read(directory / "backend_result.json")
        if public.content_sha256(raw) != result.backend_result_sha256:
            raise ValueError("child backend digest differs")
        if any(
            public._jsonable(getattr(result, k)) != public._jsonable(v)
            for k, v in public._evidence(request, raw).items()
        ):
            raise ValueError("child result/backend evidence differs")
    return result


def inspect_child_fit(directory: Path) -> dict:
    """Read-only verification; reports never invoke the optimizer."""
    frozen, request, _, _ = _load(directory)
    result = _read_result(directory, frozen, request)
    return {
        "identity": frozen["identity"],
        "seed": frozen["seed"],
        "result": public._jsonable(result),
    }


def execute_child_fit(directory: Path) -> PublicFitResult:
    """Run the existing backend once; started but incomplete attempts are consumed."""
    with public._lock(directory):
        frozen, request, train, val = _load(directory)
        saved = _read_result(directory, frozen, request)
        if saved is not None:
            return saved
        base = public._result_base(
            {**frozen["public_fit"], "identity": frozen["identity"]}, request
        )
        started = directory / "started.json"
        model, _, _ = public._lower(request)
        issue = public._capability(request, model)
        if started.exists():
            if public._read(started)["identity"] != frozen["identity"]:
                raise ValueError("child started marker differs")
            result = PublicFitResult(
                **base,
                status="interrupted",
                message="Started child attempt has no result; allocation consumed.",
            )
        elif issue:
            result = PublicFitResult(
                **base, status="capability_unsupported", message=issue
            )
        else:
            public._write(
                started,
                {
                    "identity": frozen["identity"],
                    "utc_seconds": time(),
                    "runtime": public._runtime(),
                },
            )
            try:
                raw = public._run_backend(
                    request,
                    model,
                    public.unpack_split(train),
                    public.unpack_split(val),
                    frozen["seed"]["parameters"],
                    frozen["public_fit"]["settings"],
                    directory,
                )
                public._write(directory / "backend_result.json", raw)
                evidence = public._evidence(request, raw)
                complete = (
                    evidence["parameters"] is not None
                    and evidence["training"].available
                    and evidence["validation"].available
                )
                result = PublicFitResult(
                    **base,
                    **evidence,
                    status="complete" if complete else "fit_failed",
                    backend_result_sha256=public.content_sha256(raw),
                    message=(
                        "Child evidence at retained parameters; accuracy and "
                        "budget status are separate."
                    ),
                )
            except Exception as error:
                result = PublicFitResult(
                    **base,
                    status="fit_failed",
                    message=f"{type(error).__name__}: {error}"[:6000],
                )
        value = result.model_dump(mode="json")
        public._write(
            directory / "result.json",
            {"result": value, "sha256": public.content_sha256(value)},
        )
        return result
