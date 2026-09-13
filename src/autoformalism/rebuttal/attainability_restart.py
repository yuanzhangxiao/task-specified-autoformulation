"""Copy-on-write recovery of reference arms without rerunning completed fits."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from autoformalism.rebuttal.attainability_campaign import (
    AttainabilityPlan,
    _checkpoint,
    _identity,
    code_identity,
    verify,
)
from autoformalism.rebuttal.attainability_reference import (
    reference_problem,
    reference_skeleton,
)
from autoformalism.rebuttal.fitter_diagnostic import (
    _write_bytes,
    read_json,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.piecewise_campaign import safe_path
from autoformalism.staged_topology import content_hash


def prepare_reference_recovery(
    source: Path, output: Path, plan: AttainabilityPlan
) -> dict:
    """Freeze the same matrix; retain completed public arms with old identities.

    Only reference arms with no previous numerical fit are eligible. A failed
    generation audit can be attempted under the explicitly revised gate. This
    never grants a new fitting budget to an interrupted numerical optimization.
    """
    if source.resolve().is_relative_to(
        output.resolve()
    ) or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("source/output must be separate")
    original = read_json(source / "freeze.json")
    _identity(original)
    previous = AttainabilityPlan.model_validate(original["plan"])
    runtime = identity_runtime()
    old_runtime = original["runtime"]
    if (
        old_runtime["packages"] != runtime["packages"]
        or old_runtime["casadi"] != runtime["casadi"]
        or old_runtime["python"].split(".")[:2] != runtime["python"].split(".")[:2]
    ):
        raise ValueError(
            "reference recovery requires the original numerical environment"
        )
    if (
        previous.protocol != "fitter-attainability-1"
        or plan.protocol != "fitter-attainability-2"
    ):
        raise ValueError(
            "reference recovery requires an original v1 campaign and v2 plan"
        )
    if (
        previous.fit != plan.fit
        or previous.generation_seconds != plan.generation_seconds
        or original.get("test_data_opened") is not False
        or original.get("proposer_access") is not False
    ):
        raise ValueError("recovery must preserve fitting settings and data isolation")
    for relative, digest in original["assets"].items():
        if sha256(safe_path(source, relative)) != digest:
            raise ValueError("source input hash differs: " + relative)
    cases, tasks = original["cases"], original["tasks"]
    if any(c["index"] != i for i, c in enumerate(cases)) or any(
        t["index"] != i for i, t in enumerate(tasks)
    ):
        raise ValueError("source matrix indices differ")
    reference = [c for c in cases if c["reference_skeleton"]]
    public = [c for c in cases if not c["reference_skeleton"]]
    if len(reference) != 1 or not public:
        raise ValueError("expected one reference and existing proposed models")
    reference_index = reference[0]["index"]
    bundle = read_json(source / "reference_input.json")
    _identity(bundle)
    if (
        bundle.get("test_data_opened") is not False
        or bundle.get("proposer_access") is not False
    ):
        raise ValueError("reference export isolation differs")
    candidate, context, truth = reference_skeleton(bundle["spec"])
    if any(
        bundle[k] != value
        for k, value in (
            ("candidate", candidate),
            ("context", context),
            ("truth", truth),
        )
    ):
        raise ValueError("reference export does not match its generating specification")
    template = read_json(source / f"problems/{public[0]['index']:03d}.json")
    reference_input = reference_problem(template, bundle)
    if reference_input != read_json(source / f"problems/{reference_index:03d}.json"):
        raise ValueError("reference public protocol or initial conditions differ")

    assets, retained_results, retained_generations = {}, {}, {}

    def copy(relative, destination=None):
        destination = destination or relative
        data = safe_path(source, relative).read_bytes()
        path = safe_path(output, destination)
        _write_bytes(path, data, immutable=True)
        assets[destination] = sha256(path)
        return destination

    for relative in original["assets"]:
        copy(relative)
    copy("freeze.json", "retained/freeze.json")
    if (source / "submission.json").exists():
        copy("submission.json", "retained/submission.json")
    selected = []
    for case in cases:
        i = case["index"]
        relative = f"generated/{i:03d}/result.json"
        path = safe_path(source, relative)
        record = _checkpoint(path, content_hash([original["identity"], "generate", i]))
        if not case["reference_skeleton"]:
            if record["status"] != "complete":
                raise ValueError("proposed-model generation must already be complete")
            retained_generations[str(i)] = copy(relative, "retained/" + relative)
        else:
            if (
                record.get("error")
                != "ValueError: original generator does not reproduce public data"
            ):
                raise ValueError(
                    "reference recovery requires the original replay-gate failure"
                )
            copy(relative, "retained/" + relative)
            audit = f"generated/{i:03d}/native_audit.json"
            if (source / audit).exists():
                copy(audit, "retained/" + audit)
    for task in tasks:
        i = task["index"]
        directory = source / f"results/task_{i:03d}"
        path = directory / "result.json"
        identity = content_hash([original["identity"], task])
        if task["case_index"] == reference_index:
            if any(
                (directory / n).exists()
                for n in ("fit.json", "fit_started.json", "initializer_started.json")
            ):
                raise ValueError(
                    "reference task has a prior numerical attempt; no fresh budget"
                )
            if path.exists():
                record = _checkpoint(path, identity)
                if record["status"] != "generation_unavailable":
                    raise ValueError(
                        "reference task has a terminal result beyond generation failure"
                    )
                copy(
                    str(path.relative_to(source)),
                    "retained/" + str(path.relative_to(source)),
                )
            selected.append(i)
        else:
            record = _checkpoint(path, identity)
            if record["task"] != task or record["status"] != "complete":
                raise ValueError(
                    "proposed-model fits must be complete and match the matrix"
                )
            relative = str(path.relative_to(source))
            retained_results[str(i)] = copy(relative, "retained/" + relative)
    if len(selected) != 11:
        raise ValueError("expected exactly 11 reference arms")
    frozen = deepcopy(original)
    frozen.update(
        plan=plan.model_dump(mode="json"),
        assets=assets,
        code=code_identity(),
        runtime=runtime,
        recovery_of=original["identity"],
        selected_tasks=selected,
        selected_generations=[reference_index],
        retained_results=retained_results,
        retained_generations=retained_generations,
        recovery_policy="reference-audit-only-no-new-fitting-attempts",
    )
    frozen.pop("identity", None)
    frozen["identity"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return verify(output)
