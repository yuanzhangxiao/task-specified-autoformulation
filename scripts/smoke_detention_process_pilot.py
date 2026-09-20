#!/usr/bin/env python3
"""Prescribed scientific replies -> real constructor -> frozen fit and resume.

Temporary public synthetic fixture only; no benchmark labels or LLM endpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import detention_process_pilot as pilot
from autoformalism.search.process_review import POLICY
from autoformalism.staged_topology import content_hash


def fixture(source: Path, root: Path):
    """Build a tiny separate fixture in temporary storage, with private poison."""
    files = {}
    for case in ("coupled", "independent"):
        (source / "public" / case / "noise1").mkdir(parents=True, exist_ok=True)
    for case in ("coupled", "independent"):
        for split in ("train", "val"):
            rows = []
            for i in range(2):
                time = np.linspace(0, 4, 17)
                upstream, local = 0.4 + i * 0.2, 0.1 + i * 0.05
                sol = solve_ivp(
                    lambda t, x, upstream=upstream, local=local, case=case: [
                        upstream - 0.8 * x[0],
                        (local + (0.8 * x[0] if case == "coupled" else 0) - 0.5 * x[1])
                        / 2,
                    ],
                    (0, 4),
                    [0.6, 0.3],
                    t_eval=time,
                    rtol=1e-10,
                    atol=1e-12,
                )
                rows.append(
                    {
                        "trajectory_id": f"{split}_{i}",
                        "time": time.tolist(),
                        "targets": {"h_down": sol.y[1].tolist()},
                        "external_inputs": {
                            "inflow_up": [upstream] * len(time),
                            "inflow_down": [local] * len(time),
                        },
                        "fixed_covariates": {
                            "area_up": 1.0,
                            "area_down": 2.0,
                            "crest_up": 0.0,
                            "crest_down": 0.0,
                            "initial_up": 0.6,
                            "warning_depth": 1.0,
                        },
                    }
                )
            value = {"name": split, "fingerprint": content_hash(rows), "rows": rows}
            relative = f"public/{case}/noise1/{split}.json"
            public._write(
                source / relative, {"value": value, "sha256": content_hash(value)}
            )
            files[relative] = hashlib.sha256(
                (source / relative).read_bytes()
            ).hexdigest()

        spec = {
            "public_prompt": (
                pilot.REPO / f"configs/detention_prompts/{case}.md"
            ).read_text(),
            "targets": ["h_down"],
            "external_inputs": ["inflow_up", "inflow_down"],
            "fixed_covariates": list(rows[0]["fixed_covariates"]),
        }
        relative = f"public/{case}/noise1/specification.json"
        public._write(source / relative, {"value": spec, "sha256": content_hash(spec)})
        files[relative] = hashlib.sha256((source / relative).read_bytes()).hexdigest()
    plan = {
        "protocol": "detention-development-1",
        "generation_gate_passed": True,
        "files": files,
        "diagnostic": "PRIVATE_ORACLE_POISON",
    }
    public._write(source / "plan.json", {"value": plan, "sha256": content_hash(plan)})
    (source / "diagnostic").mkdir(exist_ok=True)
    (source / "diagnostic" / "do-not-read.txt").write_text("PRIVATE_ORACLE_POISON")
    return pilot.freeze(
        source, root, pilot.REPO / "configs/detention_process_pilot_v1.json"
    )


def transport_for(calls, mode="add"):
    """Only the optional stage proposes q; fallback returns viable inline laws."""

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        payload = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        calls.append({"body": body, "payload": payload})
        if payload.get("protocol") == POLICY:
            reply = {
                "processes": []
                if mode in {"empty", "independent"}
                else [
                    {
                        "name": "q",
                        "scientific_role": "shared volumetric transfer",
                        "drivers": ["h_up"],
                        "consumers": ["h_up", "h_down"],
                    }
                ]
            }
        elif "selected_state" in payload:
            reply = {
                "initial": {
                    "mode": "causal_map",
                    "expression": "initial_up",
                    "parameters": [],
                }
            }
        elif "selected_equation" in payload:
            lhs = payload["selected_equation"]["lhs"]
            q = any(v.get("name") == "q" for v in payload["frozen_inventory"])

            def f(expression, name=None):
                return {
                    "expression": expression,
                    "parameters": (
                        [{"name": name, "role": "coefficient"}] if name else []
                    ),
                }

            funcs = {
                "q": [f("k*h_up", "k")],
                "h_up": [
                    f("inflow_up/area_up"),
                    f("q/area_up") if q else f("k*h_up/area_up", "k"),
                ],
                "h_down": [
                    f("inflow_down/area_down"),
                    f("q/area_down") if q else f("k*h_up/area_down", "k"),
                    f("d*h_down/area_down", "d"),
                ],
            }
            if mode == "independent":
                funcs["h_down"].pop(1)
            reply = {"functions": funcs[lhs]}
            if mode == "bad_function" and q:
                reply = {"functions": [f("missing")]}
        elif "selected_lhs" in payload:
            lhs = payload["selected_lhs"]["name"]
            q = any(v["name"] == "q" for v in payload["frozen_inventory"])
            rate_sources = ["q"] if q else ["h_up"]
            terms = {
                "q": [(["h_up"], "positive")],
                "h_up": [
                    (["inflow_up", "area_up"], "positive"),
                    ([*rate_sources, "area_up"], "negative"),
                ],
                "h_down": [
                    (["inflow_down", "area_down"], "positive"),
                    ([*rate_sources, "area_down"], "positive"),
                    (["h_down", "area_down"], "negative"),
                ],
            }
            if mode == "independent":
                terms["h_down"].pop(1)
            reply = {
                "terms": [
                    {
                        "sources": sources,
                        "outer_weight_sign": sign,
                        "scientific_role": "area converted water balance",
                    }
                    for sources, sign in terms[lhs]
                ],
                "inventory_revision": None,
            }
            if mode == "bad_topology" and q:
                reply = {
                    "terms": [
                        {
                            "sources": ["missing"],
                            "outer_weight_sign": "positive",
                            "scientific_role": "bad",
                        }
                    ],
                    "inventory_revision": None,
                }
        else:
            reply = {
                "variables": [
                    {
                        "name": n,
                        "definition": "differential",
                        "scientific_role": "physical depth storage",
                    }
                    for n in (
                        ("h_down",) if mode == "independent" else ("h_down", "h_up")
                    )
                ]
            }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {
                "total_tokens": 100,
                "prompt_tokens": 70,
                "completion_tokens": 30,
            },
        }

    return transport


def run(root):
    plan = fixture(root / "source", root / "pilot")
    output = root / "pilot"
    task = next(
        t
        for t in plan["tasks"]
        if t["case"] == "coupled" and t["review"] and t["arm"] == "full"
    )
    calls = []
    proposal = pilot.construct(
        output,
        plan,
        task,
        pilot.make_client(
            output, plan, task, "offline", transport=transport_for(calls)
        ),
    )
    assert proposal["status"] == "constructed", proposal
    result = pilot.fit_task(output, task["index"])
    assert result["fit"]["training"]["available"], result
    assert result["fit"]["validation"]["normalized_mse"] < 1e-4, result
    assert result["replay"]["replay_agreement"], result
    assert pilot.fit_task(output, task["index"]) == result
    count = len(calls)
    assert (
        pilot.construct(
            output,
            plan,
            task,
            pilot.make_client(
                output, plan, task, "offline", transport=transport_for(calls)
            ),
        )
        == proposal
    )
    assert count == len(calls)
    pilot.report(output)
    return {
        "status": "passed",
        "calls": count,
        "validation": result["fit"]["validation"],
        "resume_unchanged": True,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
