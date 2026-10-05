#!/usr/bin/env python3
"""Measure how fast this VM's model server generates for the baselines' requests.

LLM-SR asks /v1/chat/completions for four programs of up to 4096 tokens per
request (the limit its Phase C plans declare for gpt-oss, which reasons before
it answers); LLM-ODE asks /v1/responses for up to 1024 tokens. This sends
requests of those two shapes from several concurrent clients and reports
generated tokens per second, then projects the GPU hours and SUs one model's
Phase C matrix needs at that rate. The prompts are synthetic, of about the
length the methods send; no benchmark data are read.

The projection counts generation only. LLM-SR and LLM-ODE also spend CPU time
scoring each candidate, during which the server idles for that task, so treat
the projected hours as a floor.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

#: One model's Phase C matrix: 22 LLM-SR searches of 10,000 samples at four
#: samples per request, and LLM-ODE's 200 iterations x 4 islands per target.
WORKLOAD_REQUESTS = {"llm_sr": 22 * 10_000 // 4, "llm_ode": 17_600}

#: LLM-SR's generation limit per sample in its Phase C plans
#: (configs/phase_c_llm_sr_budget_pilot_v2.json, reasoning_model_adaptation).
SR_MAX_TOKENS = 4096

SR_PROMPT = '''"""
Find the mathematical function skeleton that represents the acceleration of a
damped, driven oscillator, given its position, its velocity and the measured
driving force. The friction may be linear or nonlinear in the velocity, and the
restoring force may include higher-order terms in the position.
"""

import numpy as np

#Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS


def equation_v0(
    x: np.ndarray, v: np.ndarray, F: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """ Mathematical function for acceleration in a damped, driven oscillator

    Args:
        x: A numpy array representing observations of the current position.
        v: A numpy array representing observations of the velocity.
        F: A numpy array representing observations of the driving force.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the
        mathematical function to the inputs.
    """
    dv = params[0] * x + params[1] * v + params[2] * F + params[3]
    return dv


def equation_v1(
    x: np.ndarray, v: np.ndarray, F: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """ Improved version of `equation_v0`. """
    restoring = params[0] * x + params[1] * x ** 3
    friction = params[2] * v + params[3] * np.abs(v) * v
    forcing = params[4] * F
    return restoring + friction + forcing + params[5]


def equation_v2(
    x: np.ndarray, v: np.ndarray, F: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """ Improved version of `equation_v1`. """
'''

ODE_PROMPT = (
    "You are helping to discover a system of ordinary differential equations "
    "from data. The state variables are x0, x1 and x2; the measured input is u. "
    "Earlier candidates for dx0/dt, with their validation errors, were:\n"
    + "".join(
        f"  {i}. -c0*x0 + c1*x1*x2 + c2*u**{i % 3} - c3*x0**2"
        f"  (error {0.4 / (i + 1):.3f})\n"
        for i in range(12)
    )
    + "Propose five new candidate expressions for dx0/dt that may fit better. "
    "Use only x0, x1, x2, u, constants c0..c9, + - * / ** and exp, log, sin, "
    "cos. Explain each briefly, then list them one per line."
)


def _shape(name: str, model: str) -> tuple[str, dict, Callable[[dict], int]]:
    """The endpoint, body and generated-token count for one request shape."""
    if name == "llm_sr":
        body = {
            "model": model,
            "messages": [{"role": "user", "content": SR_PROMPT}],
            "n": 4,
            "max_tokens": SR_MAX_TOKENS,
        }
        return (
            "/v1/chat/completions",
            body,
            lambda usage: int(usage["completion_tokens"]),
        )
    if name == "llm_ode":
        body = {"model": model, "input": ODE_PROMPT, "max_output_tokens": 1024}
        return "/v1/responses", body, lambda usage: int(usage["output_tokens"])
    raise ValueError(f"unknown request shape {name!r}")


def _post(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def measure(
    base_url: str, model: str, shape: str, concurrency: int, seconds: float
) -> dict:
    """Keep `concurrency` requests in flight for `seconds`; summarize them."""
    path, body, generated = _shape(shape, model)
    deadline = time.monotonic() + seconds
    records: list[tuple[float, int]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def client() -> None:
        while time.monotonic() < deadline:
            started = time.monotonic()
            try:
                answer = _post(base_url.rstrip("/") + path, body, timeout=600)
                tokens = generated(answer["usage"])
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
                time.sleep(1.0)
                continue
            with lock:
                records.append((time.monotonic() - started, tokens))

    started = time.monotonic()
    threads = [threading.Thread(target=client) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return summarize(shape, concurrency, records, time.monotonic() - started, errors)


def summarize(
    shape: str,
    concurrency: int,
    records: list[tuple[float, int]],
    elapsed: float,
    errors: list[str],
) -> dict:
    """Generated tokens per second, and what one request cost."""
    tokens = sum(count for _, count in records)
    requests = len(records)
    return {
        "shape": shape,
        "concurrency": concurrency,
        "requests": requests,
        "errors": len(errors),
        "first_error": errors[0] if errors else None,
        "seconds": round(elapsed, 1),
        "generated_tokens": tokens,
        "tokens_per_second": round(tokens / elapsed, 1) if elapsed > 0 else 0.0,
        "tokens_per_request": round(tokens / requests, 1) if requests else None,
        "mean_latency_seconds": (
            round(sum(latency for latency, _ in records) / requests, 2)
            if requests
            else None
        ),
    }


def project(results: list[dict], su_per_hour: float) -> dict:
    """GPU hours and SUs for one model's matrix, at the best measured rate."""
    projection: dict[str, dict] = {}
    for shape, requests in WORKLOAD_REQUESTS.items():
        measured = [
            item
            for item in results
            if item["shape"] == shape and item["requests"] and item["tokens_per_second"]
        ]
        if not measured:
            continue
        best = max(measured, key=lambda item: item["tokens_per_second"])
        tokens = requests * best["tokens_per_request"]
        hours = tokens / best["tokens_per_second"] / 3600
        projection[shape] = {
            "at_concurrency": best["concurrency"],
            "generated_tokens": round(tokens),
            "generation_hours": round(hours, 1),
            "su": round(hours * su_per_hour),
        }
    return projection


def main() -> None:
    """Measure each shape at each concurrency, then project the matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--shapes", nargs="+", default=["llm_sr", "llm_ode"])
    parser.add_argument("--su-per-hour", type=float, required=True)
    args = parser.parse_args()
    results = []
    for shape in args.shapes:
        for concurrency in args.concurrency:
            result = measure(
                args.base_url, args.model, shape, concurrency, args.seconds
            )
            print(json.dumps(result), flush=True)
            results.append(result)
    print(
        json.dumps(
            {
                "model": args.model,
                "su_per_hour": args.su_per_hour,
                "projection": project(results, args.su_per_hour),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
