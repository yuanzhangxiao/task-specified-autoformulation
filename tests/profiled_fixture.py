"""Small independent terminal-output ODE for M14 tests; not benchmark data."""

from copy import deepcopy
from itertools import pairwise

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.coordinates import training_coordinates
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

TRUTH = {"a": 0.7, "b": 0.3, "c": 1.8, "d": 0.4, "init_z_value": 0.4}


def make_base():
    request = controls.request("nonlinear", 0).model_dump(mode="json")
    request["base_candidate"]["state_equations"] = [
        {"state": "y", "rhs": "-b*y+c*tanh(z)-d*u"},
        {"state": "z", "rhs": "-a*z+u"},
    ]
    request["parameter_guesses"] = {"a": 1.1, "b": 0.8, "c": 1.1, "d": 0.2}
    request["initialization_plan"]["rules"]["z"]["initial"]["guess"] = -0.1
    request = PublicFitRequest.model_validate(request)
    splits = {}
    for split in ("train", "val"):
        rows = []
        for j in range(2):
            time = np.linspace(0, 6, 61)
            knots = np.array([0, 0.4, 0.5, 0.6, 1.5, 2, 3, 4.5, 6])
            command = np.array([0, 0, 2, 0, 1, -0.5, 0.8, -1, 0]) * (1 + 0.2 * j)
            if split == "val":
                command = -command * 0.7
            forcing = np.interp(time, knots, command)
            y0 = 0.2 - 0.6 * j

            def rhs(t, x, time=time, forcing=forcing):
                y, z = x
                u = np.interp(t, time, forcing)
                return [
                    -TRUTH["b"] * y + TRUTH["c"] * np.tanh(z) - TRUTH["d"] * u,
                    -TRUTH["a"] * z + u,
                ]

            current = np.array([y0, TRUTH["init_z_value"]])
            states = [current.copy()]
            for left, right in pairwise(time):
                sol = solve_ivp(
                    rhs, (left, right), current, method="DOP853", rtol=1e-11, atol=1e-13
                )
                assert sol.success
                current = sol.y[:, -1]
                states.append(current.copy())
            rows.append(
                {
                    "trajectory_id": f"{split}_{j}",
                    "time": time.tolist(),
                    "targets": {"y": np.array(states)[:, 0].tolist()},
                    "external_inputs": {"u": forcing.tolist()},
                    "auxiliaries": {},
                }
            )
        splits[split] = {
            "name": split,
            "rows": rows,
            "fingerprint": public.content_sha256(rows),
        }
    model, start, _ = public._lower(request)
    training = public.unpack_split(PublicSplit.model_validate(splits["train"]))
    base = {
        "request": request.model_dump(mode="json"),
        "training": splits["train"],
        "start": start,
        "nodes": {},
        "coordinates": training_coordinates(model, training, start).model_dump(
            mode="json"
        ),
    }
    return deepcopy(base), splits["val"]
