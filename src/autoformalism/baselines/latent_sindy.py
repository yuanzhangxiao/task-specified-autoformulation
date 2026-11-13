"""Latent-state SINDy: sparse polynomial dynamics with unobserved states.

SINDy (Brunton, Proctor and Kutz, 2016) writes each state's time derivative as
a sparse combination of library functions and makes it sparse by sequential
thresholding. Unobserved states have no derivative data, so, as in the
trajectory-matching approaches to SINDy with hidden variables, the coefficients
are fitted to the observations through the model's own rollouts. The targets
are augmented with ``k`` latent coordinates:

    x = [y, h],   dx/dt = Xi^T theta([x, u(t)]) / tau,   h(t0) = A [y(t0), u(t0), 1]

``theta`` is the degree-2 polynomial library of the scaled state and supplied
channels: the constant, every coordinate, and every product of two. The targets
are standardized with training statistics; the measured inputs and supplied
auxiliaries ``u``, linear between samples, are centred on their training means
and divided by their largest training deviation, so that a short pulse does not
enter a squared term at many times its typical size. ``tau`` is a fixed tenth
of the training horizon. The initializer ``A`` is causal: it reads only the first
readings and inputs, as a forecast receives them, and is fitted on training
trajectories only. Every latent state evolves by its own sparse equation.

Fitting (``train``): Adam on the mean squared error of the standardized targets
along RK4 rollouts of every training trajectory from its first sample, with an
L1 penalty on ``Xi``; then, for each threshold, the coefficients whose magnitude
is below it are fixed at zero and the rest refitted without the penalty. The
validation free-rollout error stops each phase early; per seed it then chooses
the latent dimension and the threshold. No part of the method reads the task
specification. ``LatentSINDy`` replays a saved model in numpy with an adaptive
solver, one sample interval at a time, which is how it is selected and scored.
"""

from __future__ import annotations

import math
from itertools import combinations_with_replacement
from time import monotonic

import numpy as np
from pydantic import Field
from scipy.integrate import solve_ivp

from autoformalism.baselines.latent_ode import Arrays, standardization
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "latent-sindy-1"


class LatentSINDyConfig(StrictSchema):
    """Every setting of the method, fixed before any model is trained."""

    latent_dimensions: tuple[int, ...] = Field(default=(1, 2, 3), min_length=1)
    thresholds: tuple[float, ...] = Field(default=(0.01, 0.03, 0.1), min_length=1)
    l1: float = Field(default=1e-3, ge=0)
    learning_rate: float = Field(default=1e-2, gt=0)
    iterations: int = Field(default=2000, ge=1)
    refit_iterations: int = Field(default=500, ge=1)
    evaluate_every: int = Field(default=25, ge=1)
    gradient_clip: float = Field(default=1.0, gt=0)
    initial_scale: float = Field(default=1e-2, gt=0)
    restarts_after_divergence: int = Field(default=3, ge=0)
    rk4_substeps: int = Field(default=2, ge=1, le=16)
    horizon_fraction_for_tau: float = Field(default=0.1, gt=0, le=1)
    replay_rtol: float = Field(default=1e-8, gt=0)
    replay_atol: float = Field(default=1e-10, gt=0)
    replay_seconds: float = Field(default=45.0, gt=0)


def library_terms(n: int) -> tuple[tuple[int, ...], ...]:
    """The degree-2 library: (), (i,), then (i, j) with i <= j, in that order."""
    return ((), *((i,) for i in range(n)), *combinations_with_replacement(range(n), 2))


def _pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(combinations_with_replacement(range(n), 2))
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def theta(z: np.ndarray) -> np.ndarray:
    """Library features of ``z`` (``(..., n)``), in ``library_terms`` order."""
    i, j = _pairs(z.shape[-1])
    ones = np.ones((*z.shape[:-1], 1))
    return np.concatenate([ones, z, z[..., i] * z[..., j]], -1)


# --- training (PyTorch) --------------------------------------------------------


def _theta_torch(z, i, j, torch):
    ones = torch.ones((*z.shape[:-1], 1), dtype=z.dtype)
    return torch.cat([ones, z, z[..., i] * z[..., j]], -1)


def _rollout(params, y, u, time, tau, substeps, pairs, torch):
    """Targets along the grid from the first sample; inputs linear in between."""
    coefficients, initializer, mask = params
    effective = coefficients * mask
    n_y = y.shape[-1]
    i, j = pairs

    def derivative(x, w):
        return _theta_torch(torch.cat([x, w], -1), i, j, torch) @ effective / tau

    ones = torch.ones((y.shape[0], 1), dtype=y.dtype)
    x = torch.cat(
        [y[:, 0], torch.cat([y[:, 0], u[:, 0], ones], -1) @ initializer.T], -1
    )
    out = [x[:, :n_y]]
    for k in range(len(time) - 1):
        h = float(time[k + 1] - time[k]) / substeps
        for s in range(substeps):

            def at(fraction, k=k):
                return u[:, k] + (u[:, k + 1] - u[:, k]) * fraction

            a = s / substeps
            mid = at(a + 0.5 / substeps)
            k1 = derivative(x, at(a))
            k2 = derivative(x + 0.5 * h * k1, mid)
            k3 = derivative(x + 0.5 * h * k2, mid)
            k4 = derivative(x + h * k3, at(a + 1 / substeps))
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x[:, :n_y])
    return torch.stack(out, 1)


def scaling(train: Arrays) -> dict:
    """Target means and SDs; input means and largest deviations (a constant keeps 1)."""
    scale = standardization(train)
    flat = train.inputs.reshape(-1, train.inputs.shape[-1])
    deviation = np.abs(flat - flat.mean(axis=0)).max(axis=0)
    scale["input_sd"] = np.where(deviation > 1e-12, deviation, 1.0).tolist()
    scale["input_scaling"] = "largest absolute deviation from the training mean"
    return scale


def _standardized(data: Arrays, scale: dict, torch):
    y = (data.targets - np.asarray(scale["target_mean"])) / np.asarray(
        scale["target_sd"]
    )
    u = (data.inputs - np.asarray(scale["input_mean"])) / np.asarray(scale["input_sd"])
    return (
        torch.tensor(y, dtype=torch.float64),
        torch.tensor(u, dtype=torch.float64),
    )


def _phase(
    params, train_t, valid_t, time, tau, *, iterations, l1, config, pairs, torch
) -> dict:
    """One fitting phase; keep the checkpoint with the best validation rollout."""
    coefficients, initializer, mask = params
    (y, u), (vy, vu), (t_train, t_valid) = train_t, valid_t, time
    learning_rate = config.learning_rate
    optimizer = torch.optim.Adam([coefficients, initializer], lr=learning_rate)
    best = {"validation_nmse_rk4": math.inf, "iteration": None, "snapshot": None}
    restarts, history = 0, []
    last = (coefficients.detach().clone(), initializer.detach().clone())
    iteration = 0
    while iteration < iterations:
        iteration += 1
        optimizer.zero_grad()
        predicted = _rollout(
            params, y, u, t_train, tau, config.rk4_substeps, pairs, torch
        )
        mse = torch.mean((predicted[:, 1:] - y[:, 1:]) ** 2)
        loss = mse + l1 * torch.mean(torch.abs(coefficients * mask))
        if not torch.isfinite(loss):
            # A polynomial rollout can diverge: return to the last parameters
            # whose rollout was finite with a tenth of the step, a bounded
            # number of times.
            if restarts >= config.restarts_after_divergence:
                history.append({"iteration": iteration, "diverged": True})
                break
            restarts += 1
            with torch.no_grad():
                coefficients.copy_(last[0])
                initializer.copy_(last[1])
            learning_rate /= 10
            optimizer = torch.optim.Adam([coefficients, initializer], lr=learning_rate)
            history.append(
                {"iteration": iteration, "restart_learning_rate": learning_rate}
            )
            continue
        last = (coefficients.detach().clone(), initializer.detach().clone())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [coefficients, initializer], config.gradient_clip
        )
        optimizer.step()
        with torch.no_grad():
            coefficients.mul_(mask)
        if iteration % config.evaluate_every == 0 or iteration == iterations:
            with torch.no_grad():
                held = _rollout(
                    params, vy, vu, t_valid, tau, config.rk4_substeps, pairs, torch
                )
                score = float(torch.mean((held - vy) ** 2, dim=(0, 1)).mean())
            history.append(
                {
                    "iteration": iteration,
                    "training_mse": mse.item(),
                    "validation": score,
                }
            )
            if math.isfinite(score) and score < best["validation_nmse_rk4"]:
                best = {
                    "validation_nmse_rk4": score,
                    "iteration": iteration,
                    "snapshot": {
                        "coefficients": (coefficients * mask).detach().tolist(),
                        "initializer": initializer.detach().tolist(),
                        "mask": mask.detach().tolist(),
                    },
                }
    return {**best, "history": history, "restarts": restarts}


def train(
    train_data: Arrays,
    validation: Arrays,
    n_h: int,
    seed: int,
    config: LatentSINDyConfig,
) -> dict:
    """Fit one latent dimension and seed: a penalized phase, then each threshold."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    scale = scaling(train_data)
    tau = config.horizon_fraction_for_tau * float(
        train_data.time[-1] - train_data.time[0]
    )
    n_y, n_u = train_data.targets.shape[-1], train_data.inputs.shape[-1]
    n = n_y + n_h + n_u
    terms = library_terms(n)
    pairs = tuple(torch.tensor(a) for a in _pairs(n))
    coefficients = torch.nn.Parameter(
        torch.randn(len(terms), n_y + n_h, dtype=torch.float64) * config.initial_scale
    )
    initializer = torch.nn.Parameter(
        torch.randn(n_h, n_y + n_u + 1, dtype=torch.float64) * config.initial_scale
    )
    mask = torch.ones_like(coefficients)
    data = (
        _standardized(train_data, scale, torch),
        _standardized(validation, scale, torch),
        (train_data.time, validation.time),
    )
    start = monotonic()
    dense = _phase(
        (coefficients, initializer, mask),
        *data,
        tau,
        iterations=config.iterations,
        l1=config.l1,
        config=config,
        pairs=pairs,
        torch=torch,
    )
    sparse = []
    for threshold in config.thresholds:
        if dense["snapshot"] is None:
            sparse.append({"threshold": threshold, "validation_nmse_rk4": math.inf})
            continue
        with torch.no_grad():
            start_values = torch.tensor(
                dense["snapshot"]["coefficients"], dtype=torch.float64
            )
            keep = (start_values.abs() >= threshold).to(torch.float64)
            coefficients.copy_(start_values * keep)
            initializer.copy_(
                torch.tensor(dense["snapshot"]["initializer"], dtype=torch.float64)
            )
        refit = _phase(
            (coefficients, initializer, keep),
            *data,
            tau,
            iterations=config.refit_iterations,
            l1=0.0,
            config=config,
            pairs=pairs,
            torch=torch,
        )
        sparse.append(
            {
                "threshold": threshold,
                "kept_terms": int(keep.sum().item()),
                "validation_nmse_rk4": refit["validation_nmse_rk4"],
                "iteration": refit["iteration"],
                "restarts": refit["restarts"],
                "snapshot": refit["snapshot"],
            }
        )
    return {
        "protocol": PROTOCOL,
        "latent_dimension": n_h,
        "seed": seed,
        "tau": tau,
        "standardization": scale,
        "library": [list(term) for term in terms],
        "dense": {
            k: dense[k] for k in ("validation_nmse_rk4", "iteration", "restarts")
        },
        "dense_history": dense["history"],
        "thresholds": sparse,
        "seconds": monotonic() - start,
        "torch": torch.__version__,
    }


# --- replay (numpy) ------------------------------------------------------------


class LatentSINDy:
    """A saved latent-state SINDy model, replayed adaptively between samples."""

    def __init__(self, model: dict):
        self.targets = tuple(model["targets"])
        self.inputs = tuple(model["inputs"])
        self.tau = float(model["tau"])
        scale = model["standardization"]
        self.ym, self.ys = (np.asarray(scale[k]) for k in ("target_mean", "target_sd"))
        self.um, self.us = (np.asarray(scale[k]) for k in ("input_mean", "input_sd"))
        self.coefficients = np.asarray(model["coefficients"], dtype=float) * np.asarray(
            model["mask"], dtype=float
        )
        self.initializer = np.asarray(model["initializer"], dtype=float)
        self.n_y = len(self.targets)
        self.n_h = self.initializer.shape[0]
        n = self.n_y + self.n_h + len(self.inputs)
        if self.coefficients.shape != (len(library_terms(n)), self.n_y + self.n_h):
            raise ValueError("saved coefficients do not match the library")

    def derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return theta(np.concatenate([x, u])) @ self.coefficients / self.tau

    def initial(self, y0: np.ndarray, u0: np.ndarray) -> np.ndarray:
        y = (y0 - self.ym) / self.ys
        u = (u0 - self.um) / self.us
        return np.concatenate([y, self.initializer @ np.concatenate([y, u, [1.0]])])

    def simulate(
        self,
        time: np.ndarray,
        y0: np.ndarray,
        inputs: np.ndarray,
        *,
        method: str = "Radau",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        seconds: float = 45.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Targets ``(N, n_y)`` in their units and latent coordinates ``(N, n_h)``."""
        deadline = monotonic() + seconds
        time = np.asarray(time, dtype=float)
        u = (np.asarray(inputs, dtype=float) - self.um) / self.us
        x = self.initial(
            np.asarray(y0, dtype=float), np.asarray(inputs[0], dtype=float)
        )
        states = [x]
        for k in range(len(time) - 1):
            t0, t1, ua, ub = time[k], time[k + 1], u[k], u[k + 1]

            def rhs(t, state, t0=t0, t1=t1, ua=ua, ub=ub):
                if monotonic() > deadline:
                    raise TimeoutError("latent SINDy replay wall-clock limit reached")
                return self.derivative(state, ua + (ub - ua) * (t - t0) / (t1 - t0))

            sol = solve_ivp(rhs, (t0, t1), x, method=method, rtol=rtol, atol=atol)
            if not sol.success or not np.isfinite(sol.y[:, -1]).all():
                raise ValueError(f"latent SINDy replay failed: {sol.message}")
            x = sol.y[:, -1]
            states.append(x)
        states = np.asarray(states)
        return states[:, : self.n_y] * self.ys + self.ym, states[:, self.n_y :]

    def equations(self) -> dict[str, str]:
        """Each state's equation in standardized coordinates, nonzero terms only."""
        names = (*self.targets, *(f"h{j + 1}" for j in range(self.n_h)), *self.inputs)
        labels = []
        for term in library_terms(len(names)):
            labels.append("1" if not term else "*".join(names[i] for i in term))
        out = {}
        for column, state in enumerate(names[: self.n_y + self.n_h]):
            parts = [
                f"{c:+.6g}*{label}"
                for c, label in zip(self.coefficients[:, column], labels, strict=True)
                if c != 0.0
            ]
            body = " ".join(parts) if parts else "0"
            out[state] = f"d{state}/dt = ({body}) / {self.tau:g}"
        return out
