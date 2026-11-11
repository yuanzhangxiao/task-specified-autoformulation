"""A latent neural ODE driven by the measured inputs: latent states, no specification.

The model is the conventional continuous-time latent-state model of the neural
ODE literature: the measured targets are augmented with latent coordinates
(augmented neural ODEs, Dupont et al. 2019), the latent part of the initial state
is encoded from the initial readings (the latent ODE's encoder, Rubanova et al.
2019, restricted to the first observation so that a rollout is causal), and one
network gives the time derivative of the whole state from the state and the
supplied channels, as in continuous-time neural state-space identification.

    x = [y, h],   dx/dt = F([x, u(t)]) / tau,   h(t0) = E([y(t0), u(t0)])

``y`` are the targets and ``u`` the measured inputs and supplied auxiliaries,
both standardized with training statistics; ``u`` is interpolated linearly
between samples, as every other model's supplied channels are. ``tau`` is a
fixed tenth of the training horizon. The model never reads the task
specification, and no part of it names a physical quantity.

Training (``train``) unrolls fixed-step RK4 through every training trajectory
from its first sample and minimizes the mean squared error of the standardized
targets; validation free-rollout error decides early stopping. ``LatentODE``
replays a saved model in numpy with an adaptive solver, one sample interval at
a time, which is how the model is selected and scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from time import monotonic

import numpy as np
from pydantic import Field
from scipy.integrate import solve_ivp

from autoformalism.schemas.base import StrictSchema

PROTOCOL = "latent-neural-ode-1"


class LatentODEConfig(StrictSchema):
    """Every setting of the method, fixed before any model is trained."""

    latent_dimensions: tuple[int, ...] = Field(default=(2, 4, 8), min_length=1)
    width: int = Field(default=64, ge=4, le=512)
    depth: int = Field(default=2, ge=1, le=4)
    encoder_width: int = Field(default=32, ge=4, le=512)
    final_layer_scale: float = Field(default=0.1, gt=0, le=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    iterations: int = Field(default=2000, ge=1)
    evaluate_every: int = Field(default=25, ge=1)
    gradient_clip: float = Field(default=1.0, gt=0)
    rk4_substeps: int = Field(default=2, ge=1, le=16)
    horizon_fraction_for_tau: float = Field(default=0.1, gt=0, le=1)
    replay_rtol: float = Field(default=1e-8, gt=0)
    replay_atol: float = Field(default=1e-10, gt=0)
    replay_seconds: float = Field(default=45.0, gt=0)


@dataclass(frozen=True)
class Arrays:
    """One split on a shared grid: targets ``(B, N, n_y)``, inputs ``(B, N, n_u)``."""

    time: np.ndarray
    targets: np.ndarray
    inputs: np.ndarray


def channels(context) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Targets, then the supplied channels the dynamics may read, in order."""
    if context.lagged_targets or context.fixed_covariates:
        raise ValueError("the latent ODE reads no lagged targets or fixed covariates")
    return tuple(context.targets), (*context.external_inputs, *context.auxiliaries)


def arrays(split, targets, inputs) -> Arrays:
    """Stack a split's trajectories, which must share one increasing grid."""
    rows = split.trajectories
    time = np.asarray(rows[0].time, dtype=float)
    if any(not np.array_equal(np.asarray(r.time, dtype=float), time) for r in rows):
        raise ValueError("the latent ODE trains on trajectories with one shared grid")
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("the grid must increase")

    def stack(names, group):
        return np.stack(
            [
                np.stack(
                    [np.asarray(getattr(r, group(n))[n], float) for n in names], -1
                )
                if names
                else np.zeros((len(time), 0))
                for r in rows
            ]
        )

    def kind(name):
        return "auxiliaries" if name in rows[0].auxiliaries else "external_inputs"

    return Arrays(time, stack(targets, lambda _: "targets"), stack(inputs, kind))


def standardization(train: Arrays) -> dict:
    """Training means and population standard deviations; a constant keeps 1."""

    def stats(values):
        flat = values.reshape(-1, values.shape[-1])
        sd = flat.std(axis=0)
        return flat.mean(axis=0).tolist(), np.where(sd > 1e-12, sd, 1.0).tolist()

    (ym, ys), (um, us) = stats(train.targets), stats(train.inputs)
    return {"target_mean": ym, "target_sd": ys, "input_mean": um, "input_sd": us}


def nmse(predicted: np.ndarray, observed: np.ndarray, target_sd) -> float:
    """Mean over targets of pooled squared error over training variance."""
    error = ((predicted - observed) / np.asarray(target_sd)) ** 2
    return float(np.mean(error.reshape(-1, error.shape[-1]).mean(axis=0)))


# --- training (PyTorch) --------------------------------------------------------


def _network(n_y, n_u, n_h, config, torch):
    sizes = [n_y + n_h + n_u, *[config.width] * config.depth, n_y + n_h]
    dynamics = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in pairwise(sizes))
    with torch.no_grad():
        dynamics[-1].weight.mul_(config.final_layer_scale)
        dynamics[-1].bias.mul_(config.final_layer_scale)
    encoder = torch.nn.ModuleList(
        [
            torch.nn.Linear(n_y + n_u, config.encoder_width),
            torch.nn.Linear(config.encoder_width, n_h),
        ]
    )
    return torch.nn.ModuleDict({"dynamics": dynamics, "encoder": encoder})


def _derivative(net, x, u, tau, torch):
    z = torch.cat([x, u], -1)
    for layer in net["dynamics"][:-1]:
        z = torch.tanh(layer(z))
    return net["dynamics"][-1](z) / tau


def _initial(net, y0, u0, torch):
    hidden = torch.tanh(net["encoder"][0](torch.cat([y0, u0], -1)))
    return torch.cat([y0, net["encoder"][1](hidden)], -1)


def _rk4_rollout(net, y, u, time, tau, substeps, torch):
    """Targets along the grid from the first sample; inputs linear in between."""
    n_y = y.shape[-1]
    x = _initial(net, y[:, 0], u[:, 0], torch)
    out = [x[:, :n_y]]
    for k in range(len(time) - 1):
        h = float(time[k + 1] - time[k]) / substeps
        for s in range(substeps):

            def at(fraction, k=k):
                return u[:, k] + (u[:, k + 1] - u[:, k]) * fraction

            a = s / substeps
            mid = at(a + 0.5 / substeps)
            k1 = _derivative(net, x, at(a), tau, torch)
            k2 = _derivative(net, x + 0.5 * h * k1, mid, tau, torch)
            k3 = _derivative(net, x + 0.5 * h * k2, mid, tau, torch)
            k4 = _derivative(net, x + h * k3, at(a + 1 / substeps), tau, torch)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x[:, :n_y])
    return torch.stack(out, 1)


def _standardized(data: Arrays, scale: dict, torch):
    y = (data.targets - np.asarray(scale["target_mean"])) / np.asarray(
        scale["target_sd"]
    )
    u = (data.inputs - np.asarray(scale["input_mean"])) / np.asarray(scale["input_sd"])
    return (
        torch.tensor(y, dtype=torch.float64),
        torch.tensor(u, dtype=torch.float64),
    )


def _export(net) -> dict:
    return {
        part: [
            {
                "weight": layer.weight.detach().tolist(),
                "bias": layer.bias.detach().tolist(),
            }
            for layer in net[part]
        ]
        for part in ("dynamics", "encoder")
    }


def train(
    train_data: Arrays,
    validation: Arrays,
    n_h: int,
    seed: int,
    config: LatentODEConfig,
) -> dict:
    """Fit one latent dimension and seed; keep the best validation checkpoint."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    scale = standardization(train_data)
    tau = config.horizon_fraction_for_tau * float(
        train_data.time[-1] - train_data.time[0]
    )
    n_y, n_u = train_data.targets.shape[-1], train_data.inputs.shape[-1]
    net = _network(n_y, n_u, n_h, config, torch).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    y, u = _standardized(train_data, scale, torch)
    vy, vu = _standardized(validation, scale, torch)
    best = {"validation_nmse_rk4": math.inf, "iteration": None, "network": None}
    history = []
    start = monotonic()
    for iteration in range(1, config.iterations + 1):
        optimizer.zero_grad()
        predicted = _rk4_rollout(
            net, y, u, train_data.time, tau, config.rk4_substeps, torch
        )
        loss = torch.mean((predicted[:, 1:] - y[:, 1:]) ** 2)
        if not torch.isfinite(loss):
            history.append({"iteration": iteration, "training_loss": None})
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip)
        optimizer.step()
        if iteration % config.evaluate_every == 0 or iteration == config.iterations:
            with torch.no_grad():
                held = _rk4_rollout(
                    net, vy, vu, validation.time, tau, config.rk4_substeps, torch
                )
                score = float(torch.mean((held - vy) ** 2, dim=(0, 1)).mean())
            history.append(
                {
                    "iteration": iteration,
                    "training_loss": loss.item(),
                    "validation_nmse_rk4": score,
                }
            )
            if math.isfinite(score) and score < best["validation_nmse_rk4"]:
                best = {
                    "validation_nmse_rk4": score,
                    "iteration": iteration,
                    "network": _export(net),
                }
    return {
        "protocol": PROTOCOL,
        "latent_dimension": n_h,
        "seed": seed,
        "tau": tau,
        "standardization": scale,
        "best": best,
        "history": history,
        "seconds": monotonic() - start,
        "torch": torch.__version__,
    }


# --- replay (numpy) ------------------------------------------------------------


class LatentODE:
    """A saved latent ODE, replayed with an adaptive solver between samples."""

    def __init__(self, model: dict):
        self.targets = tuple(model["targets"])
        self.inputs = tuple(model["inputs"])
        self.tau = float(model["tau"])
        scale = model["standardization"]
        self.ym, self.ys = (np.asarray(scale[k]) for k in ("target_mean", "target_sd"))
        self.um, self.us = (np.asarray(scale[k]) for k in ("input_mean", "input_sd"))
        net = model["network"]
        self.layers = [
            (np.asarray(p["weight"]), np.asarray(p["bias"])) for p in net["dynamics"]
        ]
        self.encoder = [
            (np.asarray(p["weight"]), np.asarray(p["bias"])) for p in net["encoder"]
        ]
        self.n_y = len(self.targets)
        self.n_h = self.encoder[-1][0].shape[0]
        if self.layers[0][0].shape[1] != self.n_y + self.n_h + len(self.inputs):
            raise ValueError("saved network does not match its channels")

    def derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        z = np.concatenate([x, u])
        for weight, bias in self.layers[:-1]:
            z = np.tanh(weight @ z + bias)
        weight, bias = self.layers[-1]
        return (weight @ z + bias) / self.tau

    def initial(self, y0: np.ndarray, u0: np.ndarray) -> np.ndarray:
        (w1, b1), (w2, b2) = self.encoder
        y = (y0 - self.ym) / self.ys
        hidden = np.tanh(w1 @ np.concatenate([y, (u0 - self.um) / self.us]) + b1)
        return np.concatenate([y, w2 @ hidden + b2])

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
                    raise TimeoutError("latent ODE replay wall-clock limit reached")
                return self.derivative(state, ua + (ub - ua) * (t - t0) / (t1 - t0))

            sol = solve_ivp(rhs, (t0, t1), x, method=method, rtol=rtol, atol=atol)
            if not sol.success or not np.isfinite(sol.y[:, -1]).all():
                raise ValueError(f"latent ODE replay failed: {sol.message}")
            x = sol.y[:, -1]
            states.append(x)
        states = np.asarray(states)
        return states[:, : self.n_y] * self.ys + self.ym, states[:, self.n_y :]
