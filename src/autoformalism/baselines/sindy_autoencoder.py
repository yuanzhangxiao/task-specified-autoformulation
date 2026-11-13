"""SINDy autoencoder on time-delay coordinates, driven by the measured inputs.

The SINDy autoencoder (Champion, Lusch, Kutz and Brunton, PNAS 2019) learns at
once an encoder ``phi`` from the measurements to a few latent coordinates, a
decoder ``psi`` back to the measurements, and a sparse polynomial model of the
latent coordinates' time derivative. When a single quantity is measured, the
deep delay autoencoder (Bakarji, Champion, Kutz and Brunton, Proc. R. Soc. A
2023) feeds the encoder time-delay coordinates of it, asks the first latent
coordinate to equal the measurement, and integrates the SINDy model across the
delay window for a consistency loss. The reactor is driven by measured inputs,
so the library also holds them, as in SINDy with control (Brunton, Proctor and
Kutz, IFAC 2016):

    h_k = [y_k, y_(k-1), ..., y_(k-n+1)],   z_k = phi(h_k),   z_k1 ~ y_k,
    dz/dt = Xi^T theta([z, u(t)])

``y`` is the measured target, standardized with training statistics; ``u`` are
the measured inputs and supplied auxiliaries, linear between samples, centred on
their training means and divided by their largest training deviation, as
latent-state SINDy scales them; ``theta`` is the same degree-2 polynomial
library. The delay window is causal: it ends at the current sample, and samples
before a record's first are filled with the first reading, the reactor at rest
before the record starts. A forecast therefore reads only the first reading and
the inputs, as every other model does: ``z(t0) = phi([y0, ..., y0])``; the
SINDy model is then integrated and its first coordinate is the forecast.

Losses, per training sample (equation 15 of Bakarji et al., with the weights of
their code's real-data case, ``testcases/lorenzww_basic.py``):

    w_rec |h - psi(z)|^2 + w_dh |dh/dt - J_psi(z) theta Xi|^2
    + w_dz |J_phi(h) dh/dt - theta Xi|^2 + w_z1 (z_1 - y_k)^2
    + w_cons mean_j (zhat_1(t_(k+j)) - y_(k+j))^2 + w_l1 mean |Xi|

``dh/dt`` comes from finite differences of each record (the windows' entries
before a record are at rest); ``zhat`` integrates the SINDy model by RK4 from
``z_k`` across the next ``n - 1`` samples with the recorded inputs, clipped as
the authors' code clips it. Adam on mini-batches; sequential thresholding as in
Champion et al. removes, for good, the coefficients below a fixed threshold at
set epochs; a refinement phase then trains without the L1 penalty with the
sparsity pattern fixed, and validation free rollouts choose its checkpoint.
Per seed, validation free rollouts in the adaptive replay choose the latent
dimension and the window length. No part of the method reads the task
specification. ``SindyAutoencoder`` replays a saved model in numpy with an
adaptive solver, one sample interval at a time, which is how the model is
selected and scored.
"""

from __future__ import annotations

import math
from itertools import combinations_with_replacement, pairwise
from time import monotonic

import numpy as np
from pydantic import Field
from scipy.integrate import solve_ivp

from autoformalism.baselines.latent_ode import Arrays
from autoformalism.baselines.latent_sindy import library_terms, scaling, theta
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "sindy-autoencoder-1"


class SindyAutoencoderConfig(StrictSchema):
    """Every setting of the method, fixed before any model is trained."""

    latent_dimensions: tuple[int, ...] = Field(default=(2, 3, 4), min_length=1)
    delays: tuple[int, ...] = Field(default=(5, 10, 20), min_length=1)
    widths: tuple[int, ...] = Field(default=(64, 32), min_length=1)
    weight_reconstruction: float = Field(default=0.3, ge=0)
    weight_dh: float = Field(default=1e-3, ge=0)
    weight_dz: float = Field(default=1e-3, ge=0)
    weight_first_coordinate: float = Field(default=1e-2, ge=0)
    weight_consistency: float = Field(default=0.1, ge=0)
    weight_l1: float = Field(default=1e-5, ge=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=300, ge=1)
    threshold: float = Field(default=0.1, ge=0)
    threshold_every: int = Field(default=100, ge=1)
    refinement_epochs: int = Field(default=100, ge=1)
    evaluate_every: int = Field(default=10, ge=1)
    initial_coefficient: float = Field(default=0.0)
    initial_scale: float = Field(default=1e-2, ge=0)
    consistency_clip: float = Field(default=500.0, gt=0)
    rk4_substeps: int = Field(default=2, ge=1, le=16)
    replay_rtol: float = Field(default=1e-8, gt=0)
    replay_atol: float = Field(default=1e-10, gt=0)
    replay_seconds: float = Field(default=45.0, gt=0)


# --- delay coordinates ---------------------------------------------------------


def delay_index(n_samples: int, n: int) -> np.ndarray:
    """Positions in a record padded with ``n - 1`` leading samples: ``k - i``."""
    return np.arange(n_samples)[:, None] + n - 1 - np.arange(n)[None, :]


def windows(y: np.ndarray, n: int) -> np.ndarray:
    """Causal delay windows ``(B, N, n)`` of records ``(B, N)``: entry ``i`` at
    sample ``k`` is ``y[k - i]``; the first reading stands in before the record."""
    padded = np.concatenate([np.repeat(y[:, :1], n - 1, axis=1), y], axis=1)
    return padded[:, delay_index(y.shape[1], n)]


def derivative_windows(y: np.ndarray, time: np.ndarray, n: int) -> np.ndarray:
    """The windows' time derivatives: finite differences of each record, zero
    before it (the reactor at rest)."""
    dy = np.gradient(y, time, axis=1)
    padded = np.concatenate([np.zeros((y.shape[0], n - 1)), dy], axis=1)
    return padded[:, delay_index(y.shape[1], n)]


def horizon(values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``values[:, k + j]`` for ``j < n`` as ``(B, N, n, ...)`` and where it exists."""
    n_samples = values.shape[1]
    ahead = np.arange(n_samples)[:, None] + np.arange(n)[None, :]
    return values[:, np.minimum(ahead, n_samples - 1)], ahead < n_samples


# --- training (PyTorch) --------------------------------------------------------


def _pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(combinations_with_replacement(range(n), 2))
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _theta_torch(x, i, j, torch):
    ones = torch.ones((*x.shape[:-1], 1), dtype=x.dtype)
    return torch.cat([ones, x, x[..., i] * x[..., j]], -1)


def _network(sizes, torch):
    layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in pairwise(sizes))
    for layer in layers:
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    return layers


def _apply(layers, x, torch):
    for layer in layers[:-1]:
        x = torch.nn.functional.elu(layer(x))
    return layers[-1](x)


def _rk4(f, z, ua, ub, h, torch, clip=None):
    mid = 0.5 * (ua + ub)
    k1 = f(z, ua)
    k2 = f(z + 0.5 * h * k1, mid)
    k3 = f(z + 0.5 * h * k2, mid)
    k4 = f(z + h * k3, ub)
    z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z if clip is None else torch.clamp(z, -clip, clip)


def _export(layers) -> list[dict]:
    return [
        {"weight": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()}
        for layer in layers
    ]


def train(
    train_data: Arrays,
    validation: Arrays,
    latent_dimension: int,
    delays: int,
    seed: int,
    config: SindyAutoencoderConfig,
) -> dict:
    """Fit one latent dimension, window length and seed.

    Thresholding phase, then refinement; validation free rollouts (RK4) choose
    the refinement checkpoint.
    """
    import torch

    if train_data.targets.shape[-1] != 1:
        raise ValueError("the delay autoencoder reads one measured target")
    step = np.diff(train_data.time)
    if np.max(np.abs(step - step[0])) > 1e-9 * max(1.0, step[0]):
        raise ValueError("the delay autoencoder trains on a uniform grid")
    dt, n, d = float(step[0]), delays, latent_dimension
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    scale = scaling(train_data)

    def standardized(data):
        y = (data.targets[..., 0] - scale["target_mean"][0]) / scale["target_sd"][0]
        u = (data.inputs - np.asarray(scale["input_mean"])) / np.asarray(
            scale["input_sd"]
        )
        return y, u

    y, u = standardized(train_data)
    vy, vu = standardized(validation)
    n_u = u.shape[-1]
    future, exists = horizon(y, n)
    future_u, _ = horizon(u, n)
    tensor = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    samples = {
        "h": tensor(windows(y, n).reshape(-1, n)),
        "dh": tensor(derivative_windows(y, train_data.time, n).reshape(-1, n)),
        "u": tensor(u.reshape(-1, n_u)),
        "future": tensor(future.reshape(-1, n)),
        "future_u": tensor(future_u.reshape(-1, n, n_u)),
        "exists": tensor(exists[None].repeat(len(y), 0).reshape(-1, n)),
    }
    count = samples["h"].shape[0]
    encoder = _network([n, *config.widths, d], torch).double()
    decoder = _network([d, *config.widths[::-1], n], torch).double()
    terms = library_terms(d + n_u)
    pairs = tuple(torch.tensor(a) for a in _pairs(d + n_u))
    coefficients = torch.nn.Parameter(
        config.initial_coefficient
        + config.initial_scale * torch.randn(len(terms), d, dtype=torch.float64)
    )
    mask = torch.ones_like(coefficients)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters(), coefficients],
        lr=config.learning_rate,
    )

    def sindy(z, w):
        return _theta_torch(torch.cat([z, w], -1), *pairs, torch) @ (
            coefficients * mask
        )

    def loss_of(batch, l1):
        h, dh = samples["h"][batch], samples["dh"][batch]
        w = samples["u"][batch]
        z, dz_chain = torch.func.jvp(lambda x: _apply(encoder, x, torch), (h,), (dh,))
        dz = sindy(z, w)
        reconstruction, dh_model = torch.func.jvp(
            lambda x: _apply(decoder, x, torch), (z,), (dz,)
        )
        future, future_u = samples["future"][batch], samples["future_u"][batch]
        exists = samples["exists"][batch]
        state, error = z, (z[:, 0] - future[:, 0]) ** 2
        for j in range(n - 1):
            state = _rk4(
                sindy,
                state,
                future_u[:, j],
                future_u[:, j + 1],
                dt,
                torch,
                config.consistency_clip,
            )
            error = error + exists[:, j + 1] * (state[:, 0] - future[:, j + 1]) ** 2
        parts = {
            "reconstruction": torch.mean((h - reconstruction) ** 2),
            "dh": torch.mean((dh - dh_model) ** 2),
            "dz": torch.mean((dz_chain - dz) ** 2),
            "first_coordinate": torch.mean((z[:, 0] - h[:, 0]) ** 2),
            "consistency": torch.mean(error / exists.sum(1)),
            "l1": torch.mean(torch.abs(coefficients * mask)),
        }
        total = (
            config.weight_reconstruction * parts["reconstruction"]
            + config.weight_dh * parts["dh"]
            + config.weight_dz * parts["dz"]
            + config.weight_first_coordinate * parts["first_coordinate"]
            + config.weight_consistency * parts["consistency"]
            + l1 * parts["l1"]
        )
        return total, parts

    def validation_nmse() -> float:
        with torch.no_grad():
            first = tensor(np.repeat(vy[:, :1], n, axis=1))
            state = _apply(encoder, first, torch)
            w = tensor(vu)
            out = [state[:, 0]]
            for k in range(len(validation.time) - 1):
                h = float(validation.time[k + 1] - validation.time[k])
                h /= config.rk4_substeps
                for s in range(config.rk4_substeps):
                    a, b = s / config.rk4_substeps, (s + 1) / config.rk4_substeps
                    ua = w[:, k] + (w[:, k + 1] - w[:, k]) * a
                    ub = w[:, k] + (w[:, k + 1] - w[:, k]) * b
                    state = _rk4(sindy, state, ua, ub, h, torch)
                out.append(state[:, 0])
            score = float(torch.mean((torch.stack(out, 1) - tensor(vy)) ** 2))
        return score if math.isfinite(score) else math.inf

    best = {"validation_nmse_rk4": math.inf, "epoch": None, "snapshot": None}
    history, diverged = [], False
    start = monotonic()
    total_epochs = config.epochs + config.refinement_epochs
    for epoch in range(1, total_epochs + 1):
        refining = epoch > config.epochs
        order = torch.randperm(count)
        sums = {}
        for first in range(0, count, config.batch_size):
            batch = order[first : first + config.batch_size]
            optimizer.zero_grad()
            loss, parts = loss_of(batch, 0.0 if refining else config.weight_l1)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                coefficients.mul_(mask)
            for key, value in {"total": loss, **parts}.items():
                sums[key] = sums.get(key, 0.0) + value.item() * len(batch)
        if diverged:
            history.append({"epoch": epoch, "diverged": True})
            break
        record = {"epoch": epoch, **{k: v / count for k, v in sums.items()}}
        if not refining and epoch % config.threshold_every == 0:
            with torch.no_grad():
                mask.mul_((coefficients.abs() >= config.threshold).to(mask.dtype))
                coefficients.mul_(mask)
            record["kept_terms"] = int(mask.sum().item())
        if epoch >= config.epochs and (
            (epoch - config.epochs) % config.evaluate_every == 0
            or epoch == total_epochs
        ):
            score = validation_nmse()
            record["validation_nmse_rk4"] = score
            if score < best["validation_nmse_rk4"]:
                best = {
                    "validation_nmse_rk4": score,
                    "epoch": epoch,
                    "snapshot": {
                        "encoder": _export(encoder),
                        "decoder": _export(decoder),
                        "coefficients": (coefficients * mask).detach().tolist(),
                        "mask": mask.detach().tolist(),
                    },
                }
        history.append(record)
    return {
        "protocol": PROTOCOL,
        "latent_dimension": d,
        "delays": n,
        "seed": seed,
        "sample_interval": dt,
        "standardization": scale,
        "library": [list(term) for term in terms],
        "best": best,
        "kept_terms": int(mask.sum().item()),
        "diverged": diverged,
        "history": history,
        "seconds": monotonic() - start,
        "torch": torch.__version__,
    }


# --- replay (numpy) ------------------------------------------------------------


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


class SindyAutoencoder:
    """A saved delay SINDy autoencoder, replayed adaptively between samples."""

    def __init__(self, model: dict):
        self.targets = tuple(model["targets"])
        self.inputs = tuple(model["inputs"])
        if len(self.targets) != 1:
            raise ValueError("the delay autoencoder reads one measured target")
        scale = model["standardization"]
        self.ym, self.ys = (np.asarray(scale[k]) for k in ("target_mean", "target_sd"))
        self.um, self.us = (np.asarray(scale[k]) for k in ("input_mean", "input_sd"))
        self.delays = int(model["delays"])
        self.encoder = [
            (np.asarray(p["weight"]), np.asarray(p["bias"])) for p in model["encoder"]
        ]
        self.coefficients = np.asarray(model["coefficients"], dtype=float) * np.asarray(
            model["mask"], dtype=float
        )
        self.n_z = self.coefficients.shape[1]
        if self.encoder[0][0].shape[1] != self.delays or (
            self.encoder[-1][0].shape[0] != self.n_z
        ):
            raise ValueError("saved encoder does not match the window or latent size")
        n = self.n_z + len(self.inputs)
        if self.coefficients.shape != (len(library_terms(n)), self.n_z):
            raise ValueError("saved coefficients do not match the library")

    def derivative(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return theta(np.concatenate([z, u])) @ self.coefficients

    def initial(self, y0: np.ndarray) -> np.ndarray:
        """Latent state of a record's first reading: the window at rest."""
        x = np.full(self.delays, (float(y0[0]) - self.ym[0]) / self.ys[0])
        for weight, bias in self.encoder[:-1]:
            x = _elu(weight @ x + bias)
        weight, bias = self.encoder[-1]
        return weight @ x + bias

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
        """The target ``(N, 1)`` in its units and the other coordinates ``(N, d-1)``."""
        deadline = monotonic() + seconds
        time = np.asarray(time, dtype=float)
        u = (np.asarray(inputs, dtype=float) - self.um) / self.us
        z = self.initial(np.asarray(y0, dtype=float))
        states = [z]
        for k in range(len(time) - 1):
            t0, t1, ua, ub = time[k], time[k + 1], u[k], u[k + 1]

            def rhs(t, state, t0=t0, t1=t1, ua=ua, ub=ub):
                if monotonic() > deadline:
                    raise TimeoutError(
                        "SINDy autoencoder replay wall-clock limit reached"
                    )
                return self.derivative(state, ua + (ub - ua) * (t - t0) / (t1 - t0))

            sol = solve_ivp(rhs, (t0, t1), z, method=method, rtol=rtol, atol=atol)
            if not sol.success or not np.isfinite(sol.y[:, -1]).all():
                raise ValueError(f"SINDy autoencoder replay failed: {sol.message}")
            z = sol.y[:, -1]
            states.append(z)
        states = np.asarray(states)
        return states[:, :1] * self.ys + self.ym, states[:, 1:]

    def equations(self) -> dict[str, str]:
        """Each latent coordinate's equation, nonzero terms only; ``z1`` stands
        for the standardized measured target."""
        names = (*(f"z{j + 1}" for j in range(self.n_z)), *self.inputs)
        labels = [
            "1" if not term else "*".join(names[i] for i in term)
            for term in library_terms(len(names))
        ]
        out = {}
        for column, state in enumerate(names[: self.n_z]):
            parts = [
                f"{c:+.6g}*{label}"
                for c, label in zip(self.coefficients[:, column], labels, strict=True)
                if c != 0.0
            ]
            out[state] = f"d{state}/dt = " + (" ".join(parts) if parts else "0")
        return out
