"""Multilayer perceptrons a hybrid model declares, fitted with its parameters.

HDTwinGen (github.com/samholt/HDTwinGen) lets its model add black-box
multilayer perceptrons to a white-box model, written as PyTorch code that the
harness executes. We never execute model-written code. A model here declares
each network in a fixed form instead: what it sees (input expressions in the
restricted grammar), how many hidden layers, how wide, and which activation.
The harness builds the network, fits its weights together with the model's
parameters, and evaluates it. A network has one output, which the model's
equations and algebraics use as a plain symbol named after the network.

A network's inputs may use the observed states, the available inputs, the
model's parameters and ``t``; not an algebraic, and not another network's
output. Inputs are used exactly as written: nothing rescales them.

Weights start from PyTorch's default initialization of a linear layer under
the fit's seed. Only the scalar parameters are shown back to the model, as
upstream's ``get_model_parameters`` shows only top-level parameters.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

import numpy as np
from pydantic import Field

from autoformalism.expressions.parser import BASELINE_FUNCTION_ARITY, baseline_parser
from autoformalism.schemas.base import Identifier, NonEmptyText, StrictSchema

#: At most this many networks in one model.
MAX_NETWORKS = 4

Activation = Literal["tanh", "relu", "sigmoid", "softplus", "elu", "silu"]


class NetworkSpec(StrictSchema):
    """One declared multilayer perceptron with a single output."""

    name: Identifier
    inputs: tuple[NonEmptyText, ...] = Field(min_length=1, max_length=8)
    hidden_layers: int = Field(ge=1, le=3)
    hidden_width: int = Field(ge=2, le=64)
    activation: Activation


class NetworkError(ValueError):
    """A declared network cannot be built or evaluated."""


def layer_shapes(spec: NetworkSpec) -> tuple[tuple[int, int], ...]:
    """Each linear layer's (outputs, inputs), input layer first."""
    sizes = (len(spec.inputs), *([spec.hidden_width] * spec.hidden_layers), 1)
    return tuple(zip(sizes[1:], sizes[:-1], strict=True))


def weight_count(spec: NetworkSpec) -> int:
    """How many weights and biases the network has."""
    return sum(rows * columns + rows for rows, columns in layer_shapes(spec))


def check_networks(
    networks: Iterable[NetworkSpec],
    *,
    input_symbols: set[str],
    taken: set[str],
) -> dict[str, tuple[ast.AST, ...]]:
    """Parse every network's inputs, refusing clashes and unavailable symbols.

    `input_symbols` are the names an input may use; `taken` are names a
    network may not have (states, inputs, parameters, algebraics, ``t``).
    Returns each network's parsed input expressions by name.
    """
    networks = tuple(networks)
    if len(networks) > MAX_NETWORKS:
        raise NetworkError(f"at most {MAX_NETWORKS} networks may be declared")
    parser = baseline_parser()
    parsed: dict[str, tuple[ast.AST, ...]] = {}
    for spec in networks:
        if spec.name in parsed:
            raise NetworkError(f"network {spec.name} is declared twice")
        if spec.name in taken or spec.name in BASELINE_FUNCTION_ARITY:
            raise NetworkError(f"network name {spec.name} is already in use")
        trees = []
        for position, source in enumerate(spec.inputs):
            expression = parser.parse(
                source, location=f"network:{spec.name}:input{position}"
            )
            unknown = expression.symbols - input_symbols
            if unknown:
                raise NetworkError(
                    f"network {spec.name} input {position} uses unavailable "
                    f"symbols: {sorted(unknown)}"
                )
            trees.append(expression.tree.body)
        parsed[spec.name] = tuple(trees)
    return parsed


def _torch_activation(torch, name: str) -> Callable:
    functional = torch.nn.functional
    return {
        "tanh": torch.tanh,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "softplus": functional.softplus,
        "elu": functional.elu,
        "silu": functional.silu,
    }[name]


def _numpy_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    def sigmoid(value):
        return np.exp(-np.logaddexp(0.0, -value))

    return {
        "tanh": np.tanh,
        "relu": lambda value: np.maximum(value, 0.0),
        "sigmoid": sigmoid,
        # torch's softplus returns its input above 20.
        "softplus": lambda value: np.where(
            value > 20.0, value, np.logaddexp(0.0, value)
        ),
        "elu": lambda value: np.where(
            value > 0.0, value, np.expm1(np.minimum(value, 0.0))
        ),
        "silu": lambda value: value * sigmoid(value),
    }[name]


def _stack(values: list, backend) -> Any:
    """Inputs as one array with the input index last; scalars broadcast."""
    if backend is np:
        arrays = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in values])
        return np.stack(arrays, axis=-1)
    tensors = [backend.as_tensor(v, dtype=backend.float64) for v in values]
    shape = backend.broadcast_shapes(*(t.shape for t in tensors))
    return backend.stack([backend.broadcast_to(t, shape) for t in tensors], dim=-1)


class TorchNetworks:
    """The declared networks as trainable torch layers, built under one seed."""

    def __init__(
        self,
        specs: Iterable[NetworkSpec],
        parsed: Mapping[str, tuple[ast.AST, ...]],
        torch,
        weights: Mapping[str, list[dict]] | None = None,
    ) -> None:
        self._torch = torch
        self.specs = tuple(specs)
        self._parsed = dict(parsed)
        self.layers: dict[str, list] = {}
        for spec in self.specs:
            layers = [
                torch.nn.Linear(columns, rows, dtype=torch.float64)
                for rows, columns in layer_shapes(spec)
            ]
            if weights is not None:
                saved = weights[spec.name]
                if len(saved) != len(layers):
                    raise NetworkError(f"saved weights of {spec.name} differ in depth")
                with torch.no_grad():
                    for layer, values in zip(layers, saved, strict=True):
                        weight = torch.tensor(values["weight"], dtype=torch.float64)
                        bias = torch.tensor(values["bias"], dtype=torch.float64)
                        if (
                            weight.shape != layer.weight.shape
                            or bias.shape != layer.bias.shape
                        ):
                            raise NetworkError(
                                f"saved weights of {spec.name} differ in shape"
                            )
                        layer.weight.copy_(weight)
                        layer.bias.copy_(bias)
            self.layers[spec.name] = layers

    def parameters(self) -> list:
        """Every weight and bias, for the optimizer."""
        return [
            tensor
            for layers in self.layers.values()
            for layer in layers
            for tensor in (layer.weight, layer.bias)
        ]

    def outputs(self, environment: Mapping[str, Any], evaluate: Callable) -> dict:
        """Each network's output for the environment's rows."""
        values = {}
        for spec in self.specs:
            inputs = [
                evaluate(tree, environment, self._torch)
                for tree in self._parsed[spec.name]
            ]
            hidden = _stack(inputs, self._torch)
            activation = _torch_activation(self._torch, spec.activation)
            layers = self.layers[spec.name]
            for layer in layers[:-1]:
                hidden = activation(layer(hidden))
            values[spec.name] = layers[-1](hidden)[..., 0]
        return values

    def snapshot(self) -> dict[str, list]:
        """A detached copy of every layer, for early stopping."""
        return {
            name: [
                (layer.weight.detach().clone(), layer.bias.detach().clone())
                for layer in layers
            ]
            for name, layers in self.layers.items()
        }

    def restore(self, snapshot: Mapping[str, list]) -> None:
        """Load a snapshot taken by ``snapshot``."""
        with self._torch.no_grad():
            for name, layers in self.layers.items():
                for layer, (weight, bias) in zip(
                    layers, snapshot[name], strict=True
                ):
                    layer.weight.copy_(weight)
                    layer.bias.copy_(bias)

    def export(self) -> dict[str, list[dict]]:
        """The weights as plain lists, which ``numpy_outputs`` reads."""
        return {
            name: [
                {
                    "weight": layer.weight.detach().cpu().numpy().tolist(),
                    "bias": layer.bias.detach().cpu().numpy().tolist(),
                }
                for layer in layers
            ]
            for name, layers in self.layers.items()
        }


def check_weights(
    specs: Iterable[NetworkSpec], weights: Mapping[str, list[dict]]
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Saved weights as arrays, once their shapes match the declarations."""
    specs = tuple(specs)
    if set(weights) != {spec.name for spec in specs}:
        raise NetworkError("saved weights must name exactly the declared networks")
    arrays = {}
    for spec in specs:
        saved = weights[spec.name]
        shapes = layer_shapes(spec)
        if len(saved) != len(shapes):
            raise NetworkError(f"saved weights of {spec.name} differ in depth")
        layers = []
        for values, (rows, columns) in zip(saved, shapes, strict=True):
            weight = np.asarray(values["weight"], dtype=float)
            bias = np.asarray(values["bias"], dtype=float)
            if weight.shape != (rows, columns) or bias.shape != (rows,):
                raise NetworkError(f"saved weights of {spec.name} differ in shape")
            if not (np.isfinite(weight).all() and np.isfinite(bias).all()):
                raise NetworkError(f"saved weights of {spec.name} are not finite")
            layers.append((weight, bias))
        arrays[spec.name] = layers
    return arrays


def numpy_outputs(
    specs: Iterable[NetworkSpec],
    parsed: Mapping[str, tuple[ast.AST, ...]],
    arrays: Mapping[str, list[tuple[np.ndarray, np.ndarray]]],
    environment: Mapping[str, Any],
    evaluate: Callable,
    backend,
) -> dict[str, Any]:
    """Each network's output with numpy, as ``TorchNetworks.outputs`` computes it."""
    values = {}
    for spec in specs:
        inputs = [evaluate(tree, environment, backend) for tree in parsed[spec.name]]
        hidden = _stack(inputs, np)
        activation = _numpy_activation(spec.activation)
        layers = arrays[spec.name]
        for weight, bias in layers[:-1]:
            hidden = activation(hidden @ weight.T + bias)
        weight, bias = layers[-1]
        values[spec.name] = (hidden @ weight.T + bias)[..., 0]
    return values
