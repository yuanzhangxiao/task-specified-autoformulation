"""HDTwinGen: D3's loop with HDTwinGen's prompts and declared black-box networks.

What is under test: the prompts differ from D3's exactly where HDTwinGen's
differ; a reply may declare networks, which are checked, built, fitted and
replayed without executing anything the model wrote; their weights reach the
selected model; and D3's own loop is unchanged by any of it.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from autoformalism.baselines import d3_upstream as loop
from autoformalism.baselines import hdtwingen
from autoformalism.baselines.d3_native import NativeD3Fit, _evaluate
from autoformalism.baselines.d3_rollout import _NUMPY, evaluate_validation
from autoformalism.baselines.hybrid_networks import (
    NetworkError,
    NetworkSpec,
    TorchNetworks,
    check_networks,
    check_weights,
    layer_shapes,
    numpy_outputs,
    weight_count,
)
from autoformalism.rebuttal.phase_c_baselines import load_cell
from autoformalism.schemas import CandidateModel
from tests.test_phase_c_vendored_campaign import CELL, _release

NETWORK = {
    "name": "r",
    "inputs": ["h_down", "t / 100"],
    "hidden_layers": 2,
    "hidden_width": 8,
    "activation": "tanh",
}


def _reply(rhs: str = "-k * h_down + r", networks=(NETWORK,)):
    return hdtwingen.HybridModelReply.model_validate(
        {
            "model_description": f"hybrid: {rhs}",
            "model": {
                "candidate_id": "m",
                "states": [
                    {
                        "name": "h_down",
                        "kind": "observed",
                        "observed_channel": "h_down",
                        "rhs": rhs,
                    }
                ],
                "parameters": [{"name": "k"}],
            },
            "parameter_initial_values": [{"name": "k", "value": 0.1}],
            "networks": list(networks),
        }
    )


class Chat:
    """Answers every model call with the next scripted reply."""

    def __init__(self, replies) -> None:
        self.replies = iter(replies)
        self.messages: list[list[dict]] = []

    def reflect(self, messages, *, generation):
        self.messages.append(messages)
        return "add a residual"

    def write_model(self, messages, *, generation):
        self.messages.append(messages)
        return next(self.replies)


@pytest.fixture
def cell(tmp_path):
    return load_cell(_release(tmp_path / "release"), CELL)


def test_hdtwingens_prompts_restore_the_black_box_lines_d3s_port_drops(cell):
    form = hdtwingen.model_format(cell.context, ("inflow_up", "inflow_down"))
    first = hdtwingen.first_task_prompt("desc", form, generations=20)
    d3_first = loop.first_task_prompt(
        "desc", loop.model_format(cell.context, ("inflow_up",)), generations=20
    )
    assert "black box neural network components (multi-layer perceptrons)" in first
    assert "switch to hybrid white and black box models for the residuals" in first
    assert "black box components cannot be expressed" not in first
    assert "black box components cannot be expressed" in d3_first
    assert "Black box components: declare each multi-layer perceptron" in form
    assert "multi-layer perceptions where required" in hdtwingen.SYSTEM_PROMPT
    assert "perceptions" not in loop.SYSTEM_PROMPT
    reflection = hdtwingen.reflection_prompt([], [], iteration=1, generations=20)
    assert "only add black box models to the white box models" in reflection
    assert "x_next = x + rhs" in form


def test_network_declarations_are_checked_before_anything_is_built():
    spec = NetworkSpec.model_validate(NETWORK)
    parsed = check_networks(
        [spec], input_symbols={"h_down", "t"}, taken={"h_down", "k", "t"}
    )
    assert isinstance(parsed["r"][0], ast.AST)
    assert layer_shapes(spec) == ((8, 2), (8, 8), (1, 8))
    assert weight_count(spec) == 8 * 2 + 8 + 8 * 8 + 8 + 8 + 1
    with pytest.raises(NetworkError, match="unavailable symbols"):
        check_networks([spec], input_symbols={"h_down"}, taken=set())
    with pytest.raises(NetworkError, match="already in use"):
        check_networks([spec], input_symbols={"h_down", "t"}, taken={"r"})
    exp = NetworkSpec.model_validate({**NETWORK, "name": "exp"})
    with pytest.raises(NetworkError, match="already in use"):
        check_networks([exp], input_symbols={"h_down", "t"}, taken=set())
    with pytest.raises(NetworkError, match="declared twice"):
        check_networks([spec, spec], input_symbols={"h_down", "t"}, taken=set())
    with pytest.raises(NetworkError, match="at most"):
        check_networks(
            [
                NetworkSpec.model_validate({**NETWORK, "name": f"n{i}"})
                for i in range(5)
            ],
            input_symbols={"h_down", "t"},
            taken=set(),
        )
    with pytest.raises(ValueError):
        NetworkSpec.model_validate({**NETWORK, "hidden_width": 65})
    with pytest.raises(ValueError):
        NetworkSpec.model_validate({**NETWORK, "activation": "gelu"})


@pytest.mark.parametrize(
    "activation", ["tanh", "relu", "sigmoid", "softplus", "elu", "silu"]
)
def test_numpy_replays_the_torch_network_exactly(activation):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    spec = NetworkSpec.model_validate({**NETWORK, "activation": activation})
    parsed = check_networks(
        [spec], input_symbols={"h_down", "t"}, taken={"h_down", "t"}
    )
    built = TorchNetworks([spec], parsed, torch)
    values = np.linspace(-30.0, 30.0, 41)
    environment = {
        "h_down": torch.tensor(values, dtype=torch.float64),
        "t": torch.tensor(values * 3.0, dtype=torch.float64),
    }
    expected = built.outputs(environment, _evaluate)["r"].detach().numpy()
    arrays = check_weights([spec], built.export())
    for index, (h, t) in enumerate(zip(values, values * 3.0, strict=True)):
        got = numpy_outputs([spec], parsed, arrays, {"h_down": h, "t": t},
                            _evaluate, _NUMPY)["r"]
        assert got == pytest.approx(expected[index], rel=1e-12, abs=1e-12)
    rebuilt = TorchNetworks([spec], parsed, torch, built.export())
    again = rebuilt.outputs(environment, _evaluate)["r"].detach().numpy()
    assert np.array_equal(again, expected)


def test_saved_weights_must_match_the_declarations():
    spec = NetworkSpec.model_validate(NETWORK)
    weights = {
        "r": [
            {"weight": np.zeros(shape).tolist(), "bias": np.zeros(shape[0]).tolist()}
            for shape in layer_shapes(spec)
        ]
    }
    assert len(check_weights([spec], weights)["r"]) == 3
    with pytest.raises(NetworkError, match="exactly the declared"):
        check_weights([spec], {})
    weights["r"][1]["bias"] = [0.0]
    with pytest.raises(NetworkError, match="differ in shape"):
        check_weights([spec], weights)
    weights["r"][1]["bias"] = [float("nan")] * 8
    with pytest.raises(NetworkError, match="not finite"):
        check_weights([spec], weights)


def test_the_loop_fits_a_hybrid_reply_and_keeps_its_networks(
    cell, tmp_path, monkeypatch
):
    started = []

    def fit(candidate, training, validation, *, targets, seed, initial_values,
            networks):
        built = networks(_FakeTorch())
        started.append(built)
        return NativeD3Fit({"k": 0.2}, 0.1, 0.05, 20, {"h_down": 1.0},
                           {"r": "weights"})

    def losses(candidate, split, parameters, *, networks):
        assert networks(_FakeTorch()).weights == {"r": "weights"}
        return 0.01, {"h_down": 0.01}

    monkeypatch.setattr(loop, "fit_native_d3", fit)
    monkeypatch.setattr(loop, "raw_state_losses", losses)
    monkeypatch.setattr(loop, "TorchNetworks", _FakeNetworks)
    chat = Chat([_reply(), _reply("-k * h_down + 2 * r")])
    result = loop.run_d3_upstream(
        cell.dataset,
        cell.context,
        task_prompt=cell.prompt,
        chat=chat,
        settings=loop.D3Settings(generations=2, patience=2),
        seed=0,
        work_directory=tmp_path / "task",
        identity={"task": 0},
        variant=hdtwingen.HDTWINGEN,
    )
    assert result.selected_hyperparameters["loop"] == "hdtwingen_nsdt"
    assert result.selection_payload["networks"] == [NETWORK]
    assert result.selection_payload["network_weights"] == {"r": "weights"}
    assert chat.messages[0][0]["content"] == hdtwingen.SYSTEM_PROMPT
    reflection = chat.messages[1][-1]["content"]
    assert '"networks": [{"name": "r"' in reflection
    assert "weights" not in reflection
    assert "optimized_parameters = {'k': 0.2}" in reflection


def test_a_white_box_reply_needs_no_networks_and_records_none(
    cell, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        loop,
        "fit_native_d3",
        lambda *a, **k: NativeD3Fit({"k": 0.2}, 0.1, 0.05, 20, {"h_down": 1.0}),
    )
    monkeypatch.setattr(
        loop, "raw_state_losses", lambda *a, **k: (0.01, {"h_down": 0.01})
    )
    result = loop.run_d3_upstream(
        cell.dataset,
        cell.context,
        task_prompt=cell.prompt,
        chat=Chat([_reply("-k * h_down", networks=())]),
        settings=loop.D3Settings(generations=1, patience=1),
        seed=0,
        work_directory=tmp_path / "task",
        identity={"task": 0},
        variant=hdtwingen.HDTWINGEN,
    )
    assert "networks" not in result.selection_payload


def test_a_network_that_reads_an_unknown_symbol_costs_its_generation(
    cell, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        loop,
        "fit_native_d3",
        lambda *a, **k: pytest.fail("a refused network must not be fitted"),
    )
    bad = {**NETWORK, "inputs": ["volume"]}
    with pytest.raises(loop.D3AdapterError, match="unavailable symbols"):
        loop.run_d3_upstream(
            cell.dataset,
            cell.context,
            task_prompt=cell.prompt,
            chat=Chat([_reply(networks=(bad,))]),
            settings=loop.D3Settings(generations=1, patience=1),
            seed=0,
            work_directory=tmp_path / "task",
            identity={"task": 0},
            variant=hdtwingen.HDTWINGEN,
        )


def test_one_variants_checkpoint_is_never_resumed_by_another(cell, tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(
        loop,
        "fit_native_d3",
        lambda *a, **k: NativeD3Fit({"k": 0.2}, 0.1, 0.05, 20, {"h_down": 1.0}),
    )
    monkeypatch.setattr(
        loop, "raw_state_losses", lambda *a, **k: (0.01, {"h_down": 0.01})
    )
    options = {
        "task_prompt": cell.prompt,
        "settings": loop.D3Settings(generations=1, patience=1),
        "seed": 0,
        "work_directory": tmp_path / "task",
        "identity": {"task": 0},
    }
    loop.run_d3_upstream(
        cell.dataset, cell.context, chat=Chat([_reply("-k * h_down", networks=())]),
        variant=hdtwingen.HDTWINGEN, **options,
    )
    with pytest.raises(loop.D3AdapterError, match="fingerprint"):
        loop.run_d3_upstream(cell.dataset, cell.context, chat=Chat([]), **options)


def test_a_reply_of_the_wrong_format_costs_its_generation(cell, tmp_path):
    d3_reply = loop.D3ModelReply.model_validate(
        _reply(networks=()).model_dump(exclude={"networks"})
    )
    with pytest.raises(loop.D3AdapterError, match="not the variant's"):
        loop.run_d3_upstream(
            cell.dataset,
            cell.context,
            task_prompt=cell.prompt,
            chat=Chat([d3_reply]),
            settings=loop.D3Settings(generations=1, patience=1),
            seed=0,
            work_directory=tmp_path / "task",
            identity={"task": 0},
            variant=hdtwingen.HDTWINGEN,
        )


def test_a_real_hybrid_fit_replays_exactly_and_rolls_out(cell, tmp_path):
    pytest.importorskip("torch")
    result = loop.run_d3_upstream(
        cell.dataset,
        cell.context,
        task_prompt=cell.prompt,
        chat=Chat([_reply()]),
        settings=loop.D3Settings(generations=1, patience=1),
        seed=0,
        work_directory=tmp_path / "task",
        identity={"task": 0},
        variant=hdtwingen.HDTWINGEN,
    )
    payload = result.selection_payload
    evaluation = evaluate_validation(
        CandidateModel.model_validate(payload["candidate"]),
        payload["parameters"],
        cell.dataset.train,
        cell.dataset.validation,
        saved_one_step_nmse=result.validation_normalized_mse,
        networks=tuple(NetworkSpec.model_validate(n) for n in payload["networks"]),
        network_weights=payload["network_weights"],
    )
    assert evaluation["saved_one_step_matches"] is True
    assert evaluation["phase_b_rollout"]["status"] == "complete"
    with pytest.raises(NetworkError):
        evaluate_validation(
            CandidateModel.model_validate(payload["candidate"]),
            payload["parameters"],
            cell.dataset.train,
            cell.dataset.validation,
            networks=tuple(
                NetworkSpec.model_validate(n) for n in payload["networks"]
            ),
            network_weights={},
        )


class _FakeTorch:
    """Stands in for torch where the fit itself is stood in for."""


class _FakeNetworks:
    """Records what the loop builds, in place of TorchNetworks."""

    def __init__(self, specs, parsed, torch, weights=None) -> None:
        self.specs, self.parsed, self.weights = tuple(specs), parsed, weights
