"""Native optimizer success cannot hide rejected/missing final checkpoints."""

import numpy as np
import pytest

from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import transcription_solver as solver
from tests.test_fitting_strategies import small_payload


@pytest.mark.parametrize("rejection", [None, "bounds", "nonfinite"])
def test_final_publication_is_explicit(tmp_path, monkeypatch, rejection):
    payload = small_payload(controls.make_inputs(), "collocation")
    payload.update(
        checkpoint_mode="compact",
        retain_inner_nodes=True,
        strict_parameter_bounds=True,
        seconds=20,
    )
    build = solver.build_problem
    options = []

    def build_proxy(*args):
        problem = build(*args)
        original = problem.opti

        class OptiProxy:
            def __getattr__(self, name):
                return getattr(original, name)

            def solver(self, name, plugin, opts):
                options.append(opts)
                return original.solver(name, plugin, opts)

            def solve(self):
                solution = original.solve()

                class SolutionProxy:
                    def stats(self):
                        return solution.stats()

                    def value(self, expression):
                        result = solution.value(expression)
                        if rejection and expression is problem.theta:
                            result = np.asarray(result).copy()
                            result[0] = (
                                problem.layout.lower[0] - 1e-4
                                if rejection == "bounds"
                                else np.nan
                            )
                        return result

                return SolutionProxy()

        problem.opti = OptiProxy()
        return problem

    monkeypatch.setattr(solver, "build_problem", build_proxy)
    result = solver.solve(payload, tmp_path)
    assert result["native_success"]
    assert options[0]["bound_relax_factor"] == 0
    status = public._read(tmp_path / "final_checkpoint_status.json")
    assert result["final_checkpoint"] == status
    assert status["published"] is (rejection is None)
    assert (tmp_path / "final_checkpoint_diagnostics.json").exists() is (
        rejection is None
    )
    if rejection == "bounds":
        assert status["status"] == "parameter_bounds_violated"
        assert status["lower_bound_violation"] == pytest.approx(1e-4)
    if rejection == "nonfinite":
        assert status["status"] == "nonfinite_parameter_or_objective"
        assert None in status["raw_parameters"].values()
