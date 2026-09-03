from pathlib import Path

from scripts.run_profiled_latent_basis_fitting_smoke import run_smoke


def test_profiled_latent_basis_fitting_smoke(tmp_path: Path) -> None:
    result = run_smoke(tmp_path / "result.json")

    assert result["status"] == "pass"
    assert result["latent_values_supplied_to_fitter"] is False
    assert result["latent_derivatives_supplied_to_fitter"] is False
    assert result["certified_parameter_transformations"] == [
        "reciprocal:tau=1/tau"
    ]
    assert result["test_normalized_mse"] < 1e-9
