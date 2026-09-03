from pathlib import Path

from scripts.run_fixed_latent_basis_fitting_smoke import run_smoke


def test_fixed_latent_basis_fitting_smoke(tmp_path: Path) -> None:
    result = run_smoke(tmp_path / "result.json")

    assert result["status"] == "pass"
    assert result["latent_values_supplied_to_fitter"] is False
    assert result["latent_derivatives_supplied_to_fitter"] is False
    assert result["function_evaluations"] == 1
