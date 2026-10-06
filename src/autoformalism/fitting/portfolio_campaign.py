"""M13 matched-budget rollout portfolios; post-fit evaluation stays separate."""

from pathlib import Path

from pydantic import Field, model_validator

from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import portfolio_fit

PROTOCOL = "phase-c-start-portfolio-1"
ARMS = ("rollout_only", "rollout_portfolio", "checkpoint_portfolio")


class PortfolioPolicy(common.RecoveryPolicy):
    """Frozen three-candidate trials, local failure recovery and one time ceiling."""

    probe_seconds: float = Field(default=120, ge=5, le=300)
    continuation_seconds: float = Field(default=240, ge=5, le=600)
    probe_calls: int = Field(default=60, ge=5, le=200)
    continuation_calls: int = Field(default=120, ge=5, le=400)
    perturbation_spread: float = Field(default=0.7, gt=0, le=2)
    progress_floor: float = Field(default=0.01, gt=0, lt=1)
    stagnant_visits: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def portfolio_limits(self):
        minimum = (
            3 * self.probe_seconds
            + self.native_seconds
            + 2 * self.point_seconds
            + self.certificate_seconds
        )
        if minimum >= self.seconds:
            raise ValueError("budget must leave time after native/screens/three trials")
        if self.maximum_rollout_calls < 3 * self.probe_calls + 5:
            raise ValueError("call budget must permit all initial trials")
        return self


def prepare(root: Path, inputs: Path, policy: PortfolioPolicy) -> dict:
    """Reuse exact generic inputs and freeze all three original seeds."""
    return common.prepare(root, inputs, policy, protocol=PROTOCOL, arms=ARMS)


def verify(root: Path, *, runtime=True):
    return common.verify(root, runtime=runtime, protocol=PROTOCOL)


def run_task(root: Path, index: int) -> dict:
    return common.run_task(root, index, protocol=PROTOCOL, fitter=portfolio_fit.fit)


def report(root: Path) -> dict:
    return common.report(root, protocol=PROTOCOL, arms=ARMS)
