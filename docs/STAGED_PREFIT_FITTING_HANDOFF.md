# Staged pre-fit fitting handoff

This development milestone fits the exact six candidates that passed the
staged function pre-fit handoff. It does not regenerate or select structures.

Before fitting, the prepare step freezes each complete source result, its
canonical candidate, public prompt/train/validation files, candidate identity,
equations, parameter roles, latent initializers, deterministic pre-fit
certificate, repair count, and fitting route. The route is selected without
validation outcomes:

1. certify the profiled latent-basis parameterization structurally;
2. require exact observed derivative provenance in the public training input;
3. use profiled variable projection only when both contracts pass; otherwise
   use bounded causal rollout fitting.

The current Phase-B public tidy tables contain finite-difference derivative
estimates, not exact derivatives. The frozen router therefore records any
structural profiled compatibility but chooses bounded rollout. It never creates
an exact-derivative overlay from private equations.

Parameters are fit on public training trajectories only. Validation trajectories
are used only for the post-fit causal rollout metric. The summary keeps source
structural validity, fitting success, train NMSE, validation NMSE, integration
failures, function evaluations, and CPU/wall resources separate. It defines no
weighted score or automatic winner. Test data, private references, the proposer,
and the scientific judge remain closed.
