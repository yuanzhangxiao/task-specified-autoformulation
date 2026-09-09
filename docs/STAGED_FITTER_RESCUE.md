# Staged fitter-rescue test

This matched development test keeps the six candidates from the passed public
pre-fit construction campaign exactly fixed. It asks whether the first fitting
handoff failed because of numerical policy before any topology or interaction
is revised.

For each candidate, the runtime performs the following train-only sequence:

1. causally rescore the prior fit's finite parameter vector under a fresh
   deadline;
2. obtain a warm-start proposal from estimated derivatives in the public
   training table using the structurally certified profiled system;
3. construct two runtime role/time-scale starts;
4. screen those starts on short causal public-training rollouts;
5. fit the two best starts independently with bounded causal rollout; and
6. select by fresh full-training NMSE before evaluating validation once.

Estimated derivatives are initializer evidence only. They never define the
reported fit, never enter validation, and are not relabeled as exact. Each
rescue optimization receives training in both required API slots so it cannot
inspect real validation. Real validation is used only for descriptive rescoring
of the frozen source vector and the training-selected final rescue vector.
Every reported train or validation NMSE is obtained by causal rollout. Test
data, private references, proposer calls, scientific-judge calls, topology
changes, function changes, and automatic model selection are absent.

The attribution labels are deliberately limited:

- `source_timeout_policy_limited`: the old retained vector scores successfully
  under a fresh deadline;
- `fitter_initialization_or_search_limited`: the old vector fails, but the
  bounded rescue produces a fully scoreable candidate;
- `unresolved_after_bounded_rescue`: this bounded test cannot distinguish a
  poor model from a harder fitting problem; and
- `source_fit_already_finite`: the control fit had already succeeded.

This is the last single-iteration numerical attribution milestone. Once its
result selects a usable fitting policy, the next experiment should hold that
policy fixed and run multiple feedback-routed proposal rounds so topology and
interaction revisions can respond to fitting, mechanism, and judge feedback.
