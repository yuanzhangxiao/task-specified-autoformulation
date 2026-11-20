# Graph-based numerical critique of fitted equations

The critic receives a public task, the executable parent and saved training endpoints.
It asks where the free rollout fails, which measured signals carry that error to other
outputs, and whether the fitting evidence warrants an equation investigation. Both
feedback arms use the same measurements. A hosted GPT-OSS-120B supplies one additional
review per comparison parent.

## Frozen experiment

Use c15-2 to develop and reproduce the earlier glucose and insulin substitution.
The primary evaluation contains the other three original candidates in the November 3
targeted fitting roster, 10, 12 and 14. Candidates 10 and 14 share their equations
and have different parameter domains. All three use the same 16 public T2-easy
training records. Candidates 10 and 14 contribute two fixed-saturation trials.
Candidate 12 contributes two freely fitted starts. The separately reported c15-3
control has completed short- and long-memory fits and was previously studied.
Freshness here concerns application of the frozen critic.

The development run produced freeze SHA-256
`aabb9d386ba986038b25a24725897a1b4eee81b9b933aa6389b1c9764e0e558b`
before any fresh-parent probe ran.

1. Replay the original equations, physical coefficients, fitted initial values and
   public input contract using Radau and DOP853. Compare the saved training NMSE
   and the two solvers' outputs. Retain failure reasons and charged numerical work.
2. For every nonconstant target, trajectory and elapsed-time third with two or
   more samples, rank the absolute error in the end-to-start change divided by
   the pooled training standard deviation. Choose the largest under a fixed tie
   rule. A score below 0.05 supplies no localized change finding. This statistic
   can miss offsets and within-window excursions.
3. Trace the validated state, process and readout graph. On the selected window
   replay each eligible identity-observed state as a piecewise-linear recorded
   driver. Integrate all remaining states from their fitted values with the same
   coefficients. Compare window RMSE and terminal absolute error on other reachable
   outputs. Each reported path gives a shortest syntactic connection. A clamp can
   act through several paths, including an instantaneous readout. Signed RHS
   integrals for directly observed winning states additionally require agreement
   across solvers, temporal coarsening and balance closure.
4. Across starts, compare the winning trajectory, interval, target and error
   direction. A repeatable directional effect needs a change above 5% in window
   RMSE and a consistent change in terminal error. The route to conditional
   equation investigation additionally requires two unrestricted, unexhausted
   fit endpoints. The alternative routes ask for further fitting, retain an
   accurate model or request missing evidence. The thresholds govern this pilot.
5. Send the deterministic feedback, evidence identifiers, equations and public
   task to the critic. The review names one action, an alternative and an
   observable way to disconfirm it. Compare its guidance with the unchanged
   deterministic feedback. Check the meaning of claims separately from the
   existence of their citations. Judge the usefulness of an actual model edit
   in a subsequent matched refitting experiment on reserved trajectories.

## Reproduce the comparison

From the project checkout with the previously downloaded fitting artifacts,
run the following commands in order. `scripts/graph_critic_parents.py` lists
the input paths and pinned digests. The fresh stage requires the completed
development replay and its frozen rule, code and model identities.

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
AF_GRAPH_RUN=output/graph-critic-paired-v1
.venv/bin/python scripts/run_graph_critic_probes.py freeze --root "$AF_GRAPH_RUN"
.venv/bin/python scripts/run_graph_critic_probes.py run --root "$AF_GRAPH_RUN"
.venv/bin/python scripts/compare_graph_critic_feedback.py --root "$AF_GRAPH_RUN" --prepare-only
.venv/bin/python scripts/compare_graph_critic_feedback.py --root "$AF_GRAPH_RUN" --hosted-js2
```

The final command uses the existing `js2` SSH alias to reach Jetstream's hosted
model by HTTPS. All artifacts and the request cache are local. There is at most
one generation per parent, including failures and interrupted requests. The first
three commands require no provider. Repeating the stages on the same root reuses
completed work. The resulting `comparison/result.json` reports the three primary
reviews separately from the transfer control. Each parent directory contains the
paired feedback, exact request, response status, duration and reported token usage.
