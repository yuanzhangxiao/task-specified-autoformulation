# T2-easy breadth pilot

Generate fifteen alternatives with GPT-OSS-120B on the corrected public T2-easy task. Report initial admission, admissions recovered by a correction, unavailable branches and distinct fitting representatives. Shared parameters, training trajectories and the eight-minute forecast-prefix contract travel with the inspection archive. Orion can qualify fits after construction.

| Origin | First response | Evidence and scope |
| --- | --- | --- |
| Candidate 15, fitted signed-action descendant | One request requiring five independent patches | The primary concentration-constrained endpoint, its outward rate at zero insulin, measured and fitted ingestion windows, and the earlier undefined symbol. Each option can replace the insulin RHS or its production process with the supporting declarations it needs. |
| Candidate 21 | One request requiring five independent patches | Orion’s verified training fit, fasting glucose with positive disposal, and the rising post-meal source. A parent-relative patch can change dynamics, processes and eligible initializers. |
| Fresh T2-easy constructions | One independent topology per seed, 100 through 104 | Public task, corrected forcing, training observations and prefix policy. The proposer chooses differential states, algebraic quantities and their dependencies. Each accepted graph receives a separate call for complete functions and latent initialization. |

The first seven requests cover the two fitted parents and five fresh topology seeds. The worker finishes the original one-correction workflow across all fifteen branches before spending a second correction on any branch. It completes that second pass before starting the third. Each assessable, unresolved branch may use up to three corrections across topology and functions combined. An identical rejected option from the same batch shares its sibling’s correction opportunity. An admitted branch stops. A corrected fresh topology still receives its first formula request, including when the third correction resolved its topology. Operational failures and truncated batches are recorded separately.

The campaign-wide ceiling is 57 physical generation requests. This includes two five-option calls, five first topology calls, at most five first formula calls, and at most 45 corrections. Reused v1 requests and interrupted provider reservations count toward the same total. Commit `be80173` contains the one-correction implementation. Its saved outcomes remain in the original ACES result directory.

## Comparing repair depths

The inspection contains separate branch snapshots, chronological call lists, and summaries under `depths/1/`, `depths/2/`, and `depths/3/`. Each summary gives the cumulative physical requests, admitted branches by origin, and distinct representatives. `depths/1/summary.json` is the original stopping rule, which can finish before call 27. The final `summary.json` also reports which models were admitted within the first 27 physical calls, or all calls if fewer were needed. `call-order.json` identifies the source plan of each call.

## Admission and fitting handoff

The same public path, expression, initialization, forcing and qualified training checks apply to the assembled models. Candidate 15 also uses the previously verified depletion witness, spelling normalization and clock-family counterexample. The glucose check identifies models that imply exactly zero net supply at zero meal input, wherever the tested polynomial is defined. Such equations enforce declining glucose while disposal stays positive. The observed training record includes a fasting plateau and positive disposal at every sampled minute. The new gate excludes retention of that targeted structural property. Unsupported algebra or a different glucose readout leaves this particular check unresolved. A nonzero expression can still have a poor sign, magnitude or time course.

The inspection retains every branch and both rejected and accepted equations. Two eligible branches can share a fitting representative if their normalized model families agree or if their expanded balances, readouts, initializers, parameter domains and modeled auxiliary identities match under the implemented checks. Inconclusive comparisons keep distinct representatives. These are conservative counts of model representations. They can overcount physiological mechanisms. No fitting or automatic promotion occurs in this job.

The archive includes the complete models in `results/`, the representative packets in `handoff/`, per-stage prompts and responses in `steps/`, `summary.json`, and a browsable `REPORT.html`. `inputs.json` contains the matched public training rows and both fitted parents, including the saved per-trajectory initial coordinates. Orion can independently replay training NMSE and compare positivity, fasting balance, timing, dose shifts and mechanism compliance under matched fitting resources.

The completed run produced five distinct fitting packets in 39 calls. The equation audit and two-solver training replay identify `c15-4` as the first refit to prioritize. [Results and fitting handoff for Orion](../work/agent-comms/2026-11-08-t2-breadth-results-and-fitting-handoff.md)

## Fitting the five admitted equations

The November 8 fitting campaign uses the 16 public T2-easy training trajectories. It includes all five returned equations and a sixth task with the basal amplitude in `c15-4` fixed to zero. This control recovers Candidate 15's parent dynamics. Each task starts by reproducing Astra's per-target and pooled scores with the supplied shared coefficients and 32 training initial values. The 8-minute observation prefix applies to a future forecast evaluation.

The fitter searches original or algebraically reduced coordinates and returns all coefficients and initial values in the original names. It uses exact affine propagation and a conditional initial-value start for `c15-3` and `c21-2`. Nonlinear trials integrate the supplied continuous-time equations with eight RK4 substeps per minute. Smooth equations use bounded least squares. The clipped `c15-0` uses a bounded, derivative-free Powell search. Public training scales determine the additional starts, including the glucose threshold and slope of the sigmoid.

Each task reserves up to 1,000 whole-training evaluations and 900 numerical seconds for each of two stages. The first minimizes the mean of the three variance-normalized target errors. The second adds a penalty for negative predicted concentrations or disposal rates. Retention uses the unpenalized pooled error and saves both the lowest-error endpoint and the lowest-error endpoint passing the sampled domain check. Two adaptive solvers, Radau and DOP853, must agree on the original equations at every observation. Each input corner splits the integration interval. For the clipped law the original-equation replay additionally uses a two-minute internal step cap and tighter tolerances.

The domain check samples all three targets at 0.125-minute spacing, accepting numerical undershoot of up to `1e-8` times that target's pooled training standard deviation. Peak times, rates at zero insulin, and a 30-minute zero-input extension from the 150 g meal are reported separately. Training ends at minute 300.

### Reproducing the fits on Delta

The verified local results and fitted models are available in the [numerical handoff](../work/agent-comms/2026-11-08-t2-breadth-fitting-results.md). For an independent Delta run, upload the bundle from your local terminal.

```bash
scp "$HOME/Projects/autoformalism/output/t2-breadth-fitting-20261108/t2-breadth-fitting-bundle.tar.gz" \
  yxiao2@login.delta.ncsa.illinois.edu:/projects/bibo/yxiao2/
```

On Delta, unpack the code and construction handoff into one checkout. Retain the checkout until the workers and report finish.

```bash
AF_REPO=$(mktemp -d /projects/bibo/yxiao2/repos/t2-breadth-fitting.XXXXXX)
tar -xzf /projects/bibo/yxiao2/t2-breadth-fitting-bundle.tar.gz -C "$AF_REPO"
(cd "$AF_REPO" && sha256sum --quiet -c SHA256SUMS && bash scripts/hpc/start_t2_breadth_fitting_delta.sh run)
```

This submits a six-element CPU array with concurrency two and a dependent report. In any subsequent Delta login, recover the original checkout and export the fitted coefficients from the completed run.

```bash
export AF_T2_BREADTH_FIT_ROOT=/work/hdd/bibo/yxiao2/phase_c/t2-breadth-fitting-v1
AF_REPO=$(jq -er '.value.repo' "$AF_T2_BREADTH_FIT_ROOT/launcher_execution.json")
AF_LAUNCHER="$AF_REPO/scripts/hpc/start_t2_breadth_fitting_delta.sh"
bash "$AF_LAUNCHER" status
bash "$AF_LAUNCHER" export
bash "$AF_LAUNCHER" inspect
```

From your local terminal, retrieve the portable inspection. It contains the plan, authenticated input models, training observations, search checkpoints and full verification records.

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/t2-breadth-fitting-v1/inspection.tar.gz \
  "$HOME/Downloads/t2-breadth-fitting-inspection.tar.gz"
```

## ACES

Upload `phase-c-t2-breadth-v2.tar.gz` to the group scratch directory. From the ACES login shell:

```bash
cd /scratch/group/p.nairr260351.000/u.yx126462 && tar -xzf phase-c-t2-breadth-v2.tar.gz && cd phase-c-t2-breadth-v2 && sha256sum --quiet -c SHA256SUMS && bash scripts/hpc/start_t2_breadth_aces.sh run
```

One allocation requests two H100s for up to two hours. The v2 launcher reads the original `phase-c-t2-breadth-results-v1` root if it exists and publishes to `phase-c-t2-breadth-results-v2`. It checks ambiguous acknowledgments against the live queue and accounting, including submission time, owner, worker and campaign log paths. The Slurm singleton dependency uses the original job name and waits behind an earlier same-name job even if its ID was lost. An identified active predecessor also gets an explicit job dependency. [Slurm dependency semantics](https://slurm.schedmd.com/sbatch.html#OPT_dependency)

At worker startup, v2 snapshots any original durable call records and verifies each reused prompt, response schema, seed and physical receipt against its slot. It completes missing slots with the v2 namespace. The original model outcomes and scheduler receipts are preserved. Repeating `run` reconciles or returns the existing v2 submission. If the scheduler explicitly rejects a v2 submission, its response is printed. After resolving that condition, `resume` can submit a new wave using the same plan and budget. Use the following commands from any directory.

```bash
AF_LAUNCH=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-t2-breadth-v2/scripts/hpc/start_t2_breadth_aces.sh
bash "$AF_LAUNCH" status
bash "$AF_LAUNCH" inspect
```

Download `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-t2-breadth-results-v2/inspection.tar.gz`. For a terminal allocation with unfinished branches, `bash "$AF_LAUNCH" resume` reuses the frozen plan, cached calls and original allowances. A request interrupted in flight counts against its slot. Inspect an ambiguous scheduler acknowledgement via `squeue`, `sacct` and the saved submission intent.
