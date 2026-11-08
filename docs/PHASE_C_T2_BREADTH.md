# T2-easy breadth pilot

Generate fifteen alternatives with GPT-OSS-120B on the corrected public T2-easy task. Report initial admission, admissions recovered by a correction, unavailable branches and distinct fitting representatives. Shared parameters, training trajectories and the eight-minute forecast-prefix contract travel with the inspection archive. Orion can qualify fits after construction.

| Origin | First response | Evidence and scope |
| --- | --- | --- |
| Candidate 15, fitted signed-action descendant | One request requiring five independent patches | The primary concentration-constrained endpoint, its outward rate at zero insulin, measured and fitted ingestion windows, and the earlier undefined symbol. Each option can replace the insulin RHS or its production process with the supporting declarations it needs. |
| Candidate 21 | One request requiring five independent patches | Orion’s verified training fit, fasting glucose with positive disposal, and the rising post-meal source. A parent-relative patch can change dynamics, processes and eligible initializers. |
| Fresh T2-easy constructions | One independent topology per seed, 100 through 104 | Public task, corrected forcing, training observations and prefix policy. The proposer chooses differential states, algebraic quantities and their dependencies. Each accepted graph receives a separate call for complete functions and latent initialization. |

The first seven requests cover all three origins before any correction. Each distinct, assessable rejected patch can receive one correction. An identical rejected sibling from the same parent shares the first opportunity. Each fresh seed has one correction across topology and functions combined. A rejected fresh function can be returned with a revised graph. A failed or truncated batch is recorded as unavailable. A response with five slots is assessed slot by slot. A formula request following an accepted topology is ordinary construction. The hard maximum is 27 generation requests across all allocations and resumes.

The single correction is a budget choice for this pilot. In the October replay, 16 of 60 initial options passed, and 8 of 39 selected follow-ups passed. Those observations measure yield through one correction. A later depth experiment can start from the saved rejected branches and compare the extra yield from corrections two and three.

## Admission and fitting handoff

The same public path, expression, initialization, forcing and qualified training checks apply to the assembled models. Candidate 15 also uses the previously verified depletion witness, spelling normalization and clock-family counterexample. The glucose check identifies models that imply exactly zero net supply at zero meal input, wherever the tested polynomial is defined. Such equations enforce declining glucose while disposal stays positive. The observed training record includes a fasting plateau and positive disposal at every sampled minute. The new gate excludes retention of that targeted structural property. Unsupported algebra or a different glucose readout leaves this particular check unresolved. A nonzero expression can still have a poor sign, magnitude or time course.

The inspection retains every branch and both rejected and accepted equations. Two eligible branches can share a fitting representative if their normalized model families agree or if their expanded balances, readouts, initializers, parameter domains and modeled auxiliary identities match under the implemented checks. Inconclusive comparisons keep distinct representatives. These are conservative counts of model representations. They can overcount physiological mechanisms. No fitting or automatic promotion occurs in this job.

The archive includes the complete models in `results/`, the representative packets in `handoff/`, per-stage prompts and responses in `steps/`, `summary.json`, and a browsable `REPORT.html`. `inputs.json` contains the matched public training rows and both fitted parents, including the saved per-trajectory initial coordinates. Orion can independently replay training NMSE and compare positivity, fasting balance, timing, dose shifts and mechanism compliance under matched fitting resources.

## ACES

Upload `phase-c-t2-breadth-v1.tar.gz` to the group scratch directory. From the ACES login shell:

```bash
cd /scratch/group/p.nairr260351.000/u.yx126462 && tar -xzf phase-c-t2-breadth-v1.tar.gz && cd phase-c-t2-breadth-v1 && sha256sum --quiet -c SHA256SUMS && bash scripts/hpc/start_t2_breadth_aces.sh run
```

One allocation requests two H100s for at most two hours. A repeated `run` returns the existing submission. Use the following commands from any directory.

```bash
AF_LAUNCH=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-t2-breadth-v1/scripts/hpc/start_t2_breadth_aces.sh
bash "$AF_LAUNCH" status
bash "$AF_LAUNCH" inspect
```

Download `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-t2-breadth-results-v1/inspection.tar.gz`. For a terminal allocation with unfinished branches, `bash "$AF_LAUNCH" resume` reuses the frozen plan, cached calls and original allowances. A request interrupted in flight counts against its slot. Inspect an ambiguous scheduler acknowledgement via `squeue`, `sacct` and the saved submission intent.
