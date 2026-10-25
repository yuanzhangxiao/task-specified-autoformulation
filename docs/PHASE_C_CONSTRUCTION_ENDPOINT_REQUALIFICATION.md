# Construction endpoint requalification

25 October 2026. This is an opt-in, training-only numerical follow-up for the
seventh saved T2-easy construction candidate (zero-based model 6). It changes no
model equations, parameter domains, initial-value families, benchmark data or
production defaults. No LLM, GPU, validation or test access is involved.

## What produced the improvement

The returned `construction-timescale-controls-2` archive has SHA256
`bd1a4c8f9d0474ecb879bed4e2a4a8c6714f9905ab36a554799a0a3ef9ff874c`.
Its source and all nine saved terminals were authenticated against their
operation receipts. The tenth task has no numerical operation receipt.

The model has 27 effective fitting coordinates after exact removal of redundant
parameter factorizations. For each choice of six dynamic coordinates, all three
observed trajectories depend affinely on five forcing gains and sixteen hidden
initial values. Bounded least squares solves these 21 inner coordinates together;
the outer optimizer searches only six coordinates. This is conditional profiling,
not treating the hidden trajectories as observed or fitting separate time samples.

Three changes made the different starts usable: project the inner coordinates
before evaluating a new dynamic start; check derivatives using genuinely local
physical perturbations; and try predetermined slow, intermediate and resolved
timescales instead of only continuing the old very fast solution. Two independent
starts reached essentially the same new solution. Their mean training NMSE fell
from 1.042632756 to 0.305843813 (70.7%). The per-target changes were:

| Target | Inherited endpoint | Improved endpoint |
|---|---:|---:|
| Glucose Gp | 0.298979 | 0.317857 |
| Insulin I | 1.840638 | 0.410812 |
| Utilization U | 0.988282 | 0.188863 |

The insulin self coefficient changed from approximately -2.36 million to
-0.0061145, corresponding to a self-decay time of approximately 164 minutes.
This is a fitted model timescale, not recovered physiological ground truth.
Exact affine propagation and independent original-equation Radau integration
agreed to roughly 6e-12 in the reported normalized prediction discrepancy.
No equations were revised or terms pruned to obtain this improvement. This
experiment does not isolate the contribution of each numerical change.

## Why a further numerical check is needed

The old numerical coordinate units remained anchored to the fast initial model.
A unit of 2.36 million is unsuitable for interpreting a rate now near 0.0061.
It distorts relative SVD rank thresholds and scaled gradient magnitudes. The
reported rank five therefore does not establish 22 exactly unidentified
coordinates. We need a new, reproducible local assessment, with unchanged
stationarity and derivative tolerances.

One improved endpoint preceded the last accepted optimizer step. Its exact
73-step accepted-history prefix is available, but the earlier diagnostic
required it to be the last step and discarded the history. The corrected path
uses only the prefix ending at the retained parameter hash. A termination reason
from a later, different point does not transfer to that earlier endpoint.

Finally, an extremely poor but finite projected start can fail strict absolute
solver agreement despite close relative agreement. Search eligibility is now
reported separately from strict endpoint certification. Derivatives must still
qualify before search; only strict independent verification permits retention.

## Frozen next experiment

Three tasks use separate new operation ledgers:

1. Reverify and diagnose the exact retained endpoint from source row 7.
2. Reverify and diagnose the exact retained endpoint from source row 8.
3. Recover source row 9 from its original resolved-timescale start.

The first two allow bounded warm profiling only if the new diagnosis remains
unresolved. The third retains its original six overrides, rather than adopting
the successful endpoints. Every source row, including the absent one, remains
in the input provenance. An absent operation is distinguished from a reserved,
interrupted operation; consumed budgets are never renewed. The pre-operation
wall time of the missing job is unknown and is not reported as zero cost.

Numerical coordinate units are fixed for each local operation at
`max(abs(anchor_coordinate), declared_floor)`. Floors use only the training time
grid, observed channel scales and certified equation roles. Rates use 1/T; gains use the appropriate output / (input * T) scale;
the reciprocal coefficient in `-U/tau_G` uses `T*S_U/S_Gp`; and hidden initials
use their certified state scale. For this
candidate the fixed `I_mem` contribution to `U'` supplies the hidden scale
`S_I_mem = S_U / T`. These are optimizer coordinates, not bounds or scientific
priors. They are recorded with the anchor and never adapted from the current
Jacobian during an operation.

New full-coordinate diagnoses save the residual, Jacobian, gradient, coordinate
units and endpoint identity so their SVD and stationarity summaries can be
recomputed. Original diagnostics remain historical evidence. Numerical rank
and local stationarity are separate from scientific validity and a global
optimum. No structural feedback is automatically released.

Search has a new maximum allowance of 600 seconds per task (including its local
derivative check; the missing start's first projection shares that allowance).
Independent verification has 180 seconds per operation; diagnosis has 600.
Inclusive setup/checkpoint overhead is separately bounded and recorded. The
scheduler requests one CPU, 8 GB and 75 minutes per task, with two concurrent
tasks; the read-only report receives 30 minutes. These scheduler ceilings are
not optimizer targets. Workers print progress before imports, source validation,
locking and numerical operations to make any pre-operation stalls visible.

## Run on Delta

The portable bundle contains authenticated inputs; the old remote campaign does
not need to be modified or copied. Extract the delivered commit-specific bundle
under `/projects/bibo/yxiao2/repos`, enter it, and verify `CONTENTS.sha256`.
Then:

```bash
export AF_PLATFORM=delta
export AF_ACCOUNT=bibo-delta-cpu
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
export AF_EXCLUDE_NODES=cn064
bash scripts/hpc/start_construction_endpoint_requalification.sh run
```

The explicit optional exclusion avoids the node on which both the missing task
and the old read-only report timed out. Their exact failure cause is unproven.
The exclusion is sealed with the launcher identity and must be preserved for
resume. The default result root is
`/work/hdd/bibo/yxiao2/phase_c/construction-endpoint-requalification-v1`;
`AF_ENDPOINT_REQUALIFICATION_ROOT` overrides it. Old `AF_OUTPUT_ROOT` and
`AF_TIMESCALE_CONTROLS_ROOT` variables cannot redirect this campaign.

```bash
bash scripts/hpc/start_construction_endpoint_requalification.sh status
bash scripts/hpc/start_construction_endpoint_requalification.sh inspect
```

`inspect` writes `inspection.tar.gz` under that result root. Scheduler replies
that do not confirm a job ID require reconciliation; automatic blind resubmission
is forbidden. Remote results are pending. The delivered `VALIDATION.json` records
local verification and any limitations; local checks do not establish completion
of this three-task campaign.

## Local saved-endpoint checks

Both improved saved endpoints were reevaluated without optimization. Their NMSE
remained 0.3058438133 and independent affine/Radau verification passed (normalized
prediction gaps about 6.1e-12). The corrected chart yields raw projected gradient
maxima 3.58e-9 and 8.15e-9, and relative numerical rank 27 at the declared 1e-8
threshold. Those calculations establish that the old chart distorted the earlier
rank/gradient summaries; the new rank is still chart-dependent and is not proof
of physical identifiability. Complete uncertainty-qualified derivative assessment
and any warranted endpoint polishing remain the purpose of the Delta run.

The portable preparation check also identified floating-point cost summation
that depended on JSON dictionary order. Historical operations are reconstructed
in their original declared stage order before comparing terminal digests. This
preserves old receipts exactly, rather than changing their recorded costs.
