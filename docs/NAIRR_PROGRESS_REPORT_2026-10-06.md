# Autoformalism progress report and allocation extension request

NSF NAIRR Pilot | Texas A&M ACES | 6 October 2026

**Request.** We request a six-month extension of our TAMU ACES allocation, from
4 November 2026 through 4 May 2027, to complete controlled evaluations and develop
the next stage of our research. Approximately 5 percent of the allocation has
been consumed. We request additional time to use the remaining allocation, with
no increase in awarded service units.

**Research progress.** Autoformalism converts scientific task specifications and
partially observed time-series data into executable dynamical models. We have
implemented staged variable and equation construction, shared processes across
equations, latent-state initialization, parameter fitting, deterministic validity
checks, LLM scientific critique, model revision and pruning. Cached model calls,
checkpointed experiments and versioned artifacts support reproducibility. We
have submitted a research manuscript and obtained baseline and ablation results,
including comparisons with SINDy, PySR, D3 and a GPT-based modeling agent. The
main component campaign was designed around nine benchmark cases and 160 search
lineages; campaign closeout and corrected evaluations remain in progress.

**Scientific findings.** Our studies cover glucose–insulin physiology, chemical
reactor dynamics, a synthetic nonlinear device and stormwater storage basins.
Equation inspection showed that good prediction error or graph validity alone
does not establish mechanism recovery. We therefore developed independent
mechanism assessments and now examine both predictive accuracy and scientific
behavior. Benchmark audits identified input-pulse and reference-trajectory
defects, leading to separately versioned development data. Current studies
isolate model construction from fitting, inspect proposer prompts and responses,
and test numerical scaling, initialization and solver reliability. These results
have clarified limitations that require controlled follow-up rather than simply
larger runs.

**Utilization and operational constraints.** GPU scheduling delays and file-count
quota exhaustion interrupted experiments. Environment and runtime debugging also
consumed development time. Larger campaigns were configured for parallel GPU
proposal and CPU fitting workers, but recent diagnostic pilots deliberately used
small batches with sequential construction. Some complementary work ran on Delta
and Jetstream2 and therefore did not consume ACES units. The low allocation
consumption consequently reflects both operational constraints and our staged
development process; we do not attribute it solely to GPU availability. Compact
traces and cache archiving address file-count pressure. We will expand independent
single-GPU workers and CPU arrays within account limits as protocols stabilize,
tracking queue delays, completed work per allocation and failure rates.

**Plan during the extension.** In the first two months, we will complete campaign
closeout, qualify construction and fitting on corrected development benchmarks,
and expand parallel execution where independent work is ready. In months three
and four, we will run matched baseline and component evaluations, broaden domain
coverage, and evaluate frozen models on held-out data after development choices
are fixed. In months five and six, we will pilot discovery under fallible
specifications: distinguishing unsupported scientific assumptions from
insufficient observations or unsuccessful fitting, proposing minimal revisions,
and identifying informative follow-up interventions. We will release qualified
code, benchmark protocols and evaluation artifacts and prepare follow-up
publications. Continued ACES access will support reproducible open-weight LLM
inference and controlled replication while complementary resources handle
suitable fitting and orchestration workloads.

We appreciate NAIRR and Texas A&M HPRC support for this research.
