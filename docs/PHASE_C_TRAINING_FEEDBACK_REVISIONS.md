# Phase C: training-feedback construction revisions

## Execution order update, 21 October 2026

The user has selected fitting qualification **before** residual-driven proposal
revision. This implemented opt-in pilot accepts budget-limited endpoints and
does not enforce a local optimization-adequacy gate. Hold new launches for this
comparison until unchanged-parent fitting is qualified, then freeze versioned
inputs/gating rather than relabeling the old receipts. Existing saved campaigns
remain unchanged. See `PHASE_C_CONSTRUCTION_FITTING_HANDOFF.md`, especially
"Fitting-first qualification and division of responsibilities", for exact gain
reduction, scoped affine fitting and runtime/judge responsibilities. This is an
execution-plan update, not an implemented scheduler or runtime block.

## Purpose and bounded experiment

The first fitted construction cohort contains eight T2-easy models, from two
topology seeds and eight accepted function/initialization proposals. All eight
searches exhausted their 600-second allowances. Poor error at a saved endpoint
does not distinguish inadequate model structure from incomplete optimization.
Orion's independent fitting work addresses the latter. This opt-in construction
experiment tests whether the proposer can make a justified dependency/formula
revision from public requirements and verified training residuals.

Use all eight parents, each with three revision seeds, without ranking or
preselecting by error. Each of the 24 episodes requests one coherent revision;
up to two additional calls repair invalid replies. The maximum is 72 physical
generation requests and 2,359,296 charged tokens. An unchanged-model response is
allowed when the evidence does not justify an edit. A model that lacks a checked
public path cannot qualify merely by returning no edits.

This is a T2-easy diagnostic, not a new multi-benchmark performance campaign.
It launches no fitting, numerical rollouts, test evaluation, model promotion,
or automatic follow-up. Historical campaigns and their costs are unchanged.

## Dependency and formula edits together

The original interaction stage remains a fixed-topology construction protocol.
The new `construction-feedback-revision-1` protocol explicitly permits revisiting
dependencies. For example, a process initially declared as `k*I` with sources
`[I]` may become `k*I*Gp` with sources `[I,Gp]`. The proposer supplies its scientific
hypothesis, cited training/public evidence, and predicted observable effect.
Multiple arguments and nonlinear expressions are permitted, not mandatory.

An edit replaces the **complete** existing differential RHS or algebraic process
definition. All edits in a reply apply together or are rejected together. Exact
expression symbols must match the declared sources after separating parameters.
New parameters use the existing trusted role-to-domain/start mapping. The runtime
does not invent a scientific dependency or silently change a parameter role.

For this milestone the variable roster, observation mappings, constraints,
parameter scopes, and complete scoped initialization graph remain fixed. A
shared-process definition is edited once for its consumers; checks prevent an
edit to that definition from silently deleting an existing shared use. Changes
requiring new states or different initialization are reported as a broader
follow-up rather than executed in this pilot. This separation keeps the
initialization interface identical to the one Orion is qualifying.

## Evidence and acceptance

The training projection authenticates the original fitting campaign, the
saved-parameter verification supplement, and the returned construction archive.
It contains only the training split, the public brief, the fixed prefix/forcing
policy, the saved candidate and fitted parameters, bounded residual statistics,
and provenance hashes. Neither validation arrays nor scores enter the bundle or
the proposer request. A strict positive schema rejects extra evidence fields.

Seven endpoints have independent training verification; one timed out and has
an explicit unavailable status instead of a substitute error score. Verified
endpoints include the three largest absolute normalized training residuals per
target with exact trajectory/time/value witnesses. The original budget-limited
stop reason is visible in every request. Replay agreement verifies numerical
execution at saved parameters; it does not certify predictive quality or an
optimization optimum. These residuals describe full training reconstruction
using fitted training initials, not held-out forecasting after the prefix.

The restricted expression parser, typed compilation, scoped initializer
validation, algebraic cycle checks, exact dependency declarations, and public
reachability checks run before export. Public-to-internal variable aliases come
from authenticated construction handoffs. In particular, a modeled `Uii` may
appear as `af_internal_aux_1`; the checker must preserve that identity rather
than demand a new measured-input edge. An initial-time dependency alone is not
an ongoing dynamic path. Unknown identity evidence is not a passed check.

These static checks do not establish nonzero influence, adequate delay, correct
signs, conservation, mechanism compliance, or improved fit. A syntactic path
may have a fitted coefficient of zero. Later independent mechanism evaluation
must be reported separately from eligibility and NMSE.

## Bookkeeping and fitting handoff

Each request includes one public task, one assembled model, training evidence,
and (when needed) a specific rejected-edit diagnostic. Repeated invalid replies
are identified. Rejected transactions leave the parent unchanged. Every physical
call, measured token reservation and response is recorded. Resume reuses cached
calls without renewing budgets. Terminal results are reconstructed from the
exact requests/replies, executable source, and sealed inputs before reporting.

The output retains eight unchanged controls and every episode, including
retained, rejected, unavailable, and pending outcomes. Accepted children export
`construction-revision-fit-handoff-1` packets containing a `PrefixFitProblem`
and the original fixed prefix policy. A missing or failed episode is not dropped
from the denominator. No validation-based child selection occurs here.

The next comparison should fit parents and accepted children under the same
fitter version, numerical settings, starts/budgets, and information-access
policy. Report paired error changes, numerical failures, complexity, and
independent mechanism compliance. Use medians and unscaled MAD for NMSE, and
mean (SD) for compliance. Improvements from fitting an unchanged parent and
from changing its model are distinct experimental contrasts, not an additive
percentage attribution of all error.

## Operation

The portable delivery contains a verified training-only input projection and
offline Harmony vocabulary. `run_revision.sh run` prepares an immutable output
root and submits one H100 job (8 CPUs, 64 GiB, two hours; 90-minute worker).
Compiler caches use node-local job storage. `status`, `report`, and `inspect`
never initiate generation. `inspect` creates `inspection.tar.gz`.

Set `AF_REVISION_ROOT` to override this campaign's output directory. The launcher
ignores an inherited generic `AF_OUTPUT_ROOT` (which may still name an earlier
fitting campaign), then exports its resolved revision directory to the shared
worker. Existing fitting inputs use a different seal format; a revision reader's
`artifact digest differs` at a fitting directory does not establish corruption.
Do not delete or rewrite those inputs. For the original `3627487` bundle, which
predates this launcher fix, explicitly set both `AF_REVISION_ROOT` and
`AF_OUTPUT_ROOT` to `phase-c-construction-revision-v1` for run/status/inspect.

The submitter records intent before submission and refuses another submission
wave after an ambiguous scheduler reply. Reconcile `squeue`/`sacct` and the saved
intent instead of blindly submitting again. The worker itself supports
deterministic continuation with the same plan and recorded budget. Allocation
expiry leaves pending work visible; it does not launch another job.

The exact initial requests and response schema are generated by the `preview`
command into `PROMPTS.html`; its marked repair example is synthetic. Review
artifacts are generated outputs, not source-controlled experiment results.
