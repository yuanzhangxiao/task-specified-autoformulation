# Phase-B frozen-candidate replay pilot

## Purpose

This pilot asks whether historical, already-paid candidate structures can be
reused under the redesigned Phase-B open-loop protocol before purchasing any
new proposer calls. It is a development diagnostic, not a benchmark result.
It uses train and validation data only; test data remain sealed.

Only exact public-contract mappings are allowed in this first replay:

- historical `original_b1` to named canonical Dalla Man T1;
- historical `perturbed_b1` to named perturbed Dalla Man T1.

Historical T2--T4 pools do not exist, and candidates are not silently renamed
across named, obfuscated, CSTR, or alien-device contracts.

## Eligibility audit

| Phase-B cell | Historical artifacts | Compile eligible | Unique eligible structures |
|---|---:|---:|---:|
| canonical named easy | 106 | 78 | 74 |
| canonical named hard | 222 | 90 | 85 |
| perturbed named easy | 66 | 50 | 45 |
| perturbed named hard | 64 | 44 | 43 |

Most ineligible candidates use historical public covariates, especially
`body_weight_kg`, that are intentionally absent from the new contract. They are
excluded rather than adapted.

## Minimal numerical smoke test

The smoke test replayed the two historically best eligible structures per cell
with one start, three function evaluations, fixed-step RK4, and a 30-second
per-fit limit. It made no LLM calls and did not load test data.

| Phase-B cell | Successful / replayed | Best train NMSE | Best validation NMSE |
|---|---:|---:|---:|
| canonical named easy | 1 / 2 | 0.182 | 0.211 |
| canonical named hard | 2 / 2 | 0.267 | 0.602 |
| perturbed named easy | 2 / 2 | 0.129 | 0.0582 |
| perturbed named hard | 2 / 2 | 0.154 | 0.217 |

These numbers are deliberately under-optimized and must not be reported as
Phase-B test performance. Their role is to establish feasibility: historical
structures can be compiled and refitted under true open-loop validation.

The pilot also exposed and fixed a fitter defect: fixed or analytic latent
initial conditions were incorrectly treated as requiring optimization ranges.
The simulator already handled those initializers correctly, so the optimizer
now introduces a latent-initial variable only for an explicit range.

## Decision

Proceed with a larger development-only replay sweep before making new LLM
calls. Use the replay to calibrate numerical budgets and establish a cached
structure baseline. Do not open Phase-B test data until candidate selection and
all numerical settings are frozen. New proposer calls remain necessary for
T2--T4, obfuscated contracts, CSTR, and alien device because there is no exact
historical candidate pool for those cells.

## Staged-budget calibration

A naive larger run with ten function evaluations and a 90-second deadline was
discarded: under four-way laptop contention, some structures that had passed
the smoke test timed out. Consequently, rollout fitting must not be classified
under an oversubscribed wall-clock budget.

The replacement policy screens structures with three function evaluations and
a 60-second limit, then warm-starts a ten-evaluation, 120-second refinement for
the best valid screening candidate. Refinement is accepted only when it is
valid and improves validation NMSE; it can never erase a better screening fit.
Five structures per cell were run sequentially on the local machine:

| Phase-B cell | Valid screens | Screen-best validation NMSE | Frozen validation NMSE | Refinement retained? |
|---|---:|---:|---:|---:|
| canonical named easy | 5 / 5 | 0.0846 | 0.0697 | yes |
| canonical named hard | 5 / 5 | 0.249 | 0.239 | yes |
| perturbed named easy | 5 / 5 | 0.0582 | 0.0582 | no; refinement was worse |
| perturbed named hard | 5 / 5 | 0.194 | 0.194 | no; refinement was worse |

Median screening time was 11--20 seconds per structure. The full set of 247
unique eligible structures therefore requires roughly 1.0--1.5 single-core
hours at the screening budget, plus finalist refinement. It should be run as
isolated CPU jobs (with numerical-library thread counts pinned to one), not as
four contending processes on the laptop. This is a suitable first Delta CPU
array once the ACCESS exchange becomes active.
