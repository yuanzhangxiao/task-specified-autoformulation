# Dalla demonstration rescue implementation

The requested six-task CPU campaign is implemented. Its runbook is
`docs/DALLA_DEMONSTRATION_RESCUE.md`; the CLI is `scripts/dalla_rescue.py` and the
submission entry point is `scripts/submit_dalla_rescue.py`.

It covers Brief R9, Full R4 with two starts, Full R13, and the unchanged/corrected
T1-hard delay pair. The new `collocation-rescue-v1` allocation is 300 seconds
collocation plus 900 seconds refinement, with 60-second recovery screens. It
preserves all old profiles and uses the existing solver and paired pruning rule.
Every seed is replayed, retained against a worse refit, and exported with the
pre/pruning/control alternatives. No intervention data enter selection.

The input packet is prepared and all six models compile and pass their hard
public checks. Its content identity is
`817598e2f0b8616618f0d09cf71dad56e05e88a105c678531446140b8ff0159f`.
The corrected patch creates `tau_meal_rescue` and a fitted M initializer; all
existing parameter declarations remain unchanged. The correction reuses the
saved proposal and makes no new LLM call.

Verification: 65 targeted tests passed; the synthetic smoke performed three real
fits, selected pruning, and resumed unchanged. The full suite was interrupted
after 677 passes / five skips without a reported failure; it was not completed.
Changed Python files pass Ruff. The 37 whole-tree Ruff findings are in unrelated
`analysis/claude/` files. No Dalla numerical fits were run locally.

The source/data upload archive will be supplied with the commit in the user
response. It needs no GitHub login on ACES. Submission uses group scratch, the
existing Python environment, one CPU per task, three tasks concurrently, and no
GPU. Uncertain scheduler submissions can be adopted only after job verification.
The report exports `models.json` for the next fitted-equation and intervention
inspection. A positive intervention example remains to be established.
