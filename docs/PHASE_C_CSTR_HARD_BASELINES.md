# External baselines on Phase C CSTR-hard

On 2026-11-11 the user asked for the external baselines to be scored with the
reactor intervention study's response metrics: direction, peak time and
response RSE under interventions the models never saw. That study scored our
method on CSTR-hard
(`phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1`,
release `phase-c-development-2`), where only T is measured. The baselines had
run only on CSTR-easy, where C and Tj are supplied measurements, so they now
run on CSTR-hard with seeds 0, 1 and 2, the study's three seeds. The user chose
all nine baselines, with SINDy, PySR and D3 first.

## Plans

Each plan is a frozen Phase C plan with only its cell, its seeds and the wording
that describes them changed (`tests/test_phase_c_cstr_hard_plans.py`). All six
were frozen on 2026-11-11 on the user's sign-off.

| Plan (`configs/…`) | Frozen source | Tasks |
|---|---|---|
| `phase_c_cstr_hard_public_baseline_jetstream_cpu_v1` | `phase_c_public_baseline_jetstream_cpu_v1` | 6 (3 SINDy, 3 PySR) |
| `phase_c_cstr_hard_d3_hosted_120b_v2` | `phase_c_d3_hosted_120b_v2` | 3, at most 117 requests |
| `phase_c_cstr_hard_llm_ode_hosted_120b_v1` | `phase_c_llm_ode_hosted_120b_v1` | 3, at most 2,400 requests |
| `phase_c_cstr_hard_llm_ode_published_prompt_hosted_120b_v1` | `phase_c_llm_ode_published_prompt_hosted_120b_v1` | 3, at most 2,400 requests |
| `phase_c_cstr_hard_llm_sr_hosted_120b_v1` | `phase_c_llm_sr_hosted_120b_v1` | 3 searches, 30,000 samples |
| `phase_c_cstr_hard_llm_sr_expression_hosted_120b_v1` | `phase_c_llm_sr_expression_hosted_120b_v1` | 3 searches, 30,000 samples |

Dry prepares on the Mac against the released public CSTR-hard files give these
counts. HDTwinGen and spec-informed SINDy run on the nine-case branch's code and
get their own plans.

## Code

`phase_c_baselines.INTERVENTION_CELLS` adds CSTR-hard to the cells a baseline
plan may name (`BASELINE_CELLS`). The eight-cell roster
(`research.phase_c_inputs.ROSTER`) and everything else that reads it are
unchanged. On js2 the cell's four public files sit beside the eight roster
cells in `~/af/release/phase-c-public-v2/public`; the release receipt already
lists them, so the loader verifies them as it verifies every other cell.

## Order on js2

1. SINDy and PySR, which need the VM to themselves, then D3. These come first
   because the user wants them first.
2. LLM-SR in both forms and LLM-ODE under both prompts.
3. HDTwinGen and spec-informed SINDy.
4. The held D3 and LLM-ODE seeds 2 to 4 ([PHASE_C_SEEDS_2_4.md](PHASE_C_SEEDS_2_4.md)),
   from the commit that froze them.

`scripts/jetstream/run_phase_c_cstr_hard.sh` runs items 1 and 2: SINDy and
PySR, then D3, each alone, then LLM-SR and LLM-ODE side by side, each step once
the previous one has ended. From a clean checkout of this branch on js2:

```bash
nohup bash scripts/jetstream/run_phase_c_cstr_hard.sh > ~/af/runs/cstr-hard-chain.log 2>&1 < /dev/null &
```

## HDTwinGen and spec-informed SINDy

These two ran first on the nine Phase B cases, whose staged inputs their
runners read. A plan may now name the Phase C release instead
(`inputs_protocol: phase-c-development-2` with its `release_summary_sha256`);
its cell is then read from the release as Phase C D3 reads it, with that cell's
public context. The nine-case plans and runs are unchanged
(`tests/test_phase_c_cstr_hard_hybrid_plans.py`).

| Plan (`configs/…`) | Frozen source | Tasks |
|---|---|---|
| `phase_c_cstr_hard_hdtwingen_hosted_120b_v1` | `phase_b_nine_case_hdtwingen_hosted_120b_v1` | 3, at most 117 requests |
| `phase_c_cstr_hard_spec_sindy_hosted_120b_v1` | `phase_b_nine_case_spec_sindy_hosted_120b_v1` | 3, 3 requests |

Only the cell and where it is read from differ; the nine-case plans already
ran seeds 0, 1 and 2. From a clean checkout of branch
`phase-c-cstr-hard-hybrid` on js2:

```bash
bash scripts/jetstream/run_phase_b_hosted.sh hdtwingen configs/phase_c_cstr_hard_hdtwingen_hosted_120b_v1.json phase-c-cstr-hard-hdtwingen-hosted-120b-v1 3
bash scripts/jetstream/run_phase_b_hosted.sh spec_sindy configs/phase_c_cstr_hard_spec_sindy_hosted_120b_v1.json phase-c-cstr-hard-spec-sindy-hosted-120b-v1 3
```

Scoring is a separate step: every pick is frozen first, failures included, and
then scored on the study's held-out intervention schedules.
