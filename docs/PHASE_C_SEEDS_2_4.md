# Seeds 2 to 4 of the external baselines

On 2026-10-27 the user asked for three more seeds of every method, to reach the
five the draft promises, LLM-SR's expression form first. That form's plan, run
and reseal are recorded in [PHASE_C_LLM_SR.md](PHASE_C_LLM_SR.md). On
2026-11-03 the user asked for the other methods next. The gpt-oss-20b arm
waits until there is a GPU allocation.

## Plans

Each plan is a frozen seeds 0 and 1 plan with seeds 2, 3 and 4.
`tests/test_phase_c_seeds_2_4.py` checks that the only other difference is
the wording that describes the seeds (and, within a budget, only its
rationale). All five were proposed in `a40df71` and frozen on 2026-11-03 on
the user's sign-off, with SINDy and PySR to run on js2 first; no setting
changed.

| Plan (`configs/…_seeds_2_4.json`) | Seeds 0 and 1 ran from | Tasks | At most |
|---|---|---|---|
| `phase_c_d3_hosted_120b_v2` | `25814f3` | 24 | 936 requests |
| `phase_c_llm_ode_hosted_120b_v1` | `7cb7088` | 24 (33 searches) | 26,400 requests |
| `phase_c_llm_ode_published_prompt_hosted_120b_v1` | `7cb7088` | 24 (33 searches) | 26,400 requests |
| `phase_c_llm_sr_hosted_120b_v1` | `231a97f` | 24 (33 searches) | 330,000 samples, about 82,500 requests |
| `phase_c_public_baseline_jetstream_cpu_v1` | `d74a223` | 48 (24 SINDy, 24 PySR) | no model calls |

Each method runs what its seeds 0 and 1 ran:

- **D3** and **LLM-ODE**: their code is unchanged since those commits. Of what
  D3 imports, only the model-list check has changed, in how it names an
  endpoint that cannot answer.
- **LLM-SR, program form**: the 33 specifications LLM-SR is given are byte for
  byte those of `231a97f`, and the search code has since gained only the
  expression form's switch, off in this plan. The sealing is the current one,
  as in the expression form's reseals.
- **SINDy and PySR**: their runners, plan, launcher and setup are unchanged.
  The plan now also accepts a status of "proposed_pending_review", which its
  freeze step refuses, so a proposed plan can be reviewed but not run.

Dry prepares on the Mac give the counts above, and every LLM-ODE and LLM-SR row
is the frozen plan's apart from its seed. D3's prepare records its PyTorch
version, which the Mac lacks; its rows are the eight cells times the seeds.

## Where and in what order

Everything runs on js2 against the hosted gpt-oss-120b, one method at a time,
as seeds 0 and 1 ran. Every method has wall-clock limits that a busier VM could
make bind more often: D3's 120 s a trajectory, LLM-ODE's 30 s a trajectory when
it chooses, LLM-SR's 30 s a sample and 60 s a seal rollout, and SINDy's and
PySR's whole-task limits. SINDy and PySR need the CPUs to themselves, and
their launcher refuses to start beside another of our runs.

| Order | Run | Seeds 0 and 1 took | Expected here |
|---|---|---|---|
| 1 | SINDy and PySR, one task at a time on js2's 16 CPUs | 1.5 h on js2b, four tasks at a time on 64 CPUs | 2.5 to 7 h, after a one-time setup |
| 2 | LLM-SR, program form, 33 searches at once | 1.5 to 4 days a search, with outages | one to four days |
| 3 | D3, four tasks at a time | 2.8 h | about 4 h |
| 4 | LLM-ODE, both campaigns side by side, eight tasks each | 7.6 h and 10.5 h | about 11 h and 16 h |

SINDy and PySR go first because they are short and need the VM alone. LLM-SR
goes next because it is the longest, and its picks that read the history then
need transcription and review, which can proceed while D3 and LLM-ODE run.
A second CPU VM could instead run SINDy, PySR, D3 and LLM-ODE beside LLM-SR,
saving about a day at the cost of its allocation.

From a clean checkout at the commit that freezes the plans:

```bash
bash scripts/jetstream/setup_classical.sh
bash scripts/jetstream/run_phase_c_classical.sh configs/phase_c_public_baseline_jetstream_cpu_v1_seeds_2_4.json phase-c-classical-seeds-2-4
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_hosted_120b_v1_seeds_2_4.json phase-c-llm-sr-hosted-120b-v1-seeds-2-4 33
bash scripts/jetstream/run_phase_c_hosted.sh d3 configs/phase_c_d3_hosted_120b_v2_seeds_2_4.json phase-c-d3-hosted-120b-v2-seeds-2-4 4
bash scripts/jetstream/run_phase_c_hosted.sh llm_ode configs/phase_c_llm_ode_hosted_120b_v1_seeds_2_4.json phase-c-llm-ode-hosted-120b-v1-seeds-2-4 8
bash scripts/jetstream/run_phase_c_hosted.sh llm_ode configs/phase_c_llm_ode_published_prompt_hosted_120b_v1_seeds_2_4.json phase-c-llm-ode-published-prompt-hosted-120b-v1-seeds-2-4 8
```

## Limitations

- **A different CPU VM for SINDy and PySR.** Seeds 0 and 1 ran on js2b, an
  m3.2xl since deleted. js2 is an m3.xl with the same 16 CPUs per task, and
  their environment is built on it anew. Its setup records the package and
  Julia versions; any difference from js2b's record is reported.
- **SINDy ignores the seed.** Its three new runs should repeat seed 0's results.
  They are kept so the plan differs from the frozen one only in its seeds, and
  they take about an hour.
- **LLM-ODE is unseeded**, as upstream is: its repetitions are labels.
- **The hosted service sets the pace.** LLM-SR resets its weakest islands on a
  four-hour clock, so a faster or slower service changes how many resets a
  search sees. The expression form's seeds 2 to 4 ran about twice as fast as
  its seeds 0 and 1.
- **Test scoring waits** for the shared test path and, for LLM-SR's program
  form, for the new picks' transcriptions and their review.
