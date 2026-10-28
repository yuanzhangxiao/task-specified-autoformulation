# Phase C baselines with gpt-oss-20b

The LLM baselines ran with gpt-oss-120b on the Jetstream2 hosted service. Our
method's proposer is gpt-oss-20b. This arm reruns every LLM baseline with
gpt-oss-20b, so the comparison also holds with the same model. The user asked
for it on 2026-10-27, with five seeds per method.

The model is served by vLLM on a Jetstream2 GPU VM at the revision our method
pins (`configs/phase_c_construction_v1.json`: `openai/gpt-oss-20b` at
`6cee5e8`). Everything else is the 120b plans'. Each 20b plan is a 120b plan
with only these changed: the model, the endpoint, the seeds of a full plan and
the sentences that name them. `tests/test_phase_c_20b_arm.py` checks this.
SINDy and PySR use no model and are not repeated.

## Plans

| Method | Smoke | Full plan, seeds 0 to 4 |
| --- | --- | --- |
| LLM-SR, expression form | `phase_c_llm_sr_expression_smoke_vm_20b_v1.json` | `phase_c_llm_sr_expression_vm_20b_v1.json` |
| LLM-SR, program form | `phase_c_llm_sr_smoke_vm_20b_v3.json` | `phase_c_llm_sr_vm_20b_v1.json` |
| LLM-ODE, with the description | `phase_c_llm_ode_smoke_vm_20b_v1.json` | `phase_c_llm_ode_vm_20b_v1.json` |
| LLM-ODE, prompt as published | `phase_c_llm_ode_published_prompt_smoke_vm_20b_v1.json` | `phase_c_llm_ode_published_prompt_vm_20b_v1.json` |
| D3 | `phase_c_d3_smoke_vm_20b_v2.json` | `phase_c_d3_vm_20b_v1.json` |

- **Status.** Every plan is proposed, pending review, except D3's smoke, which
  was frozen on 2026-10-08 and never ran. Each smoke is the 120b smoke with
  only the model moved, and so is each full plan, with five seeds. The two
  earlier LLM-SR 20b smokes (`phase_c_llm_sr_smoke_vm_20b_v1` and `_v2`) are
  replaced: they predate the time-ordered table, the program rollout and the
  paper's settings, and never ran.
- **Dry prepares.** On the Mac (D3 on js2) the smokes give 2, 2, 8, 8 and 1
  tasks. Each full plan gives 40 tasks; each LLM-SR form gives 55 searches
  and at most 550,000 samples.

## Settings carried over, and what the smokes check

- **Reasoning effort.** None is sent, as at 120b, so the server's default
  applies.
- **LLM-SR's limit.** Each sample has 4,096 new tokens, the limit measured for
  gpt-oss-120b. The smokes record how often gpt-oss-20b reaches it. If it
  often does, whether to raise the limit for 20b is decided before the full
  plans freeze.
- **LLM-ODE's limit.** It keeps upstream's 1,024 tokens (decided on
  2026-10-21). D3 keeps 8,192.
- **Serving.** The server is the pinned vLLM 0.27.1 image
  (`scripts/jetstream/gpu_server.sh`), with a 32,768-token context, on this
  VM's loopback address only.

## How many searches run at once

LLM-SR resets its weakest islands every four hours of wall-clock time, so a
search's pace changes the search. On one GPU, the searches running at once
share the server's throughput. The number at once is therefore chosen from the
smoke's pace, so that a search runs within the 120b runs' range: 106 to 278
samples an hour for the program form, 230 to 265 for the expression form.

`run_phase_c_tasks.sh` runs each target's search as its own process, as the
hosted launcher does, and seals no more tasks at once than the VM has CPUs.
The expression form runs first, then the program form. LLM-ODE and D3 run
beside LLM-SR; together they generate about 3% of the arm's tokens.

## The VM

1. **Create it in Exosphere.**
   - Size `g3.xl`: one A100 with 40 GB, 32 vCPUs and 120 GB RAM, at 64 SU an
     hour.
   - Image: Ubuntu 24.04, as on js2.
   - Root disk: volume-backed, 200 GB. The default 60 GB cannot hold the vLLM
     image, five seeds of runs and the system.
   - Weights: a 50 GB volume named `af-models`, attached at
     `/media/volume/af-models`.
2. **Set it up.** Copy the public release bundle from js2. Clone the
   repository at the frozen commit and run `gpu_server.sh setup 20b`.
3. **Measure the GPU's speed.** Run `gpu_server.sh start 20b 64`, then
   `scripts/jetstream/vllm_throughput.py --model openai/gpt-oss-20b
   --su-per-hour 64 --concurrency 8 16 32 55 --seconds 300`. It prints
   tokens per second at each concurrency, and the generation hours and SUs the
   arm needs (`WORKLOAD_REQUESTS`: both LLM-SR forms and both LLM-ODE prompts
   at five seeds). That projection is a floor, since it counts generation only.
4. **Run the smokes,** then the full plans, each frozen on sign-off.

Shelve the VM whenever it will sit idle for hours. A shelved VM is not
charged, but resuming it depends on free capacity.

## SU estimate, 2026-10-28

No A100 measurement of gpt-oss-20b exists yet. The only self-hosted figures
are from one H100 serving one sequence at a time, about 200 to 250 tokens a
second. So the throughput below is an assumption, to be replaced by step 3.

What the 120b runs used for two seeds (read on js2):

| Run | Requests | Tokens |
| --- | --- | --- |
| LLM-SR, program form | 55,000 | 794M |
| LLM-SR, expression form | 55,000 | 463M |
| LLM-ODE, with the description | 17,992 | 31M |
| LLM-ODE, prompt as published | 17,998 | 26M |
| D3 | 621 | 8M |

- **Prompt tokens.** Prompts are about a third of these tokens: about 2,900
  per expression request and 5,300 per program request. They are estimated
  from the specification and the two examples each LLM-SR prompt shows.
- **Generated tokens.** At five seeds the 120b rates give about 2.1 billion
  generated tokens. gpt-oss-20b may reason longer: the range below assumes
  1.0 to 1.4 times as many.
- **Throughput assumed.** 2,500 to 5,000 generated tokens a second on the
  A100, with 22 to 55 searches at once.
- **Allowances.**
  - About 10 hours for setup, measurement and smokes.
  - One idle day.
  - 15% of the run time for reruns.

| A100 (`g3.xl`, 64 SU an hour) | Optimistic | Middle | Pessimistic |
| --- | --- | --- | --- |
| Generated tokens | 2.1 billion | 2.5 billion | 2.9 billion |
| Tokens a second | 5,000 | 3,500 | 2,500 |
| Run time | 5 days | 9 days | 14 days |
| SUs, with allowances | 11,300 | 17,500 | 27,000 |

On an H100 (`g5.xl`, 128 SU an hour), if the allocation can launch one, the
SUs would be about the same (12,000, 17,000 and 24,000), with about
two and a half times the throughput assumed. The run would take 2 to 6 days.

CPU SUs are separate:

- **js2** (`m3.large`) costs 384 SU a day while it runs. Seeds 2 to 4 of the
  120b runs, the mechanism tests and test scoring keep it busy until about
  mid-November, about 7,000 SU.
- **SINDy and PySR seeds 2 to 4** need a CPU VM again, as their one-hour
  limits assume dedicated CPUs: an `m3.2xl` (64 SU an hour) for about four
  hours, about 300 SU.

## Limitations

- **Throughput is assumed.** It is measured in step 3, and the projection
  there counts generation only.
- **20b token use per sample is unknown** until the smokes run.
- **The pace of an LLM-SR search depends on how many share the GPU.** It is
  kept within the 120b runs' range, not equal to it, and the hosted runs'
  pace varied too.
- **Transcriptions.** Program-form picks that read the history each need a
  transcription, reviewed before test scoring. At 120b, 11 of 15 picks did.
