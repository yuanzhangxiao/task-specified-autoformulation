# Benchmark appendix revision

## Scope

Revised `paper_to_revise/sections/appendix_benchmarks.tex` for the current
nine-case study: four T1-easy dynamics/naming combinations, canonical named
T1-hard and T2-easy/hard, named CSTR-easy, and functional alien-device easy.
Removed unused T3/T4 task descriptions and the medium tier. No benchmark data,
public prompt, generator, or experimental configuration was modified.

The old appendix also contained obsolete judge, baseline, and experimental
settings (a four-category judge, five seeds, 50 rounds, and one-trajectory
splits). Those paragraphs were removed from this benchmark appendix; the
methodology and experimental sections own those settings.

## Corrections verified against implementation

- T1 predicts plasma glucose **mass** `Gp`, not concentration `G`. Easy supplies
  `EGP,Uii,E,Gt`; hard still supplies `Gt`.
- The actual T1 clause requires causal meal influence but does not prescribe
  three gastrointestinal compartments, a named `Ra`, a particular latent count,
  or a generated `Gt` state. Do not retrospectively label omitted optional
  mechanisms as specification failures.
- T2 predicts `Gp,I,U` in easy and `Gp,I` in hard. Imposed insulin forcing is an
  external input; future observed `I` is not a supplied auxiliary. Hard supplies
  `Uii`, while disposal remains a required internal mechanism.
- The T1 dynamics perturbation modifies both plasma/tissue exchange laws using
  the frozen tanh multipliers (0.35 and -0.25, 20% basal scales). It does not
  perturb gastric emptying or insulin action.
- CSTR easy predicts `T`, supplies `C,Tj`, and receives `Cf,Tf,Tjf`. Its explicit
  requirement separates transport, reaction heat, and jacket exchange; extra
  hidden states are not mandatory under this availability contract.
- Alien easy requires input-driven memory. Its private reference has five
  internal states plus a dynamic output, two of those internal states exposed
  as telemetry; the public prompt does not reveal the dimension or demand the
  complete hard-tier nonlinear-feedback architecture.
- All designs use 16/4/6 trajectories; horizons/sample intervals are 300/1 for
  Dalla Man, 30/0.1 for CSTR, and 60/0.1 for the alien device.

Reference dynamics are explanatory, not additional information provided to
proposers. The appendix does not claim unique recovery of private coordinates.

## Known limitation retained

The historical CSTR reference-pulse defect was confirmed in
`work/agent-comms/2026-09-22-codex-cstr-reference-pulse-ground-truth.md`.
The current generator still integrates a whole horizon without explicit forcing
boundaries. The appendix distinguishes the intended physical task from the
validity of affected stored trajectories; it does not claim a corrected release
or matched reruns. No new benchmark experiments or saved test-array reads were
performed here; existing regression tests exercise the simulator separately.

## Sources

- `configs/final_component_campaign_v1.json`: exact evaluated roster.
- `src/autoformalism/benchmarks/phase_b_public.py`: rendered public prompt
  requirements, channel roles, units, and common six-section scaffold.
- `src/autoformalism/benchmarks/phase_b_generation.py`: schedules, splits,
  horizons, CSTR and alien reference laws.
- `src/autoformalism/benchmarks/phase_b_gates.py`: training-only telemetry
  selection and Dalla Man observability definitions.
- `src/autoformalism/rebuttal/dalla_man.py`: reference mass/concentration
  conventions, meal jumps, insulin forcing, and exchange perturbations.
- `data_raw/benchmark5_anonymous_nonlinear_process/private/system_specification.json`
  and `data_raw/benchmark6_alien_device/private/selected_system_spec.json`:
  existing reference specifications inspected solely for paper documentation.

## Preview

From the project root:

```bash
mkdir -p tmp/pdfs/benchmark-appendix output/pdf
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=tmp/pdfs/benchmark-appendix \
  -jobname=autoformalism-benchmark-appendix \
  docs/AUTOFORMALISM_BENCHMARKS_PREVIEW.tex
# Run the same command a second time to resolve references.
cp tmp/pdfs/benchmark-appendix/autoformalism-benchmark-appendix.pdf output/pdf/
```

The standalone preview uses an article layout. The section preserves
`app:benchmark` for insertion into the paper; pagination in the conference
template will differ.

## Verification

- `pytest tests/test_phase_b_public.py tests/test_phase_b_generation.py
  tests/test_component_campaign.py -q`: **68 passed**.
- `ruff check .`: **37 pre-existing findings** confined to unrelated
  `analysis/claude` Python files. This revision changes only LaTeX/Markdown.
- The standalone LaTeX preview compiled twice to five pages, with resolved
  citations and cross-references and no overfull/underfull box warnings. All five
  final pages were rendered and visually inspected.
- Scope check: the appendix contains no unused task or medium-tier sections.
- The frozen alien coupling matrix was checked to be skew-symmetric.
