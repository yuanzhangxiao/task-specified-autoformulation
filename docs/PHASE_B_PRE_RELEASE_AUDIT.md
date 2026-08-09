# Phase-B v1 pre-release simulator audit

## Scope

This audit uses only trusted private simulators and the prespecified training
and validation protocols. It uses no discovery-method output and does not open
or generate sealed test references. The 20 numerical cases cover four Dalla
Man tasks, two dynamics conditions, two tiers, CSTR easy/hard, and alien-device
easy/hard. Because semantic controls share numerical data, these 20 cases
support all 40 eventual public benchmark cells.

The reproducible entry point is `scripts/audit_phase_b_suite.py`. It writes a
machine-readable report for every case plus JSON and Markdown summaries.

## Gates

Each cell must pass:

1. full claimed mechanism-subspace rank at relative singular value `1e-3`;
2. claimed-subspace condition number at most 1,000 (easy) or 5,000 (hard);
3. stable rank at least 50% (easy) or 35% (hard) of claimed dimension;
4. intact-versus-ablated validation discrepancy NMSE at least 0.20 (easy) or
   0.15 (hard) for every required mechanism;
5. input Gram eigenvalue ratio at least `1e-3`;
6. persistence NMSE at least 0.25 for Dalla Man or 0.5 for CSTR/alien; and
7. 100% finite reference rollouts.

Sensitivity columns correspond to fractional perturbations of named private
mechanism groups, not individual raw parameters. Outputs and mechanism
directions are scaled before the singular spectrum is interpreted. Ablation
scores use training-derived channel scales and validation protocols.

## Simulator-only redesign decisions

The first audit found 15/20 cases ready. The following five failures were
resolved using the remedies frozen in the protocol—minimum added public
information or a weaker identifiable-subspace claim. Thresholds were not
changed.

| Initial failure | Evidence | Protocol remedy |
|---|---|---|
| Perturbed T1 hard | Full rank and condition passed, but target-only stable rank was 1.047 versus 1.05 required | Supply `Gt` as the minimum task-relevant auxiliary; retain the three-dimensional claim |
| Canonical T4 easy | Four directions had full rank and good conditioning, but stable rank 1.525 was below 2.0 | Claim recovery only on the three-dimensional identifiable flux subspace; keep all four mechanisms as structural/ablation requirements |
| CSTR easy | Three mechanism directions reduced to rank two and condition 1,480 | Claim a two-dimensional mechanism equivalence subspace; retain all three component checks |
| CSTR hard | Five directions reduced to rank two and condition 20,874; adding private-state outputs did not restore the weak directions | Claim a two-dimensional mechanism equivalence subspace; retain all five structural and intervention checks |
| Alien hard | Four directions had full rank and condition 22.2, but stable rank 1.107 was below 1.4 | Claim a three-dimensional identifiable subspace; retain all four mechanism checks |

The alien easy auxiliaries selected by empirical initial-state influence on the
output are `z5` and `z1`. Public names will be assigned only when semantic
assets are generated.

## Final result

After applying those simulator-only remedies, all 20/20 numerical cases pass
all pre-release gates. Representative margins are:

| Case | Claimed dimension | Condition number | Stable rank | Minimum ablation discrepancy NMSE |
|---|---:|---:|---:|---:|
| Dalla T1 perturbed hard | 3 | 4.33 | 1.184 | 1.918 |
| Dalla T4 canonical easy | 3 | 4.43 | 1.525 | 0.496 |
| CSTR easy | 2 | 179.32 | 1.000 | 112.854 |
| CSTR hard | 2 | 189.47 | 1.000 | 0.769 |
| Alien easy | 2 | 1.80 | 1.309 | 1.102 |
| Alien hard | 3 | 6.52 | 1.107 | 1.015 |

Passing these local empirical gates does not prove global structural
identifiability. It establishes that the frozen excitation design supports the
specific subspace and intervention claims used by the benchmark. Test assets
remain unsealed and no production discovery run is authorized until deliberate
test sealing and production-registry integration are complete. Public channel
mappings, all semantic-control prompts, the shared judge prompt, and leakage
checks have now been generated and manually reviewed.
