# Phase-B v1 release integration

## Scope

Phase-B v1 now has a production registry and loader contract without modifying
historical benchmark files. The default registry contains the six historical
benchmarks plus 40 Phase-B cells. Historical entries retain their legacy
observation/derivative/input layout; Phase B uses one canonical tidy CSV per
split.

## Release command

The production assets are deliberately materialized only by an explicit
command:

```bash
python scripts/release_phase_b_public_suite.py \
  --private-data-root data_raw \
  --public-data-root /path/to/public-release-root
```

The command creates `phase_b_v1/<benchmark_id>/` packages. Every package
contains the frozen proposer prompt, shared judge prompt, train, validation,
and sealed test tables, plus a production manifest. It fails unless all 40
cells pass leakage checks, all tests are sealed, and each semantic pair has an
identical channel-name-independent numeric commitment.

The repository does not contain the materialized release tables. Generated
benchmark data remain external artifacts and must not be committed.

## Loader isolation

The production loader:

1. rejects a Phase-B staging manifest;
2. requires the frozen production schema and status;
3. requires all train, validation, and sealed-test fingerprints;
4. verifies every split hash before loading development data;
5. rejects eager `BenchmarkLoader.load()` for Phase B;
6. loads only train and validation during discovery; and
7. requires `FrozenTestAccess` matching the benchmark and tier before opening
   test data.

The search controller constructs this access object from the validation-frozen
selection hash after the checkpoint store grants its one-time test claim.

## Numerical protocol

Phase-B registry entries set `one_step_target_history=False`, so target reset is
disabled and the execution plan reports `prediction_protocol: open_loop`.
Derivative columns are not public assets. The tidy loader derives deterministic
finite-difference arrays only to satisfy the common typed trajectory contract;
execution forcibly disables derivative-regression fitting for Phase B. Search
and final fitting therefore use rollout simulation.

## Verification

A complete temporary release rehearsal produced and loaded all 40 cells:

- released cells: 40/40;
- leakage-clean cells: 40/40;
- sealed test cells: 40/40;
- semantic numeric-commitment mismatches: 0;
- registry/loader development and authorized-test loads: 40/40; and
- representative CLI dry run: open-loop with derivative fast path disabled.

No production test metric was computed and no released package was written to
the project data tree during this rehearsal.
