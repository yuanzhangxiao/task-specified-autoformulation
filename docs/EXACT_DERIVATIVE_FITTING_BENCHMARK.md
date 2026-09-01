# Exact-derivative affine-parameter fitting benchmark

This CPU-only synthetic benchmark isolates the oracle fitting milestone. Every
state and exact state derivative is observed, nonlinear basis-shape constants
are fixed in proposer-owned expressions, and all optimized graph weights enter
the ODE right-hand sides affinely.

Three systems are predeclared:

- source plus decay;
- a fixed-shape exponential basis with an external input;
- a coupled two-state affine graph.

For three frozen repetitions, the benchmark compares:

- `exact_derivative_linear_ridge`, which evaluates the design matrix and solves
  Eq. (11) once; and
- `bounded_nonlinear`, which uses the existing iterative bounded derivative
  regression on the same exact labels.

It reports success, parameter recovery error, function evaluations, fitting
wall time, validation NMSE, and test NMSE separately. Fitted parameters are
written to a content-addressed freeze file before the synthetic test split is
evaluated. This benchmark does not claim realistic performance with estimated
derivatives.

Run locally:

```bash
.venv/bin/python scripts/run_exact_derivative_fitting_benchmark.py \
  --config configs/exact_derivative_fitting_benchmark_v1.json \
  --output-root artifacts/exact-derivative-fitting-benchmark-v1
```

Run on ACES after the repository and Python environment are available there:

```bash
repo=/scratch/group/p.nairr260351.000/autoformalism
cd "$repo"

export AF_REPO_ROOT="$repo"
export AF_PYTHON="$repo/.venv/bin/python"
export AF_OUTPUT_ROOT=/scratch/user/$USER/exact-derivative-fitting-benchmark-v1

sbatch --account=156264627414 \
  scripts/hpc/exact_derivative_fitting_benchmark.slurm
```

The job requests the ACES `cpu` partition, four CPU cores, 8 GB of memory, and
30 minutes. It requests no GPU.
