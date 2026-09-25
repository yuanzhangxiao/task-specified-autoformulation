# Dalla R4 sign pilot: scheduler recovery

ACES returned a socket timeout without a job ID for preparation, but accounting
confirmed job 2162020 completed successfully. The original recovery adopted that
job and retained an `afterok` dependency on it. The next review submission was
explicitly rejected with `Job dependency problem`, consistent with a completed
job leaving the live scheduler's dependency records. Review 2161544 is older
than this preparation and must not be adopted based on its name alone.

A standalone recovery helper verifies the frozen plan and completed preparation,
preserves original receipts, and submits the rejected review without the satisfied
preparation dependency. It retains fit-after-success and report-after-termination
ordering. New receipts record every attempt separately. Exact owned accounting
matches can recover jobs accepted despite a timeout, including fit arrays;
accounting delays stop submission instead of triggering an ambiguous retry.

The helper must run outside the original dbb51ed checkout. Existing scientific
code, prompts, models, budgets, plan hashes and data boundaries are unchanged.
Tests simulate the retired preparation, success, rejection races, timeout
adoption, delayed accounting, arrays and deterministic recovery without resending
jobs. The coding agent does not access or submit to ACES directly.
