# ACES review-cache recovery

The 23 October 2026 inventory reports 498,628 of 500,000 group file entries,
with 22,096 entries below `runtime-cache/review`. Byte capacity is ample. Both
the group and HOME copies of `phase-c-directional-family-cf21fa0` have experienced
failed extraction. Neither an existing directory nor quota headroom proves a
complete installation.

The reviewed launcher chain is `submit_review_integrity_aces.sh` (also
`submit_review_response_aces.sh` and `submit_review_multi_aces.sh`) through
`run_review_continuation_aces.sh`, `run_review_deadline_aces.sh`, and
`run_staged_topology_server.sh`. It assigns numeric job directories below
`runtime-cache/review` to GPU compilation caches. Scientific responses, experiment
results, Python environments, and model weights have separate paths.

## Bounded maintenance

`archive_finished_component_caches.py --profile review` accepts only a physical
root ending in `runtime-cache/review`. `--all-jobs` discovers numeric directories
there; unexpected entries stop the operation. Each selected allocation must
have an exact owned terminal accounting record, and the user's whole queue must
be empty. The helper accepts only the existing compiler-cache namespaces and
rejects links, special files, mounts, and foreign ownership. Discovery does not
make an unverified directory disposable.

With `--apply`, each cache is archived and read back, with every file's SHA-256
verified. Durable per-job receipts and the archives are retained. The helper
rechecks scheduler state and live contents before removing loose cache copies.
Missing accounting, active jobs, or unexpected contents stop cleanup. Without
`--apply`, it produces a read-only plan. No automatic job cancellation occurs.
The archives are recoverable by normal tar extraction to a new directory; do
not restore them over an active cache. Retain their checksum receipts.

Use the existing Python 3.11 executable with `-S`; do not use ACES's default
Python 3.6. These maintenance helpers need no scientific imports or compiler
module setup. They can be delivered as one zipapp, requiring only one upload
entry and no source-tree extraction. The standalone dispatcher rejects old
Python before importing helpers. Keep the maintenance revision separate from
the unchanged experiment revision.

## Installation after reclamation

The authenticated installer now has an explicit `--profile directional-family`
for regular-file `phase-c-directional-family-<commit>` bundles with `SHA256SUMS`.
The default paired-fit profile is unchanged. Both profiles check the externally
provided archive SHA-256, complete manifest/member correspondence, and every
member hash before writing. Installation uses a private temporary directory,
verifies the complete tree, and publishes it atomically. Existing copies are
reused only if their entire contents match; partial copies are not overwritten.

Install the unchanged `phase-c-directional-family-cf21fa0.tar.gz` under a fresh
parent such as `$AF_GROUP/directional-recovered-code`. Its SHA-256 is
`32c11d32284ec3bad6724d877a697d9f6a79e812bf85ccace91cbfee0d486284`.
Leave both earlier failed extractions untouched. The bundle requires 601 file
and directory entries, before experiment outputs. It uses node-local temporary
compiler caches, so this new run does not repopulate the retired persistent cache.

Only after successful installation, run that copy's `run_revision.sh run` with
`AF_DIRECTIONAL_FAMILY_ROOT=$AF_GROUP/phase-c-directional-family-retest-v1`.
That launcher preserves existing plans/submission receipts and does not reset
budgets. The recovery helpers themselves submit no jobs, modify no experiment
results, and make no model, optimizer, or test-data calls.

The reported 22,096 entries are an upper estimate of potential reclamation until
remote validation succeeds. Archive files/receipts consume a few entries. Check
`showquota` after the operation; allocation-wide use can change concurrently.
