# Jetstream2 construction serving compatibility

This is a bounded serving check, not a new construction campaign. The ACES
five-seed experiment remains pinned to its original code and plan. The new VM
is a single H100 80GB HBM3, observed driver 580.159.03, with a 200GB root disk.
It needs no additional volume for this check. The existing CPU baseline VM and
its running experiments are separate.

## Scope and evidence

The delivered `probe-requests.json` freezes four complete projected requests
from the human-reviewed `focused-prompts-review-20261010-r2/PROMPTS.json`.
Selection takes the largest serialized request in each of variables, shared
processes, ordinary topology, and global repair. Byte size is a selection
heuristic, not a tokenizer count. Replies are evaluated independently against
their displayed historical predecessors; they do not form a new construction.
The payloads are generated artifacts distributed in the upload bundle, not Git.

The script retains model `openai/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`, low reasoning, temperature 0.2,
8192 output tokens and the original request seeds. Serving uses vLLM 0.27.1,
32768 context tokens, tensor parallelism 1, maximum sequences 1, and memory
utilization 0.90. The pinned Docker image is:

```
vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
```

The image is the Docker source used by the project's serving setup; Docker
execution on this H100 is a different recorded environment from the ACES SIF.
The script records image identity, driver, Torch/CUDA/vLLM versions, a successful
GPU matrix multiplication, and downloaded weights verified against the published
pinned-revision hashes. `verify_weights.py` reuses the baseline agent's existing
verification tool from the `phase-c-baselines` branch.

The tokenizer must report the 32768-token context and at most 24000 input
tokens for each complete request. Each generation receives exactly one physical
attempt, with no hidden retries. A durable inflight record precedes HTTP. A rerun
reuses completed outcomes; interrupted calls remain unavailable with unknown
usage. Changed inputs or serving records require a fresh result directory.
Requests, responses, tokenization, elapsed time and token usage remain available.
The current stage schema validates final JSON; topology/scientific correctness
is not assessed. Passing four probes does not establish equal outputs across
hardware, full 32K occupancy, campaign throughput, or scientific improvement.
No model promotion, fitting, test access, or automatic campaign follows.

## Connect from a Mac

Use native SSH in Ghostty. Exosphere calls its account `exouser`; without a
selected SSH key, its password is available privately under Credentials.
Do not change that password: Exosphere's own access depends on it.

```bash
ssh -o ServerAliveInterval=30 exouser@149.165.152.222
```

Use Command-V to paste and Control-C to interrupt. On a new SSH connection,
Ghostty's terminal type may be unknown to the VM. If terminal programs complain,
connect with a widely supported terminal description:

```bash
TERM=xterm-256color ssh -o ServerAliveInterval=30 exouser@149.165.152.222
```

To install an existing Mac public key once, run this on the Mac and enter the
VM password when prompted. This appends the public key without replacing any
existing authorized keys. Never transfer the private key.

```bash
ssh exouser@149.165.152.222 'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_ed25519.pub
```

[Jetstream's native SSH instructions](https://docs.jetstream-cloud.org/ui/exo/access-instance/)
explain both login methods and adding keys after creation.

## Run the delivered bundle

Upload the compatibility bundle from the Mac with `scp`, then extract it on the
VM. Use its own checkout; do not run the ACES Slurm launcher on this machine.
From the extracted directory:

```bash
bash scripts/jetstream/check_construction.sh run
tail -f "$HOME/af-construction/launcher.log"
```

Control-C exits the log viewer only. The background worker continues through
installation, weight download, GPU check, server startup and four probes. First
startup may take several minutes; model/image downloads depend on connectivity.
The dedicated environment and outputs live in `~/af-construction`. The server
is published at `127.0.0.1:8000` only. Setup uses `sudo` for packages and Docker;
it registers the NVIDIA runtime with a daemon reload, without stopping the
Exosphere web containers. It does not change firewall rules or SSH configuration.

Inspect:

```bash
bash scripts/jetstream/check_construction.sh status
cat "$HOME/af-construction/probe/summary.json"
```

`compatibility_passed: true` means all four final responses passed their schemas
and had measured usage, after GPU and tokenizer checks. Otherwise retain logs
and inspect the failure; do not repeatedly create fresh roots to hide failures.
The status includes physical generations started and unknown usage.

Package the saved evidence without model weights, containers or caches:

```bash
bash scripts/jetstream/check_construction.sh inspect
```

This produces `~/af-construction/inspection.tar.gz`, including startup logs even
when setup failed before any request. It preserves logs from earlier invocations.

The server remains loaded afterward. To stop this named server while preserving
records:

```bash
bash scripts/jetstream/check_construction.sh stop
```

Stopping a container does not stop VM allocation charges. Shelving an idle VM
stops compute charging, but resuming it depends on available capacity. The full
Jetstream construction campaign launcher remains a later step after this check.

## Offline verification

```bash
PYTHONPATH=src .venv/bin/python scripts/jetstream_construction_probe.py prepare \
  --review output/focused-prompts-review-20261010-r2/PROMPTS.json \
  --requests output/jetstream-construction-compatibility/probe-requests.json
.venv/bin/pytest -q tests/test_jetstream_construction_probe.py
bash -n scripts/jetstream/check_construction.sh
```

Infrastructure references:
[NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
[Docker runtime reload](https://docs.docker.com/reference/cli/dockerd/),
[Jetstream accounting](https://docs.jetstream-cloud.org/general/access/).
