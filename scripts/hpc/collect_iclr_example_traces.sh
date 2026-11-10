#!/usr/bin/env bash
# Package selected existing Phase B search receipts on ACES without running a job.
set -euo pipefail
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
AF_CAMPAIGN=${1:-"$AF_GROUP/final-components-v1"}
AF_PARENT=${2:-"$AF_GROUP"}
python3 - "$AF_CAMPAIGN" "$AF_PARENT" <<'PY'
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

campaign, dest = map(Path, sys.argv[1:])
tasks = ["cell%02d_seed0_full_c1v1s1" % n for n in (0,4,5,8)]
plan = campaign / "plan.json"
if not plan.is_file():
    sys.exit("Campaign plan is missing at %s" % plan)
data = json.loads(plan.read_bytes())
if data.get("artifact_sha256") != "cdfbd68d07a88f510a869c595c457dfcdf6595d42cf6fb8983bf10a901d5b950":
    sys.exit("The selected directory does not contain the original component campaign.")
if not dest.is_dir():
    sys.exit("Destination directory is missing: %s" % dest)
present, missing = [], []
for task in tasks:
    for i in range(6):
        for f in ["proposal.json","critic.json","response.json","result.json"]:
            rel = "results/%s/round_%02d/%s" % (task,i,f)
            (present if (campaign / rel).is_file() else missing).append(rel)
    for f in ["result.json","critic.json"]:
        rel = "pruning/%s/%s" % (task,f)
        (present if (campaign / rel).is_file() else missing).append(rel)
if not present:
    sys.exit("No requested source receipts were found. No archive created.")
fd, path = tempfile.mkstemp(prefix="iclr-phase-b-first-six-",suffix=".tar.gz",dir=str(dest))
os.close(fd)
try:
    signatures = {}
    with tarfile.open(path, "w:gz") as tf:
        def write(name, contents):
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            tf.addfile(info, io.BytesIO(contents))
        for name in ["plan.json"] + present:
            content = (campaign / name).read_bytes()
            signatures[name] = hashlib.sha256(content).hexdigest()
            write(name,content)
        write("TRACE_INDEX.json",json.dumps({"method":"final-component-campaign-1","tasks":tasks,
            "prepruning_round_indices":[0,1,2,3,4,5],"missing":missing,"files":signatures},
            sort_keys=True, indent=2).encode())
    print(json.dumps({"archive":path,"receipts":len(present),"missing":len(missing)},indent=2))
except BaseException:
    os.unlink(path)
    raise
PY
