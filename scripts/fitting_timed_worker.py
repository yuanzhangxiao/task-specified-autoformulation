#!/usr/bin/env python3
"""Start the diagnostic clock before importing the fitting package."""

import os
import socket
import sys
from pathlib import Path
from time import monotonic, process_time


def main():
    wall, cpu = monotonic(), process_time()
    # Deliberately delayed: python -m package.worker imports the package first.
    from autoformalism.fitting import mesh_refinement_worker as worker
    from autoformalism.fitting import public_fitting as public
    from autoformalism.fitting.operation_timing import Recorder

    imported_wall, imported_cpu = monotonic(), process_time()
    recorder = Recorder()
    recorder.add("imports", imported_wall - wall, imported_cpu - cpu, False)
    folder = Path(sys.argv[2])
    status = "raised"
    try:
        with recorder.active():
            worker.run(sys.argv[1], folder)
        status = "returned"
    finally:
        report = {
            "protocol": "fitting-worker-timing-1",
            "status": status,
            "mode": sys.argv[1],
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "entry_monotonic": wall,
            "worker_wall_seconds": monotonic() - wall,
            "worker_cpu_seconds": process_time() - cpu,
            **recorder.snapshot(),
            "limitation": "Inclusive spans; wall/CPU gaps do not identify their cause. "
            "Interpreter startup and final timing-file write are excluded. "
            "A killed worker may leave no timing report; this is missing, not zero.",
        }
        public._write(folder / "timing.json", report)


if __name__ == "__main__":
    main()
