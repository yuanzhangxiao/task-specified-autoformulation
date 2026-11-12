"""Exit 0 only if every nine-case smoke summary clears the full campaigns.

    python scripts/check_phase_b_smokes.py RUN/summary.json [RUN/summary.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from autoformalism.rebuttal.phase_b_nine_case import smoke_problems


def main() -> None:
    summaries = {}
    for argument in sys.argv[1:]:
        path = Path(argument)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries[path.parent.name] = summary
        print(
            path.parent.name,
            json.dumps(summary.get("counts")),
            json.dumps(summary.get("accounting")),
        )
    problems = smoke_problems(summaries) if summaries else ["no summaries given"]
    if problems:
        print("a smoke failed its check; no campaign was started")
        print("\n".join(problems))
        sys.exit(1)
    print("every smoke passed")


if __name__ == "__main__":
    main()
