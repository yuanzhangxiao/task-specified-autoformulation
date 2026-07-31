#!/usr/bin/env python3
"""Resume one checkpointed autoformalism experiment."""

from __future__ import annotations

import json

from autoformalism.execution import (
    arguments_from_namespace,
    build_experiment_parser,
    execute,
)


def main() -> None:
    """Parse the shared CLI with resume enabled by default."""
    parser = build_experiment_parser(
        description=__doc__ or "Resume an experiment.",
        default_resume=True,
    )
    result = execute(arguments_from_namespace(parser.parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
