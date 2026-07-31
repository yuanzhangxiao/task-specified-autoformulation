#!/usr/bin/env python3
"""Run one checkpointed autoformalism experiment."""

from __future__ import annotations

import json

from autoformalism.execution import (
    arguments_from_namespace,
    build_experiment_parser,
    execute,
)


def main(*, default_resume: bool = False) -> None:
    """Parse arguments and run, dry-run, or resume an experiment."""
    parser = build_experiment_parser(
        description=__doc__ or "Run an experiment.",
        default_resume=default_resume,
    )
    result = execute(arguments_from_namespace(parser.parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
