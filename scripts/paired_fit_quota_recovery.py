"""Dispatch maintenance helpers after a check parseable by ACES Python 3.6."""

import sys


def main():
    """Reject old interpreters before importing helpers or touching any files."""
    if sys.version_info < (3, 9):  # noqa: UP036 - bootstrap runs outside project Python
        raise SystemExit(
            "Quota recovery needs Python 3.9 or newer; this is "
            f"Python {sys.version_info[0]}.{sys.version_info[1]}. "
            "On ACES, use the existing Python 3.11 virtualenv executable "
            "instead of the default python3. No files were changed."
        )
    if len(sys.argv) < 2 or sys.argv[1] not in {"cache", "install"}:
        raise SystemExit(
            "Usage: paired-fit-quota-recovery.pyz {cache|install} [options]"
        )
    command = sys.argv.pop(1)
    if command == "cache":
        from archive_finished_component_caches import main as dispatch
    else:
        from install_verified_paired_bundle import main as dispatch
    dispatch()


if __name__ == "__main__":
    main()
