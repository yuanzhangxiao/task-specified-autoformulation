"""Scoped scientific-gate ablation; never changes expression/execution safety.

The public brief and all proposer-visible specifications remain intact. Only
explicit scientific requirement enforcement is optional. ContextVar keeps the
switch local to a worker/async context; ordinary and historical callers default
to enforcement. Post-hoc assessment runs outside this scope.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_ENABLED = ContextVar("scientific_verification", default=True)


def enabled() -> bool:
    """Whether scientific gates are active in the current search operation."""
    return _ENABLED.get()


@contextmanager
def scope(active: bool):
    """Restore enforcement even if a provider, compiler or worker raises."""
    if type(active) is not bool:
        raise TypeError("scientific verification must be a boolean")
    token = _ENABLED.set(active)
    try:
        yield
    finally:
        _ENABLED.reset(token)
