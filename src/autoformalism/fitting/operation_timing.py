"""Opt-in bounded worker timings; disabled instrumentation changes no budgets."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import monotonic, process_time

_ACTIVE = ContextVar("fitting_operation_timing", default=None)


class Recorder:
    """Aggregate inclusive spans and retain only the twenty longest observations.

    CPU is process CPU, including all threads. Wall minus CPU is not an I/O
    measurement: descheduling and other waits also contribute. Nested categories
    must not be added together. Recording does no file I/O inside a span.
    """

    def __init__(self):
        self.groups = {}
        self.longest = []

    def add(self, name, wall, cpu, failed):
        group = self.groups.setdefault(
            name,
            {
                "count": 0,
                "failures": 0,
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "maximum_wall_seconds": 0.0,
            },
        )
        group["count"] += 1
        group["failures"] += int(failed)
        group["wall_seconds"] += wall
        group["cpu_seconds"] += cpu
        group["maximum_wall_seconds"] = max(group["maximum_wall_seconds"], wall)
        self.longest.append(
            {"stage": name, "wall_seconds": wall, "cpu_seconds": cpu, "failed": failed}
        )
        self.longest.sort(key=lambda item: item["wall_seconds"], reverse=True)
        del self.longest[20:]

    @contextmanager
    def active(self):
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def snapshot(self):
        return {
            "groups": self.groups,
            "longest_spans": self.longest,
            "spans_are_inclusive": True,
            "contains_parameters_or_data": False,
        }


def timed(name):
    """Observe a trusted operation, preserving return values and exceptions."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            recorder = _ACTIVE.get()
            if recorder is None:
                return function(*args, **kwargs)
            wall, cpu, failed = monotonic(), process_time(), True
            try:
                value = function(*args, **kwargs)
                failed = False
                return value
            finally:
                recorder.add(name, monotonic() - wall, process_time() - cpu, failed)

        return wrapped

    return decorate
