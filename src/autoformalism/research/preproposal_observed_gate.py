"""Necessary training checks on fully wired, deterministic ODE/algebraic graphs."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import comb

from autoformalism.search.mechanism_realizations import _memory_target_witness
from autoformalism.search.preproposal_screen import _forcing_value
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    _closure,
    _common_missing,
    _value,
    validate_evidence,
)


class ObservedGraphGate:
    """Reuse training observations and conditional equality proofs across graphs."""

    def __init__(self, evidence: TrainingCompatibilityEvidence):
        validate_evidence(evidence)
        self.evidence = evidence
        self.rows = evidence.trajectories
        ctx = evidence.context
        self.measured = set(ctx.targets + ctx.auxiliaries)
        self.bounds = evidence.contract.absolute_error_bounds
        self.missing = tuple(_common_missing(evidence))
        self.unique = bool(
            evidence.contract.uniqueness_assumed
            and evidence.contract.reset_impulse_free
        )
        self.samples = tuple(
            (r, i) for r, row in enumerate(self.rows) for i in range(len(row.time))
        )
        self.knots = sorted({t for row in self.rows for t in row.time})
        grids = [(r.time[0], set(r.time)) for r in self.rows]
        self.unmatched_pairs = sum(
            a != b or (len(first & second) <= 1 and max(len(first), len(second)) > 1)
            for (a, first), (b, second) in combinations(grids, 2)
        )
        # Cache results on this campaign's immutable evidence only.
        self.column = lru_cache(maxsize=256)(self._column)
        self.match = lru_cache(maxsize=2048)(self._match)
        self.increments = lru_cache(maxsize=512)(self._increments)

    def _column(self, mode, name):
        """Give every sample the same key whenever equality can be certified."""
        clock = self.evidence.context.time_symbol
        if mode == "origin":
            return [self.rows[r].time[0] for r, _ in self.samples]
        if mode == "current":
            return [_value(self.rows[r], name, i, clock) for r, i in self.samples]
        if mode == "initial":
            return [_value(self.rows[r], name, 0, clock) for r, _ in self.samples]

        # Exact complete forcing histories are compared on the union of knots.
        # Intern prefixes so storage and comparisons stay linear in sample count
        # on a common grid. Held inputs at the right endpoint do not drive an ODE
        # before that instant; a linear input depends on both endpoints.
        interpolation = self.evidence.contract.interpolation[name]
        interned, values = {}, []
        for row in self.rows:
            sample_indices = set(row.time)
            start, end = row.time[0], row.time[-1]
            prefix, previous = 0, None
            for t in self.knots:
                if t < start or t > end:
                    continue
                current = _forcing_value(row, name, t, interpolation)
                if previous is not None:
                    segment = (
                        previous,
                        current if interpolation == "linear_complete_knots" else None,
                    )
                    key = (prefix, t, segment)
                    prefix = interned.setdefault(key, len(interned) + 1)
                if t in sample_indices:
                    values.append(prefix)
                previous = current
        return values

    def assess(self, target, graph, states, initialization, gap):
        """Test the realized target with its own unit-specific observation margin."""
        ctx = self.evidence.context
        # At sample times exact observations of other generated quantities can
        # condition a necessary test of joint exact reproduction. They never
        # become supplied rollout inputs.
        differential = target in states
        dynamics, current, pending = set(), set(), [target]
        visited = set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            if name != target and name in self.measured and self.bounds.get(name) == 0:
                current.add(name)
            elif name in states:
                dynamics.add(name)
            elif name in graph:
                pending.extend(graph[name])  # algebraic acyclicity was checked
            else:
                current.add(name)

        initial_sources = set()
        for state in dynamics:
            initial_sources.update(initialization[state]["public_initials"])
        upstream = set().union(*(_closure(s, graph) for s in dynamics))
        histories = upstream & (
            set(ctx.auxiliaries + ctx.external_inputs) - graph.keys()
        )
        initial_ongoing = set(initial_sources) | (upstream & set(ctx.fixed_covariates))
        for state in upstream & states:
            initial_ongoing.update(initialization[state]["public_initials"])

        at_start = {("current", n) for n in current} | {
            ("initial", n) for n in initial_sources
        }
        ongoing = {("current", n) for n in current} | {
            ("initial", n) for n in initial_ongoing
        }
        if dynamics:
            ongoing.update({("history", n) for n in histories})
            ongoing.update({("current", ctx.time_symbol), ("origin", ctx.time_symbol)})

        unknown = bool(any(initialization[s]["trajectory_roots"] for s in dynamics))
        unknown_ongoing = any(
            initialization[s]["trajectory_roots"] for s in upstream & states
        )
        result = self.match(
            target,
            gap,
            tuple(sorted(at_start)),
            tuple(sorted(ongoing)),
            unknown,
            unknown_ongoing or bool(dynamics and not self.unique),
        ).copy()
        result["kind"] = "differential" if differential else "algebraic"
        result["conditioned_measured_arguments"] = sorted(current & graph.keys())
        if differential:
            if self.unmatched_pairs or len(self.rows) < 2:
                result["unavailable"] = [
                    *result["unavailable"],
                    "Too few runs or incomplete common-grid comparisons",
                ]
                result["unmatched_pairs"] = self.unmatched_pairs
                if result["status"] == "screened":
                    result["status"] = "unresolved"
            direct = set(graph[target])
            if not (direct & graph.keys() or ctx.time_symbol in direct):
                increments = self.increments(target, tuple(sorted(direct)), gap)
                result["constant_driver_increments"] = increments
                if increments["status"] == "contradicted":
                    result["status"] = "contradicted"
        return result

    def _unavailable(self, recipe):
        clock = self.evidence.context.time_symbol
        for mode, name in recipe:
            if name != clock and self.bounds.get(name) != 0:
                return True
            if (
                mode == "history"
                and self.evidence.contract.interpolation.get(name, "unknown")
                == "unknown"
            ):
                return True
        return False

    def _match(self, target, gap, at_start, ongoing, unknown_start, unknown_later):
        """Compare all samples sharing a certified complete argument tuple."""
        bound = self.bounds.get(target)
        missing = list(self.missing)
        if bound is None:
            missing.append(f"Unknown observation error for {target}")
        skip_start = unknown_start or self._unavailable(at_start)
        skip_later = unknown_later or self._unavailable(ongoing)
        if skip_start:
            missing.append("Initial argument equality is unavailable")
        if skip_later:
            missing.append(
                "Ongoing argument or complete forcing equality is unavailable"
            )
        start_cols = [] if skip_start else [self.column(*c) for c in at_start]
        later_cols = [] if skip_later else [self.column(*c) for c in ongoing]
        groups, starts, members = {}, {}, {}
        conflicting_pairs = set()
        witness = None
        largest = Fraction(0)
        compared = conflicts = initial_pairs = 0
        if bound is not None and not self.missing:
            threshold = 2 * Fraction(bound) + Fraction(gap)
            values = self.column("current", target)
            # When there is no dynamical dependence, compare the origin to all
            # other samples as well. With memory, equality at t0 needs no forcing.
            separate_origin = at_start != ongoing
            for n, (_, i) in enumerate(self.samples):
                if (i == 0 and skip_start) or (i > 0 and skip_later):
                    continue
                cols = start_cols if i == 0 else later_cols
                key = (i == 0 and separate_origin, *(col[n] for col in cols))
                members.setdefault(key, []).append(n)
                if i == 0:
                    starts.setdefault(key, []).append(n)
                if key not in groups:
                    groups[key] = [n, n]
                    continue
                compared += 1
                low, high = groups[key]
                if values[n] < values[low]:
                    groups[key][0] = n
                if values[n] > values[high]:
                    groups[key][1] = n
            for key, (low, high) in groups.items():
                if values[high] == values[low]:
                    continue
                spread = Fraction(values[high]) - Fraction(values[low])
                if spread > threshold:
                    conflicts += 1
                    self._record_pairs(
                        conflicting_pairs, members[key], values, threshold
                    )
                if spread > largest:
                    largest = spread
                    witness = [
                        {
                            "trajectory": self.rows[r].trajectory_id,
                            "time": self.rows[r].time[i],
                            "value": values[n],
                        }
                        for n in (low, high)
                        for r, i in [self.samples[n]]
                    ]
            for indices in starts.values():
                for a, b in combinations(indices, 2):
                    if values[a] != values[b] and (
                        not threshold
                        or abs(Fraction(values[a]) - Fraction(values[b])) > threshold
                    ):
                        initial_pairs += 1

        return {
            "status": "contradicted"
            if conflicts
            else "unresolved"
            if missing
            else "screened",
            "unavailable": missing,
            "minimum_target_gap": gap,
            "equal_argument_repeats": compared,
            "conflicting_argument_groups": conflicts,
            "initial_conflicting_pairs": initial_pairs,
            "conflicting_pairs": len(conflicting_pairs),
            "trajectory_pairs": comb(len(self.rows), 2),
            "max_observed_group_gap": float(largest),
            "witness": witness if conflicts else None,
        }

    def _record_pairs(self, found, indices, values, threshold):
        """Count trajectory pairs once, even when they conflict at many times."""
        ranges = {}
        for n in indices:
            r, _ = self.samples[n]
            low, high = ranges.get(r, (values[n], values[n]))
            ranges[r] = (min(low, values[n]), max(high, values[n]))
        for first, second in combinations(sorted(ranges), 2):
            if (first, second) in found:
                continue
            lo1, hi1 = ranges[first]
            lo2, hi2 = ranges[second]
            if threshold:
                disjoint = (
                    max(Fraction(hi1) - Fraction(lo2), Fraction(hi2) - Fraction(lo1))
                    > threshold
                )
            else:
                disjoint = hi1 > lo2 or hi2 > lo1
            if disjoint:
                found.add((first, second))

    def _increments(self, target, direct, gap):
        """Equal supplied drivers over equally long intervals imply equal changes."""
        modes = self.evidence.contract.interpolation
        fixed = set(self.evidence.context.fixed_covariates)
        bound = self.bounds.get(target)
        if (
            self.missing
            or bound is None
            or not self.evidence.contract.reset_impulse_free
            or any(self.bounds.get(s) != 0 for s in direct)
            or any(modes.get(s, "unknown") == "unknown" for s in set(direct) - fixed)
        ):
            return {"status": "unresolved", "compared_intervals": 0}
        # Each increment uses two endpoints. Apply twice the pair-difference
        # sensitivity to a difference of increments; source bounds stay exact.
        limit = 4 * Fraction(bound) + 2 * Fraction(gap)
        groups, compares = {}, 0
        for row in self.rows:
            vals = row.targets.get(target, row.auxiliaries.get(target))
            for i, t in enumerate(row.time[:-1]):
                start = tuple(
                    _value(row, s, i, self.evidence.context.time_symbol) for s in direct
                )
                if any(
                    modes.get(s) == "linear_complete_knots"
                    and a != _value(row, s, i + 1, self.evidence.context.time_symbol)
                    for s, a in zip(direct, start, strict=True)
                ):
                    continue
                key = (Fraction(row.time[i + 1]) - Fraction(t), start)
                delta = Fraction(vals[i + 1]) - Fraction(vals[i])
                location = (row.trajectory_id, t, row.time[i + 1])
                if key not in groups:
                    groups[key] = [(delta, location), (delta, location)]
                    continue
                compares += 1
                low, high = groups[key]
                if delta < low[0]:
                    low = (delta, location)
                if delta > high[0]:
                    high = (delta, location)
                if high[0] - low[0] > limit:
                    return {
                        "status": "contradicted",
                        "compared_intervals": compares,
                        "witness": [low[1], high[1]],
                        "increments": [float(low[0]), float(high[0])],
                    }
                groups[key] = [low, high]
        return {"status": "screened", "compared_intervals": compares}


def check_bound_memories(graph, states, brief, bindings):
    """Keep each declared memory on its claimed public driver-to-effect route."""
    if not bindings:
        return
    requirements = {r.id: r for r in brief.requirements}
    outgoing = {}
    for receiver, sources in graph.items():
        for source in sources:
            outgoing.setdefault(source, set()).add(receiver)
    for record in bindings:
        rid = record["requirement_id"]
        requirement = requirements.get(rid)
        if requirement is None:
            raise ValueError(f"Unknown mechanism binding {rid}")
        drivers = frozenset(requirement.drivers)
        targets = frozenset(requirement.targets)
        for memory in record["memory_states"]:
            if memory not in states or memory in drivers | targets:
                raise ValueError(
                    f"{rid} needs a distinct differential memory state: {memory}"
                )
            if not any(
                _path_via_memory(outgoing, source, memory, target, drivers)
                for source in sorted(drivers)
                for target in sorted(targets)
            ):
                raise ValueError(
                    f"{rid} has no qualified path from a public driver through "
                    f"assigned memory {memory} to a public target"
                )


def _path_via_memory(outgoing, source, memory, target, drivers):
    """Require a simple path through memory, allowing a self-feedback endpoint."""
    if not _memory_target_witness(outgoing, memory, target, drivers):
        return False
    # Distinct legs must be simultaneously realizable. Two independent
    # reachability results can share a node, making a dangling cycle look causal.
    pending = [(source, frozenset({source}))]
    examined = 0
    while pending:
        current, prefix = pending.pop()
        examined += 1
        if examined > 8192:
            raise ValueError(
                "Could not certify the bound memory path within 8192 prefixes"
            )
        for child in sorted(outgoing.get(current, ())):
            if child == memory:
                if _memory_target_witness(outgoing, memory, target, prefix | drivers):
                    return True
            elif child not in prefix and child != target:
                pending.append((child, prefix | {child}))
    return False
