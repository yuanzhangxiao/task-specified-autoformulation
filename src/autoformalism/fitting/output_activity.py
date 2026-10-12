"""Conservative structural output dependencies, without sampled-Jacobian pruning."""

import ast

from autoformalism.fitting.sensitivity_contract import certify_expressions


def certificate(system) -> dict:
    """Trace every output backwards through expanded RHSs and causal initial maps.

    Syntactic dependencies deliberately overestimate activity (e.g. p-p stays
    active). Initial observation names are data, not dynamical state references.
    The certificate concerns outputs, not feasibility of disconnected dynamics.
    """
    rhs, observations, audit = certify_expressions(
        system.model, allow_piecewise=system.allow_piecewise
    )

    def symbols(expression):
        return {n.id for n in ast.walk(expression.tree) if isinstance(n, ast.Name)}

    states, parameters = set(system.model.state_names), set(system.names)
    needed = set().union(*(symbols(e) for e in observations.values()))
    reached = set()
    while new := (needed & states) - reached:
        reached |= new
        needed |= set().union(*(symbols(rhs[n]) for n in new))
    initial_parameters = set()
    direct = system.model.direct_state_observation_channels
    for name in reached - set(direct):
        if name in audit.initial_expressions:
            initial_parameters |= symbols(audit.initial_expressions[name]) & parameters
    active = (needed & parameters) | initial_parameters
    return {
        "protocol": "structural-output-activity-1",
        "active_parameters": [n for n in system.names if n in active],
        "inactive_parameters": [n for n in system.names if n not in active],
        "output_reachable_states": sorted(reached),
        "disconnected_states": sorted(states - reached),
        "initializer_parameters": sorted(initial_parameters),
        "scope": "All declared outputs; syntactic dependencies, not identifiability.",
        "equations_changed": False,
        "disconnected_state_feasibility_certified": False,
    }


class ActiveProfile:
    """Keep the original kernel and drop only certified inactive Jacobian columns."""

    def __init__(self, profile, names, active):
        self.profile = profile
        self.columns = [k for k, i in enumerate(profile.outer) if names[i] in active]
        self.outer = tuple(profile.outer[k] for k in self.columns)
        self.audit = profile.audit | {
            "active_outer_names": [names[i] for i in self.outer],
            "fully_profiled_output": not self.outer,
        }

    def evaluate(self, vector, deadline):
        fitted, r, j, audit = self.profile.evaluate(vector, deadline)
        return fitted, r, j[:, self.columns], audit
