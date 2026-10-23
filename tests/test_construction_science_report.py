"""Human inspection preserves proof scope and never executes saved model text."""

from copy import deepcopy
from html.parser import HTMLParser

from autoformalism.research.construction_science_report import render


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attributes = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend(attrs)


def test_escaped_model_proof_parameter_and_title(tmp_path):
    attack = '</pre><script>alert("x")</script><img src=x onerror=alert(1)>'
    check = {
        "property": "affine_state_derivative_bound",
        "target": "U",
        "status": "conditional_conflict",
        "model_relation": "d(U)/dt = 1+k*M " + attack,
        "derivative_interval": [1, "infinity"],
        "proven_nonnegative_states": ["I", "M"],
        "witnesses": [
            {
                "trajectory": attack,
                "times": [0, 2],
                "observations": [1, 1],
                "observed_change_interval": [0, 0],
                "required_change": {"comparison": "at_least", "value": 2},
            }
        ],
        "repair_feedback": attack,
    }
    models = [
        {
            "model_id": attack,
            "source_role": attack,
            "status": "revised",
            "parameter_audit": {
                "equations": [
                    {
                        "lhs": "d(U)/dt",
                        "declared_expression": attack,
                        "expanded_expression": "1+k*M",
                        "status": "affine_collected",
                    }
                ],
                "simplification_candidates": [
                    {
                        "location": "rhs:U",
                        "argument": "M",
                        "effective_coefficient": attack,
                        "parameters": [attack],
                    }
                ],
            },
            "training_consistency": {
                "status": "deferred",
                "function_consistency": {
                    "status": "conflict",
                    "conflicts": [check],
                    "checks": [check],
                },
            },
        }
    ]
    before = deepcopy(models)
    target = tmp_path / "REPORT.html"
    render(target, {"protocol": attack}, models, title=attack)
    text = target.read_text()
    parsed = Elements()
    parsed.feed(text)
    assert "script" not in parsed.tags and "img" not in parsed.tags
    assert not any(name.startswith("on") for name, _ in parsed.attributes)
    assert "&lt;script&gt;" in text and "&lt;/pre&gt;" in text
    assert "Blocked: qualified training conflict" in text
    assert "The model requires a change <strong>at least 2</strong>" in text
    assert "observed change interval is [0, 0]" in text
    assert models == before


def test_empty_results_have_no_false_acceptance_or_external_resources(tmp_path):
    target = tmp_path / "new" / "REPORT.html"
    render(target, {}, [], title="No returned cases")
    text = target.read_text()
    assert "0 model records" in text
    assert "No model records were supplied" in text
    assert "No category establishes full scientific validity" in text
    parsed = Elements()
    parsed.feed(text)
    assert "svg" in parsed.tags
    assert not any(name in {"src", "href"} for name, _ in parsed.attributes)
    assert "script" not in parsed.tags


def test_coverage_parent_equations_and_inactive_coordinate_caveats(tmp_path):
    models = [
        {
            "model_id": "child-08",
            "parent_id": "parent-03",
            "status": "revised",
            "parameter_audit": {
                "equations": [{"lhs": "d(U)/dt", "expanded_expression": "Uii+k*M"}],
                "inactive_parameter_certificate": {
                    "parameters": ["tau_unused"],
                    "execution_domain_safety_proven": False,
                },
            },
            "parent_parameter_audit": {
                "equations": [{"lhs": "d(U)/dt", "expanded_expression": "Uii+M-U"}]
            },
            "training_consistency": {
                "status": "deferred",
                "checks": [
                    {
                        "property": "closed_subsystem_equality",
                        "target": "U",
                        "status": "deferred",
                        "unavailable": [
                            "relevant trajectory roots are not established equal"
                        ],
                    }
                ],
            },
        },
        {"model_id": "unassessed", "training_consistency": {}},
    ]
    target = tmp_path / "REPORT.html"
    render(
        target,
        {"nmse": 999, "fitted_parameters": {"secret_parameter": 5}},
        models,
        title="Read-only check",
    )
    text = target.read_text()
    assert "No qualified conflict; some checks deferred" in text
    assert "Assessment unavailable or incomplete" in text
    assert "Parent equations for comparison" in text and "Uii+M-U" in text
    assert "tau_unused" in text
    assert "No optimizer coordinates were changed" in text
    assert "secret_parameter" not in text
    first = text
    render(target, {}, models, title="Read-only check")
    assert target.read_text() == first


def test_combined_conflicts_are_not_counted_twice(tmp_path):
    check = {"property": "slope", "target": "U", "status": "conditional_conflict"}
    report = {
        "status": "conflict",
        "conflicts": [check],
        "function_consistency": {"status": "conflict", "conflicts": [check]},
    }
    path = tmp_path / "report.html"
    render(
        path,
        {},
        [{"model_id": "candidate", "training_consistency": report}],
        title="Audit",
    )
    assert "Qualified conflicts (1 checks)" in path.read_text()


def test_pending_and_rejected_equation_scopes_are_explicit(tmp_path):
    models = [
        {
            "model_id": "pending-case",
            "status": "pending",
            "audit_scope": "unchanged_parent",
            "parameter_audit": {
                "equations": [{"lhs": "d(x)/dt", "expanded_expression": "parent_rhs"}]
            },
            "training_consistency": {"status": "deferred"},
        },
        {
            "model_id": "rejected-case",
            "status": "rejected",
            "audit_scope": "last_rejected_candidate",
            "training_consistency": {"status": "conflict"},
            "parent_parameter_audit": {
                "equations": [
                    {"lhs": "d(x)/dt", "expanded_expression": "unchanged_parent_rhs"}
                ]
            },
        },
    ]
    path = tmp_path / "report.html"
    render(path, {}, models, title="Partial run")
    html = path.read_text()
    assert "unchanged parent, not a new child" in html
    assert "not installed" in html
    assert "No equation view was available" in html
    assert "rather than distinct completed proposals" in html
    assert html.index("No equation view was available") < html.index(
        "unchanged_parent_rhs"
    )


def test_compact_rejected_evidence_keeps_witness_and_equation(tmp_path):
    report = {
        "status": "conflict",
        "status_counts": {"conditional_conflict": 16, "deferred": 32},
        "conflicts": [
            {
                "target": "y",
                "model_relation": "d(y)/dt = u+k*m",
                "witness": {
                    "trajectory": "train0",
                    "times": [0, 3],
                    "observations": [2, 2],
                    "required_change": {"comparison": "at_least", "value": 3},
                    "observed_change_interval": [0, 0],
                },
            }
        ],
    }
    path = tmp_path / "report.html"
    render(
        path,
        {},
        [
            {
                "audit_scope": "last_rejected_candidate",
                "parameter_audit": {"expanded_equations": ["d(y)/dt = u+k*m"]},
                "training_consistency": {
                    "status": "conflict",
                    "function_consistency": report,
                },
            }
        ],
        title="Compact rejected evidence",
    )
    text = path.read_text()
    assert "Compact equation view retained" in text
    assert "compact coverage summary" in text
    assert "Training witness:</strong> train0" in text
    assert "at least 3" in text
