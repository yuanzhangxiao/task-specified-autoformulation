"""Standard-library artifact reader for observational component-campaign audits."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,200}\Z")


def digest(value: object, *, compact: bool = False) -> str:
    """Match campaign seals, or the fitter's compact JSON digest when requested."""
    options = {"separators": (",", ":"), "allow_nan": False} if compact else {}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, **options).encode()
    ).hexdigest()


def identifier(value: object) -> str:
    """Accept only plain identifiers before constructing artifact paths."""
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("unsafe or missing identifier")
    return value


def label(value: object) -> str | None:
    """Free-form provider text never becomes an exported status or identifier."""
    return value if isinstance(value, str) and IDENTIFIER.fullmatch(value) else None


def diagnostic(value: object) -> dict:
    """Classify emitted text conservatively; export a hash, never raw responses."""
    if value is None or value == []:
        return {"category": None, "diagnostic_sha256": None}
    text = json.dumps(value, sort_keys=True)
    lower = text.lower()
    rules = (
        (
            "delivery_or_budget",
            ("provider", "preflight", "context length", "token budget"),
        ),
        ("quota_or_storage", ("quota exceeded", "no space left", "errno 122")),
        ("expression_syntax", ("syntax_error", "cannot be parsed", "parseable scalar")),
        ("algebraic_cycle", ("algebraic cycle", "algebraic loop")),
        (
            "function_delivery",
            ("function count mismatch", "requested slot", "interaction_id"),
        ),
        ("dependency_contract", ("dependenc", "source set", "source names")),
        ("conversion_contract", ("conversion", "overlap")),
        ("initialization_contract", ("initializ", "initial value")),
        ("public_requirement", ("pathway", "target path", "public requirement")),
        ("numerical_limit", ("wall-clock", "timeout", "time limit", "worker exit")),
    )
    category = next(
        (name for name, terms in rules if any(t in lower for t in terms)),
        "other_recorded_diagnostic",
    )
    return {"category": category, "diagnostic_sha256": digest(value)}


class Snapshot:
    """Read explicit artifact paths without following symlinks or executing models.

    A file manifest includes missing paths and captures changes during collection.
    It is observational, not an execution lease or a globally atomic snapshot.
    """

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve(strict=True)
        self.files: dict[str, dict] = {}
        self.issues: list[dict] = []

    def issue(self, path: str, code: str) -> None:
        """Keep diagnostics terse and free of input payloads and credentials."""
        value = {"path": path, "code": code}
        if value not in self.issues:
            self.issues.append(value)

    def path(self, relative: str) -> Path:
        """Refuse traversal and symlinked ancestors even when targets are internal."""
        parts = Path(relative).parts
        if Path(relative).is_absolute() or any(p in {"..", "."} for p in parts):
            raise ValueError("unsafe artifact path")
        current = self.root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("symlinked artifact path")
        return current

    def children(self, relative: str) -> list[str]:
        """List one explicitly allowed directory, without recursive tree discovery."""
        try:
            path = self.path(relative)
            return sorted(p.name for p in path.iterdir()) if path.is_dir() else []
        except (OSError, ValueError):
            self.issue(relative, "directory_unreadable_or_symlinked")
            return []

    def read(self, relative: str, *, sealed: bool = False):
        """Read one bounded JSON artifact; corrupt files remain explicit evidence."""
        try:
            path = self.path(relative)
            if not path.exists():
                self.files.setdefault(relative, {"path": relative, "state": "missing"})
                return None
            before = path.stat()
            if not path.is_file() or before.st_size > 128 * 1024 * 1024:
                raise ValueError("artifact size or type")
            data = path.read_bytes()
            after = path.stat()
            entry = {
                "path": relative,
                "state": "read",
                "bytes": len(data),
                "mtime_ns": after.st_mtime_ns,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            previous = self.files.get(relative)
            if previous and previous.get("state") == "read" and previous != entry:
                self.issue(relative, "changed_during_collection")
            self.files[relative] = entry
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                self.issue(relative, "changed_during_read")
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("object required")
            if sealed and value.get("artifact_sha256") != digest(
                {k: v for k, v in value.items() if k != "artifact_sha256"}
            ):
                raise ValueError("artifact seal differs")
            return value
        except (OSError, ValueError, TypeError, UnicodeError):
            self.issue(relative, "unreadable_or_invalid_artifact")
            self.files.setdefault(relative, {"path": relative, "state": "invalid"})
            self.files[relative]["state"] = "invalid"
            return None

    def exists(self, relative: str) -> bool:
        """Inspect presence of markers without reading their content."""
        try:
            path = self.path(relative)
            present = path.is_file()
            self.files.setdefault(
                relative,
                {"path": relative, "state": "present" if present else "missing"},
            )
            return present
        except (OSError, ValueError):
            self.issue(relative, "marker_unreadable_or_symlinked")
            return False

    def finish(self) -> None:
        """Detect observed file changes; a stable read is not a global lock."""
        for name, entry in self.files.items():
            try:
                path = self.path(name)
                present = path.is_file()
                if entry["state"] == "missing" and present:
                    self.issue(name, "appeared_during_collection")
                elif entry["state"] in {"read", "present"} and not present:
                    self.issue(name, "disappeared_during_collection")
                elif entry["state"] == "read":
                    stat = path.stat()
                    if (stat.st_size, stat.st_mtime_ns) != (
                        entry["bytes"],
                        entry["mtime_ns"],
                    ):
                        self.issue(name, "changed_during_collection")
            except (OSError, ValueError):
                self.issue(name, "changed_during_collection")
