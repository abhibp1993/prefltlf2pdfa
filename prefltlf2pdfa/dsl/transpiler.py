from pathlib import Path
from .models import Spec


class Transpiler:
    """Converts a parsed Spec into the index-based .prefltlf format."""

    def __init__(self, spec: Spec):
        self._spec = spec
        # Build index lookup: both formula names AND formula bodies → index
        self._name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(spec.formulas)
        }
        self._body_to_idx: dict[str, int] = {
            body: i for i, body in enumerate(spec.formulas.values())
        }

    def _resolve_term(self, term: str) -> int:
        """Resolve a preference term (name or exact body) to its 0-based index."""
        if term in self._name_to_idx:
            return self._name_to_idx[term]
        if term in self._body_to_idx:
            return self._body_to_idx[term]
        raise ValueError(f"Cannot resolve term '{term}' to an index (bug: should have been caught in parser)")

    def to_string(self) -> str:
        """Return the .prefltlf format string."""
        lines = []

        # Header
        n = len(self._spec.formulas)
        lines.append(f"prefltlf {n}")
        lines.append("")

        # Formulas (one per line, in declaration order)
        for ltlf_str in self._spec.formulas.values():
            lines.append(ltlf_str)
        lines.append("")

        # Preference relations
        for stmt in self._spec.preferences:
            i = self._resolve_term(stmt.lhs)
            j = self._resolve_term(stmt.rhs)
            lines.append(f"{stmt.op}, {i}, {j}")

        return "\n".join(lines) + "\n"

    def to_file(self, path: Path) -> None:
        """Write the .prefltlf string to a file."""
        Path(path).write_text(self.to_string(), encoding="utf-8")

    def to_prefltlf(self, **kwargs):
        """Parse the emitted string directly into a PrefLTLf object."""
        from prefltlf2pdfa import PrefLTLf
        return PrefLTLf(self.to_string(), **kwargs)
