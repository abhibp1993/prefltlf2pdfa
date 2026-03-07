from pathlib import Path
from .models import Spec, SpecOptions
from .errors import DSLError
from prefltlf2pdfa.semantics import (
    semantics_forall_exists, semantics_exists_forall, semantics_forall_forall,
    semantics_mp_forall_exists, semantics_mp_exists_forall, semantics_mp_forall_forall,
)
from ltlf2dfa.parser.ltlf import LTLfParser as _LTLfParser
import prefltlf2pdfa.utils as _utils

_LTLF_PARSER = _LTLfParser()

_SEMANTICS_MAP = {
    "AE": "semantics_forall_exists",
    "forall-exists": "semantics_forall_exists",
    "EA": "semantics_exists_forall",
    "exists-forall": "semantics_exists_forall",
    "AA": "semantics_forall_forall",
    "forall-forall": "semantics_forall_forall",
    "MaxAE": "semantics_mp_forall_exists",
    "max-forall-exists": "semantics_mp_forall_exists",
    "MaxEA": "semantics_mp_exists_forall",
    "max-exists-forall": "semantics_mp_exists_forall",
    "MaxAA": "semantics_mp_forall_forall",
    "max-forall-forall": "semantics_mp_forall_forall",
}

_VALID_AUTO_COMPLETE = {"none", "minimal", "incomparable"}


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
        self._options = self._parse_options()
        self._validate_propositions()
        self._alphabet = self._build_alphabet()

    def _parse_options(self) -> SpecOptions:
        if self._spec.options is None:
            return SpecOptions()
        opts = SpecOptions()
        for line in self._spec.options.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise DSLError(f"Invalid option line: '{line}' (expected 'key = value')")
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "semantics":
                if value not in _SEMANTICS_MAP:
                    raise DSLError(
                        f"Unknown semantics alias '{value}'. "
                        f"Valid aliases: {sorted(_SEMANTICS_MAP.keys())}"
                    )
                opts.semantics = value
            elif key == "auto-complete":
                if value not in _VALID_AUTO_COMPLETE:
                    raise DSLError(
                        f"Unknown auto-complete value '{value}'. "
                        f"Valid values: {sorted(_VALID_AUTO_COMPLETE)}"
                    )
                opts.auto_complete = value
            else:
                raise DSLError(f"Unknown option key '{key}'. Valid keys: semantics, auto-complete")
        return opts

    def _validate_propositions(self) -> None:
        """If propositions block present, verify every atom in every formula is declared."""
        if not self._spec.propositions:
            return
        declared = set(self._spec.propositions)
        for name, formula_str in self._spec.formulas.items():
            parsed = _LTLF_PARSER(formula_str.strip())
            atoms = set(parsed.find_labels())
            undeclared = atoms - declared
            if undeclared:
                raise DSLError(
                    f"Formula '{name}' uses undeclared proposition(s): {sorted(undeclared)}. "
                    f"Declared: {sorted(declared)}"
                )

    def _build_alphabet(self) -> list | None:
        """Build alphabet from alphabet block or default to powerset(propositions)."""
        declared = set(self._spec.propositions) if self._spec.propositions else None

        # No alphabet block
        if self._spec.alphabet is None:
            if declared:
                return _utils.powerset(declared)
            return None

        raw = self._spec.alphabet.strip()

        # powerset() shorthand as sole content
        if raw == "powerset()":
            if not declared:
                raise DSLError(
                    "'powerset()' in alphabet block requires a propositions block"
                )
            return _utils.powerset(declared)

        # Parse explicit sets (and possibly inline powerset() tokens)
        tokens: list[str] = []
        for line in raw.splitlines():
            for token in line.split(";"):
                token = token.strip()
                if token:
                    tokens.append(token)

        alphabet: list[set] = []
        for token in tokens:
            if token == "powerset()":
                if not declared:
                    raise DSLError(
                        "'powerset()' in alphabet block requires a propositions block"
                    )
                alphabet.extend(_utils.powerset(declared))
                continue
            if not (token.startswith("{") and token.endswith("}")):
                raise DSLError(
                    f"Invalid alphabet entry: '{token}'. "
                    f"Expected '{{prop, ...}}' or 'powerset()'"
                )
            inner = token[1:-1].strip()
            s = set() if not inner else {p.strip() for p in inner.split(",")}
            if declared is not None:
                undeclared = s - declared
                if undeclared:
                    raise DSLError(
                        f"Alphabet entry {token} contains undeclared proposition(s): "
                        f"{sorted(undeclared)}"
                    )
            alphabet.append(s)

        return alphabet

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
        """Parse the emitted string into a PrefLTLf object.

        Automatically injects self._alphabet unless caller explicitly passes alphabet=.
        All kwargs are forwarded to PrefLTLf.__init__() (e.g. auto_complete='minimal').
        """
        from prefltlf2pdfa import PrefLTLf
        if "alphabet" not in kwargs and self._alphabet is not None:
            kwargs["alphabet"] = self._alphabet
        return PrefLTLf(self.to_string(), **kwargs)

    def to_pdfa(self, **kwargs):
        """Translate to a PrefAutomaton.

        Options from the spec's options block are used as defaults.
        Any kwargs passed here override the options block values.

        kwargs:
            semantics: Callable — overrides options block 'semantics'
            auto_complete: str  — overrides options block 'auto-complete'
            (plus any kwargs accepted by PrefLTLf.translate())
        """
        _sem_fn_map = {
            "AE": semantics_forall_exists,
            "forall-exists": semantics_forall_exists,
            "EA": semantics_exists_forall,
            "exists-forall": semantics_exists_forall,
            "AA": semantics_forall_forall,
            "forall-forall": semantics_forall_forall,
            "MaxAE": semantics_mp_forall_exists,
            "max-forall-exists": semantics_mp_forall_exists,
            "MaxEA": semantics_mp_exists_forall,
            "max-exists-forall": semantics_mp_exists_forall,
            "MaxAA": semantics_mp_forall_forall,
            "max-forall-forall": semantics_mp_forall_forall,
        }
        semantics = kwargs.pop("semantics", _sem_fn_map[self._options.semantics])
        auto_complete = kwargs.pop("auto_complete", self._options.auto_complete)
        pf = self.to_prefltlf(auto_complete=auto_complete)
        return pf.translate(semantics=semantics, **kwargs)
