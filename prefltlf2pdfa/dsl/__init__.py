"""
PrefLTLf DSL front-end.

Parses .spec files (keyword-based DSL) and transpiles them to the
index-based .prefltlf format consumed by PrefLTLf.

Public API:
    parse_spec(text)          → Spec
    transpile(text)           → .prefltlf string
    DSLError                  → exception class
"""

from .errors import DSLError
from .models import Spec, FormulaDecl, PrefStmt
from .parser import parse_spec
from .transpiler import Transpiler


def transpile(text: str, **prefltlf_kwargs):
    """Parse DSL text and return a PrefLTLf object."""
    spec = parse_spec(text)
    return Transpiler(spec).to_prefltlf(**prefltlf_kwargs)


__all__ = [
    "DSLError",
    "Spec",
    "FormulaDecl",
    "PrefStmt",
    "parse_spec",
    "Transpiler",
    "transpile",
]
