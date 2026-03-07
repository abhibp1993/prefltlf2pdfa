from dataclasses import dataclass, field


@dataclass
class FormulaDecl:
    name: str
    ltlf_str: str
    line: int = 0


@dataclass
class PrefStmt:
    lhs: str    # formula name OR exact LTLf string (from paren ref)
    op: str     # one of: ">", ">=", "~", "<>"
    rhs: str
    line: int = 0


@dataclass
class Spec:
    formulas: dict[str, str]          # name → raw LTLf string, insertion-ordered
    preferences: list[PrefStmt]
    propositions: list[str] = field(default_factory=list)
    alphabet: str | None = None       # raw stub content
    options: str | None = None        # raw stub content
