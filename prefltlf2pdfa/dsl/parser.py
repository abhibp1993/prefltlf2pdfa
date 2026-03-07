import re
import difflib
from pathlib import Path

import lark
from lark import Transformer, Token, Tree

from .errors import DSLError
from .models import FormulaDecl, PrefStmt, Spec

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
_PARSER = lark.Lark.open(
    str(_GRAMMAR_PATH),
    parser="earley",
    start="spec",
    propagate_positions=True,
)

# Maps verbatim phrases to canonical operators
_VERBATIM_TO_OP = {
    "strictly preferred": ">",
    "weakly preferred": ">=",
    "indifferent": "~",
    "incomparable": "<>",
}


def _extract_stubs(text: str) -> tuple[str, dict[str, str]]:
    """Remove alphabet/options blocks from text; return cleaned text + raw block contents."""
    stubs: dict[str, str] = {}
    for block in ("alphabet", "options"):
        pattern = rf"\b{block}\b(.*?)\bend\s+{block}\b"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            stubs[block] = m.group(1).strip()
            text = text[: m.start()] + text[m.end():]
    return text, stubs


class _PrefLTLfTransformer(Transformer):
    """Transforms Lark parse tree into intermediate Python structures."""

    # --- Propositions ---

    def propositions_block(self, items):
        props = []
        for item in items:
            props.extend(item)
        return ("propositions", props)

    def prop_item(self, items):
        return [str(t) for t in items]

    # --- Formulas ---

    def formulas_block(self, items):
        decls = list(items)
        return ("formulas", decls)

    def formula_decl(self, items):
        name_token = items[0]
        body_token = items[1]
        return FormulaDecl(
            name=str(name_token),
            ltlf_str=str(body_token).strip(),
            line=getattr(name_token, "line", 0),
        )

    # --- Preferences ---

    def preferences_block(self, items):
        stmts = []
        for item in items:
            if isinstance(item, list):
                stmts.extend(item)
            else:
                stmts.append(item)
        return ("preferences", stmts)

    def pref_stmt(self, items):
        return items[0]

    def pref_chain(self, items):
        # items = [term, op, term, op, term, ...] interleaved
        result = []
        for i in range(0, len(items) - 1, 2):
            lhs_tok = items[i]
            op = str(items[i + 1])
            rhs_tok = items[i + 2]
            result.append(self._make_stmt(lhs_tok, op, rhs_tok))
        return result

    def pref_verbatim(self, items):
        lhs_tok, verb_tok, rhs_tok = items[0], items[1], items[2]
        op = _VERBATIM_TO_OP[str(verb_tok)]
        return [self._make_stmt(lhs_tok, op, rhs_tok)]

    def pref_name(self, items):
        return items[0]  # Token('IDENT', ...)

    def pref_exact(self, items):
        return items[0]  # Token('PAREN_BODY', ...)

    def _make_stmt(self, lhs_tok, op: str, rhs_tok) -> PrefStmt:
        lhs = str(lhs_tok).strip()
        rhs = str(rhs_tok).strip()
        line = getattr(lhs_tok, "line", 0)
        # Normalize reverse operators
        if op == "<":
            return PrefStmt(lhs=rhs, op=">", rhs=lhs, line=line)
        if op == "<=":
            return PrefStmt(lhs=rhs, op=">=", rhs=lhs, line=line)
        return PrefStmt(lhs=lhs, op=op, rhs=rhs, line=line)

    # --- Top-level ---

    def spec(self, items):
        propositions = []
        formulas_decls = []
        preferences = []
        for item in items:
            if isinstance(item, tuple):
                tag = item[0]
                if tag == "propositions":
                    propositions = item[1]
                elif tag == "formulas":
                    formulas_decls = item[1]
                elif tag == "preferences":
                    preferences = item[1]
        return (propositions, formulas_decls, preferences)


def _validate(
    propositions: list[str],
    formula_decls: list[FormulaDecl],
    preferences: list[PrefStmt],
) -> Spec:
    """Semantic validation: duplicates, unknown names, exact ref lookup."""
    # Check for duplicate formula names
    seen: dict[str, int] = {}
    for decl in formula_decls:
        if decl.name in seen:
            raise DSLError(
                f"Duplicate formula name '{decl.name}'",
                line=decl.line,
            )
        seen[decl.name] = decl.line

    formulas = {d.name: d.ltlf_str for d in formula_decls}
    formula_values = list(formulas.values())

    # Validate preference terms
    for stmt in preferences:
        for term in (stmt.lhs, stmt.rhs):
            if term in formulas:
                continue  # named reference: OK
            if term in formula_values:
                continue  # exact formula reference: OK
            # Unknown — suggest close match
            close = difflib.get_close_matches(term, list(formulas.keys()), n=1)
            suggestion = close[0] if close else None
            raise DSLError(
                f"Unknown formula reference '{term}'",
                line=stmt.line,
                suggestion=suggestion,
            )

    return Spec(
        propositions=propositions,
        formulas=formulas,
        preferences=preferences,
    )


def parse_spec(text: str) -> Spec:
    """Parse a DSL spec string into a Spec dataclass."""
    text, stubs = _extract_stubs(text)
    try:
        tree = _PARSER.parse(text)
    except lark.exceptions.UnexpectedInput as exc:
        raise DSLError(
            f"Syntax error: {exc.get_context(text)}",
            line=getattr(exc, "line", None),
        ) from exc

    transformer = _PrefLTLfTransformer()
    propositions, formula_decls, preferences = transformer.transform(tree)

    spec = _validate(propositions, formula_decls, preferences)
    spec.alphabet = stubs.get("alphabet")
    spec.options = stubs.get("options")
    return spec
