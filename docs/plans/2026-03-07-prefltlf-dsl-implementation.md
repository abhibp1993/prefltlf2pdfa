# PrefLTLf DSL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `.spec` DSL front-end that parses keyword-based PrefLTLf specs and transpiles them to the existing index-based `.prefltlf` format.

**Architecture:** A `prefltlf2pdfa/dsl/` sub-package contains a Lark grammar, a Transformer-based parser producing `Spec` dataclasses, a `Transpiler` class that emits `.prefltlf` strings/files, and a CLI entry point. Alphabet and options blocks are parsed as raw stubs (not deeply processed). The core `PrefLTLf` pipeline is untouched.

**Tech Stack:** Python 3.10+, `lark` (already in requirements.txt), `pytest` (dev dependency), `difflib` (stdlib), `argparse` (stdlib).

**Working branch:** `feature/dsl-grammar`

---

## Background: How the Existing Format Works

The existing `.prefltlf` format (consumed by `PrefLTLf` in `prefltlf2pdfa/prefltlf.py`) looks like:

```
prefltlf 3

F a
G b
!(F(a) | G(b))

>, 0, 1
>=, 0, 2
>=, 1, 2
```

The new DSL transpiles to this format. `PrefLTLf(spec_string)` accepts this string directly.

## Key Grammar Notes

- `FORMULA_BODY: /[^\n#]+/` captures raw LTLf text to end-of-line — Lark matches terminals greedily before `%ignore` strips separators between tokens, so internal spaces are preserved.
- `PAREN_BODY: /[^)]+/` captures formula text inside `()` for exact formula references in preferences.
- `VERB_PHRASE.2: /strictly preferred|weakly preferred|indifferent|incomparable/` has priority 2 > `IDENT`'s default 0, so it wins over `IDENT` at those positions.
- Alphabet and options blocks are extracted via regex **before** Lark parsing to avoid grammar complexity.
- Use `parser="earley"` with `propagate_positions=True` for flexible parsing and line numbers.

---

## Task 1: Scaffold

**Files:**
- Create: `prefltlf2pdfa/dsl/__init__.py`
- Create: `prefltlf2pdfa/dsl/grammar.lark` (empty placeholder)
- Create: `prefltlf2pdfa/dsl/parser.py` (empty placeholder)
- Create: `prefltlf2pdfa/dsl/transpiler.py` (empty placeholder)
- Create: `prefltlf2pdfa/dsl/cli.py` (empty placeholder)
- Create: `tests/__init__.py`
- Create: `tests/fixtures/` (directory)
- Create: `tests/fixtures/erroneous/` (directory)
- Create: `requirements-dev.txt`

**Step 1: Create directory structure**

```bash
mkdir -p prefltlf2pdfa/dsl tests/fixtures/erroneous
touch prefltlf2pdfa/dsl/__init__.py
touch prefltlf2pdfa/dsl/grammar.lark
touch prefltlf2pdfa/dsl/parser.py
touch prefltlf2pdfa/dsl/transpiler.py
touch prefltlf2pdfa/dsl/cli.py
touch tests/__init__.py
```

**Step 2: Create `requirements-dev.txt`**

```
pytest
pytest-cov
```

**Step 3: Install dev dependencies**

```bash
pip install -r requirements-dev.txt
```

**Step 4: Verify pytest works**

```bash
python -m pytest --version
```
Expected: `pytest X.Y.Z`

**Step 5: Commit**

```bash
git add prefltlf2pdfa/dsl/ tests/ requirements-dev.txt
git commit -m "#add: scaffold dsl package and test structure"
```

---

## Task 2: DSLError and Spec Dataclasses

**Files:**
- Create: `prefltlf2pdfa/dsl/errors.py`
- Create: `prefltlf2pdfa/dsl/models.py`
- Create: `tests/test_dsl_models.py`

**Step 1: Write failing tests**

Create `tests/test_dsl_models.py`:

```python
import pytest
from prefltlf2pdfa.dsl.errors import DSLError
from prefltlf2pdfa.dsl.models import FormulaDecl, PrefStmt, Spec


class TestDSLError:
    def test_message_only(self):
        err = DSLError("something went wrong")
        assert str(err) == "something went wrong"

    def test_with_line(self):
        err = DSLError("bad token", line=7)
        assert str(err) == "Line 7: bad token"

    def test_with_suggestion(self):
        err = DSLError("unknown formula 'f1'", line=3, suggestion="f0")
        assert str(err) == "Line 3: unknown formula 'f1'. Did you mean 'f0'?"

    def test_is_value_error(self):
        assert isinstance(DSLError("x"), ValueError)


class TestModels:
    def test_formula_decl(self):
        fd = FormulaDecl(name="safety", ltlf_str="G safe", line=2)
        assert fd.name == "safety"
        assert fd.ltlf_str == "G safe"
        assert fd.line == 2

    def test_pref_stmt(self):
        ps = PrefStmt(lhs="f0", op=">", rhs="f1", line=5)
        assert ps.lhs == "f0"
        assert ps.op == ">"
        assert ps.rhs == "f1"

    def test_spec_defaults(self):
        spec = Spec(formulas={"f0": "G safe"}, preferences=[])
        assert spec.propositions == []
        assert spec.alphabet is None
        assert spec.options is None
```

**Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_dsl_models.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`

**Step 3: Implement `errors.py`**

Create `prefltlf2pdfa/dsl/errors.py`:

```python
class DSLError(ValueError):
    def __init__(self, message: str, line: int | None = None, suggestion: str | None = None):
        self.msg = message
        self.line = line
        self.suggestion = suggestion
        super().__init__(self._format())

    def _format(self) -> str:
        parts = []
        if self.line is not None:
            parts.append(f"Line {self.line}:")
        parts.append(self.msg)
        if self.suggestion:
            parts.append(f"Did you mean '{self.suggestion}'?")
        return " ".join(parts)

    def __str__(self) -> str:
        return self._format()
```

**Step 4: Implement `models.py`**

Create `prefltlf2pdfa/dsl/models.py`:

```python
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
```

**Step 5: Run to verify they pass**

```bash
python -m pytest tests/test_dsl_models.py -v
```
Expected: All green.

**Step 6: Commit**

```bash
git add prefltlf2pdfa/dsl/errors.py prefltlf2pdfa/dsl/models.py tests/test_dsl_models.py
git commit -m "#add: DSLError and Spec dataclasses with tests"
```

---

## Task 3: Grammar File

**Files:**
- Modify: `prefltlf2pdfa/dsl/grammar.lark`

**Step 1: Write the grammar**

Replace `prefltlf2pdfa/dsl/grammar.lark` with:

```lark
// PrefLTLf DSL Grammar
// Transpiles to index-based .prefltlf format.
// NOTE: alphabet and options blocks are extracted by regex pre-processing
//       BEFORE this grammar is applied. They do not appear here.

spec: propositions_block? formulas_block preferences_block

// ─── Propositions (optional) ─────────────────────────────────────────────────
propositions_block: "propositions" prop_item+ "end" "propositions"
prop_item: IDENT ("," IDENT)*

// ─── LTLf formulas ───────────────────────────────────────────────────────────
formulas_block: "ltlf-formulas" formula_decl+ "end" "ltlf-formulas"
formula_decl: IDENT ":" FORMULA_BODY

// ─── Preferences ─────────────────────────────────────────────────────────────
preferences_block: "preferences" pref_stmt+ "end" "preferences"
pref_stmt: pref_chain | pref_verbatim
pref_chain: pref_term (PREF_OP pref_term)+
pref_verbatim: pref_term "is" VERB_PHRASE "to" pref_term
pref_term: IDENT       -> pref_name
         | "(" PAREN_BODY ")" -> pref_exact

// ─── Terminals ───────────────────────────────────────────────────────────────
// PREF_OP: longer tokens listed first so ">=" matches before ">"
PREF_OP: ">=" | "<=" | "<>" | ">" | "<" | "~"

// VERB_PHRASE: priority 2 beats IDENT (priority 0) at these positions
VERB_PHRASE.2: /strictly preferred|weakly preferred|indifferent|incomparable/

// FORMULA_BODY captures rest-of-line (spaces allowed; stops at # or newline)
FORMULA_BODY: /[^\n#]+/

// PAREN_BODY captures content inside parentheses for exact formula references
PAREN_BODY: /[^)]+/

IDENT: /[a-zA-Z_][a-zA-Z0-9_]*/

%ignore /[ \t]+/
%ignore /\r?\n/
%ignore /#[^\n]*/
```

**Step 2: Write a basic grammar smoke test**

Add to `tests/test_dsl_models.py` (or create `tests/test_dsl_grammar.py`):

```python
# tests/test_dsl_grammar.py
from pathlib import Path
import lark

GRAMMAR_PATH = Path(__file__).parent.parent / "prefltlf2pdfa" / "dsl" / "grammar.lark"

MINIMAL_SPEC = """
ltlf-formulas
  f0: G safe
  f1: F clean
end ltlf-formulas

preferences
  f0 > f1
end preferences
"""


def test_grammar_loads():
    parser = lark.Lark.open(str(GRAMMAR_PATH), parser="earley", propagate_positions=True)
    assert parser is not None


def test_grammar_parses_minimal_spec():
    parser = lark.Lark.open(str(GRAMMAR_PATH), parser="earley", propagate_positions=True)
    tree = parser.parse(MINIMAL_SPEC)
    assert tree.data == "spec"
```

**Step 3: Run to verify**

```bash
python -m pytest tests/test_dsl_grammar.py -v
```
Expected: Both tests pass (grammar loads and parses without error).

**Step 4: Commit**

```bash
git add prefltlf2pdfa/dsl/grammar.lark tests/test_dsl_grammar.py
git commit -m "#add: Lark grammar for PrefLTLf DSL"
```

---

## Task 4: Parser — Formulas Block

**Files:**
- Modify: `prefltlf2pdfa/dsl/parser.py`
- Create: `tests/test_dsl_parser.py`

**Step 1: Write failing tests**

Create `tests/test_dsl_parser.py`:

```python
import pytest
from prefltlf2pdfa.dsl.parser import parse_spec
from prefltlf2pdfa.dsl.models import Spec
from prefltlf2pdfa.dsl.errors import DSLError


MINIMAL = """
ltlf-formulas
  safety: G safe
  liveness: F clean
end ltlf-formulas

preferences
  safety > liveness
end preferences
"""


class TestFormulasBlock:
    def test_formula_names_parsed(self):
        spec = parse_spec(MINIMAL)
        assert list(spec.formulas.keys()) == ["safety", "liveness"]

    def test_formula_bodies_parsed(self):
        spec = parse_spec(MINIMAL)
        assert spec.formulas["safety"] == "G safe"
        assert spec.formulas["liveness"] == "F clean"

    def test_formula_declaration_order_preserved(self):
        src = """
ltlf-formulas
  z_last: true
  a_first: false
end ltlf-formulas

preferences
  z_last >= a_first
end preferences
"""
        spec = parse_spec(src)
        assert list(spec.formulas.keys()) == ["z_last", "a_first"]

    def test_single_formula(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas

preferences
  f0 >= f0
end preferences
"""
        spec = parse_spec(src)
        assert len(spec.formulas) == 1
        assert spec.formulas["f0"] == "G p"

    def test_formula_body_with_complex_ltlf(self):
        src = """
ltlf-formulas
  complex: F(a) & G(b | !c)
end ltlf-formulas

preferences
  complex >= complex
end preferences
"""
        spec = parse_spec(src)
        assert spec.formulas["complex"] == "F(a) & G(b | !c)"

    def test_underscore_in_name(self):
        src = """
ltlf-formulas
  safe_first: G safe
end ltlf-formulas

preferences
  safe_first >= safe_first
end preferences
"""
        spec = parse_spec(src)
        assert "safe_first" in spec.formulas

    def test_comments_ignored(self):
        src = """
# This is a top-level comment
ltlf-formulas
  f0: G safe  # inline comment
end ltlf-formulas

preferences
  f0 >= f0
end preferences
"""
        spec = parse_spec(src)
        assert spec.formulas["f0"] == "G safe"
```

**Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_dsl_parser.py::TestFormulasBlock -v
```
Expected: `ImportError` — `parse_spec` not defined.

**Step 3: Implement `parser.py` (formulas block only)**

Create `prefltlf2pdfa/dsl/parser.py`:

```python
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
            text = text[: m.start()] + text[m.end() :]
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
```

**Step 4: Run to verify formulas tests pass**

```bash
python -m pytest tests/test_dsl_parser.py::TestFormulasBlock -v
```
Expected: All green.

**Step 5: Commit**

```bash
git add prefltlf2pdfa/dsl/parser.py tests/test_dsl_parser.py
git commit -m "#add: parser Transformer with formulas block support"
```

---

## Task 5: Parser — Preferences (Operator Form + Normalization)

**Files:**
- Modify: `tests/test_dsl_parser.py` (add tests)
- No code changes needed — parser.py already handles this

**Step 1: Add preference operator tests to `tests/test_dsl_parser.py`**

```python
class TestPreferenceOperators:
    def _spec_with_prefs(self, pref_line: str) -> Spec:
        return parse_spec(f"""
ltlf-formulas
  f0: G p
  f1: F q
end ltlf-formulas

preferences
  {pref_line}
end preferences
""")

    def test_strict_preference(self):
        spec = self._spec_with_prefs("f0 > f1")
        assert len(spec.preferences) == 1
        assert spec.preferences[0].op == ">"
        assert spec.preferences[0].lhs == "f0"
        assert spec.preferences[0].rhs == "f1"

    def test_weak_preference(self):
        spec = self._spec_with_prefs("f0 >= f1")
        assert spec.preferences[0].op == ">="

    def test_indifferent(self):
        spec = self._spec_with_prefs("f0 ~ f1")
        assert spec.preferences[0].op == "~"

    def test_incomparable(self):
        spec = self._spec_with_prefs("f0 <> f1")
        assert spec.preferences[0].op == "<>"

    def test_reverse_strict_normalized(self):
        # f0 < f1  →  PrefStmt(lhs="f1", op=">", rhs="f0")
        spec = self._spec_with_prefs("f0 < f1")
        p = spec.preferences[0]
        assert p.op == ">"
        assert p.lhs == "f1"
        assert p.rhs == "f0"

    def test_reverse_weak_normalized(self):
        spec = self._spec_with_prefs("f0 <= f1")
        p = spec.preferences[0]
        assert p.op == ">="
        assert p.lhs == "f1"
        assert p.rhs == "f0"

    def test_multiple_preference_statements(self):
        src = """
ltlf-formulas
  f0: G p
  f1: F q
  f2: true
end ltlf-formulas

preferences
  f0 > f1
  f1 >= f2
end preferences
"""
        spec = parse_spec(src)
        assert len(spec.preferences) == 2
```

**Step 2: Run**

```bash
python -m pytest tests/test_dsl_parser.py::TestPreferenceOperators -v
```
Expected: All green (already implemented in parser.py).

**Step 3: Commit**

```bash
git add tests/test_dsl_parser.py
git commit -m "#add: parser tests for preference operators and normalization"
```

---

## Task 6: Parser — Chain Expansion

**Files:**
- Modify: `tests/test_dsl_parser.py` (add tests)

**Step 1: Add chain expansion tests**

```python
class TestChainExpansion:
    def test_three_term_chain(self):
        src = """
ltlf-formulas
  f0: G p
  f1: F q
  f2: true
end ltlf-formulas

preferences
  f0 > f1 >= f2
end preferences
"""
        spec = parse_spec(src)
        assert len(spec.preferences) == 2
        assert spec.preferences[0] == PrefStmt(lhs="f0", op=">", rhs="f1", line=spec.preferences[0].line)
        assert spec.preferences[1] == PrefStmt(lhs="f1", op=">=", rhs="f2", line=spec.preferences[1].line)

    def test_four_term_chain(self):
        src = """
ltlf-formulas
  f0: G p
  f1: F q
  f2: true
  f3: false
end ltlf-formulas

preferences
  f0 > f1 >= f2 ~ f3
end preferences
"""
        spec = parse_spec(src)
        assert len(spec.preferences) == 3
        ops = [p.op for p in spec.preferences]
        assert ops == [">", ">=", "~"]

    def test_chain_with_normalization(self):
        # f0 < f1 > f2  →  [PrefStmt(f1,>,f0), PrefStmt(f1,>,f2)]
        src = """
ltlf-formulas
  f0: G p
  f1: F q
  f2: true
end ltlf-formulas

preferences
  f0 < f1 > f2
end preferences
"""
        spec = parse_spec(src)
        assert len(spec.preferences) == 2
        assert spec.preferences[0].op == ">"
        assert spec.preferences[0].lhs == "f1"
        assert spec.preferences[0].rhs == "f0"
```

**Step 2: Run**

```bash
python -m pytest tests/test_dsl_parser.py::TestChainExpansion -v
```
Expected: All green.

**Step 3: Commit**

```bash
git add tests/test_dsl_parser.py
git commit -m "#add: parser tests for chain preference expansion"
```

---

## Task 7: Parser — Verbatim Preferences

**Files:**
- Modify: `tests/test_dsl_parser.py` (add tests)

**Step 1: Add verbatim tests**

```python
class TestVerbatimPreferences:
    def _two_formula_spec(self, pref_line: str) -> Spec:
        return parse_spec(f"""
ltlf-formulas
  f0: G p
  f1: F q
end ltlf-formulas

preferences
  {pref_line}
end preferences
""")

    def test_strictly_preferred(self):
        spec = self._two_formula_spec("f0 is strictly preferred to f1")
        assert spec.preferences[0].op == ">"
        assert spec.preferences[0].lhs == "f0"
        assert spec.preferences[0].rhs == "f1"

    def test_weakly_preferred(self):
        spec = self._two_formula_spec("f0 is weakly preferred to f1")
        assert spec.preferences[0].op == ">="

    def test_indifferent(self):
        spec = self._two_formula_spec("f0 is indifferent to f1")
        assert spec.preferences[0].op == "~"

    def test_incomparable(self):
        spec = self._two_formula_spec("f0 is incomparable to f1")
        assert spec.preferences[0].op == "<>"

    def test_mixed_verbatim_and_operator(self):
        src = """
ltlf-formulas
  f0: G p
  f1: F q
  f2: true
end ltlf-formulas

preferences
  f0 is strictly preferred to f1
  f1 >= f2
end preferences
"""
        spec = parse_spec(src)
        assert len(spec.preferences) == 2
        assert spec.preferences[0].op == ">"
        assert spec.preferences[1].op == ">="
```

**Step 2: Run**

```bash
python -m pytest tests/test_dsl_parser.py::TestVerbatimPreferences -v
```
Expected: All green.

**Step 3: Commit**

```bash
git add tests/test_dsl_parser.py
git commit -m "#add: parser tests for verbatim preference phrases"
```

---

## Task 8: Parser — Exact Formula References and Optional Propositions

**Files:**
- Modify: `tests/test_dsl_parser.py` (add tests)

**Step 1: Add exact formula reference and propositions tests**

```python
class TestExactFormulaRefs:
    def test_exact_ref_in_preference(self):
        src = """
ltlf-formulas
  safety: G safe
  liveness: F clean
end ltlf-formulas

preferences
  (G safe) > (F clean)
end preferences
"""
        spec = parse_spec(src)
        p = spec.preferences[0]
        assert p.lhs == "G safe"
        assert p.rhs == "F clean"
        assert p.op == ">"

    def test_exact_ref_must_exist(self):
        src = """
ltlf-formulas
  safety: G safe
end ltlf-formulas

preferences
  (G safe) > (F nonexistent)
end preferences
"""
        with pytest.raises(DSLError, match="Unknown formula reference"):
            parse_spec(src)

    def test_mixed_name_and_exact_ref(self):
        src = """
ltlf-formulas
  safety: G safe
  liveness: F clean
end ltlf-formulas

preferences
  safety > (F clean)
end preferences
"""
        spec = parse_spec(src)
        assert spec.preferences[0].lhs == "safety"
        assert spec.preferences[0].rhs == "F clean"


class TestOptionalPropositions:
    def test_no_propositions_block(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas

preferences
  f0 >= f0
end preferences
"""
        spec = parse_spec(src)
        assert spec.propositions == []

    def test_propositions_comma_separated(self):
        src = """
propositions
  clean, charged, safe
end propositions

ltlf-formulas
  f0: G safe
end ltlf-formulas

preferences
  f0 >= f0
end preferences
"""
        spec = parse_spec(src)
        assert set(spec.propositions) == {"clean", "charged", "safe"}

    def test_propositions_one_per_line(self):
        src = """
propositions
  clean
  charged
  safe
end propositions

ltlf-formulas
  f0: G safe
end ltlf-formulas

preferences
  f0 >= f0
end preferences
"""
        spec = parse_spec(src)
        assert "safe" in spec.propositions
```

**Step 2: Run**

```bash
python -m pytest tests/test_dsl_parser.py::TestExactFormulaRefs tests/test_dsl_parser.py::TestOptionalPropositions -v
```
Expected: All green.

**Step 3: Commit**

```bash
git add tests/test_dsl_parser.py
git commit -m "#add: parser tests for exact refs and optional propositions"
```

---

## Task 9: Parser — Stub Blocks and Semantic Errors

**Files:**
- Modify: `tests/test_dsl_parser.py` (add tests)

**Step 1: Add stub and error tests**

```python
class TestStubBlocks:
    def _base_spec(self, extra_block: str = "") -> str:
        return f"""
{extra_block}

ltlf-formulas
  f0: G safe
end ltlf-formulas

preferences
  f0 >= f0
end preferences
"""

    def test_alphabet_stub_stored_raw(self):
        src = self._base_spec("""
alphabet
  powerset()
  exclude charged
end alphabet
""")
        spec = parse_spec(src)
        assert spec.alphabet is not None
        assert "powerset" in spec.alphabet

    def test_options_stub_stored_raw(self):
        src = self._base_spec("""
options
  semantics = MaxAE
  auto-complete = minimal
end options
""")
        spec = parse_spec(src)
        assert spec.options is not None
        assert "MaxAE" in spec.options

    def test_no_stubs(self):
        spec = parse_spec(self._base_spec())
        assert spec.alphabet is None
        assert spec.options is None


class TestSemanticErrors:
    def test_duplicate_formula_name_raises(self):
        src = """
ltlf-formulas
  f0: G p
  f0: F q
end ltlf-formulas

preferences
  f0 > f0
end preferences
"""
        with pytest.raises(DSLError, match="Duplicate formula name 'f0'"):
            parse_spec(src)

    def test_duplicate_formula_name_has_line_number(self):
        src = """
ltlf-formulas
  f0: G p
  f0: F q
end ltlf-formulas

preferences
  f0 > f0
end preferences
"""
        with pytest.raises(DSLError) as exc_info:
            parse_spec(src)
        assert exc_info.value.line is not None

    def test_unknown_formula_name_raises(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas

preferences
  f0 > f_typo
end preferences
"""
        with pytest.raises(DSLError, match="Unknown formula reference 'f_typo'"):
            parse_spec(src)

    def test_unknown_name_suggests_close_match(self):
        src = """
ltlf-formulas
  safety: G safe
end ltlf-formulas

preferences
  safety > safty
end preferences
"""
        with pytest.raises(DSLError) as exc_info:
            parse_spec(src)
        assert "safety" in str(exc_info.value)  # suggestion present

    def test_syntax_error_raises_dsl_error(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas

preferences
  f0 ??? f0
end preferences
"""
        with pytest.raises(DSLError):
            parse_spec(src)

    def test_missing_formulas_block_raises(self):
        src = """
preferences
  f0 > f1
end preferences
"""
        with pytest.raises(DSLError):
            parse_spec(src)

    def test_missing_preferences_block_raises(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
"""
        with pytest.raises(DSLError):
            parse_spec(src)
```

**Step 2: Run**

```bash
python -m pytest tests/test_dsl_parser.py::TestStubBlocks tests/test_dsl_parser.py::TestSemanticErrors -v
```
Expected: All green.

**Step 3: Commit**

```bash
git add tests/test_dsl_parser.py
git commit -m "#add: parser tests for stub blocks and semantic errors"
```

---

## Task 10: Transpiler

**Files:**
- Modify: `prefltlf2pdfa/dsl/transpiler.py`
- Create: `tests/test_dsl_transpiler.py`

**Step 1: Write failing tests**

Create `tests/test_dsl_transpiler.py`:

```python
import pytest
from pathlib import Path
from prefltlf2pdfa.dsl.parser import parse_spec
from prefltlf2pdfa.dsl.transpiler import Transpiler
from prefltlf2pdfa.dsl.models import Spec, PrefStmt


def _transpile(src: str) -> str:
    spec = parse_spec(src)
    return Transpiler(spec).to_string()


SIMPLE_SPEC = """
ltlf-formulas
  safety: G safe
  liveness: F clean
  charge: charged U clean
end ltlf-formulas

preferences
  safety > liveness
  liveness >= charge
end preferences
"""


class TestTranspilerHeader:
    def test_header_formula_count(self):
        out = _transpile(SIMPLE_SPEC)
        first_line = out.strip().splitlines()[0]
        assert first_line == "prefltlf 3"

    def test_single_formula_header(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
"""
        out = _transpile(src)
        assert out.strip().startswith("prefltlf 1")


class TestTranspilerFormulas:
    def test_formulas_emitted_in_declaration_order(self):
        out = _transpile(SIMPLE_SPEC)
        lines = out.strip().splitlines()
        assert lines[1] == "G safe"
        assert lines[2] == "F clean"
        assert lines[3] == "charged U clean"

    def test_formula_bodies_match_input(self):
        src = """
ltlf-formulas
  f0: F(a) & G(b | !c)
end ltlf-formulas
preferences
  f0 >= f0
end preferences
"""
        out = _transpile(src)
        assert "F(a) & G(b | !c)" in out


class TestTranspilerPreferences:
    def test_strict_preference_emitted(self):
        out = _transpile(SIMPLE_SPEC)
        assert ">, 0, 1" in out

    def test_weak_preference_emitted(self):
        out = _transpile(SIMPLE_SPEC)
        assert ">=, 1, 2" in out

    def test_indifferent_emitted(self):
        src = """
ltlf-formulas
  f0: G p
  f1: F q
end ltlf-formulas
preferences
  f0 ~ f1
end preferences
"""
        out = _transpile(src)
        assert "~, 0, 1" in out

    def test_incomparable_emitted(self):
        src = """
ltlf-formulas
  f0: G p
  f1: F q
end ltlf-formulas
preferences
  f0 <> f1
end preferences
"""
        out = _transpile(src)
        assert "<>, 0, 1" in out

    def test_exact_ref_preference_uses_correct_index(self):
        src = """
ltlf-formulas
  safety: G safe
  liveness: F clean
end ltlf-formulas
preferences
  (G safe) > (F clean)
end preferences
"""
        out = _transpile(src)
        assert ">, 0, 1" in out

    def test_indices_reflect_declaration_order(self):
        src = """
ltlf-formulas
  z_last: false
  a_first: true
end ltlf-formulas
preferences
  z_last > a_first
end preferences
"""
        out = _transpile(src)
        assert ">, 0, 1" in out


class TestTranspilerOutput:
    def test_to_string_is_valid_prefltlf(self):
        """Output string must be parseable by PrefLTLf (without MONA call)."""
        from prefltlf2pdfa import PrefLTLf
        spec = parse_spec(SIMPLE_SPEC)
        result = Transpiler(spec).to_string()
        pf = PrefLTLf(result)
        assert len(pf.phi) == 3

    def test_to_file_writes_correct_content(self, tmp_path):
        spec = parse_spec(SIMPLE_SPEC)
        t = Transpiler(spec)
        out_file = tmp_path / "test_output.prefltlf"
        t.to_file(out_file)
        assert out_file.exists()
        assert out_file.read_text() == t.to_string()
```

**Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_dsl_transpiler.py -v
```
Expected: `ImportError` — `Transpiler` not defined.

**Step 3: Implement `transpiler.py`**

Create `prefltlf2pdfa/dsl/transpiler.py`:

```python
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
```

**Step 4: Run to verify tests pass**

```bash
python -m pytest tests/test_dsl_transpiler.py -v
```
Expected: All green.

**Step 5: Commit**

```bash
git add prefltlf2pdfa/dsl/transpiler.py tests/test_dsl_transpiler.py
git commit -m "#add: Transpiler (Spec → .prefltlf) with tests"
```

---

## Task 11: Public API

**Files:**
- Modify: `prefltlf2pdfa/dsl/__init__.py`

**Step 1: Write the public API**

Replace `prefltlf2pdfa/dsl/__init__.py` with:

```python
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
```

**Step 2: Smoke test the public API**

```bash
python -c "from prefltlf2pdfa.dsl import parse_spec, transpile, DSLError; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add prefltlf2pdfa/dsl/__init__.py
git commit -m "#add: public API for prefltlf2pdfa.dsl"
```

---

## Task 12: CLI and setup.py Entry Point

**Files:**
- Modify: `prefltlf2pdfa/dsl/cli.py`
- Modify: `setup.py`
- Create: `tests/test_dsl_cli.py`

**Step 1: Write failing CLI tests**

Create `tests/test_dsl_cli.py`:

```python
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
MINIMAL_SPEC = FIXTURES / "minimal.spec"


class TestCLI:
    def test_stdout_flag_prints_prefltlf(self):
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli", str(MINIMAL_SPEC), "--stdout"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert result.stdout.startswith("prefltlf")

    def test_output_file_flag(self, tmp_path):
        out_file = tmp_path / "out.prefltlf"
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli", str(MINIMAL_SPEC), "-o", str(out_file)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert out_file.exists()
        assert out_file.read_text().startswith("prefltlf")

    def test_default_output_next_to_input(self, tmp_path):
        import shutil
        spec_copy = tmp_path / "test.spec"
        shutil.copy(MINIMAL_SPEC, spec_copy)
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli", str(spec_copy)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert (tmp_path / "test.prefltlf").exists()

    def test_missing_file_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli", "nonexistent.spec"],
            capture_output=True, text=True
        )
        assert result.returncode != 0

    def test_dsl_error_exits_nonzero(self, tmp_path):
        bad = tmp_path / "bad.spec"
        bad.write_text("ltlf-formulas\n  f0: G p\nend ltlf-formulas\npreferences\n  f0 > unknown\nend preferences\n")
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli", str(bad), "--stdout"],
            capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "unknown" in result.stderr.lower() or "unknown" in result.stdout.lower()
```

**Step 2: Create `tests/fixtures/minimal.spec`**

```
ltlf-formulas
  safety: G safe
  liveness: F clean
end ltlf-formulas

preferences
  safety > liveness
end preferences
```

**Step 3: Run to verify tests fail**

```bash
python -m pytest tests/test_dsl_cli.py -v
```
Expected: Failures (CLI not implemented yet).

**Step 4: Implement `cli.py`**

Create `prefltlf2pdfa/dsl/cli.py`:

```python
"""
CLI entry point for prefltlf-compile.

Usage:
    prefltlf-compile input.spec
    prefltlf-compile input.spec -o output.prefltlf
    prefltlf-compile input.spec --stdout
"""

import argparse
import sys
from pathlib import Path

from .errors import DSLError
from .parser import parse_spec
from .transpiler import Transpiler


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="prefltlf-compile",
        description="Compile a .spec DSL file to the .prefltlf index-based format.",
    )
    parser.add_argument("input", type=Path, help="Input .spec file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .prefltlf file path")
    parser.add_argument("--stdout", action="store_true", help="Print output to stdout instead of a file")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        text = args.input.read_text(encoding="utf-8")
        spec = parse_spec(text)
        t = Transpiler(spec)
        result = t.to_string()
    except DSLError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.stdout:
        print(result, end="")
    else:
        out_path = args.output or args.input.with_suffix(".prefltlf")
        out_path.write_text(result, encoding="utf-8")
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
```

**Step 5: Add console_scripts to `setup.py`**

In `setup.py`, find the `entry_points` comment block and replace it with:

```python
    entry_points={
        "console_scripts": [
            "prefltlf-compile=prefltlf2pdfa.dsl.cli:main",
        ],
    },
```

Then reinstall:
```bash
pip install -e .
```

**Step 6: Run tests**

```bash
python -m pytest tests/test_dsl_cli.py -v
```
Expected: All green.

**Step 7: Commit**

```bash
git add prefltlf2pdfa/dsl/cli.py setup.py tests/test_dsl_cli.py tests/fixtures/minimal.spec
git commit -m "#add: CLI (prefltlf-compile) with tests"
```

---

## Task 13: Test Fixtures and Integration Tests

**Files:**
- Create: All remaining `tests/fixtures/*.spec` files
- Create: `tests/test_dsl_integration.py`

**Step 1: Create fixture files**

`tests/fixtures/full.spec`:
```
propositions
  safe, clean, charged
end propositions

alphabet
  powerset()
end alphabet

options
  semantics = MaxAE
end options

ltlf-formulas
  safety: G safe
  liveness: F clean
  charge: charged U clean
end ltlf-formulas

preferences
  safety > liveness >= charge
end preferences
```

`tests/fixtures/chain_prefs.spec`:
```
ltlf-formulas
  f0: G p
  f1: F q
  f2: true
  f3: false
end ltlf-formulas

preferences
  f0 > f1 >= f2 ~ f3
end preferences
```

`tests/fixtures/verbatim_prefs.spec`:
```
ltlf-formulas
  f0: G p
  f1: F q
  f2: true
  f3: false
end ltlf-formulas

preferences
  f0 is strictly preferred to f1
  f1 is weakly preferred to f2
  f2 is indifferent to f3
  f3 is incomparable to f0
end preferences
```

`tests/fixtures/exact_refs.spec`:
```
ltlf-formulas
  safety: G safe
  liveness: F clean
end ltlf-formulas

preferences
  (G safe) > (F clean)
end preferences
```

`tests/fixtures/reverse_ops.spec`:
```
ltlf-formulas
  f0: G p
  f1: F q
  f2: true
end ltlf-formulas

preferences
  f0 < f1
  f1 <= f2
end preferences
```

`tests/fixtures/no_propositions.spec`:
```
ltlf-formulas
  safety: G safe
  liveness: F clean
end ltlf-formulas

preferences
  safety > liveness
end preferences
```

`tests/fixtures/erroneous/unknown_name.spec`:
```
ltlf-formulas
  f0: G p
end ltlf-formulas

preferences
  f0 > f_does_not_exist
end preferences
```

`tests/fixtures/erroneous/duplicate_name.spec`:
```
ltlf-formulas
  f0: G p
  f0: F q
end ltlf-formulas

preferences
  f0 > f0
end preferences
```

`tests/fixtures/erroneous/syntax_error.spec`:
```
ltlf-formulas
  f0: G p
end ltlf-formulas

preferences
  f0 ??? f0
end preferences
```

**Step 2: Write integration tests**

Create `tests/test_dsl_integration.py`:

```python
import pytest
from pathlib import Path
from prefltlf2pdfa.dsl import parse_spec, Transpiler, DSLError
from prefltlf2pdfa import PrefLTLf

FIXTURES = Path(__file__).parent / "fixtures"


class TestAllFixturesParse:
    """Smoke test: all valid fixtures parse without error."""

    @pytest.mark.parametrize("spec_file", [
        "minimal.spec",
        "full.spec",
        "chain_prefs.spec",
        "verbatim_prefs.spec",
        "exact_refs.spec",
        "reverse_ops.spec",
        "no_propositions.spec",
    ])
    def test_fixture_parses(self, spec_file):
        text = (FIXTURES / spec_file).read_text()
        spec = parse_spec(text)
        assert spec is not None
        assert len(spec.formulas) > 0

    @pytest.mark.parametrize("spec_file", [
        "erroneous/unknown_name.spec",
        "erroneous/duplicate_name.spec",
        "erroneous/syntax_error.spec",
    ])
    def test_erroneous_fixture_raises(self, spec_file):
        text = (FIXTURES / spec_file).read_text()
        with pytest.raises(DSLError):
            parse_spec(text)


class TestEquivalenceToLegacyFormat:
    """Transpiled output must match hand-written .prefltlf equivalents."""

    def test_minimal_spec_matches_legacy(self):
        text = (FIXTURES / "minimal.spec").read_text()
        spec = parse_spec(text)
        result = Transpiler(spec).to_string()

        expected = "prefltlf 2\n\nG safe\nF clean\n\n>, 0, 1\n"
        assert result == expected

    def test_chain_prefs_expands_correctly(self):
        text = (FIXTURES / "chain_prefs.spec").read_text()
        spec = parse_spec(text)
        result = Transpiler(spec).to_string()

        assert "prefltlf 4" in result
        assert ">, 0, 1" in result
        assert ">=, 1, 2" in result
        assert "~, 2, 3" in result

    def test_reverse_ops_normalized_in_output(self):
        text = (FIXTURES / "reverse_ops.spec").read_text()
        spec = parse_spec(text)
        result = Transpiler(spec).to_string()

        # f0 < f1 → f1 > f0 → >, 1, 0
        assert ">, 1, 0" in result
        # f1 <= f2 → f2 >= f1 → >=, 2, 1
        assert ">=, 2, 1" in result


class TestEndToEndPipeline:
    """Full pipeline: .spec text → PrefLTLf object (no MONA needed)."""

    def test_parse_to_prefltlf_object(self):
        text = (FIXTURES / "minimal.spec").read_text()
        spec = parse_spec(text)
        pf = Transpiler(spec).to_prefltlf()
        assert isinstance(pf, PrefLTLf)
        assert len(pf.phi) == 2

    def test_full_spec_to_prefltlf(self):
        text = (FIXTURES / "full.spec").read_text()
        spec = parse_spec(text)
        pf = Transpiler(spec).to_prefltlf()
        assert len(pf.phi) == 3

    def test_to_file_then_read_back(self, tmp_path):
        text = (FIXTURES / "minimal.spec").read_text()
        spec = parse_spec(text)
        t = Transpiler(spec)
        out = tmp_path / "output.prefltlf"
        t.to_file(out)
        pf = PrefLTLf.from_file(out)
        assert len(pf.phi) == 2
```

**Step 3: Run all integration tests**

```bash
python -m pytest tests/test_dsl_integration.py -v
```
Expected: All green.

**Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: All tests pass across all test files.

**Step 5: Commit**

```bash
git add tests/fixtures/ tests/test_dsl_integration.py
git commit -m "#add: test fixtures and integration tests for DSL pipeline"
```

---

## Task 14: Final Wiring and Smoke Check

**Files:**
- Verify: `prefltlf2pdfa/__init__.py` (no changes needed — dsl is a sub-package)

**Step 1: Verify CLI is accessible after install**

```bash
pip install -e .
prefltlf-compile --help
```
Expected: Help text printed.

**Step 2: End-to-end CLI smoke test**

```bash
prefltlf-compile tests/fixtures/minimal.spec --stdout
```
Expected:
```
prefltlf 2

G safe
F clean

>, 0, 1
```

**Step 3: Run full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All green.

**Step 4: Final commit**

```bash
git add -A
git commit -m "#add: PrefLTLf DSL (.spec) grammar, parser, transpiler, CLI — complete"
```

---

## Grammar Troubleshooting Notes

If grammar tests fail unexpectedly, check these common Lark issues:

1. **`PREF_OP` ambiguity** — `>=` must be listed before `>` in the terminal definition (already done). If Lark still tokenizes `>=` as `>` then `=`, give `>=` explicit higher priority: `PREF_GE.1: ">="`.

2. **`VERB_PHRASE` not matching** — if "strictly preferred" isn't recognized, confirm `VERB_PHRASE.2` has priority 2 and that the regex uses `|` without spaces around it: `/strictly preferred|weakly preferred|.../`.

3. **`FORMULA_BODY` eating too much** — if formula bodies bleed across lines, check that `%ignore /\r?\n/` is NOT consuming newlines before `FORMULA_BODY` can stop at them. If this happens, change `%ignore /[ \t\r\n]+/` to `%ignore /[ \t]+/` only, and add explicit newline handling.

4. **Exact paren refs not matching** — if `(G safe)` isn't parsed, check that `PAREN_BODY: /[^)]+/` has higher priority than `IDENT` at that position. Add `.1` priority: `PAREN_BODY.1: /[^)]+/`.
