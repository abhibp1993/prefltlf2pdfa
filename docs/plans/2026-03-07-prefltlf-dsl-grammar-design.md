# PrefLTLf DSL Grammar — Design Document

**Date:** 2026-03-07
**Status:** Approved

---

## Overview

Implement a new keyword-based DSL (`.spec` files) that transpiles to the existing index-based `.prefltlf` format. The new DSL is a front-end only — the core `PrefLTLf` pipeline is unchanged.

---

## Module Layout

```
prefltlf2pdfa/
    dsl/
        __init__.py          # public API: parse_dsl, transpile, DSLError
        grammar.lark         # Lark grammar file
        parser.py            # Lark + Transformer → Spec dataclass
        transpiler.py        # Spec → .prefltlf string / file / PrefLTLf object
        cli.py               # CLI entry point (argparse)

tests/
    test_dsl_parser.py
    test_dsl_transpiler.py
    test_dsl_integration.py
    fixtures/
        minimal.spec
        full.spec
        chain_prefs.spec
        verbatim_prefs.spec
        exact_refs.spec
        reverse_ops.spec
        no_propositions.spec
        erroneous/
            unknown_name.spec
            duplicate_name.spec
            syntax_error.spec
```

---

## Data Flow

```
.spec file / string
      │
      ▼
grammar.lark + Lark parser
      │
      ▼
Transformer (parser.py)  →  semantic validation
      │
      ▼
Spec dataclass
      │
      ├──► to_string() → .prefltlf string → PrefLTLf(spec_string)   [in-memory]
      └──► to_file()   → .prefltlf file on disk                      [file output]
```

---

## Grammar (grammar.lark)

```lark
spec: propositions_block? alphabet_block? options_block? formulas_block preferences_block

propositions_block: "propositions" prop_item+ "end" "propositions"
prop_item: IDENT ("," IDENT)*

// Stubs — parsed, stored raw, not deeply processed
alphabet_block: "alphabet" alphabet_stmt* "end" "alphabet"
options_block:  "options"  option_stmt*  "end" "options"

formulas_block: "ltlf-formulas" formula_decl+ "end" "ltlf-formulas"
formula_decl: IDENT ":" ltlf_expr
ltlf_expr: /[^\n]+/        // raw passthrough to ltlf2dfa

preferences_block: "preferences" pref_stmt+ "end" "preferences"
pref_stmt: pref_chain | pref_verbatim
pref_chain: pref_term (PREF_OP pref_term)+
pref_verbatim: pref_term "is" VERB_PHRASE "to" pref_term
pref_term: IDENT | "(" ltlf_expr ")"

PREF_OP: ">" | ">=" | "<" | "<=" | "~" | "<>"
VERB_PHRASE: "strictly preferred" | "weakly preferred" | "indifferent" | "incomparable"
IDENT: /[a-zA-Z_][a-zA-Z0-9_]*/
%ignore /\s+/
%ignore /#[^\n]*/
```

---

## Dataclasses (parser.py)

```python
@dataclass
class FormulaDecl:
    name: str
    ltlf_str: str
    line: int

@dataclass
class PrefStmt:
    lhs: str       # formula name or exact LTLf string
    op: str        # ">", ">=", "~", "<>"  (< and <= normalized at parse time)
    rhs: str
    line: int

@dataclass
class Spec:
    propositions: list[str]          # empty if block absent
    formulas: dict[str, str]         # name → raw LTLf string (ordered)
    preferences: list[PrefStmt]      # chains expanded pairwise
    alphabet: list | None            # stub: raw tokens
    options: dict | None             # stub: raw key-value pairs
```

---

## Semantic Validation (parser.py, post-Transformer)

1. **Duplicate formula names** → `DSLError` with line number
2. **Unknown formula name in preference** → `DSLError` with line number + `difflib.get_close_matches` suggestion
3. **Exact formula reference** (`(G safe)`) → string-matched against `formulas` values; `DSLError` if not found
4. **Chain expansion** → `f0 > f1 >= f2` expands pairwise to `[PrefStmt(f0,>,f1), PrefStmt(f1,>=,f2)]`
5. **Operator normalization** → `a < b` → `PrefStmt(b, >, a)`; `a <= b` → `PrefStmt(b, >=, a)`

---

## Transpiler (transpiler.py)

### Mapping to .prefltlf

| DSL element | .prefltlf output |
|-------------|-----------------|
| `propositions` block | discarded |
| `ltlf-formulas` declarations | `prefltlf N` header + one formula per line in declaration order |
| `preferences` statements | `op, i, j` lines using 0-based declaration indices |
| `alphabet` / `options` | passed as kwargs to `PrefLTLf()`, not emitted |

### API

```python
class Transpiler:
    def __init__(self, spec: Spec): ...
    def to_string(self) -> str
    def to_file(self, path: Path) -> None
    def to_prefltlf(self, **kwargs) -> PrefLTLf
```

---

## Error Handling

```python
class DSLError(ValueError):
    def __init__(self, message: str, line: int | None = None, suggestion: str | None = None): ...
    def __str__(self):
        # "Line 7: Unknown formula 'f1'. Did you mean 'f0'?"
```

Lark parse errors (`UnexpectedInput`) are caught and re-raised as `DSLError` with line/column from Lark's exception.

---

## CLI (cli.py)

```bash
prefltlf-compile input.spec                   # writes input.prefltlf next to input
prefltlf-compile input.spec -o out.prefltlf   # explicit output path
prefltlf-compile input.spec --stdout          # print to stdout
```

Registered as a `console_scripts` entry point in `setup.py`.

---

## Test Strategy

### test_dsl_parser.py

| Category | Cases |
|----------|-------|
| Valid minimal | formulas + preferences only |
| Valid full | all 5 blocks present |
| Propositions | optional block absent; comma-separated; one-per-line |
| Formula declarations | single; multiple; names with underscores |
| Preference operators | `>`, `>=`, `<`, `<=`, `~`, `<>` all parse correctly |
| Chain expansion | 3-term and 4-term chains expanded pairwise |
| Verbatim preferences | all 4 phrase variants |
| Exact formula refs | resolves when formula exists |
| Normalization | `<` and `<=` swapped and converted |
| Comments | `#` ignored mid-block and end-of-line |
| Whitespace | extra blank lines, leading/trailing spaces |
| Error: duplicate name | `DSLError` with correct line number |
| Error: unknown name | `DSLError` with `difflib` suggestion |
| Error: exact ref not found | `DSLError` with line number |
| Error: missing formulas block | `DSLError` |
| Error: missing preferences block | `DSLError` |
| Error: syntax error | `DSLError` with line/column |

### test_dsl_transpiler.py

| Category | Cases |
|----------|-------|
| Formula ordering | emitted index matches declaration order |
| Operator mapping | all 4 output operators emitted correctly |
| Header | `prefltlf N` matches formula count |
| Single formula | edge case: 1 formula |
| 10-formula spec | matches spec8.prefltlf structure |
| `to_string()` | output valid input for `PrefLTLf()` |
| `to_file()` | file written, content matches `to_string()` |

### test_dsl_integration.py

| Case | What it checks |
|------|----------------|
| Parse `.spec` → `PrefLTLf` | pipeline works without MONA |
| Equivalence to hand-written `.prefltlf` | transpiled output matches manual equivalent |
| All fixture specs | smoke test across all `tests/fixtures/*.spec` |
| CLI file output | writes correct `.prefltlf` file |
| CLI stdout | prints valid `.prefltlf` string |
| CLI missing file | exits non-zero with error message |

---

## File Extension

New DSL files use the `.spec` extension.
