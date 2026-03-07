# Design: DSL-to-PDFA Full Pipeline

**Date:** 2026-03-07
**Branch:** feature/dsl-grammar
**Status:** Approved

## Overview

Extend the `.spec` DSL front-end so that a single command produces a complete PDFA
(semi-automaton + preference graph), generating the same artifacts as the existing
`.prefltlf` Python pipeline. This involves:

1. Propositions validation — atoms in LTLf formulas must be declared
2. Alphabet construction — from `alphabet` block or `powerset(propositions)`
3. Options parsing — `semantics` and `auto-complete` from the `options` block
4. `Transpiler.to_pdfa()` — new method producing a `PrefAutomaton`
5. CLI `--translate` flag — drives the full pipeline with output format selection

---

## Architecture & Data Flow

```
.spec text
    │
    ▼
parse_spec()
    │  Spec {formulas, preferences, propositions,
    │         alphabet(stub), options(stub)}
    ▼
Transpiler.__init__()
    ├─ _parse_options()          → SpecOptions {semantics, auto_complete}
    ├─ _validate_propositions()  → DSLError if atom undeclared
    └─ _build_alphabet()         → list[frozenset[str]] or None
    │
    ├─ .to_string()     → .prefltlf text          (existing, unchanged)
    ├─ .to_file()       → writes .prefltlf         (existing, unchanged)
    ├─ .to_prefltlf()   → PrefLTLf object          (gains auto_complete param)
    └─ .to_pdfa()       → PrefAutomaton            (NEW)
           │  calls to_prefltlf(auto_complete=...)
           │  then .translate(semantics=..., **kwargs)
           ▼
       PrefAutomaton
           │
    ┌──────┴──────────────────┐
    ▼                         ▼
  json                  dot / png / svg
(jsonpickle)       (paut2dot / paut2png / paut2svg)
```

**Key decisions:**
- `_parse_options()`, `_validate_propositions()`, `_build_alphabet()` all run eagerly
  in `__init__()` so errors surface immediately, not deferred to `to_pdfa()`.
- `to_pdfa(**kwargs)` kwargs override anything parsed from the `options` block —
  call-site always wins.
- Existing `to_string()` / `to_file()` / `to_prefltlf()` are unchanged in behavior.

---

## Section 1: Propositions Validation

When the `propositions` block is present, `Transpiler.__init__()` extracts all atomic
propositions from each LTLf formula body and verifies every atom is declared.
Undeclared atoms raise `DSLError` with the line number from `FormulaDecl`.

```
propositions
  safe, clean
end propositions

ltlf-formulas
  f0: G safe
  f1: F robot   ← DSLError: 'robot' not declared in propositions (line 7)
end ltlf-formulas
```

If the `propositions` block is absent, no validation is performed (same behavior as today).

---

## Section 2: Alphabet Construction

Three cases:

| Condition | Alphabet used |
|---|---|
| No `alphabet` block, `propositions` present | `powerset(propositions)` |
| No `alphabet` block, no `propositions` | `None` (existing behavior) |
| `alphabet` block contains `powerset()` | `powerset(propositions)` — requires `propositions` block; raises `DSLError` if absent |
| Explicit sets in `alphabet` block | Parsed as `list[frozenset[str]]`, validated against `propositions` if declared |

**Alphabet block syntax** — both formats supported, freely mixed:

```
# One set per line
alphabet
  {}
  {safe}
  {clean}
  {safe, clean}
end alphabet

# Multiple sets separated by semicolons on one line
alphabet
  {}; {safe}; {clean}; {safe, clean}
end alphabet

# Powerset shorthand
alphabet
  powerset()
end alphabet
```

When `propositions` are declared, any atom appearing in an explicit alphabet set that
was not declared raises `DSLError`.

---

## Section 3: Options Parsing

A new `SpecOptions` dataclass:

```python
@dataclass
class SpecOptions:
    semantics: Callable = semantics_mp_forall_exists   # default: MaxAE
    auto_complete: str = "none"                         # default: none
```

The raw options stub is parsed line-by-line with `key = value` splitting.
Unknown keys raise `DSLError`.

**Supported keys:**

| Key | Accepted values |
|---|---|
| `semantics` | See alias table below |
| `auto-complete` | `minimal`, `incomparable` |

**Semantics aliases:**

| Alias(es) | Function |
|---|---|
| `AE`, `forall-exists` | `semantics_forall_exists` |
| `EA`, `exists-forall` | `semantics_exists_forall` |
| `AA`, `forall-forall` | `semantics_forall_forall` |
| `MaxAE`, `max-forall-exists` | `semantics_mp_forall_exists` |
| `MaxEA`, `max-exists-forall` | `semantics_mp_exists_forall` |
| `MaxAA`, `max-forall-forall` | `semantics_mp_forall_forall` |

Note: `EE` and `MaxEE` are not yet implemented in `semantics.py` and are unsupported.

**Example options block:**

```
options
  semantics = MaxAE
  auto-complete = minimal
end options
```

---

## Section 4: `Transpiler.to_pdfa()`

```python
def to_pdfa(self, **kwargs) -> PrefAutomaton:
    semantics = kwargs.pop("semantics", self._options.semantics)
    auto_complete = kwargs.pop("auto_complete", self._options.auto_complete)
    pf = self.to_prefltlf(auto_complete=auto_complete)
    return pf.translate(semantics=semantics, **kwargs)
```

`to_prefltlf()` gains an `auto_complete` parameter (previously hardcoded to `"none"`).
The constructed `PrefLTLf` object also receives the `alphabet` built in `__init__()`.

---

## Section 5: CLI Changes

```
prefltlf-compile input.spec [--translate] [--output-format FORMAT] [-o OUTPUT_DIR]
```

| Flag | Description |
|---|---|
| `--translate` | Run full pipeline to `PrefAutomaton` instead of stopping at `.prefltlf` |
| `--output-format` | `json` (default), `dot`, `all-artifacts` |
| `-o / --output` | Output directory (default: same directory as input file) |

**Output filenames** (base = input stem, e.g. `spec`):

| `--output-format` | Files produced |
|---|---|
| `json` | `spec.json` |
| `dot` | `spec_sa.dot`, `spec_pg.dot` |
| `all-artifacts` | `spec.json` + `spec_sa.dot` + `spec_pg.dot` + `spec_sa.png` + `spec_pg.png` + `spec_sa.svg` + `spec_pg.svg` |

Without `--translate`, existing behavior is unchanged (produces `.prefltlf`).

---

## Section 6: Testing

### Unit tests (`tests/test_dsl_transpiler.py`)

| Test | What it checks |
|---|---|
| `test_undeclared_proposition_raises` | `DSLError` when formula uses atom not in `propositions` |
| `test_valid_propositions_passes` | No error when all atoms are declared |
| `test_alphabet_explicit_parsed` | Explicit sets parsed correctly to `list[frozenset]` |
| `test_alphabet_powerset_keyword` | `powerset()` expands to full powerset of propositions |
| `test_alphabet_powerset_requires_propositions` | `DSLError` when `powerset()` used without `propositions` block |
| `test_alphabet_validated_against_propositions` | `DSLError` when explicit alphabet contains undeclared atom |
| `test_options_parsed_semantics` | All 6 semantics aliases map to correct function |
| `test_options_unknown_key_raises` | `DSLError` on unknown option key |
| `test_options_override_at_call_site` | `to_pdfa(semantics=...)` overrides options block value |

### Integration tests (`tests/test_dsl_integration.py` — new class)

| Test | What it checks |
|---|---|
| `test_to_pdfa_returns_pref_automaton` | `to_pdfa()` returns a `PrefAutomaton` instance |
| `test_to_pdfa_with_explicit_alphabet` | Alphabet from `alphabet` block is respected |
| `test_to_pdfa_semantics_from_options` | Semantics set in `options` block is used |

All tests run by default. `@pytest.mark.slow` is used as a label for MONA-dependent
tests but does not exclude them from the default test run.
