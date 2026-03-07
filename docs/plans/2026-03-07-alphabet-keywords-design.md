# Design: Extended Alphabet Block Keywords

**Date:** 2026-03-07
**Branch:** feature/dsl-grammar
**Status:** Approved

---

## Problem

The current `alphabet` block in the PrefLTLf DSL only supports two entry forms:
- `powerset()` — full powerset of declared propositions
- `{p, q}` — explicit set literals, semicolon-separated

The language spec (`docs/LANGUAGE.md`) defines a richer set of keywords (`singletons`, `emptyset`, `exclude`) that are not yet implemented.

---

## Approach

**Approach B: Rewrite `_build_alphabet` as a line-by-line two-pass parser.**

Keep the existing architecture (alphabet block extracted as a raw string stub by regex in `parser.py`; parsed in `transpiler.py`). Replace the current `;`-split token loop with a line-oriented parser and a two-pass execution model.

No changes to `parser.py`, `grammar.lark`, `models.py`, or the CLI.

---

## Alphabet Command Syntax

Seven line forms, one per line inside `alphabet ... end alphabet`:

| Line | Meaning |
|------|---------|
| `powerset()` | All 2^\|AP\| subsets of declared propositions |
| `powerset([p, q])` | All subsets of the specified subset `{p, q}` |
| `singletons()` | `{ {p} | p ∈ AP }` — one singleton set per declared proposition |
| `singletons([p, q])` | Singletons restricted to `{p, q}` |
| `emptyset` | Adds `{}` (the empty set) |
| `{p, q}` | Explicit set literal (existing syntax, preserved) |
| `exclude {a}, {b, c}` | Remove exact sets from the accumulated alphabet |
| `exclude p` | Shorthand for `exclude {p}` |

Blank lines and `#` comments are ignored. Multiple generators may appear in any order.

---

## Two-Pass Algorithm

**Pass 1 — generators** (process all non-exclude lines):
- `powerset()` / `powerset([subset])` → add all subsets of AP / subset
- `singletons()` / `singletons([subset])` → add each `{p}` for p in AP / subset
- `emptyset` or `{}` → add `set()`
- `{p, q, ...}` → add that explicit set
- Duplicates are silently deduplicated

**Pass 2 — exclusions** (after all generators):
- `exclude {a}, {b, c}` → remove each listed set (exact match)
- `exclude p` → treated as `exclude {p}`
- Silently skips sets not present in the accumulated alphabet

**Rationale:** Commands are freely mixed in the file; the two-pass model gives consistent results regardless of order.

---

## Validation Rules

Raised as `DSLError`:
- `powerset()` or `singletons()` without arguments when no `propositions` block is present
- Any proposition used in `[subset]`, `{literal}`, or `exclude` that is not declared in the `propositions` block (only enforced when a `propositions` block exists)
- Unrecognized line → error showing the offending text

---

## Code Structure

All changes confined to `prefltlf2pdfa/dsl/transpiler.py` and `TUTORIAL.md`.

`_build_alphabet` is replaced by:

1. `_parse_alphabet_lines(raw: str) -> list[tuple[str, any]]`
   - Scans each line with regex
   - Returns `(kind, args)` tuples where `kind` ∈ `{"powerset", "singletons", "emptyset", "literal", "exclude"}`
   - Raises `DSLError` on unrecognized lines

2. `_build_alphabet(self) -> list | None`
   - Calls `_parse_alphabet_lines`
   - Runs two passes
   - Returns final `list[set]` or `None`

---

## Tests

New tests in `tests/test_dsl_transpiler.py` (`TestTranspilerAlphabet`):

| Test | Input | Expected |
|------|-------|----------|
| `test_singletons_default` | `singletons()` + props `p, q` | `[{p}, {q}]` |
| `test_singletons_subset` | `singletons([p])` + props `p, q` | `[{p}]` |
| `test_powerset_subset` | `powerset([p])` + props `p, q` | `[{}, {p}]` |
| `test_emptyset_keyword` | `emptyset` | `[set()]` |
| `test_exclude_exact` | `powerset()` + `exclude {p, q}` | powerset minus `{p,q}` |
| `test_exclude_shorthand` | `powerset()` + `exclude p` | powerset minus `{p}` |
| `test_exclude_multi` | `powerset()` + `exclude {p}, {}` | powerset minus `{p}` and `{}` |
| `test_exclude_missing_is_silent` | exclude absent set | no error |
| `test_mixed_generators` | `singletons()` + `emptyset` | `[{}, {p}, {q}]` |
| `test_generators_and_exclude_mixed_order` | exclude before powerset | same as exclude after |
| `test_unknown_keyword_raises` | `blah()` | `DSLError` |
| `test_singletons_without_props_raises` | `singletons()`, no props | `DSLError` |

New fixture: `tests/fixtures/with_alphabet_keywords.spec`

`TUTORIAL.md` alphabet section and quick-reference table updated.
