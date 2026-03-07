# PrefLTLfSpec — Implementation Plan

## Overview

Four milestones, each buildable and testable independently:

1. **Lark parser + semantic analyzer** — clean error messages with line numbers
2. **Dash app skeleton** — `dash-ace` editor + "Run" button wired to `PrefLTLf` backend
3. **Syntax highlighting** — color keywords, LTLf operators, formula names
4. **Live error annotations** — parse on keystroke, show squiggles

---

## Milestone 1: Lark Parser + Semantic Analyzer

### 1.1 Grammar file

Create `prefltlf.lark`. Transcribe the EBNF from the spec directly. Key decisions:

- Use `%ignore` for whitespace and `#`-comments.
- Define `IDENT` as `[a-zA-Z_][a-zA-Z0-9_-]*` (to allow `safe_first`, `charge_until_clean`).
- Treat `ltlf_expr` as a **pass-through terminal** for now — capture everything between the `:` and end-of-line as a raw string, and hand it to `ltlf2pdfa`'s own parser later. Avoids duplicating LTLf grammar.
- Use Lark's `earley` parser to handle ambiguity during prototyping; switch to `lalr` once grammar is stable for speed.

```
# prefltlf.lark (sketch)
spec          : propositions_block alphabet_block? options_block? formulas_block preferences_block

propositions_block : "propositions" prop_item+ "end" "propositions"
prop_item          : IDENT ("," IDENT)*

...etc
```

### 1.2 AST transformer

Write a `lark.Transformer` subclass that converts Lark's generic parse tree into clean Python dataclasses:

```python
@dataclass
class Spec:
    propositions: list[str]
    alphabet: AlphabetBlock | None
    options: OptionsBlock | None
    formulas: dict[str, str]       # name -> raw LTLf string
    preferences: list[PrefStmt]

@dataclass
class PrefStmt:
    lhs: str
    op: str          # ">", ">=", "~", "<>"
    rhs: str
    line: int        # for error reporting
```

Always store `.line` and `.column` from the Lark token so every error can point to a source location.

### 1.3 Semantic analyzer

A separate pass over the AST (not mixed into parsing). Checks in order:

| Check | Error example |
|---|---|
| Duplicate proposition names | `"clean" declared twice (line 3)` |
| Duplicate formula names | `"f0" declared twice (line 12)` |
| Undefined formula in preferences | `"unknown" not declared in ltlf-formulas (line 18)` |
| Invalid `semantics =` value | `"MaxXY" is not a valid semantics value (line 7)` |
| Invalid `auto-complete =` value | `"full" is not a valid auto-complete value (line 8)` |
| Preference references proposition name | `"clean" is a proposition, not a formula (line 20)` |
| Self-preference warning | `Warning: f0 ~ f0 is trivially true (line 22)` |

Return a list of structured diagnostics:

```python
@dataclass
class Diagnostic:
    severity: Literal["error", "warning"]
    message: str
    line: int        # 1-indexed
    column: int      # 1-indexed
    end_line: int
    end_column: int
```

### 1.4 Desugaring pass

After semantic analysis passes, lower the AST:

- Verbatim sentences → operator form
- `<` → swap and use `>`; `<=` → swap and use `>=`
- Chains → pairwise pairs
- Formula names → indices for `PrefLTLf`

### 1.5 Testing

Write `pytest` unit tests for the parser and semantic analyzer before touching the Dash app. Cover:

- Valid full spec (full example from the doc)
- Missing required block (`formulas_block`)
- Undefined formula reference
- All verbatim forms
- All chain forms
- All desugaring rewrites

---

## Milestone 2: Dash App Skeleton

### 2.1 Dependencies

```
dash>=2.14
dash-ace
prefltlf2pdfa     # your existing backend
lark-parser
```

### 2.2 Layout

```
┌─────────────────────────────────────────────────────┐
│  PrefLTLfSpec Editor                    [Run ▶]      │
├───────────────────────────┬─────────────────────────┤
│                           │                         │
│   Ace Editor (left ~60%)  │   Output panel (right)  │
│                           │   - Errors list         │
│                           │   - PDFA result / stats │
│                           │   - Download button     │
└───────────────────────────┴─────────────────────────┘
│  Status bar:  ● 0 errors  ● 2 warnings              │
└─────────────────────────────────────────────────────┘
```

### 2.3 Callbacks

**Run button callback** (`Input: btn-run`, `State: editor-content`):

```
editor text
  → Milestone 1 parser
  → semantic analyzer
  → desugarer
  → PrefLTLf(...) call
  → format result for output panel
```

If any `Diagnostic` with `severity="error"` exists, skip calling `PrefLTLf` and show errors only.

**Store for editor content** — use a `dcc.Store` + `dcc.Interval` (300ms debounce) to buffer keystrokes before triggering parse callbacks. Do **not** run the full `PrefLTLf` backend on every keystroke — only run the parser + semantic analyzer.

### 2.4 Error panel

Show diagnostics as a table with line number, severity icon, and message. Clicking a row should move the editor cursor to that line (via a clientside callback that calls `editor.gotoLine(row)`).

### ❓ Questions for you — Milestone 2

1. **Does `prefltlf2pdfa` return a Python object, a graph, or serializable data?** This affects what the output panel shows (graph visualization vs. text stats vs. a downloadable file).
2. **Should "Run" be the only way to execute, or should there be a "Check" button** that runs the parser/semantic analyzer only (without calling `PrefLTLf`)? Useful for slow backends.
3. **Do you need authentication / multi-user isolation?** If multiple users share the Dash app simultaneously, each needs their own editor state. `dcc.Store(storage_type='session')` handles this, but worth confirming.

---

## Milestone 3: Syntax Highlighting

### 3.1 Approach

Define a custom **Ace Editor mode** in JavaScript. Inject it into the Dash app via `app.clientside_callback` or a custom `assets/prefltlf_mode.js` file (Dash automatically serves files in the `assets/` folder).

### 3.2 Token categories and colors

| Token category | Examples | Suggested color |
|---|---|---|
| Block keywords | `propositions`, `alphabet`, `options`, `ltlf-formulas`, `preferences` | Bold blue |
| `end` keyword | `end propositions`, `end ltlf-formulas`, ... | Muted blue |
| Alphabet keywords | `powerset`, `singletons`, `emptyset`, `exclude` | Teal |
| Options keywords | `semantics`, `auto-complete` | Teal |
| Semantics values | `MaxAE`, `EA`, `forall-exists`, ... | Orange |
| LTLf operators | `G`, `F`, `U`, `X`, `W`, `R` | Purple / bold |
| Boolean operators | `&`, `\|`, `!`, `->`, `<->` | Purple |
| Pref operators | `>`, `>=`, `<`, `<=`, `~`, `<>` | Red |
| Verbatim phrases | `is strictly preferred to`, `is indifferent to`, ... | Italic gray |
| Formula names | identifiers after `:` in ltlf-formulas, or lhs/rhs in preferences | Green |
| Propositions | identifiers declared in propositions block | Lighter green |
| Comments | `# ...` | Gray italic |
| Errors / unknown | anything unrecognized | Red underline |

### 3.3 Stateful highlighting (formula names)

Ace's basic regex tokenizer doesn't know which identifiers are formula names vs. plain identifiers. Two options:

- **Option A (simple):** Highlight all `IDENT` after a `:` in `ltlf-formulas` context as formula names. Requires a stateful Ace mode with a stack (medium complexity).
- **Option B (dynamic):** After every parse, send the list of declared formula names back to the editor via a clientside callback that updates the Ace session's highlight rules dynamically. This is cleaner and pairs naturally with Milestone 4.

**Recommendation: Option B.** Treat the static Ace mode as a "dumb" tokenizer for keywords and operators, and push formula names as a dynamic word list.

---

## Milestone 4: Live Error Annotations

### 4.1 Debounced parse callback

```python
@app.callback(
    Output("annotation-store", "data"),
    Input("editor-content", "value"),
    prevent_initial_call=True
)
def parse_and_annotate(text):
    if not text:
        return []
    diagnostics = run_parser_and_semantic_analyzer(text)
    return [
        {
            "row": d.line - 1,      # Ace is 0-indexed
            "column": d.column - 1,
            "text": d.message,
            "type": "error" if d.severity == "error" else "warning"
        }
        for d in diagnostics
    ]
```

### 4.2 Clientside callback to inject annotations

```javascript
// Runs in browser, no round-trip needed after store update
window.dash_clientside.prefltlf.setAnnotations = function(annotations, editorId) {
    const editor = ace.edit(editorId);
    editor.session.setAnnotations(annotations);
    return null;
}
```

### 4.3 Debounce strategy

Use `dcc.Interval` disabled by default, re-enabled on each keystroke, and disabled again when it fires — this gives a clean 300ms trailing debounce without a dependency on `dash-extensions`.

### 4.4 Error recovery in the parser

For annotations to be useful, the parser must **not** bail on the first error. Configure Lark with `ambiguity='resolve'` and wrap block-level rules with error recovery so the parser can report multiple errors per edit. This is the hardest part of Milestone 4 — budget extra time here.

### ❓ Questions for you — Milestone 4

4. **What Lark version are you targeting?** Lark 1.x has improved error recovery vs 0.x; the approach differs.
5. **How should the LTLf expression body be validated?** Right now the plan treats it as a raw string. If `ltlf2pdfa` has its own parser that can return errors with positions, those should be mapped back to the source line and surfaced as annotations too. Is that parser accessible?

---

## File Structure

```
prefltlf_app/
├── app.py                   # Dash app entry point
├── prefltlf.lark            # Lark grammar
├── parser/
│   ├── __init__.py
│   ├── grammar.py           # Loads and exposes the Lark parser
│   ├── transformer.py       # Lark Transformer → dataclasses
│   ├── semantic.py          # Semantic analyzer → list[Diagnostic]
│   └── desugar.py           # Lowering pass → PrefLTLf-ready IR
├── backend/
│   └── runner.py            # Wraps PrefLTLf call, returns result
├── assets/
│   └── prefltlf_mode.js     # Custom Ace mode (Milestone 3)
├── components/
│   ├── editor.py            # dash-ace wrapper component
│   └── output_panel.py      # Result display component
└── tests/
    ├── test_parser.py
    ├── test_semantic.py
    └── test_desugar.py
```

---

## Open Questions Summary

| # | Question | Blocks |
|---|---|---|
| 1 | What does `prefltlf2pdfa` return — object, graph, serializable data? | Milestone 2 output panel |
| 2 | Should there be a separate "Check" button vs. "Run" only? | Milestone 2 UX |
| 3 | Single user or multi-user Dash deployment? | Milestone 2 state management |
| 4 | Lark version? | Milestone 4 error recovery |
| 5 | Can the LTLf sub-parser return errors with positions? | Milestone 4 annotations |
