# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`prefltlf2pdfa` converts PrefLTLf (Preference Linear Temporal Logic over Finite Traces) formulas into preference deterministic finite automata (PDFA). It depends on the `ltlf2dfa` library, which itself requires the **MONA** binary to be installed system-wide.

## Installation & Setup

```bash
pip install -r requirements.txt
pip install .
```

MONA must be installed separately (Ubuntu only — this project is only tested on Ubuntu 22.04):
- Follow instructions at http://www.brics.dk/mona/download.html

Docker alternative:
```bash
docker pull abhibp1993/prefltlf2pdfa
```

## Running Examples

There are no formal tests. Validation is done by running example scripts:

```bash
# Parse and inspect a spec
python examples/demo_parsing.py

# Translate a spec to a preference automaton
python examples/demo_translate.py

# Visualize the automaton
python examples/demo_viz.py

# Serialize/deserialize formulas (pickle and JSON)
python examples/demo_formula_io.py

# Serialize/deserialize automata
python examples/demo_automaton_io.py
```

## Architecture

### Core Pipeline

1. **`.prefltlf` spec file** → `PrefLTLf.from_file()` or `PrefLTLf(spec_string)`
2. **`PrefLTLf.translate()`** → builds a `PrefAutomaton`
3. **`paut2dot()` / `paut2png()` / `paut2svg()` / `paut2base64()`** → visualization

### Key Classes

**`prefltlf2pdfa/prefltlf.py`** — main module:
- `PrefLTLf`: parses and validates a PrefLTLf specification. Holds the list of LTLf formulas (`phi`), the partial order (as `nx.DiGraph`), and the alphabet. Key methods: `from_file()`, `translate(semantics=..., alphabet=...)`, `serialize()`, `deserialize()`.
- `PrefAutomaton`: the output automaton. Has a semi-automaton (states, transitions, init_state) and a preference graph (`pref_graph` as `nx.DiGraph`). States are partitioned into equivalence classes. Key methods: `get_states()`, `serialize()`, `deserialize()`.

**`prefltlf2pdfa/semantics.py`** — six semantic functions used as callbacks in `translate()`:
- `semantics_forall_exists`, `semantics_exists_forall`, `semantics_forall_forall`
- `semantics_mp_forall_exists`, `semantics_mp_exists_forall`, `semantics_mp_forall_forall` (maximal-preorder variants)

**`prefltlf2pdfa/utils.py`** — helpers: `ltlf2dfa()` (converts LTLf formula → DFA dict via MONA), `outcomes()`, `maximal_outcomes()`, `vectorize()`, `powerset()`.

**`prefltlf2pdfa/viz.py`** — visualization via `pygraphviz`: produces DOT graphs for the semi-automaton and preference graph separately.

### `.prefltlf` File Format (current)

```
prefltlf <N>        # header: number of formulas

# Formulas (one per line, LTLf syntax)
F a
G b
!(F(a) | G(b))

# Specification (preference relations)
>, 0, 1             # formula 0 strictly preferred to formula 1
>=, 0, 2            # formula 0 weakly preferred to formula 2
>=, 1, 2
```

Operators: `>` (strict), `>=` (weak), `~` (indifferent), `<>` (incomparable).

### Auto-completion

When a spec is not a total preorder, `auto_complete` can be passed to `PrefLTLf`:
- `"minimal"` — adds the minimal element
- `"incomparable"` — marks unspecified pairs as incomparable

### Planned DSL (docs/LANGUAGE.md)

A new keyword-based DSL is being designed with `propositions ... end propositions`, `ltlf-formulas ... end ltlf-formulas`, `preferences ... end preferences` blocks, named formulas, alphabet declarations, and options. Implementation plan is in `docs/prefltlf_implementation_plan.md` and `docs/webapp_implementation_plan.md`. This DSL is not yet implemented.

## Important Notes

- `translate()` supports parallel construction via `ProcessPoolExecutor` (controlled by `show_progress` kwarg and `pqdm`).
- State names in `PrefAutomaton` are tuples of DFA states (one per LTLf formula).
- The `pref_graph` nodes correspond to equivalence classes of the preorder; edges encode strict preference.
- `spot` is an optional dependency (imported with try/except); MONA via `ltlf2dfa` is required for translation.