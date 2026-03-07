# PrefLTLf DSL Tutorial

This tutorial explains how to write `.spec` files using the PrefLTLf DSL and how to compile them using the `prefltlf-compile` CLI.

---

## Overview

A `.spec` file expresses preference over LTLf (Linear Temporal Logic over Finite Traces) formulas using a readable, keyword-based syntax. The compiler converts this to the index-based `.prefltlf` format used internally, and optionally builds a full preference automaton (PDFA).

---

## File Structure

A `.spec` file contains up to five blocks, each delimited by `<keyword> ... end <keyword>`. Lines beginning with `#` are comments.

```
propositions        # optional
  ...
end propositions

ltlf-formulas       # required
  ...
end ltlf-formulas

preferences         # required
  ...
end preferences

alphabet            # optional
  ...
end alphabet

options             # optional
  ...
end options
```

**Required blocks:** `ltlf-formulas`, `preferences`
**Optional blocks:** `propositions`, `alphabet`, `options`

---

## Block Reference

### `propositions` (optional)

Declares the atomic propositions used in your formulas. When present, the compiler validates that every atom used in `ltlf-formulas` is declared here, and automatically computes the alphabet as the powerset of these propositions (unless an explicit `alphabet` block overrides it).

```
propositions
  safe, clean, charged
end propositions
```

- Multiple propositions are separated by commas.
- Identifiers: letters, digits, underscores; must start with a letter or underscore.

---

### `ltlf-formulas` (required)

Declares named LTLf formulas. Each line has the form:

```
<name>: <LTLf formula>
```

```
ltlf-formulas
  safety: G safe
  liveness: F clean
  charge: charged U clean
end ltlf-formulas
```

- Names must be unique identifiers.
- Formula bodies use standard LTLf syntax: `G` (always), `F` (eventually), `U` (until), `!` (not), `&` (and), `|` (or), `->` (implies), `<->` (iff), `X` (next).
- Formula body extends to end of line; everything after `#` is treated as a comment.

---

### `preferences` (required)

Declares preference relations over named formulas. Each line is a **chain** of preferences:

```
preferences
  safety > liveness >= charge
end preferences
```

A chain `A op1 B op2 C ...` is expanded into binary statements `A op1 B`, `B op2 C`, etc.

**Preference operators:**

| Operator | Meaning | Normalized as |
|----------|---------|---------------|
| `>`  | strictly preferred | `>` |
| `>=` | weakly preferred   | `>=` |
| `~`  | indifferent        | `~` |
| `<>` | incomparable       | `<>` |
| `<`  | strictly less (reversed) | `>` with operands swapped |
| `<=` | weakly less (reversed)   | `>=` with operands swapped |

You can also use English phrases (verbatim form):

```
preferences
  safety is strictly preferred to liveness
  liveness is weakly preferred to charge
  charge is incomparable to liveness
end preferences
```

Valid verbatim phrases: `strictly preferred`, `weakly preferred`, `indifferent`, `incomparable`.

---

### `alphabet` (optional)

Specifies which symbol sets (subsets of propositions) form the alphabet for automaton construction. Each line is one command; order between commands does not matter. Blank lines and `#` comments are ignored.

**Generator commands** (add sets to the alphabet):

| Line | What it adds |
|------|-------------|
| `powerset()` | All 2^\|AP\| subsets of declared propositions |
| `powerset([p, q])` | All subsets of the specified subset `{p, q}` |
| `singletons()` | One singleton `{p}` per declared proposition |
| `singletons([p, q])` | Singletons restricted to `{p, q}` |
| `emptyset` | The empty set `{}` |
| `{p, q}` | An explicit set (existing syntax) |

**Exclusion command** (remove exact sets from the accumulated alphabet):

| Line | What it removes |
|------|----------------|
| `exclude {a, b}, {c}` | Removes `{a, b}` and `{c}` exactly |
| `exclude p` | Shorthand for `exclude {p}` |

`exclude` silently ignores sets that are not in the alphabet. Multiple generators may be combined freely.

**Examples:**

```
# Full powerset of all propositions (default when propositions block is present)
alphabet
  powerset()
end alphabet
```

```
# Only singletons plus the empty set
alphabet
  singletons()
  emptyset
end alphabet
```

```
# Powerset of {p}, plus {p, q}, minus {}
alphabet
  powerset([p])
  {p, q}
  exclude {}
end alphabet
```

```
# Explicit sets, two excludes on one line
alphabet
  powerset()
  exclude {p, q}, {q}
end alphabet
```

---

### `options` (optional)

Controls semantics and auto-completion behavior. Each line has the form `key = value`.

```
options
  semantics = MaxAE
  auto-complete = minimal
end options
```

**`semantics`** — the preference semantics to use when building the automaton:

| Alias | Long form | Description |
|-------|-----------|-------------|
| `AE` | `forall-exists` | For-all/Exists |
| `EA` | `exists-forall` | Exists/For-all |
| `AA` | `forall-forall` | For-all/For-all |
| `MaxAE` | `max-forall-exists` | Maximal-preorder For-all/Exists (default) |
| `MaxEA` | `max-exists-forall` | Maximal-preorder Exists/For-all |
| `MaxAA` | `max-forall-forall` | Maximal-preorder For-all/For-all |

Default: `MaxAE`

**`auto-complete`** — how to handle preference specifications that don't cover all traces:

| Value | Behavior |
|-------|----------|
| `none` | No auto-completion (default) |
| `minimal` | Adds a minimal (least-preferred) element |
| `incomparable` | Marks unspecified pairs as incomparable |

---

## Worked Example: `full.spec`

```
propositions
  safe, clean, charged
end propositions

alphabet
  powerset()
end alphabet

options
  semantics = MaxAE
  auto-complete = minimal
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

This spec:
1. Declares three propositions, giving an alphabet of 8 symbol sets (2^3).
2. Uses `MaxAE` semantics and adds a minimal element for auto-completion.
3. Defines three named formulas.
4. Expresses `safety > liveness` and `liveness >= charge`.

---

## CLI Usage

Install the package first:

```bash
pip install .
```

The CLI is `prefltlf-compile`. MONA must be installed to use `--translate`.

---

### Compile to `.prefltlf` format (no MONA needed)

```bash
# Write to <input>.prefltlf alongside the input file
prefltlf-compile my_spec.spec

# Write to a specific output file
prefltlf-compile my_spec.spec -o output/my_spec.prefltlf

# Print the .prefltlf text to stdout (useful for piping)
prefltlf-compile my_spec.spec --stdout
```

---

### Build a preference automaton (requires MONA)

```bash
# Build automaton and save as JSON (default) in the same directory as the input
prefltlf-compile my_spec.spec --translate

# Save JSON to a specific output directory
prefltlf-compile my_spec.spec --translate -o output/

# Save DOT graph files (<stem>_sa.dot and <stem>_pg.dot)
prefltlf-compile my_spec.spec --translate --output-format dot -o output/

# Save all artifacts: JSON + DOT + PNG + SVG
prefltlf-compile my_spec.spec --translate --output-format all-artifacts -o output/
```

Output files produced by `--translate`:

| Format | Files |
|--------|-------|
| `json` (default) | `<stem>.json` |
| `dot` | `<stem>_sa.dot`, `<stem>_pg.dot` |
| `all-artifacts` | JSON + DOT + PNG + SVG |

---

### Using `full.spec` end-to-end

```bash
# Assuming full.spec is in tests/fixtures/
prefltlf-compile tests/fixtures/full.spec --stdout

prefltlf-compile tests/fixtures/full.spec --translate --output-format all-artifacts -o /tmp/full_output/
```

---

## Python API

You can also use the DSL programmatically:

```python
from prefltlf2pdfa.dsl import parse_spec, Transpiler

text = open("my_spec.spec").read()
spec = parse_spec(text)
t = Transpiler(spec)

# Get the index-based .prefltlf string
print(t.to_string())

# Get a PrefLTLf object (no MONA needed)
pf = t.to_prefltlf()

# Get a PrefAutomaton (requires MONA)
aut = t.to_pdfa()
```

---

## Quick Reference

### Block Keywords

| Block | Required | Purpose |
|-------|----------|---------|
| `propositions` / `end propositions` | No | Declare atomic propositions |
| `ltlf-formulas` / `end ltlf-formulas` | **Yes** | Named LTLf formulas |
| `preferences` / `end preferences` | **Yes** | Preference ordering |
| `alphabet` / `end alphabet` | No | Explicit alphabet |
| `options` / `end options` | No | Semantics and auto-complete |

### LTLf Operators

| Operator | Symbol | Example |
|----------|--------|---------|
| Always (globally) | `G` | `G safe` |
| Eventually | `F` | `F clean` |
| Until | `U` | `p U q` |
| Next | `X` | `X p` |
| Not | `!` | `!p` |
| And | `&` | `p & q` |
| Or | `\|` | `p \| q` |
| Implies | `->` | `p -> q` |
| Iff | `<->` | `p <-> q` |

### Preference Operators

`>` `>=` `~` `<>` `<` `<=` and verbatim phrases: `strictly preferred`, `weakly preferred`, `indifferent`, `incomparable`

### Semantics Aliases

`AE`, `forall-exists`, `EA`, `exists-forall`, `AA`, `forall-forall`, `MaxAE` (default), `max-forall-exists`, `MaxEA`, `max-exists-forall`, `MaxAA`, `max-forall-forall`

### Auto-complete Values

`none` (default), `minimal`, `incomparable`

### Alphabet Keywords

**Generators** (one per line; may be combined):
`powerset()`, `powerset([p, q])`, `singletons()`, `singletons([p, q])`, `emptyset`, `{p, q}`

**Exclusion:**
`exclude {a, b}, {c}` or `exclude p` (removes exact sets; silent if not present)

### CLI Flags

| Flag | Description |
|------|-------------|
| `input` | Path to `.spec` file (required) |
| `-o`, `--output` | Output file (no `--translate`) or output directory (`--translate`) |
| `--stdout` | Print `.prefltlf` to stdout instead of writing a file |
| `--translate` | Run full pipeline: `.spec` → `PrefAutomaton` (requires MONA) |
| `--output-format` | `json` (default), `dot`, or `all-artifacts`; only with `--translate` |
