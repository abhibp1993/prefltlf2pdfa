# PrefLTLfSpec: Updated `keyword ... end keyword` DSL

This revision adopts your requested surface syntax while keeping a direct lowering path to current `prefltlf2pdfa` semantics.

## What changed

- `ap ... end` -> `propositions ... end propositions`.
- `formulas ... end` -> `ltlf-formulas ... end ltlf-formulas`.
- Formula declaration style is now `<name>: <ltlf_formula>`.
- Added `alphabet ... end alphabet` stub with `singletons([P])`, `emptyset`, `powerset([P])`, and `exclude`.
- Added `options ... end options` stub for semantics and auto-completion settings.
- Unified preference declarations into one `preferences ... end preferences` block.
- Preference operators include `>`, `<`, `>=`, `<=`, `<>`, `~`.
- Added optional verbatim preference sentences and chain expressions.
- No outer `model ... end model` wrapper needed.

## Core structure

```text
propositions
  ...
end propositions

alphabet
  ...
end alphabet

options
  ...
end options

ltlf-formulas
  <name>: <ltlf_formula>
  ...
end ltlf-formulas

preferences
  ...
end preferences
```

## Suggested syntax

### 1) Propositions

Both input styles are allowed:

```text
propositions
  clean, charged, safe
end propositions
```

```text
propositions
  clean
  charged
  safe
end propositions
```

### 2) Alphabet stub

The alphabet block declares which symbols (sets of propositions) are allowed in traces.

```text
alphabet
  powerset()                 # defaults to powerset(AP)
  exclude {clean, charged}
  exclude safe               # single prop, interpreted as {safe}
end alphabet
```

```text
alphabet
  singletons()               # defaults to { {a} | a in AP }
  emptyset                   # {}
  exclude charged
end alphabet
```

```text
alphabet
  powerset([clean, safe])    # powerset of selected subset only
  singletons([clean, safe])  # only singletons from subset
end alphabet
```

### 3) Options stub

The options block configures semantics and auto-completion behavior.

```text
options
  semantics = MaxAE
  auto-complete = minimal
end options
```

**Semantics values:**
- `EA` or `exists-forall`
- `AE` or `forall-exists`
- `AA` or `forall-forall`
- `EE` or `exists-exists`
- `MaxEA` or `max-exists-forall`
- `MaxAE` or `max-forall-exists`
- `MaxAA` or `max-forall-forall`
- `MaxEE` or `max-exists-exists`

**Auto-complete values:**
- `incomparable`
- `minimal`

### 4) LTLf formulas

```text
ltlf-formulas
  f0: G safe
  f1: F clean
  f2: charged U clean
end ltlf-formulas
```

### 5) Preferences

Operator form:

```text
preferences
  f0 > f1
  f1 >= f2
  f2 <> f0
  f0 ~ f0
end preferences
```

Chaining form:

```text
preferences
  f0 > f1 >= f2 ~ f3
end preferences
```

Verbatim form:

```text
preferences
  f0 is strictly preferred to f1
  f1 is weakly preferred to f2
  f2 is indifferent to f3
  f3 is incomparable to f0
end preferences
```

Reference by exact formula is also allowed:

```text
preferences
  (G safe) > (F clean)
end preferences
```

## Full example

```text
propositions
  clean
  charged
  safe
end propositions

alphabet
  powerset()
  exclude {clean, charged}
end alphabet

options
  semantics = MaxAE
  auto-complete = minimal
end options

ltlf-formulas
  safe_first: G safe
  eventually_clean: F clean
  charge_until_clean: charged U clean
end ltlf-formulas

preferences
  safe_first > eventually_clean >= charge_until_clean
  eventually_clean is incomparable to charge_until_clean
end preferences
```

## Semantic compatibility with current `prefltlf2pdfa`

Most requested changes are surface-syntax changes and can be lowered safely, but a few need explicit desugaring rules.

- `propositions`, `ltlf-formulas`, named formulas: compatible as parser sugar.
- `alphabet` block: compatible at API level (`PrefLTLf(..., alphabet=...)` already exists), but current raw spec parser does not parse alphabet text itself. You need a front-end lowering stage.
- `singletons`, `emptyset`, `powerset(...)`, `exclude`: compatible as alphabet-construction sugar if lowered to an explicit symbol list.
- `>`, `>=`, `~`, `<>`: directly match current relation operators.
- `<`, `<=`: not native today; must be lowered as `a < b  =>  b > a` and `a <= b  =>  b >= a`.
- Named references in preferences: compatible by symbol-table lookup to formula indices.
- Exact formula references in preferences: potentially ambiguous unless canonicalized; best lowered by either:
  - matching an existing declared formula exactly, or
  - auto-adding a hidden formula entry, then using its index.
- Verbatim sentences: parser sugar only; must lower to the operator set above.
- Chains (`f0 > f1 >= f2 ~ f3`): compatible if expanded pairwise (`f0 > f1`, `f1 >= f2`, `f2 ~ f3`).

## Recommended desugaring table

```text
"a is strictly preferred to b" -> a > b
"a is weakly preferred to b"   -> a >= b
"a is indifferent to b"        -> a ~ b
"a is incomparable to b"       -> a <> b
a < b                           -> b > a
a <= b                          -> b >= a
```

## Minimal grammar sketch (EBNF-like)

```text
spec              ::= sections
sections          ::= propositions_block alphabet_block? options_block? formulas_block preferences_block

propositions_block ::= "propositions" proposition_items "end" "propositions"
proposition_items ::= proposition_line (proposition_line)*
proposition_line  ::= IDENT ("," IDENT)*

alphabet_block    ::= "alphabet" alphabet_stmt* "end" "alphabet"
alphabet_stmt     ::= "singletons" ("(" proposition_list? ")")?
                   | "emptyset"
                   | "powerset" ("(" proposition_list? ")")?
                   | "exclude" (symbol | IDENT)
proposition_list  ::= IDENT ("," IDENT)*
symbol            ::= "{" proposition_list? "}"

options_block     ::= "options" option_stmt* "end" "options"
option_stmt       ::= "semantics" "=" semantics_value
                   | "auto-complete" "=" auto_complete_value
semantics_value   ::= "EA" | "exists-forall" 
                   | "AE" | "forall-exists"
                   | "AA" | "forall-forall"
                   | "EE" | "exists-exists"
                   | "MaxEA" | "max-exists-forall"
                   | "MaxAE" | "max-forall-exists"
                   | "MaxAA" | "max-forall-forall"
                   | "MaxEE" | "max-exists-exists"
auto_complete_value ::= "incomparable" | "minimal"

formulas_block    ::= "ltlf-formulas" formula_decl+ "end" "ltlf-formulas"
formula_decl      ::= IDENT ":" ltlf_expr

preferences_block ::= "preferences" preference_stmt+ "end" "preferences"
preference_stmt   ::= pref_chain | pref_verbatim
pref_chain        ::= pref_term (pref_op pref_term)+
pref_op           ::= ">" | "<" | ">=" | "<=" | "~" | "<>"
pref_term         ::= IDENT | "(" ltlf_expr ")"

pref_verbatim     ::= pref_term "is" verb_phrase "to" pref_term
verb_phrase       ::= "strictly preferred"
                   | "weakly preferred"
                   | "indifferent"
                   | "incomparable"
```

## Notes for implementation

1. Add a front-end parser for this DSL that lowers into the current index-based spec consumed by `PrefLTLf`.
2. Resolve formula names first, then expand chains and verbatim statements.
3. Normalize `<` and `<=` into `>` and `>=` before emitting core IR.
4. Build the alphabet set from the alphabet block and pass it via constructor/API.
5. Keep strict validation for conflicting preferences (already enforced in `_build_partial_order`).
