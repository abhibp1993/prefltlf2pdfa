# Extended Alphabet Block Keywords Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `singletons()`, `emptyset`, `powerset([subset])`, `singletons([subset])`, and `exclude` keywords to the `alphabet` block parser, replacing the old semicolon-separated token model with a clean line-by-line two-pass design.

**Architecture:** The `alphabet` block is already extracted as a raw string stub in `parser.py`. All changes are confined to `transpiler.py`: three module-level helper functions are added, and `_build_alphabet` is completely rewritten. Pass 1 runs all generator commands (union, dedup); Pass 2 applies all `exclude` commands (exact-match removal, silent if missing). Order of commands in the file doesn't matter.

**Tech Stack:** Python 3.11+, `re` (stdlib), `prefltlf2pdfa.utils.powerset`, `pytest`

---

### Task 1: Write failing tests, update semicolon test, update fixture

**Files:**
- Modify: `tests/test_dsl_transpiler.py` (class `TestTranspilerAlphabet`)
- Modify: `tests/fixtures/with_alphabet.spec`

The existing `test_alphabet_semicolon_separated_sets` tests `{}; {p}` syntax which will be dropped. Update it to use one-set-per-line syntax (which will still pass). Then add all new tests — they will fail until Task 3 is complete.

**Step 1: Update `test_alphabet_semicolon_separated_sets` in `TestTranspilerAlphabet`**

Replace the semicolon test with an equivalent one-per-line version (same assertions, same class, no behavioural change):

```python
def test_alphabet_semicolon_separated_sets(self):
    src = """
propositions
  p, q
end propositions

ltlf-formulas
  f0: G p
end ltlf-formulas

preferences
  f0 >= f0
end preferences

alphabet
  {}
  {p}
  {q}
  {p, q}
end alphabet
"""
    t = self._make_transpiler(src)
    assert len(t._alphabet) == 4
```

**Step 2: Update `tests/fixtures/with_alphabet.spec`**

Change `{}; {p}` to two lines:

```
propositions
  p
end propositions

ltlf-formulas
  f0: G p
  f1: !G p
end ltlf-formulas

preferences
  f0 > f1
end preferences

alphabet
  {}
  {p}
end alphabet
```

**Step 3: Add new generator tests to `TestTranspilerAlphabet`**

Add these methods to the existing `TestTranspilerAlphabet` class:

```python
def test_powerset_subset(self):
    """powerset([p]) with props p, q → only subsets of {p}: [{}, {p}]"""
    src = """
propositions
  p, q
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  powerset([p])
end alphabet
"""
    t = self._make_transpiler(src)
    assert len(t._alphabet) == 2
    assert set() in t._alphabet
    assert {"p"} in t._alphabet
    assert {"q"} not in t._alphabet

def test_singletons_default(self):
    """singletons() with props p, q → [{p}, {q}]"""
    src = """
propositions
  p, q
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  singletons()
end alphabet
"""
    t = self._make_transpiler(src)
    assert len(t._alphabet) == 2
    assert {"p"} in t._alphabet
    assert {"q"} in t._alphabet
    assert set() not in t._alphabet

def test_singletons_subset(self):
    """singletons([p]) with props p, q → [{p}] only"""
    src = """
propositions
  p, q
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  singletons([p])
end alphabet
"""
    t = self._make_transpiler(src)
    assert len(t._alphabet) == 1
    assert {"p"} in t._alphabet

def test_emptyset_keyword(self):
    """emptyset → [set()]"""
    src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  emptyset
end alphabet
"""
    t = self._make_transpiler(src)
    assert t._alphabet == [set()]

def test_mixed_generators(self):
    """singletons() + emptyset with props p, q → [{}, {p}, {q}] (deduped)"""
    src = """
propositions
  p, q
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  singletons()
  emptyset
  singletons()
end alphabet
"""
    t = self._make_transpiler(src)
    assert len(t._alphabet) == 3
    assert set() in t._alphabet
    assert {"p"} in t._alphabet
    assert {"q"} in t._alphabet
```

**Step 4: Add `exclude` tests to `TestTranspilerAlphabet`**

```python
def test_exclude_exact_set(self):
    """powerset(p,q) then exclude {p,q} → 3 sets remain"""
    src = """
propositions
  p, q
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  powerset()
  exclude {p, q}
end alphabet
"""
    t = self._make_transpiler(src)
    assert len(t._alphabet) == 3
    assert {"p", "q"} not in t._alphabet

def test_exclude_shorthand_single_prop(self):
    """exclude p removes {p} from alphabet"""
    src = """
propositions
  p
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  powerset()
  exclude p
end alphabet
"""
    t = self._make_transpiler(src)
    assert {"p"} not in t._alphabet
    assert set() in t._alphabet

def test_exclude_multiple_sets_on_one_line(self):
    """exclude {p}, {} removes both exact sets"""
    src = """
propositions
  p, q
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  powerset()
  exclude {p}, {}
end alphabet
"""
    t = self._make_transpiler(src)
    assert {"p"} not in t._alphabet
    assert set() not in t._alphabet

def test_exclude_missing_set_is_silent(self):
    """Excluding a set not in alphabet raises no error"""
    src = """
propositions
  p
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  emptyset
  exclude {p}
end alphabet
"""
    t = self._make_transpiler(src)
    assert t._alphabet == [set()]

def test_exclude_before_generator_same_result(self):
    """exclude before powerset gives same result as after"""
    src_exclude_after = """
propositions
  p
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  powerset()
  exclude {p}
end alphabet
"""
    src_exclude_before = """
propositions
  p
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  exclude {p}
  powerset()
end alphabet
"""
    t1 = self._make_transpiler(src_exclude_after)
    t2 = self._make_transpiler(src_exclude_before)
    assert t1._alphabet == t2._alphabet
```

**Step 5: Add error case tests to `TestTranspilerAlphabet`**

```python
def test_singletons_without_propositions_raises(self):
    src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  singletons()
end alphabet
"""
    with pytest.raises(DSLError, match="propositions"):
        self._make_transpiler(src)

def test_powerset_subset_undeclared_raises(self):
    src = """
propositions
  p
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  powerset([p, z])
end alphabet
"""
    with pytest.raises(DSLError, match="undeclared"):
        self._make_transpiler(src)

def test_exclude_undeclared_prop_raises(self):
    src = """
propositions
  p
end propositions
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  powerset()
  exclude {z}
end alphabet
"""
    with pytest.raises(DSLError, match="undeclared"):
        self._make_transpiler(src)

def test_unknown_alphabet_keyword_raises(self):
    src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
alphabet
  blah()
end alphabet
"""
    with pytest.raises(DSLError):
        self._make_transpiler(src)
```

**Step 6: Run all tests — new tests must FAIL, existing ones must PASS**

```bash
cd /mnt/c/MyWorld/Projects/prefltlf2pdfa
pytest tests/test_dsl_transpiler.py::TestTranspilerAlphabet -v 2>&1 | tail -30
```

Expected: ~7 existing tests pass, ~13 new tests fail with `AssertionError` or similar.

---

### Task 2: Add module-level helper functions to `transpiler.py`

**Files:**
- Modify: `prefltlf2pdfa/dsl/transpiler.py`

**Step 1: Add `import re` and three helpers after the existing imports**

At the top of `transpiler.py`, after `import prefltlf2pdfa.utils as _utils`, add:

```python
import re


def _parse_prop_list(s: str) -> list[str]:
    """Parse '[p, q, r]' bracket content into a list of stripped prop names."""
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_set_literal(token: str) -> set:
    """Parse '{p, q}' or '{}' string into a Python set of prop strings."""
    inner = token[1:-1].strip()
    return set() if not inner else {p.strip() for p in inner.split(",") if p.strip()}


def _parse_exclude_targets(rest: str) -> list[set]:
    """Parse the argument after 'exclude'.

    Accepts:
      '{a, b}, {c}'  → [{'a','b'}, {'c'}]
      'p'            → [{'p'}]
    """
    rest = rest.strip()
    if rest.startswith("{"):
        return [
            _parse_set_literal(m.group(0))
            for m in re.finditer(r'\{[^}]*\}', rest)
        ]
    if re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', rest):
        return [{rest}]
    raise DSLError(
        f"Invalid exclude argument: '{rest}'. "
        f"Expected '{{prop, ...}}' sets or a single proposition name."
    )
```

**Step 2: Verify helpers don't break existing tests**

```bash
pytest tests/test_dsl_transpiler.py -v -k "not (test_powerset_subset or test_singletons or test_emptyset or test_mixed or test_exclude or test_unknown)" 2>&1 | tail -20
```

Expected: all existing tests still pass (helpers are not yet called by `_build_alphabet`).

---

### Task 3: Rewrite `_build_alphabet` and run all tests

**Files:**
- Modify: `prefltlf2pdfa/dsl/transpiler.py`

**Step 1: Replace the body of `_build_alphabet` with the new two-pass implementation**

The method signature stays the same (`def _build_alphabet(self) -> list | None:`). Replace everything inside it:

```python
def _build_alphabet(self) -> list | None:
    """Build alphabet from alphabet block or default to powerset(propositions)."""
    declared = set(self._spec.propositions) if self._spec.propositions else None

    # No alphabet block: default to powerset of declared props, or None
    if self._spec.alphabet is None:
        if declared:
            return _utils.powerset(declared)
        return None

    generators: list[set] = []
    exclude_targets: list[set] = []

    for line in self._spec.alphabet.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # powerset() or powerset([p, q, ...])
        m = re.fullmatch(r'powerset\(\s*(?:\[([^\]]*)\])?\s*\)', line)
        if m:
            if m.group(1) is not None:
                props = set(_parse_prop_list(m.group(1)))
                if declared is not None:
                    undeclared = props - declared
                    if undeclared:
                        raise DSLError(
                            f"powerset([...]) contains undeclared proposition(s): {sorted(undeclared)}"
                        )
            else:
                if declared is None:
                    raise DSLError("'powerset()' in alphabet block requires a propositions block")
                props = declared
            generators.extend(_utils.powerset(props))
            continue

        # singletons() or singletons([p, q, ...])
        m = re.fullmatch(r'singletons\(\s*(?:\[([^\]]*)\])?\s*\)', line)
        if m:
            if m.group(1) is not None:
                props = set(_parse_prop_list(m.group(1)))
                if declared is not None:
                    undeclared = props - declared
                    if undeclared:
                        raise DSLError(
                            f"singletons([...]) contains undeclared proposition(s): {sorted(undeclared)}"
                        )
            else:
                if declared is None:
                    raise DSLError("'singletons()' in alphabet block requires a propositions block")
                props = declared
            generators.extend([{p} for p in sorted(props)])
            continue

        # emptyset
        if line == "emptyset":
            generators.append(set())
            continue

        # exclude {a, b}, {c} or exclude prop
        if line.startswith("exclude"):
            rest = line[len("exclude"):].strip()
            targets = _parse_exclude_targets(rest)
            for t in targets:
                if declared is not None:
                    undeclared = t - declared
                    if undeclared:
                        raise DSLError(
                            f"exclude target contains undeclared proposition(s): {sorted(undeclared)}"
                        )
            exclude_targets.extend(targets)
            continue

        # explicit set {p, q} or {}
        if line.startswith("{") and line.endswith("}"):
            s = _parse_set_literal(line)
            if declared is not None:
                undeclared = s - declared
                if undeclared:
                    raise DSLError(
                        f"Alphabet entry {line} contains undeclared proposition(s): {sorted(undeclared)}"
                    )
            generators.append(s)
            continue

        raise DSLError(
            f"Invalid alphabet entry: '{line}'. "
            f"Expected powerset(), singletons(), emptyset, {{...}}, or exclude ..."
        )

    # Pass 1 result: deduplicate generators (preserve first-seen order)
    result: list[set] = []
    for s in generators:
        if s not in result:
            result.append(s)

    # Pass 2: apply excludes (exact match; silent if not present)
    for ex in exclude_targets:
        try:
            result.remove(ex)
        except ValueError:
            pass

    return result
```

**Step 2: Run all alphabet tests**

```bash
pytest tests/test_dsl_transpiler.py::TestTranspilerAlphabet -v 2>&1 | tail -40
```

Expected: all tests pass (both old and new).

**Step 3: Run full test suite to check for regressions**

```bash
pytest tests/ -v -k "not slow" 2>&1 | tail -30
```

Expected: all non-slow tests pass.

**Step 4: Commit**

```bash
git add prefltlf2pdfa/dsl/transpiler.py tests/test_dsl_transpiler.py tests/fixtures/with_alphabet.spec
git commit -m "$(cat <<'EOF'
#add: extended alphabet block keywords (singletons, emptyset, powerset subset, exclude)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Add integration fixture and test

**Files:**
- Create: `tests/fixtures/with_alphabet_keywords.spec`
- Modify: `tests/test_dsl_integration.py`

**Step 1: Create `tests/fixtures/with_alphabet_keywords.spec`**

```
propositions
  p, q
end propositions

ltlf-formulas
  f0: G p
  f1: F q
end ltlf-formulas

preferences
  f0 > f1
end preferences

alphabet
  singletons()
  emptyset
  exclude {}
end alphabet
```

This exercises: `singletons()` (adds `{p}`, `{q}`), `emptyset` (adds `{}`), `exclude {}` (removes `{}`). Final alphabet: `[{p}, {q}]`.

**Step 2: Add integration test to `TestDSLtoPDFAPipeline` in `tests/test_dsl_integration.py`**

```python
def test_alphabet_keywords_fixture_parses(self):
    path = FIXTURES / "with_alphabet_keywords.spec"
    logger.info(f"[alpha-kw] Loading {path.name}")
    spec = parse_spec(path.read_text())
    t = Transpiler(spec)
    logger.info(f"[alpha-kw] Alphabet: {t._alphabet}")
    # singletons({p,q}) = [{p},{q}]; emptyset adds {}; exclude {} removes it → 2 sets
    assert len(t._alphabet) == 2
    assert {"p"} in t._alphabet
    assert {"q"} in t._alphabet
    assert set() not in t._alphabet
```

**Step 3: Run the new integration test**

```bash
pytest tests/test_dsl_integration.py::TestDSLtoPDFAPipeline::test_alphabet_keywords_fixture_parses -v
```

Expected: PASS.

**Step 4: Run full non-slow suite**

```bash
pytest tests/ -k "not slow" -v 2>&1 | tail -20
```

Expected: all pass.

**Step 5: Commit**

```bash
git add tests/fixtures/with_alphabet_keywords.spec tests/test_dsl_integration.py
git commit -m "$(cat <<'EOF'
#add: integration test and fixture for extended alphabet keywords

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Update TUTORIAL.md

**Files:**
- Modify: `TUTORIAL.md`

**Step 1: Replace the `alphabet` block section**

Find the `### \`alphabet\` (optional)` section and replace its content with:

````markdown
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
````

**Step 2: Replace the Alphabet Entry Syntax line in the Quick Reference section**

Find:

```
### Alphabet Entry Syntax

`{}` (empty set), `{p}`, `{p, q}`, `powerset()` (expands to full powerset of declared propositions)
```

Replace with:

```
### Alphabet Keywords

**Generators** (one per line; may be combined):
`powerset()`, `powerset([p, q])`, `singletons()`, `singletons([p, q])`, `emptyset`, `{p, q}`

**Exclusion:**
`exclude {a, b}, {c}` or `exclude p` (removes exact sets; silent if not present)
```

**Step 3: Run the full non-slow test suite one final time**

```bash
pytest tests/ -k "not slow" -v 2>&1 | tail -20
```

Expected: all pass. No code was changed in this task.

**Step 4: Commit**

```bash
git add TUTORIAL.md
git commit -m "$(cat <<'EOF'
#update: TUTORIAL.md alphabet section with singletons, emptyset, exclude keywords

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
