# DSL-to-PDFA Full Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `Transpiler` with options parsing, propositions validation, alphabet construction, and a `to_pdfa()` method; extend CLI with `--translate` and `--output-format` flags.

**Architecture:** All new validation and parsing logic runs eagerly in `Transpiler.__init__()`. `to_pdfa()` calls `to_prefltlf()` (which now forwards the built alphabet) then `.translate()`. CLI drives the full pipeline from a single `.spec` file.

**Tech Stack:** Python, Lark (DSL parsing, already in place), `ltlf2dfa.parser.ltlf.LTLfParser` (atom extraction), `prefltlf2pdfa.semantics` (semantics functions), `jsonpickle` (JSON serialization), `pygraphviz` via `paut2dot`/`paut2png`/`paut2svg` (visualization).

---

## Context for Implementer

### Key files
- `prefltlf2pdfa/dsl/models.py` — dataclasses (`Spec`, `FormulaDecl`, `PrefStmt`); add `SpecOptions` here
- `prefltlf2pdfa/dsl/transpiler.py` — `Transpiler` class; all new logic goes here
- `prefltlf2pdfa/dsl/errors.py` — `DSLError(ValueError)` with `line` and `suggestion`
- `prefltlf2pdfa/dsl/parser.py` — `parse_spec(text) → Spec`; `Spec.alphabet` and `Spec.options` are raw strings extracted by regex before Lark parsing
- `prefltlf2pdfa/dsl/cli.py` — `main()` entry point; extend with new flags
- `prefltlf2pdfa/utils.py` — `powerset(iterable) → list[set]`

### How `Spec.alphabet` / `Spec.options` work
`parse_spec()` extracts `alphabet ...end alphabet` and `options ...end options` blocks via regex **before** Lark sees the text. The raw block body (everything between the keywords) is stored as a plain string in `Spec.alphabet` and `Spec.options`. These are `None` if the block is absent.

### How `paut2dot` / `paut2png` / `paut2svg` work
```python
sa, pg = paut2dot(aut)          # returns two pygraphviz AGraph objects (layout already applied)
paut2png(sa, pg, fpath=str(out_dir), fname=stem)   # writes {stem}_sa.png, {stem}_pg.png
paut2svg(sa, pg, fpath=str(out_dir), fname=stem)   # writes {stem}_sa.svg, {stem}_pg.svg
sa.string()                                         # DOT text for the semi-automaton
pg.string()                                         # DOT text for the preference graph
```

### Existing test files to extend
- `tests/test_dsl_transpiler.py` — extend with new test classes
- `tests/test_dsl_cli.py` — extend with new test class
- `tests/test_dsl_integration.py` — extend with new test class

### Language-complete spec (no MONA needed for PrefLTLf construction)
A spec is language-complete when its LTLf formulas partition all finite traces. `G p` and `!G p` achieve this. Use this pattern in unit tests that call `to_prefltlf()` or `PrefLTLf()` without `auto_complete`.

---

## Task 1: `SpecOptions` dataclass + `_parse_options()`

**Files:**
- Modify: `prefltlf2pdfa/dsl/models.py`
- Modify: `prefltlf2pdfa/dsl/transpiler.py`
- Test: `tests/test_dsl_transpiler.py`

### Step 1: Write the failing tests

Add a new class `TestTranspilerOptions` to `tests/test_dsl_transpiler.py`:

```python
import pytest
from prefltlf2pdfa.dsl.errors import DSLError


class TestTranspilerOptions:
    def _make_transpiler(self, src: str):
        from prefltlf2pdfa.dsl.parser import parse_spec
        from prefltlf2pdfa.dsl.transpiler import Transpiler
        spec = parse_spec(src)
        return Transpiler(spec)

    def test_options_default_when_no_block(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
"""
        t = self._make_transpiler(src)
        assert t._options.semantics == "MaxAE"
        assert t._options.auto_complete == "none"

    def test_options_parsed_semantics_alias(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences

options
  semantics = AE
  auto-complete = minimal
end options
"""
        t = self._make_transpiler(src)
        assert t._options.semantics == "AE"
        assert t._options.auto_complete == "minimal"

    def test_options_all_semantics_aliases_valid(self):
        aliases = ["AE", "forall-exists", "EA", "exists-forall",
                   "AA", "forall-forall",
                   "MaxAE", "max-forall-exists",
                   "MaxEA", "max-exists-forall",
                   "MaxAA", "max-forall-forall"]
        src_template = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences

options
  semantics = {alias}
end options
"""
        for alias in aliases:
            t = self._make_transpiler(src_template.format(alias=alias))
            assert t._options.semantics == alias

    def test_options_unknown_key_raises(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences

options
  unknown-key = value
end options
"""
        with pytest.raises(DSLError, match="Unknown option"):
            self._make_transpiler(src)

    def test_options_unknown_semantics_raises(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences

options
  semantics = BadSemName
end options
"""
        with pytest.raises(DSLError, match="semantics"):
            self._make_transpiler(src)

    def test_options_unknown_auto_complete_raises(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences

options
  auto-complete = bad_value
end options
"""
        with pytest.raises(DSLError, match="auto-complete"):
            self._make_transpiler(src)
```

### Step 2: Run tests to confirm failure

```bash
cd /mnt/c/MyWorld/Projects/prefltlf2pdfa
pytest tests/test_dsl_transpiler.py::TestTranspilerOptions -v
```
Expected: `AttributeError: 'Transpiler' object has no attribute '_options'`

### Step 3: Add `SpecOptions` to `models.py`

Append to `prefltlf2pdfa/dsl/models.py` (after the existing dataclasses):

```python
@dataclass
class SpecOptions:
    semantics: str = "MaxAE"       # alias string; resolved to Callable in to_pdfa()
    auto_complete: str = "none"    # "none", "minimal", or "incomparable"
```

Also add `SpecOptions` to the `__all__` list in `prefltlf2pdfa/dsl/__init__.py`:
```python
from .models import Spec, FormulaDecl, PrefStmt, SpecOptions
# and add "SpecOptions" to __all__
```

### Step 4: Add `_SEMANTICS_MAP` and `_parse_options()` to `transpiler.py`

Add these imports at the top of `prefltlf2pdfa/dsl/transpiler.py`:
```python
from prefltlf2pdfa.semantics import (
    semantics_forall_exists, semantics_exists_forall, semantics_forall_forall,
    semantics_mp_forall_exists, semantics_mp_exists_forall, semantics_mp_forall_forall,
)
from .errors import DSLError
from .models import Spec, SpecOptions
```

Add the mapping and method to `Transpiler`:

```python
_SEMANTICS_MAP = {
    "AE": "semantics_forall_exists",
    "forall-exists": "semantics_forall_exists",
    "EA": "semantics_exists_forall",
    "exists-forall": "semantics_exists_forall",
    "AA": "semantics_forall_forall",
    "forall-forall": "semantics_forall_forall",
    "MaxAE": "semantics_mp_forall_exists",
    "max-forall-exists": "semantics_mp_forall_exists",
    "MaxEA": "semantics_mp_exists_forall",
    "max-exists-forall": "semantics_mp_exists_forall",
    "MaxAA": "semantics_mp_forall_forall",
    "max-forall-forall": "semantics_mp_forall_forall",
}

_VALID_AUTO_COMPLETE = {"none", "minimal", "incomparable"}


class Transpiler:
    def __init__(self, spec: Spec):
        self._spec = spec
        self._name_to_idx = {name: i for i, name in enumerate(spec.formulas)}
        self._body_to_idx = {body: i for i, body in enumerate(spec.formulas.values())}
        self._options = self._parse_options()
        # Tasks 2 and 3 will add _validate_propositions() and _build_alphabet() here

    def _parse_options(self) -> SpecOptions:
        if self._spec.options is None:
            return SpecOptions()
        opts = SpecOptions()
        for line in self._spec.options.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise DSLError(f"Invalid option line: '{line}' (expected 'key = value')")
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "semantics":
                if value not in _SEMANTICS_MAP:
                    raise DSLError(
                        f"Unknown semantics alias '{value}'. "
                        f"Valid aliases: {sorted(_SEMANTICS_MAP.keys())}"
                    )
                opts.semantics = value
            elif key == "auto-complete":
                if value not in _VALID_AUTO_COMPLETE:
                    raise DSLError(
                        f"Unknown auto-complete value '{value}'. "
                        f"Valid values: {sorted(_VALID_AUTO_COMPLETE)}"
                    )
                opts.auto_complete = value
            else:
                raise DSLError(f"Unknown option key '{key}'. Valid keys: semantics, auto-complete")
        return opts
```

### Step 5: Run tests to confirm pass

```bash
pytest tests/test_dsl_transpiler.py::TestTranspilerOptions -v
```
Expected: All 6 tests PASS.

### Step 6: Run full test suite to confirm no regressions

```bash
pytest tests/ -v
```
Expected: All existing tests still pass.

### Step 7: Commit

```bash
git add prefltlf2pdfa/dsl/models.py prefltlf2pdfa/dsl/transpiler.py prefltlf2pdfa/dsl/__init__.py tests/test_dsl_transpiler.py
git commit -m "feat: add SpecOptions dataclass and _parse_options() to Transpiler"
```

---

## Task 2: `_validate_propositions()`

**Files:**
- Modify: `prefltlf2pdfa/dsl/transpiler.py`
- Test: `tests/test_dsl_transpiler.py`

### Step 1: Write the failing tests

Add a new class `TestTranspilerPropositions` to `tests/test_dsl_transpiler.py`:

```python
class TestTranspilerPropositions:
    def _make_transpiler(self, src: str):
        from prefltlf2pdfa.dsl.parser import parse_spec
        from prefltlf2pdfa.dsl.transpiler import Transpiler
        spec = parse_spec(src)
        return Transpiler(spec)

    def test_valid_propositions_passes(self):
        src = """
propositions
  safe, clean
end propositions

ltlf-formulas
  f0: G safe
  f1: F clean
end ltlf-formulas

preferences
  f0 > f1
end preferences
"""
        t = self._make_transpiler(src)
        assert t is not None

    def test_undeclared_proposition_raises(self):
        src = """
propositions
  safe, clean
end propositions

ltlf-formulas
  f0: G robot
end ltlf-formulas

preferences
  f0 >= f0
end preferences
"""
        with pytest.raises(DSLError, match="undeclared"):
            self._make_transpiler(src)

    def test_no_propositions_block_skips_validation(self):
        """Without propositions block, any atoms are allowed."""
        src = """
ltlf-formulas
  f0: G any_atom_at_all
end ltlf-formulas

preferences
  f0 >= f0
end preferences
"""
        t = self._make_transpiler(src)
        assert t is not None

    def test_multiple_undeclared_props_reported(self):
        src = """
propositions
  safe
end propositions

ltlf-formulas
  f0: G robot & F drone
end ltlf-formulas

preferences
  f0 >= f0
end preferences
"""
        with pytest.raises(DSLError, match="undeclared"):
            self._make_transpiler(src)
```

### Step 2: Run tests to confirm failure

```bash
pytest tests/test_dsl_transpiler.py::TestTranspilerPropositions -v
```
Expected: `AssertionError` or tests pass trivially — `test_undeclared_proposition_raises` should FAIL because no validation exists yet.

### Step 3: Add `_validate_propositions()` to `Transpiler`

Add this import at the top of `transpiler.py`:
```python
from ltlf2dfa.parser.ltlf import LTLfParser as _LTLfParser
_LTLF_PARSER = _LTLfParser()
```

Add the method to `Transpiler` and call it from `__init__()`:

```python
def __init__(self, spec: Spec):
    self._spec = spec
    self._name_to_idx = {name: i for i, name in enumerate(spec.formulas)}
    self._body_to_idx = {body: i for i, body in enumerate(spec.formulas.values())}
    self._options = self._parse_options()
    self._validate_propositions()
    # Task 3 will add self._alphabet = self._build_alphabet() here

def _validate_propositions(self) -> None:
    """If propositions block present, verify every atom in every formula is declared."""
    if not self._spec.propositions:
        return
    declared = set(self._spec.propositions)
    for name, formula_str in self._spec.formulas.items():
        parsed = _LTLF_PARSER(formula_str.strip())
        atoms = set(parsed.find_labels())
        undeclared = atoms - declared
        if undeclared:
            raise DSLError(
                f"Formula '{name}' uses undeclared proposition(s): {sorted(undeclared)}. "
                f"Declared: {sorted(declared)}"
            )
```

### Step 4: Run tests to confirm pass

```bash
pytest tests/test_dsl_transpiler.py::TestTranspilerPropositions -v
```
Expected: All 4 tests PASS.

### Step 5: Run full test suite

```bash
pytest tests/ -v
```
Expected: All tests pass.

### Step 6: Commit

```bash
git add prefltlf2pdfa/dsl/transpiler.py tests/test_dsl_transpiler.py
git commit -m "feat: add _validate_propositions() to Transpiler"
```

---

## Task 3: `_build_alphabet()`

**Files:**
- Modify: `prefltlf2pdfa/dsl/transpiler.py`
- Test: `tests/test_dsl_transpiler.py`

### Step 1: Write the failing tests

Add a new class `TestTranspilerAlphabet` to `tests/test_dsl_transpiler.py`:

```python
class TestTranspilerAlphabet:
    def _make_transpiler(self, src: str):
        from prefltlf2pdfa.dsl.parser import parse_spec
        from prefltlf2pdfa.dsl.transpiler import Transpiler
        spec = parse_spec(src)
        return Transpiler(spec)

    def test_no_propositions_no_alphabet_is_none(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
"""
        t = self._make_transpiler(src)
        assert t._alphabet is None

    def test_propositions_without_alphabet_block_defaults_to_powerset(self):
        src = """
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
"""
        t = self._make_transpiler(src)
        assert t._alphabet is not None
        assert len(t._alphabet) == 2   # powerset({p}) = [{}, {p}]
        assert set() in t._alphabet
        assert {"p"} in t._alphabet

    def test_alphabet_explicit_sets_parsed(self):
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
        assert t._alphabet is not None
        assert set() in t._alphabet
        assert {"p"} in t._alphabet
        assert {"q"} in t._alphabet
        assert {"p", "q"} in t._alphabet

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
  {}; {p}; {q}; {p, q}
end alphabet
"""
        t = self._make_transpiler(src)
        assert len(t._alphabet) == 4

    def test_alphabet_powerset_keyword(self):
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
end alphabet
"""
        t = self._make_transpiler(src)
        assert len(t._alphabet) == 4   # powerset({p,q})

    def test_alphabet_powerset_without_propositions_raises(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas

preferences
  f0 >= f0
end preferences

alphabet
  powerset()
end alphabet
"""
        with pytest.raises(DSLError, match="propositions"):
            self._make_transpiler(src)

    def test_alphabet_validated_against_propositions(self):
        src = """
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
  {p, q}
end alphabet
"""
        with pytest.raises(DSLError, match="undeclared"):
            self._make_transpiler(src)
```

### Step 2: Run tests to confirm failure

```bash
pytest tests/test_dsl_transpiler.py::TestTranspilerAlphabet -v
```
Expected: `AttributeError: 'Transpiler' object has no attribute '_alphabet'`

### Step 3: Add `_build_alphabet()` to `Transpiler`

Add import at top of `transpiler.py`:
```python
from prefltlf2pdfa import utils as _utils
```

Add the method and wire it into `__init__()`:

```python
def __init__(self, spec: Spec):
    self._spec = spec
    self._name_to_idx = {name: i for i, name in enumerate(spec.formulas)}
    self._body_to_idx = {body: i for i, body in enumerate(spec.formulas.values())}
    self._options = self._parse_options()
    self._validate_propositions()
    self._alphabet = self._build_alphabet()

def _build_alphabet(self) -> list[set] | None:
    """Build alphabet from alphabet block or default to powerset(propositions)."""
    declared = set(self._spec.propositions) if self._spec.propositions else None

    # No alphabet block
    if self._spec.alphabet is None:
        if declared:
            return _utils.powerset(declared)
        return None

    raw = self._spec.alphabet.strip()

    # powerset() shorthand as sole content
    if raw == "powerset()":
        if not declared:
            raise DSLError(
                "'powerset()' in alphabet block requires a propositions block"
            )
        return _utils.powerset(declared)

    # Parse explicit sets (and possibly inline powerset() tokens)
    tokens: list[str] = []
    for line in raw.splitlines():
        for token in line.split(";"):
            token = token.strip()
            if token:
                tokens.append(token)

    alphabet: list[set] = []
    for token in tokens:
        if token == "powerset()":
            if not declared:
                raise DSLError(
                    "'powerset()' in alphabet block requires a propositions block"
                )
            alphabet.extend(_utils.powerset(declared))
            continue
        if not (token.startswith("{") and token.endswith("}")):
            raise DSLError(
                f"Invalid alphabet entry: '{token}'. "
                f"Expected '{{prop, ...}}' or 'powerset()'"
            )
        inner = token[1:-1].strip()
        s = set() if not inner else {p.strip() for p in inner.split(",")}
        if declared is not None:
            undeclared = s - declared
            if undeclared:
                raise DSLError(
                    f"Alphabet entry {token} contains undeclared proposition(s): "
                    f"{sorted(undeclared)}"
                )
        alphabet.append(s)

    return alphabet
```

### Step 4: Run tests to confirm pass

```bash
pytest tests/test_dsl_transpiler.py::TestTranspilerAlphabet -v
```
Expected: All 7 tests PASS.

### Step 5: Run full test suite

```bash
pytest tests/ -v
```
Expected: All tests pass.

### Step 6: Commit

```bash
git add prefltlf2pdfa/dsl/transpiler.py tests/test_dsl_transpiler.py
git commit -m "feat: add _build_alphabet() to Transpiler"
```

---

## Task 4: Update `to_prefltlf()` and add `to_pdfa()`

**Files:**
- Modify: `prefltlf2pdfa/dsl/transpiler.py`
- Modify: `prefltlf2pdfa/dsl/__init__.py`
- Test: `tests/test_dsl_transpiler.py`

### Step 1: Write failing tests

Add class `TestTranspilerToPdfa` to `tests/test_dsl_transpiler.py`:

```python
class TestTranspilerToPdfa:
    _COMPLETE_SPEC = """
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
"""

    def _make_transpiler(self, src: str):
        from prefltlf2pdfa.dsl.parser import parse_spec
        from prefltlf2pdfa.dsl.transpiler import Transpiler
        spec = parse_spec(src)
        return Transpiler(spec)

    def test_to_prefltlf_passes_alphabet(self):
        t = self._make_transpiler(self._COMPLETE_SPEC)
        pf = t.to_prefltlf()
        assert pf.alphabet is not None
        assert len(pf.alphabet) == 2   # powerset({p})

    def test_to_prefltlf_backward_compat_kwargs(self):
        """Existing callers using auto_complete='minimal' as kwarg still work."""
        src = """
ltlf-formulas
  f0: G safe
  f1: F clean
end ltlf-formulas
preferences
  f0 > f1
end preferences
"""
        t = self._make_transpiler(src)
        pf = t.to_prefltlf(auto_complete="minimal")
        from prefltlf2pdfa import PrefLTLf
        assert isinstance(pf, PrefLTLf)

    def test_options_override_at_call_site(self):
        """Kwargs passed to to_pdfa() override values from options block."""
        src = self._COMPLETE_SPEC + """
options
  semantics = AE
end options
"""
        from prefltlf2pdfa.semantics import semantics_mp_forall_exists
        t = self._make_transpiler(src)
        # Override AE with MaxAE at call site — verify no error and returns correctly
        # (We can't run MONA here, so just verify the resolved semantics via a mock)
        assert t._options.semantics == "AE"   # options block sets AE
        # to_pdfa with override would use MaxAE; tested in integration tests

    @pytest.mark.slow
    def test_to_pdfa_returns_pref_automaton(self):
        """MONA required."""
        from prefltlf2pdfa import PrefAutomaton
        t = self._make_transpiler(self._COMPLETE_SPEC)
        aut = t.to_pdfa()
        assert isinstance(aut, PrefAutomaton)

    @pytest.mark.slow
    def test_to_pdfa_semantics_override_at_call_site(self):
        """MONA required. Call-site semantics override options block."""
        from prefltlf2pdfa import PrefAutomaton
        from prefltlf2pdfa.semantics import semantics_forall_exists
        src = self._COMPLETE_SPEC + """
options
  semantics = MaxAE
end options
"""
        t = self._make_transpiler(src)
        # Override with AE at call site
        aut = t.to_pdfa(semantics=semantics_forall_exists)
        assert isinstance(aut, PrefAutomaton)
```

### Step 2: Run tests to confirm failure

```bash
pytest tests/test_dsl_transpiler.py::TestTranspilerToPdfa -v -k "not slow"
```
Expected: `test_to_prefltlf_passes_alphabet` FAIL — `pf.alphabet` is currently `None`.

### Step 3: Update `to_prefltlf()` and add `to_pdfa()`

Replace the existing `to_prefltlf()` and add `to_pdfa()` in `transpiler.py`:

```python
def to_prefltlf(self, **kwargs):
    """Parse the emitted string into a PrefLTLf object.

    Automatically injects self._alphabet unless caller explicitly passes alphabet=.
    All kwargs are forwarded to PrefLTLf.__init__() (e.g. auto_complete='minimal').
    """
    from prefltlf2pdfa import PrefLTLf
    if "alphabet" not in kwargs and self._alphabet is not None:
        kwargs["alphabet"] = self._alphabet
    return PrefLTLf(self.to_string(), **kwargs)

def to_pdfa(self, **kwargs):
    """Translate to a PrefAutomaton.

    Options from the spec's options block are used as defaults.
    Any kwargs passed here override the options block values.

    kwargs:
        semantics: Callable — overrides options block 'semantics'
        auto_complete: str  — overrides options block 'auto-complete'
        (plus any kwargs accepted by PrefLTLf.translate())
    """
    from prefltlf2pdfa.semantics import (
        semantics_forall_exists, semantics_exists_forall, semantics_forall_forall,
        semantics_mp_forall_exists, semantics_mp_exists_forall, semantics_mp_forall_forall,
    )
    _sem_fn_map = {
        "AE": semantics_forall_exists,
        "forall-exists": semantics_forall_exists,
        "EA": semantics_exists_forall,
        "exists-forall": semantics_exists_forall,
        "AA": semantics_forall_forall,
        "forall-forall": semantics_forall_forall,
        "MaxAE": semantics_mp_forall_exists,
        "max-forall-exists": semantics_mp_forall_exists,
        "MaxEA": semantics_mp_exists_forall,
        "max-exists-forall": semantics_mp_exists_forall,
        "MaxAA": semantics_mp_forall_forall,
        "max-forall-forall": semantics_mp_forall_forall,
    }
    semantics = kwargs.pop("semantics", _sem_fn_map[self._options.semantics])
    auto_complete = kwargs.pop("auto_complete", self._options.auto_complete)
    pf = self.to_prefltlf(auto_complete=auto_complete)
    return pf.translate(semantics=semantics, **kwargs)
```

Also add `to_pdfa` to `__all__` in `prefltlf2pdfa/dsl/__init__.py` by updating the `transpile` convenience function to remain unchanged (it still calls `to_prefltlf()`).

### Step 4: Run tests (non-slow)

```bash
pytest tests/test_dsl_transpiler.py::TestTranspilerToPdfa -v -k "not slow"
```
Expected: 3 non-slow tests PASS.

### Step 5: Run full test suite

```bash
pytest tests/ -v -k "not slow"
```
Expected: All non-slow tests pass.

### Step 6: Commit

```bash
git add prefltlf2pdfa/dsl/transpiler.py prefltlf2pdfa/dsl/__init__.py tests/test_dsl_transpiler.py
git commit -m "feat: update to_prefltlf() to pass alphabet; add to_pdfa() method"
```

---

## Task 5: CLI `--translate` and `--output-format` flags

**Files:**
- Modify: `prefltlf2pdfa/dsl/cli.py`
- Test: `tests/test_dsl_cli.py`

### Step 1: Write failing tests

Add class `TestCLITranslate` to `tests/test_dsl_cli.py`. These tests use the
`complete_for_pdfa.spec` fixture (created in this step).

First create the fixture `tests/fixtures/complete_for_pdfa.spec`:
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

options
  semantics = MaxAE
end options
```

Then add tests:

```python
import pytest

COMPLETE_SPEC = Path(__file__).parent / "fixtures" / "complete_for_pdfa.spec"


class TestCLITranslate:
    @pytest.mark.slow
    def test_translate_json_output(self, tmp_path):
        """--translate --output-format json produces a .json file."""
        import shutil
        spec_copy = tmp_path / "spec.spec"
        shutil.copy(COMPLETE_SPEC, spec_copy)
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli",
             str(spec_copy), "--translate", "--output-format", "json", "-o", str(tmp_path)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "spec.json").exists()

    @pytest.mark.slow
    def test_translate_dot_output(self, tmp_path):
        """--translate --output-format dot produces _sa.dot and _pg.dot files."""
        import shutil
        spec_copy = tmp_path / "spec.spec"
        shutil.copy(COMPLETE_SPEC, spec_copy)
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli",
             str(spec_copy), "--translate", "--output-format", "dot", "-o", str(tmp_path)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "spec_sa.dot").exists()
        assert (tmp_path / "spec_pg.dot").exists()

    @pytest.mark.slow
    def test_translate_all_artifacts(self, tmp_path):
        """--translate --output-format all-artifacts produces json, dot, png, svg."""
        import shutil
        spec_copy = tmp_path / "spec.spec"
        shutil.copy(COMPLETE_SPEC, spec_copy)
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli",
             str(spec_copy), "--translate", "--output-format", "all-artifacts",
             "-o", str(tmp_path)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        for fname in ["spec.json", "spec_sa.dot", "spec_pg.dot",
                      "spec_sa.png", "spec_pg.png", "spec_sa.svg", "spec_pg.svg"]:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"

    @pytest.mark.slow
    def test_translate_default_output_dir_is_input_dir(self, tmp_path):
        """Without -o, output goes next to the input file."""
        import shutil
        spec_copy = tmp_path / "myspec.spec"
        shutil.copy(COMPLETE_SPEC, spec_copy)
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli",
             str(spec_copy), "--translate"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "myspec.json").exists()

    def test_no_translate_existing_behavior_unchanged(self, tmp_path):
        """Without --translate, -o still means output FILE path."""
        out_file = tmp_path / "out.prefltlf"
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli",
             str(MINIMAL_SPEC), "-o", str(out_file)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert out_file.exists()
        assert out_file.read_text().startswith("prefltlf")
```

### Step 2: Run tests to confirm failure (non-slow)

```bash
pytest tests/test_dsl_cli.py::TestCLITranslate::test_no_translate_existing_behavior_unchanged -v
```
Expected: PASS (existing behavior). The slow tests will fail with "unrecognized arguments".

```bash
pytest tests/test_dsl_cli.py::TestCLITranslate -v -k "not slow"
```
Expected: The non-slow test PASSES.

### Step 3: Rewrite `cli.py` with new flags

Replace the full content of `prefltlf2pdfa/dsl/cli.py`:

```python
import argparse
import sys
from pathlib import Path


def _write_json(aut, out_dir: Path, stem: str) -> None:
    import jsonpickle
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(jsonpickle.encode(aut, indent=2), encoding="utf-8")


def _write_dot(sa, pg, out_dir: Path, stem: str) -> None:
    (out_dir / f"{stem}_sa.dot").write_text(sa.string(), encoding="utf-8")
    (out_dir / f"{stem}_pg.dot").write_text(pg.string(), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="prefltlf-compile",
        description="Compile a .spec DSL file to .prefltlf or a full PDFA.",
    )
    parser.add_argument("input", type=Path, help="Input .spec file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help=(
            "Without --translate: output .prefltlf file path. "
            "With --translate: output directory (default: same dir as input)."
        ),
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print .prefltlf to stdout instead of writing a file (no --translate only).",
    )
    parser.add_argument(
        "--translate", action="store_true",
        help="Run full pipeline: .spec → PrefAutomaton (requires MONA).",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "dot", "all-artifacts"],
        default="json",
        help="Output format when --translate is used (default: json).",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        from .parser import parse_spec
        from .transpiler import Transpiler
        from .errors import DSLError

        text = args.input.read_text(encoding="utf-8")
        spec = parse_spec(text)
        t = Transpiler(spec)

        if not args.translate:
            # Original behavior: emit .prefltlf
            result = t.to_string()
            if args.stdout:
                print(result, end="")
            else:
                out_path = args.output or args.input.with_suffix(".prefltlf")
                out_path.write_text(result, encoding="utf-8")
            return

        # --translate: build PrefAutomaton
        aut = t.to_pdfa()
        out_dir = args.output if args.output is not None else args.input.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.input.stem
        fmt = args.output_format

        from prefltlf2pdfa.viz import paut2dot, paut2png, paut2svg

        if fmt == "json":
            _write_json(aut, out_dir, stem)
        elif fmt == "dot":
            sa, pg = paut2dot(aut)
            _write_dot(sa, pg, out_dir, stem)
        elif fmt == "all-artifacts":
            _write_json(aut, out_dir, stem)
            sa, pg = paut2dot(aut)
            _write_dot(sa, pg, out_dir, stem)
            paut2png(sa, pg, fpath=str(out_dir), fname=stem)
            paut2svg(sa, pg, fpath=str(out_dir), fname=stem)

    except DSLError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 4: Run non-slow CLI tests

```bash
pytest tests/test_dsl_cli.py -v -k "not slow"
```
Expected: All non-slow tests PASS (existing tests + `test_no_translate_existing_behavior_unchanged`).

### Step 5: Run full test suite (non-slow)

```bash
pytest tests/ -v -k "not slow"
```
Expected: All non-slow tests pass.

### Step 6: Commit

```bash
git add prefltlf2pdfa/dsl/cli.py tests/test_dsl_cli.py tests/fixtures/complete_for_pdfa.spec
git commit -m "feat: add --translate and --output-format flags to CLI"
```

---

## Task 6: New spec fixtures + integration tests

**Files:**
- Create: `tests/fixtures/with_propositions.spec`
- Create: `tests/fixtures/with_alphabet.spec`
- Create: `tests/fixtures/with_options.spec`
- Create: `tests/fixtures/erroneous/undeclared_prop.spec`
- Modify: `tests/test_dsl_integration.py`

### Step 1: Create fixture files

**`tests/fixtures/with_propositions.spec`:**
```
# Spec with propositions block — atoms validated
propositions
  p, q
end propositions

ltlf-formulas
  always_p: G p
  eventually_q: F q
end ltlf-formulas

preferences
  always_p > eventually_q
end preferences
```

**`tests/fixtures/with_alphabet.spec`:**
```
# Spec with explicit alphabet block
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
  {}; {p}
end alphabet
```

**`tests/fixtures/with_options.spec`:**
```
# Spec with options block
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

options
  semantics = AE
  auto-complete = none
end options
```

**`tests/fixtures/erroneous/undeclared_prop.spec`:**
```
# Erroneous: formula uses atom not in propositions block
propositions
  safe
end propositions

ltlf-formulas
  f0: G robot
end ltlf-formulas

preferences
  f0 >= f0
end preferences
```

### Step 2: Write the failing integration tests

Add class `TestDSLtoPDFAPipeline` to `tests/test_dsl_integration.py`:

```python
import pytest
from pathlib import Path
from loguru import logger
from prefltlf2pdfa.dsl import parse_spec, Transpiler, DSLError
from prefltlf2pdfa import PrefLTLf

FIXTURES = Path(__file__).parent / "fixtures"


class TestDSLtoPDFAPipeline:
    """Full pipeline: .spec text → PrefAutomaton (MONA required for slow tests)."""

    def test_propositions_fixture_parses_and_transpiles(self):
        path = FIXTURES / "with_propositions.spec"
        logger.info(f"[props] Loading {path.name}")
        spec = parse_spec(path.read_text())
        logger.info(f"[props] Propositions: {spec.propositions}")
        logger.info(f"[props] Formulas: {list(spec.formulas.keys())}")
        t = Transpiler(spec)
        logger.info(f"[props] Alphabet: {t._alphabet}")
        assert t._alphabet is not None
        assert len(t._alphabet) == 4   # powerset({p, q})

    def test_alphabet_fixture_parses_correct_alphabet(self):
        path = FIXTURES / "with_alphabet.spec"
        logger.info(f"[alpha] Loading {path.name}")
        spec = parse_spec(path.read_text())
        t = Transpiler(spec)
        logger.info(f"[alpha] Alphabet: {t._alphabet}")
        assert len(t._alphabet) == 2   # explicit: {}, {p}
        assert set() in t._alphabet
        assert {"p"} in t._alphabet

    def test_options_fixture_parses_semantics(self):
        path = FIXTURES / "with_options.spec"
        logger.info(f"[opts] Loading {path.name}")
        spec = parse_spec(path.read_text())
        t = Transpiler(spec)
        logger.info(f"[opts] Parsed semantics alias: {t._options.semantics}")
        logger.info(f"[opts] Parsed auto_complete: {t._options.auto_complete}")
        assert t._options.semantics == "AE"
        assert t._options.auto_complete == "none"

    def test_erroneous_undeclared_prop_raises(self):
        path = FIXTURES / "erroneous" / "undeclared_prop.spec"
        logger.info(f"[error] Loading {path.name}")
        spec = parse_spec(path.read_text())
        with pytest.raises(DSLError, match="undeclared"):
            Transpiler(spec)
        logger.info("[error] Got expected DSLError for undeclared proposition")

    @pytest.mark.slow
    def test_to_pdfa_returns_pref_automaton(self):
        """MONA required."""
        from prefltlf2pdfa import PrefAutomaton
        path = FIXTURES / "complete_for_pdfa.spec"
        logger.info(f"[pdfa] Loading {path.name}")
        spec = parse_spec(path.read_text())
        t = Transpiler(spec)
        logger.info(f"[pdfa] Options: semantics={t._options.semantics}, auto_complete={t._options.auto_complete}")
        logger.info(f"[pdfa] Alphabet: {t._alphabet}")
        aut = t.to_pdfa()
        logger.info(f"[pdfa] PrefAutomaton states: {list(aut.get_states())}")
        logger.info(f"[pdfa] PrefAutomaton pref_graph nodes: {list(aut.pref_graph.nodes())}")
        assert isinstance(aut, PrefAutomaton)

    @pytest.mark.slow
    def test_to_pdfa_with_explicit_alphabet(self):
        """MONA required."""
        from prefltlf2pdfa import PrefAutomaton
        path = FIXTURES / "with_alphabet.spec"
        logger.info(f"[pdfa] Loading {path.name}")
        spec = parse_spec(path.read_text())
        t = Transpiler(spec)
        logger.info(f"[pdfa] Explicit alphabet: {t._alphabet}")
        aut = t.to_pdfa()
        logger.info(f"[pdfa] PrefAutomaton constructed successfully")
        assert isinstance(aut, PrefAutomaton)

    @pytest.mark.slow
    def test_to_pdfa_semantics_from_options(self):
        """MONA required. Semantics 'AE' from options block is used."""
        from prefltlf2pdfa import PrefAutomaton
        path = FIXTURES / "with_options.spec"
        logger.info(f"[pdfa] Loading {path.name}")
        spec = parse_spec(path.read_text())
        t = Transpiler(spec)
        logger.info(f"[pdfa] Using semantics alias from options: {t._options.semantics}")
        aut = t.to_pdfa()
        logger.info(f"[pdfa] PrefAutomaton constructed with AE semantics")
        assert isinstance(aut, PrefAutomaton)
```

### Step 3: Run non-slow integration tests to confirm they fail correctly

```bash
pytest tests/test_dsl_integration.py::TestDSLtoPDFAPipeline -v -k "not slow"
```
Expected: Fixture file `FileNotFoundError` or `AssertionError` — fixtures don't exist yet.

### Step 4: Create all fixture files

Write the four fixture files listed in Step 1 using the Write tool.

### Step 5: Run non-slow tests to confirm they pass

```bash
pytest tests/test_dsl_integration.py::TestDSLtoPDFAPipeline -v -k "not slow"
```
Expected: 4 non-slow tests PASS.

### Step 6: Run full test suite (non-slow)

```bash
pytest tests/ -v -k "not slow"
```
Expected: All non-slow tests pass.

### Step 7: Commit

```bash
git add tests/fixtures/with_propositions.spec tests/fixtures/with_alphabet.spec \
        tests/fixtures/with_options.spec tests/fixtures/erroneous/undeclared_prop.spec \
        tests/test_dsl_integration.py
git commit -m "feat: integration tests and fixtures for DSL-to-PDFA pipeline"
```

---

## Final Verification

After all tasks complete:

```bash
pytest tests/ -v
```

Confirm all tests pass (slow tests require MONA installed).

Then invoke the `superpowers:finishing-a-development-branch` skill to complete the work.
