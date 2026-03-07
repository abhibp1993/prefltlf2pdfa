import pytest
from prefltlf2pdfa.dsl.parser import parse_spec
from prefltlf2pdfa.dsl.models import Spec, PrefStmt
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
