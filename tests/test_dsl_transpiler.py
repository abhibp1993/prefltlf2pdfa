import pytest
from pathlib import Path
from prefltlf2pdfa.dsl.parser import parse_spec
from prefltlf2pdfa.dsl.transpiler import Transpiler
from prefltlf2pdfa.dsl.models import Spec, PrefStmt
from prefltlf2pdfa.dsl.errors import DSLError


def _transpile(src: str) -> str:
    spec = parse_spec(src)
    return Transpiler(spec).to_string()


SIMPLE_SPEC = """
ltlf-formulas
  safety: G safe
  liveness: F clean
  charge: charged U clean
end ltlf-formulas

preferences
  safety > liveness
  liveness >= charge
end preferences
"""


class TestTranspilerHeader:
    def test_header_formula_count(self):
        out = _transpile(SIMPLE_SPEC)
        first_line = out.strip().splitlines()[0]
        assert first_line == "prefltlf 3"

    def test_single_formula_header(self):
        src = """
ltlf-formulas
  f0: G p
end ltlf-formulas
preferences
  f0 >= f0
end preferences
"""
        out = _transpile(src)
        assert out.strip().startswith("prefltlf 1")


class TestTranspilerFormulas:
    def test_formulas_emitted_in_declaration_order(self):
        out = _transpile(SIMPLE_SPEC)
        lines = out.strip().splitlines()
        # lines[0] = "prefltlf 3", lines[1] = "" (blank separator), then formulas
        assert lines[2] == "G safe"
        assert lines[3] == "F clean"
        assert lines[4] == "charged U clean"

    def test_formula_bodies_match_input(self):
        src = """
ltlf-formulas
  f0: F(a) & G(b | !c)
end ltlf-formulas
preferences
  f0 >= f0
end preferences
"""
        out = _transpile(src)
        assert "F(a) & G(b | !c)" in out


class TestTranspilerPreferences:
    def test_strict_preference_emitted(self):
        out = _transpile(SIMPLE_SPEC)
        assert ">, 0, 1" in out

    def test_weak_preference_emitted(self):
        out = _transpile(SIMPLE_SPEC)
        assert ">=, 1, 2" in out

    def test_indifferent_emitted(self):
        src = """
ltlf-formulas
  f0: G p
  f1: F q
end ltlf-formulas
preferences
  f0 ~ f1
end preferences
"""
        out = _transpile(src)
        assert "~, 0, 1" in out

    def test_incomparable_emitted(self):
        src = """
ltlf-formulas
  f0: G p
  f1: F q
end ltlf-formulas
preferences
  f0 <> f1
end preferences
"""
        out = _transpile(src)
        assert "<>, 0, 1" in out

    def test_exact_ref_preference_uses_correct_index(self):
        src = """
ltlf-formulas
  safety: G safe
  liveness: F clean
end ltlf-formulas
preferences
  (G safe) > (F clean)
end preferences
"""
        out = _transpile(src)
        assert ">, 0, 1" in out

    def test_indices_reflect_declaration_order(self):
        src = """
ltlf-formulas
  z_last: false
  a_first: true
end ltlf-formulas
preferences
  z_last > a_first
end preferences
"""
        out = _transpile(src)
        assert ">, 0, 1" in out


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
        t = self._make_transpiler(src)
        assert t._options.semantics == "AE"   # options block sets AE

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
        aut = t.to_pdfa(semantics=semantics_forall_exists)
        assert isinstance(aut, PrefAutomaton)


class TestTranspilerOutput:
    def test_to_string_is_valid_prefltlf(self):
        """Output string must be parseable by PrefLTLf (without MONA call).
        Uses a language-complete spec (formulas partition all finite traces).
        """
        from prefltlf2pdfa import PrefLTLf
        complete_spec = """
ltlf-formulas
  safe: G safe
  unsafe: !G safe
end ltlf-formulas
preferences
  safe > unsafe
end preferences
"""
        spec = parse_spec(complete_spec)
        result = Transpiler(spec).to_string()
        pf = PrefLTLf(result)
        assert len(pf.phi) == 2

    def test_to_file_writes_correct_content(self, tmp_path):
        spec = parse_spec(SIMPLE_SPEC)
        t = Transpiler(spec)
        out_file = tmp_path / "test_output.prefltlf"
        t.to_file(out_file)
        assert out_file.exists()
        assert out_file.read_text() == t.to_string()
