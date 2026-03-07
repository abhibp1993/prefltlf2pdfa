import pytest
from pathlib import Path
from prefltlf2pdfa.dsl.parser import parse_spec
from prefltlf2pdfa.dsl.transpiler import Transpiler
from prefltlf2pdfa.dsl.models import Spec, PrefStmt


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
