import pytest
from prefltlf2pdfa.dsl.errors import DSLError
from prefltlf2pdfa.dsl.models import FormulaDecl, PrefStmt, Spec


class TestDSLError:
    def test_message_only(self):
        err = DSLError("something went wrong")
        assert str(err) == "something went wrong"

    def test_with_line(self):
        err = DSLError("bad token", line=7)
        assert str(err) == "Line 7: bad token"

    def test_with_suggestion(self):
        err = DSLError("unknown formula 'f1'", line=3, suggestion="f0")
        assert str(err) == "Line 3: unknown formula 'f1'. Did you mean 'f0'?"

    def test_is_value_error(self):
        assert isinstance(DSLError("x"), ValueError)


class TestModels:
    def test_formula_decl(self):
        fd = FormulaDecl(name="safety", ltlf_str="G safe", line=2)
        assert fd.name == "safety"
        assert fd.ltlf_str == "G safe"
        assert fd.line == 2

    def test_pref_stmt(self):
        ps = PrefStmt(lhs="f0", op=">", rhs="f1", line=5)
        assert ps.lhs == "f0"
        assert ps.op == ">"
        assert ps.rhs == "f1"

    def test_spec_defaults(self):
        spec = Spec(formulas={"f0": "G safe"}, preferences=[])
        assert spec.propositions == []
        assert spec.alphabet is None
        assert spec.options is None
