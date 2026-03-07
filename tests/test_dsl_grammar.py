# tests/test_dsl_grammar.py
from pathlib import Path
import lark

GRAMMAR_PATH = Path(__file__).parent.parent / "prefltlf2pdfa" / "dsl" / "grammar.lark"

MINIMAL_SPEC = """
ltlf-formulas
  f0: G safe
  f1: F clean
end ltlf-formulas

preferences
  f0 > f1
end preferences
"""


def test_grammar_loads():
    parser = lark.Lark.open(str(GRAMMAR_PATH), parser="earley", start="spec", propagate_positions=True)
    assert parser is not None


def test_grammar_parses_minimal_spec():
    parser = lark.Lark.open(str(GRAMMAR_PATH), parser="earley", start="spec", propagate_positions=True)
    tree = parser.parse(MINIMAL_SPEC)
    assert tree.data == "spec"
