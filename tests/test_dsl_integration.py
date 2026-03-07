import pytest
from pathlib import Path
from loguru import logger
from prefltlf2pdfa.dsl import parse_spec, Transpiler, DSLError
from prefltlf2pdfa import PrefLTLf

FIXTURES = Path(__file__).parent / "fixtures"


class TestAllFixturesParse:
    """Smoke test: all valid fixtures parse without error."""

    @pytest.mark.parametrize("spec_file", [
        "minimal.spec",
        "full.spec",
        "chain_prefs.spec",
        "verbatim_prefs.spec",
        "exact_refs.spec",
        "reverse_ops.spec",
        "no_propositions.spec",
    ])
    def test_fixture_parses(self, spec_file):
        path = FIXTURES / spec_file
        logger.info(f"[parse] Loading fixture: {path.name}")
        text = path.read_text()

        spec = parse_spec(text)
        logger.info(f"[parse] Parsed {len(spec.formulas)} formula(s): {list(spec.formulas.keys())}")
        logger.info(f"[parse] Parsed {len(spec.preferences)} preference statement(s)")
        if spec.propositions:
            logger.info(f"[parse] Propositions: {spec.propositions}")
        if spec.alphabet:
            logger.info(f"[parse] Alphabet stub: {spec.alphabet!r}")
        if spec.options:
            logger.info(f"[parse] Options stub: {spec.options!r}")

        assert spec is not None
        assert len(spec.formulas) > 0

    @pytest.mark.parametrize("spec_file", [
        "erroneous/unknown_name.spec",
        "erroneous/duplicate_name.spec",
        "erroneous/syntax_error.spec",
    ])
    def test_erroneous_fixture_raises(self, spec_file):
        path = FIXTURES / spec_file
        logger.info(f"[error] Loading erroneous fixture: {path.name}")
        text = path.read_text()

        with pytest.raises(DSLError) as exc_info:
            parse_spec(text)
        logger.info(f"[error] Got expected DSLError: {exc_info.value}")


class TestEquivalenceToLegacyFormat:
    """Transpiled output must match hand-written .prefltlf equivalents."""

    def test_minimal_spec_matches_legacy(self):
        path = FIXTURES / "minimal.spec"
        logger.info(f"[transpile] Testing {path.name} → legacy format equivalence")
        text = path.read_text()
        spec = parse_spec(text)
        result = Transpiler(spec).to_string()

        logger.info(f"[transpile] Output:\n{result}")
        expected = "prefltlf 2\n\nG safe\nF clean\n\n>, 0, 1\n"
        assert result == expected

    def test_chain_prefs_expands_correctly(self):
        path = FIXTURES / "chain_prefs.spec"
        logger.info(f"[transpile] Testing chain expansion in {path.name}")
        text = path.read_text()
        spec = parse_spec(text)
        result = Transpiler(spec).to_string()

        logger.info(f"[transpile] Output:\n{result}")
        assert "prefltlf 4" in result
        assert ">, 0, 1" in result
        assert ">=, 1, 2" in result
        assert "~, 2, 3" in result

    def test_reverse_ops_normalized_in_output(self):
        path = FIXTURES / "reverse_ops.spec"
        logger.info(f"[transpile] Testing reverse op normalization in {path.name}")
        text = path.read_text()
        spec = parse_spec(text)
        result = Transpiler(spec).to_string()

        logger.info(f"[transpile] Output:\n{result}")
        # f0 < f1 → f1 > f0 → >, 1, 0
        assert ">, 1, 0" in result
        # f1 <= f2 → f2 >= f1 → >=, 2, 1
        assert ">=, 2, 1" in result


class TestEndToEndPipeline:
    """Full pipeline: .spec text → PrefLTLf object (no MONA needed)."""

    def test_parse_to_prefltlf_object(self):
        """Passes auto_complete='minimal' since fixture formulas don't partition all traces."""
        path = FIXTURES / "minimal.spec"
        logger.info(f"[e2e] {path.name} → PrefLTLf object")
        text = path.read_text()
        spec = parse_spec(text)

        logger.info(f"[e2e] Formulas: {spec.formulas}")
        logger.info(f"[e2e] Preferences: {spec.preferences}")

        pf = Transpiler(spec).to_prefltlf(auto_complete="minimal")
        logger.info(f"[e2e] PrefLTLf.phi = {pf.phi}")
        logger.info(f"[e2e] PrefLTLf.relation = {pf.relation}")

        assert isinstance(pf, PrefLTLf)
        # auto_complete="minimal" adds one extra formula, so 2+1=3
        assert len(pf.phi) == 3

    def test_full_spec_to_prefltlf(self):
        """Passes auto_complete='minimal' since fixture formulas don't partition all traces."""
        path = FIXTURES / "full.spec"
        logger.info(f"[e2e] {path.name} → PrefLTLf object")
        text = path.read_text()
        spec = parse_spec(text)

        logger.info(f"[e2e] Formulas: {spec.formulas}")
        logger.info(f"[e2e] Preferences: {spec.preferences}")
        logger.info(f"[e2e] Propositions: {spec.propositions}")

        pf = Transpiler(spec).to_prefltlf(auto_complete="minimal")
        logger.info(f"[e2e] PrefLTLf.phi = {pf.phi}")

        # auto_complete="minimal" adds one extra formula, so 3+1=4
        assert len(pf.phi) == 4

    def test_to_file_then_read_back(self, tmp_path):
        """Passes auto_complete='minimal' since fixture formulas don't partition all traces."""
        path = FIXTURES / "minimal.spec"
        logger.info(f"[e2e] {path.name} → file → read back")
        text = path.read_text()
        spec = parse_spec(text)
        t = Transpiler(spec)

        out = tmp_path / "output.prefltlf"
        t.to_file(out)
        logger.info(f"[e2e] Written to: {out}")
        logger.info(f"[e2e] Content:\n{out.read_text()}")

        pf = PrefLTLf.from_file(out, auto_complete="minimal")
        logger.info(f"[e2e] Read back PrefLTLf.phi = {pf.phi}")
        # auto_complete="minimal" adds one extra formula, so 2+1=3
        assert len(pf.phi) == 3
