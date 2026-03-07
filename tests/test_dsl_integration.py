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
