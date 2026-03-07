import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
MINIMAL_SPEC = FIXTURES / "minimal.spec"


class TestCLI:
    def test_stdout_flag_prints_prefltlf(self):
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli", str(MINIMAL_SPEC), "--stdout"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert result.stdout.startswith("prefltlf")

    def test_output_file_flag(self, tmp_path):
        out_file = tmp_path / "out.prefltlf"
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli", str(MINIMAL_SPEC), "-o", str(out_file)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert out_file.exists()
        assert out_file.read_text().startswith("prefltlf")

    def test_default_output_next_to_input(self, tmp_path):
        import shutil
        spec_copy = tmp_path / "test.spec"
        shutil.copy(MINIMAL_SPEC, spec_copy)
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli", str(spec_copy)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert (tmp_path / "test.prefltlf").exists()

    def test_missing_file_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli", "nonexistent.spec"],
            capture_output=True, text=True
        )
        assert result.returncode != 0

    def test_dsl_error_exits_nonzero(self, tmp_path):
        bad = tmp_path / "bad.spec"
        bad.write_text("ltlf-formulas\n  f0: G p\nend ltlf-formulas\npreferences\n  f0 > unknown\nend preferences\n")
        result = subprocess.run(
            [sys.executable, "-m", "prefltlf2pdfa.dsl.cli", str(bad), "--stdout"],
            capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "unknown" in result.stderr.lower() or "unknown" in result.stdout.lower()
