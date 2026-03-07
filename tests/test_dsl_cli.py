import subprocess
import sys
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
MINIMAL_SPEC = FIXTURES / "minimal.spec"
COMPLETE_SPEC = FIXTURES / "complete_for_pdfa.spec"


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
