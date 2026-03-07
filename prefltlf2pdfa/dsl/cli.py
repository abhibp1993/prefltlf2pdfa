"""
CLI entry point for prefltlf-compile.

Usage:
    prefltlf-compile input.spec
    prefltlf-compile input.spec -o output.prefltlf
    prefltlf-compile input.spec --stdout
    prefltlf-compile input.spec --translate [--output-format json|dot|all-artifacts] [-o out_dir]
"""

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
