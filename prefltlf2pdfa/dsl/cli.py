"""
CLI entry point for prefltlf-compile.

Usage:
    prefltlf-compile input.spec
    prefltlf-compile input.spec -o output.prefltlf
    prefltlf-compile input.spec --stdout
"""

import argparse
import sys
from pathlib import Path

from .errors import DSLError
from .parser import parse_spec
from .transpiler import Transpiler


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="prefltlf-compile",
        description="Compile a .spec DSL file to the .prefltlf index-based format.",
    )
    parser.add_argument("input", type=Path, help="Input .spec file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .prefltlf file path")
    parser.add_argument("--stdout", action="store_true", help="Print output to stdout instead of a file")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        text = args.input.read_text(encoding="utf-8")
        spec = parse_spec(text)
        t = Transpiler(spec)
        result = t.to_string()
    except DSLError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.stdout:
        print(result, end="")
    else:
        out_path = args.output or args.input.with_suffix(".prefltlf")
        out_path.write_text(result, encoding="utf-8")
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
