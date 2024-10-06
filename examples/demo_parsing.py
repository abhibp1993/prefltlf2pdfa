import pprint
import os
import sys

from pathlib import Path
from prefltlf2pdfa import *
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="DEBUG")
# logger.add(sys.stdout, level="INFO")

# Define paths
CUR_DIR = Path(__file__).resolve().parent
SPECS_DIR = CUR_DIR / "specs"
OUT_DIR = CUR_DIR / "out"
SPECS = os.listdir(SPECS_DIR)


def main():
    # Simple specification
    f0 = PrefLTLf.from_file(SPECS_DIR / "spec0.prefltlf")
    print(f0)

    # Simple specification (uses all preference operators)
    f1 = PrefLTLf.from_file(SPECS_DIR / "spec1.prefltlf")
    print(f1)

    # Corner case:  trivial specification (only `true`)
    f2 = PrefLTLf.from_file(SPECS_DIR / "spec2.prefltlf")
    print(f2)

    # Corner case (only `F(a)`).
    #   This needs auto-completion.
    try:
        f3 = PrefLTLf.from_file(SPECS_DIR / "spec3.prefltlf")
    except ValueError:
        f3 = PrefLTLf.from_file(SPECS_DIR / "spec3.prefltlf", auto_complete="minimal")
    print(f3)

    # Auto-completion check of `incomparable` option
    f3 = PrefLTLf.from_file(SPECS_DIR / "spec3.prefltlf", auto_complete="incomparable")
    print(f3)

    # Inconsistent specification
    try:
        f4 = PrefLTLf.from_file(SPECS_DIR / "erroneous" / "spec4.prefltlf", auto_complete="minimal")
    except ValueError as err:
        print(err)

    # LTLf parsing error
    try:
        f5 = PrefLTLf.from_file(SPECS_DIR / "erroneous" / "spec5.prefltlf", auto_complete="minimal")
    except ValueError as err:
        logger.success("Specification successfully raised ValueError.")


if __name__ == '__main__':
    main()

    # spec = ("""
#     # test
#     prefltlf 4
#
#
#     F a
#     G b
#     !(F(a) | G(b))
#     true U a
#
#     # SPec
#     >, 0, 1
#     >, 0, 2
#     >=, 1, 2
#     """)
#
#
# if __name__ == '__main__':
#     formula = PrefLTLf(spec)
#
#     print("====================================")
#     print("formula = ")
#     pprint(formula.serialize())
#
#     print()
#     print("====================================")
#     print("aut = ")
#     aut = formula.translate(semantics=semantics_mp_forall_exists)
#     pprint(aut.serialize())
#
#     sa, pg = paut2dot(aut, show_sa_state=True, show_class=True, show_color=True, show_pg_state=True)
#     paut2png(sa, pg, fname="aut")
#     paut2svg(sa, pg, fname="aut")
