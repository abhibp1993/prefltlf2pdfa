import os
import sys

from prefltlf2pdfa import *
from loguru import logger
from pathlib import Path

logger.remove()
logger.add(sys.stdout, level="DEBUG")

# Define paths
CUR_DIR = Path(__file__).resolve().parent
SPECS_DIR = CUR_DIR / "specs"
OUT_DIR = CUR_DIR / "out"
SPECS = os.listdir(SPECS_DIR)


def main():
    # Simple specification
    # f0 = PrefLTLf.from_file(SPECS_DIR / "spec0.prefltlf")
    # f0 = PrefLTLf.from_file(
    #     SPECS_DIR / "spec6.prefltlf",
    #     alphabet=[
    #         set(),
    #         {"d1"},
    #         {"d2"},
    #         {"d3"},
    #         {"d4"},
    #         {"d5"},
    #         {"d6"},
    #         {"d7"},
    #     ]
    # )
    f0 = PrefLTLf.from_file(
        SPECS_DIR / "spec8.prefltlf",
        auto_complete="minimal",
        alphabet=[
            set(),
            {"p0"},
            {"p1"},
            {"p2"},
            {"p3"},
            {"p4"},
            {"p5"},
            {"p6"},
            {"p7"},
            {"p8"},
            {"p9"},
            # {"d10"},
        ]
    )
    aut0 = f0.translate(show_progress=True)

    # See documentation for option description.
    sa, pg = paut2dot(aut0, show_sa_state=True, show_class=True, show_color=True, show_pg_state=True)
    paut2png(sa, pg, fpath=OUT_DIR, fname="aut")
    paut2svg(sa, pg, fpath=OUT_DIR, fname="aut")


if __name__ == '__main__':
    main()
