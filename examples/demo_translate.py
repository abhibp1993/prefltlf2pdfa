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
    f0 = PrefLTLf.from_file(
        SPECS_DIR / "spec6.prefltlf",
        alphabet=[
            set(),
            {"d1"},
            {"d2"},
            {"d3"},
            {"d4"},
            {"d5"},
            {"d6"},
            {"d7"},
        ]
    )
    aut0 = f0.translate(show_progress=True)
    print(aut0)


if __name__ == '__main__':
    main()
