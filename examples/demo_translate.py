import os
import sys

from prefltlf2pdfa import *
from loguru import logger
from pathlib import Path

logger.remove()
logger.add(sys.stdout, level="INFO")

# Define paths
CUR_DIR = Path(__file__).resolve().parent
SPECS_DIR = CUR_DIR / "specs"
OUT_DIR = CUR_DIR / "out"
SPECS = os.listdir(SPECS_DIR)


def main():
    # Simple specification
    f0 = PrefLTLf.from_file(SPECS_DIR / "spec0.prefltlf")
    aut0 = f0.translate(show_progress=True)
    print(aut0)


if __name__ == '__main__':
    main()
