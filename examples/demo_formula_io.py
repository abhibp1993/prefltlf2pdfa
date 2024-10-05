import pprint
import os
import sys

import jsonpickle
import pickle

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
    f0 = PrefLTLf.from_file(SPECS_DIR / "spec0.prefltlf")

    # Pickle formula (write)
    with open(OUT_DIR / "spec0.pkl", "wb") as pkl_file:
        pickle.dump(f0, pkl_file)

    # Pickle formula (read)
    with open(OUT_DIR / "spec0.pkl", "rb") as pkl_file:
        f0_pkl = pickle.load(pkl_file)

    assert f0_pkl == f0

    # JSON formula (write)
    with open(OUT_DIR / "spec0.json", "w") as json_file:
        out = jsonpickle.encode(f0.serialize(), indent=2)
        json_file.write(out)

    # JSON formula (read)
    with open(OUT_DIR / "spec0.json", "r") as json_file:
        json_str = json_file.read()
        f0_json = jsonpickle.decode(json_str)
        f0_json = PrefLTLf.deserialize(f0_json)

    assert f0_json == f0, f"{f0_json}\n\n{f0}"


if __name__ == '__main__':
    main()
