import pprint
import os
import sys

import json
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
    # Simple specification
    f0 = PrefLTLf.from_file(SPECS_DIR / "spec0.prefltlf")
    aut0 = f0.translate(show_progress=True)

    # Write to json
    with open(OUT_DIR / "spec0.json", "w") as outfile:
        out_str = jsonpickle.encode(aut0, indent=2)
        outfile.write(out_str)

    # Read from json
    with open(OUT_DIR / "spec0.json", "r") as infile:
        in_str = infile.read()
        aut0_load = jsonpickle.decode(in_str)

    assert aut0_load == aut0

    # # JSON formula (read)
    # with open(OUT_DIR / "pdfa.json", "r") as json_file:
    #     f0_json = json.load(json_file)
    #     f0_json = PrefAutomaton.deserialize(f0_json["pdfa"])
    #
    # print(f0_json)


if __name__ == '__main__':
    main()
