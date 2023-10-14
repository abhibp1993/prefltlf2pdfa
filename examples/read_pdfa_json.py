"""
Example of reading a pdfa.json file.
"""

import ast
import ioutils
import pathlib
import os
from pprint import pprint


DIR = pathlib.Path(__file__).parent.absolute()
FILE = os.path.join("icra2023", "pdfa.json")


def read_pdfa(path) -> dict:
    base_dict = ioutils.from_json(path)
    base_dict["states"] = {int(k): v for k, v in base_dict["states"].items()}
    base_dict["transitions"] = {int(k): v for k, v in base_dict["transitions"].items()}
    base_dict["pref_graph"]["nodes"] = {ast.literal_eval(k): v for k, v in base_dict["pref_graph"]["nodes"].items()}
    base_dict["pref_graph"]["edges"] = {
        ast.literal_eval(k): {ast.literal_eval(kk) for kk in v}
        for k, v in base_dict["pref_graph"]["edges"].items()
    }
    return base_dict


def main():
    pdfa = read_pdfa(os.path.join(DIR, FILE))
    print("States")
    print("======")
    pprint(pdfa["states"])

    print()
    print("Alphabet")
    print("========")
    pprint(pdfa["alphabet"])

    print()
    print("Transitions")
    print("===========")
    pprint(pdfa["transitions"])

    print()
    print("Initial State")
    print("=============")
    pprint(pdfa["init_state"])

    print()
    print("Preferece Graph")
    print("===============")
    print("Nodes:")
    for k, v in pdfa["pref_graph"]["nodes"].items():
        print(k)
        for vv in v:
            print("\t", pdfa["states"][vv]["state"])

    print()
    print("Edges:")
    pprint(pdfa["pref_graph"]["edges"])






if __name__ == '__main__':
    main()