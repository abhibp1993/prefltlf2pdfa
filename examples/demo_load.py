import json
import prefltlf2pdfa as pdfa
from pathlib import Path

DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    # Load the saved game
    pdfa_file = DIR / 'pdfa.json'
    with open(pdfa_file, "r") as f:
        obj_dict = json.load(f)

    aut = pdfa.PrefAutomaton.deserialize(obj_dict["pdfa"])
