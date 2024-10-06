import jsonpickle
from prefltlf2pdfa import *
from pprint import pprint

if __name__ == '__main__':
    with open("pdfa.jsonpkl", "r") as file:
        web_dict = file.read()
        web_dict = jsonpickle.decode(web_dict)

    # Raw input string
    f_str = web_dict["input"]
    pprint(f_str)

    # PrefLTLf formula
    prefltlf_formula = PrefLTLf.deserialize(web_dict["formula"])
    print("========== PrefLTLf Instance ==========")
    pprint(prefltlf_formula.serialize())

    # Preference automaton
    paut = PrefAutomaton.deserialize(web_dict["pdfa"])
    print("========== PrefAutomaton Instance ==========")
    pprint(paut.serialize())
