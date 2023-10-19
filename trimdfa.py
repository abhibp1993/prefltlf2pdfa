"""
Method to trim DFA based on specific alphabet.
The transitions need to fire synchronously.

"""
import ast
import ioutils
import os
import pathlib
import translate
import spot

from pprint import pprint


DIR = pathlib.Path(__file__).parent.absolute()
FILE = os.path.join("examples", "icra2023", "pdfa.json")
PNG_FILE = os.path.join("examples", "icra2023", "trimmed.png")
PDFA_FILE = os.path.join("examples", "icra2023", "trimmed_pdfa.json")


def trim(pdfa, alphabet):
    # Remove transitions whose guard condition is always false
    rm_transitions = set()
    for source, transitions in pdfa["transitions"].items():
        for formula, target in transitions.items():
            # if not any(spot.formula(f"{formula} & {prop}") == spot.formula_ff() for prop in propositions if len(prop) != 0):
            # if not any(evaluate(spot.formula(formula), true_atoms) for true_atoms in alphabet):
            if not any(evaluate(spot.formula(formula), true_atoms) for true_atoms in alphabet):
                rm_transitions.add((source, formula, target))

    for source, formula, target in rm_transitions:
        del pdfa["transitions"][source][formula]

    # Prune the states
    init_state = pdfa["init_state"]
    reachable_states = set()
    frontier = [init_state]
    set_frontier = {init_state}
    while len(frontier) > 0:
        state = frontier.pop(0)
        set_frontier.remove(state)
        reachable_states.add(state)

        for _, target in pdfa["transitions"][state].items():
            if target not in set_frontier and target not in reachable_states:
                frontier.append(target)
                set_frontier.add(target)

    for state in set(pdfa["states"]) - reachable_states:
        del pdfa["states"][state]
        del pdfa["transitions"][state]

    rm_transitions = set()
    for state, transitions in pdfa["transitions"].items():
        for formula, target in transitions.items():
            if target not in reachable_states:
                rm_transitions.add((state, formula))

    for state, formula in rm_transitions:
        del pdfa["transitions"][state][formula]

    # Prune the preference graph
    rm_nodes = set()
    for node, partition in pdfa["pref_graph"]["nodes"].items():
        if len(set.intersection(partition, pdfa["states"])) == 0:
            rm_nodes.add(node)

    for node in rm_nodes:
        del pdfa["pref_graph"]["nodes"][node]
        del pdfa["pref_graph"]["edges"][node]

    for node in pdfa["pref_graph"]["nodes"]:
        pdfa["pref_graph"]["nodes"][node] = set.intersection(pdfa["pref_graph"]["nodes"][node], pdfa["states"])

    for source, targets in pdfa["pref_graph"]["edges"].items():
        for target in targets - set(pdfa["pref_graph"]["nodes"].keys()):
            pdfa["pref_graph"]["edges"][source].remove(target)

    return pdfa


def evaluate(spot_formula, true_atoms):
    """
    Evaluates a propositional logic formula given the set of true atoms.
    :param true_atoms: (Iterable[str]) A propositional logic formula.
    :return: (bool) True if formula is true, otherwise False.
    """

    # Define a transform to apply to AST of spot.formula.
    def transform(node: spot.formula):
        if node.is_literal():
            if "!" not in node.to_str():
                if node.to_str() in true_atoms:
                    return spot.formula.tt()
                else:
                    return spot.formula.ff()

        return node.map(transform)

    # Apply the transform and return the result.
    # Since every literal is replaced by true or false,
    #   the transformed formula is guaranteed to be either true or false.
    return True if transform(spot_formula).is_tt() else False


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
    pdfa = trim(pdfa, [set(), {"d"}, {"o"}, {"t"}])
    translate.pdfa_to_png(pdfa, PNG_FILE)
    pdfa["pref_graph"]["nodes"] = {str(k): v for k, v in pdfa["pref_graph"]["nodes"].items()}
    pdfa["pref_graph"]["edges"] = {str(k): {str(vv) for vv in v} for k, v in pdfa["pref_graph"]["edges"].items()}
    ioutils.to_json(PDFA_FILE, pdfa)


if __name__ == '__main__':
    main()