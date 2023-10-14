"""
Translator from PrefLTLf to PDFA.
"""
import ast
import copy
import itertools
import json
import os
import pathlib
import pprint
import pygraphviz
import spot

from ltlf2dfa.parser.ltlf import LTLfParser
from loguru import logger
from networkx.drawing import nx_agraph

PARSER = LTLfParser()


# =================================================================================== #
# PARSE FORMULA, BUILD PREFERENCE MODEL
# =================================================================================== #

def parse_prefltlf(file):
    """
    Read a PrefLTLf formula from a file.

    :return: (set) A set of triples (PREF_TYPE, LTLf Formula, LTLf Formula).
    """
    formula = set()
    with open(file, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            raise EOFError("Empty PrefLTLf file.")
        formula_type = lines[0].strip().lower()
        if formula_type != "prefltlf":
            raise ValueError(f"Not a PrefLTLf formula. Likely a '{formula_type}' formula.")

        for i in range(1, len(lines)):
            stmt = lines[i].split(",")
            pref_type = stmt[0].strip()
            assert pref_type in [">", ">=", "~", "<>"], f"The formula is ill-formed. Unrecognized operator `{pref_type}`"
            left = PARSER(stmt[1].strip())
            right = PARSER(stmt[2].strip())
            formula.add((pref_type, left, right))

    return formula


def build_prefltlf_model(formula):
    """
    Build the preference model $$(\Phi, \trianglerighteq)$$ from a PrefLTLf formula.

    :param formula: (set) A set of triples (PREF_TYPE, LTLf Formula, LTLf Formula).
    :return: ($$(\Phi, \trianglerighteq)$$) A set of LTLf formulas $$\Phi$ and a set of triples (PREF_TYPE, LTLf, LTLf).
    """
    atoms = set()
    phi = set()
    preorder = set()

    # Construct preorder.
    # First, process the non-incomparability formulas because they add elements to preorder relation.
    for pref_type, phi1, phi2 in (f for f in formula if f[0] != "<>"):
        phi.add(phi1)
        phi.add(phi2)
        if pref_type == ">" or pref_type == ">=":
            preorder.add((phi1, phi2))

        if pref_type == "~":
            preorder.add((phi1, phi2))
            preorder.add((phi2, phi1))

    logger.debug(f"Constructed preorder except incomparability: \n{pprint.pformat(preorder)}")

    # Second, process the incomparability formulas because it removes elements to preorder relation.
    for pref_type, phi1, phi2 in (f for f in formula if f[0] == "<>"):
        phi.add(phi1)
        phi.add(phi2)
        if (phi1, phi2) in preorder:
            logger.warning(f"{(phi1, phi2)} is removed from preorder.")
            preorder.remove((phi1, phi2))

        if (phi2, phi1) in preorder:
            logger.warning(f"{(phi2, phi1)} is removed from preorder.")
            preorder.remove((phi2, phi1))

    logger.debug(f"Constructed preorder: \n{pprint.pformat(preorder)}")

    # Construct atoms.
    for formula_ in phi:
        atoms.update(set(formula_.find_labels()))

    # Make preorder reflexive.
    for formula_ in phi:
        preorder.add((formula_, formula_))

    # Make preorder transitive.
    preorder = transitive_closure(preorder)
    logger.debug(f"Transitive closure of preorder: \n{pprint.pformat([(str(e1), str(e2)) for e1, e2 in preorder])}")

    # Return model
    return atoms, phi, preorder


def index_model(model):
    """
    Rewrites the preorder $$\trianglerighteq$$ by indexing the $$\Phi$$ set.

    :param model: (Tuple[set, set, set]) A tuple of (atoms, phi, preorder).
    :return: (Tuple[set, set, set]) A tuple of (atoms, phi, preorder).
    """
    atoms, phi, preorder = model
    phi = list(phi)
    phi_index = {f: i for i, f in enumerate(phi)}
    preorder_index = [(phi_index[f1], phi_index[f2]) for f1, f2 in preorder]
    return list(atoms), phi, preorder_index


def save_prefltlf_model(model, file):
    """
    Save the preference model to a file.
    :param model: (Tuple[set, set, set]) A tuple of (atoms, phi, preorder).
    :param file: (str or Path-like) The file to save the model. The file will be overwritten if it already exists. Format: JSON.
    :return: None
    """
    atoms, phi, preorder = model
    d = {
        "atoms": atoms,
        "phi": [str(f) for f in phi],
        "preorder": preorder
    }
    with open(file, 'w') as f:
        json.dump(d, f, indent=2)


def load_prefltlf_model(file):
    with open(file, 'r') as f:
        d = json.load(f)
        atoms = d["atoms"]
        phi = [PARSER(f) for f in d["phi"]]
        preorder = [tuple(e) for e in d["preorder"]]
    return atoms, phi, preorder


def transitive_closure(a):
    closure = set(a)
    while True:
        new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)

        closure_until_now = closure | new_relations

        if closure_until_now == closure:
            break

        closure = closure_until_now

    return closure


# =================================================================================== #
# TRANSLATE FORMULA TO PDFA
# =================================================================================== #

def translate(prefltlf_model, **kwargs):
    """
    Translates PrefLTLf formula to Preference Deterministic Finite Automaton (PDFA).

    :param prefltlf_model: (Tuple[list, list, list]) An indexed preference model: tuple of (atoms, phi, preorder).
    :return:
    """
    atoms, phi, preorder = prefltlf_model
    dfa_list = list()
    for f in phi:
        dfa = ltlf2dfa(f)
        dfa_list.append(dfa)
        logger.info(f"DFA({f}): \n{pprint.pformat(dfa)}")

    for i in range(len(dfa_list)):
        if kwargs.get("debug", False):
            dfa_to_png(dfa_list[i], f"dfa_{i}.png")

    product_dfa = union_product(*dfa_list)
    logger.info(f"Union product DFA: \n{pprint.pformat(product_dfa)}")

    pref_graph = construct_pref_graph(product_dfa, dfa_list, preorder)
    logger.info(f"Preference graph: \n{pprint.pformat(pref_graph)}")

    if kwargs.get("debug", False):
        pdfa = copy.deepcopy(product_dfa)
    else:
        pdfa = product_dfa

    del pdfa["final_states"]
    pdfa["pref_graph"] = pref_graph
    pdfa["alphabet"] = atoms

    # if kwargs.get("debug", False):
    #     return pdfa, dfa_list, product_dfa
    # else:
    return pdfa


def ltlf2dfa(ltlf_formula):
    # Use LTLf2DFA to convert LTLf formula to DFA.
    dot = ltlf_formula.to_dfa()

    # Convert dot to networkx MultiDiGraph.
    dot_graph = nx_agraph.from_agraph(pygraphviz.AGraph(dot))

    # Construct DFA dictionary using networkx MultiDiGraph.
    dfa = dict()
    dfa["states"] = set()
    dfa["transitions"] = dict()
    dfa["init_state"] = set()
    dfa["final_states"] = set()

    # Add states to DFA
    for u, d in dot_graph.nodes(data=True):
        if u == "init":
            continue
        dfa["states"].add(u)
        dfa["transitions"][u] = dict()
        if d.get('shape', None) == 'doublecircle':
            dfa["final_states"].add(u)

    for u, v, d in dot_graph.edges(data=True):
        if u == "init":
            dfa["init_state"] = v
            continue

        dfa["transitions"][u][d['label']] = v

    return dfa


def union_product(*args):
    assert len(args) > 1, "At least 2 DFAs are required for product construction."

    # Initial state
    q0 = tuple([dfa['init_state'] for dfa in args])

    # Define product dfa
    product = dict()
    product["states"] = set()
    product["transitions"] = dict()
    product["init_state"] = q0
    product["final_states"] = set()

    # Add states to product dfa (only reachable states are added)
    queue = [q0]
    explored = set()
    while queue:
        # Visit next state
        q = queue.pop()
        explored.add(q)

        # Add state to product dfa
        product["states"].add(q)
        product["transitions"][q] = dict()

        # Add transitions to product dfa
        for guard_label in itertools.product(*[args[i]['transitions'][q[i]].keys() for i in range(len(args))]):
            label = " & ".join(guard_label)
            label = spot.formula(label).simplify()

            # If label is false, then the synchronous transition is not valid.
            if label.is_ff():
                continue

            # Otherwise, add transition
            q_next = tuple([args[i]['transitions'][q[i]][guard_label[i]] for i in range(len(args))])
            if q_next not in explored:
                queue.append(q_next)

            product["transitions"][q][str(label)] = q_next

    # Index states and simplify representation
    enum_states = {q: i for i, q in enumerate(product["states"])}
    product["init_state"] = enum_states[product["init_state"]]
    product["final_states"] = set(
        i for q, i in enum_states.items()
        if any(q[i] in args[i]["final_states"] for i in range(len(args)))
    )
    transitions = dict()
    for q, d in product["transitions"].items():
        transitions[enum_states[q]] = {label: enum_states[q_next] for label, q_next in d.items()}
    product["transitions"] = transitions
    product["states"] = {i: {"state": q} for q, i in enum_states.items()}

    return product


def get_mp_outcomes(outcomes, preorder):
    return {f for f in outcomes if not any((t, f) in preorder for t in outcomes - {f})}  # no formula in (sat - f) is preferred to f


def mp_semantics(preorder, source, target):
    sat_source = {i for i in range(len(source)) if source[i] == 1}
    sat_target = {i for i in range(len(target)) if target[i] == 1}

    # Force empty set to be indifferent to each other. Required for preference graph to be preorder.
    if sat_source == sat_target == set():
        return True

    if sat_target == set():
        return False

    for alpha_to in sat_target:
        if not any((alpha_to, alpha_from) in preorder for alpha_from in sat_source):
            return False

    return True


def construct_pref_graph(product_dfa, dfa_list, preorder):
    # Initialize preference graph
    graph = dict()
    graph["nodes"] = dict()
    graph["edges"] = dict()

    # Create partition and add nodes
    # classes = dict()
    for u in product_dfa["final_states"]:
        state = product_dfa["states"][u]['state']
        outcomes = set(i if state[i] in dfa_list[i]["final_states"] else 0 for i in range(len(state)))
        mp_outcomes = get_mp_outcomes(outcomes, preorder)
        logger.debug(f"get_mp_outcomes({outcomes}, {preorder})={mp_outcomes}")
        cls = tuple(1 if i in mp_outcomes else 0 for i in range(len(state)))
        if cls in graph["nodes"]:
            graph["nodes"][str(cls)].add(u)
        else:
            graph["nodes"][str(cls)] = {u}
            graph["edges"][str(cls)] = set()

    graph["nodes"][str((0, 0, 0))] = set(product_dfa["states"]) - set(product_dfa["final_states"])
    graph["edges"][str((0, 0, 0))] = set()

    # Construct edges using mp_semantics.
    for source, target in itertools.product(graph["nodes"].keys(), graph["nodes"].keys()):
        source_tuple = ast.literal_eval(source)
        target_tuple = ast.literal_eval(target)
        if mp_semantics(preorder, source_tuple, target_tuple):
            graph["edges"][source].add(target)

    return graph


# =================================================================================== #
# PRINTING AND IMAGE GENERATION
# =================================================================================== #

def prettystring_prefltlf_model(model):
    """
    Pretty string for indexed model.

    :param model: (Tuple[list, list, list]) A tuple of (atoms, phi, preorder).
    :return:
    """
    atoms, phi, preorder = model
    pretty = ""
    pretty += "Atoms:\n"
    pretty += f"\t{atoms}\n"
    pretty += "Phi:" + "\n"
    for f in phi:
        pretty += f"\t{f}\n"
    pretty += "Preorder:" + "\n"
    for f1, f2 in preorder:
        pretty += f"\t{phi[f1]} >= {phi[f2]}\n"

    return pretty


def prettystring_pdfa(pdfa):
    pretty = ""
    pretty += "States:\n"
    pretty += pprint.pformat(pdfa["states"])
    pretty += "\n\n"
    pretty += "Alphabet:\n"
    pretty += pprint.pformat(pdfa["alphabet"])
    pretty += "\n\n"
    pretty += "Transitions:\n"
    pretty += pprint.pformat(pdfa["transitions"])
    pretty += "\n\n"
    pretty += "Initial state:\n"
    pretty += pprint.pformat(pdfa["init_state"])
    pretty += "\n\n"
    pretty += "Preference graph:\n"
    pretty += pprint.pformat(pdfa["pref_graph"])
    pretty += "\n\n"
    return pretty


def pdfa_to_png(pdfa, file):
    pref_graph = pdfa["pref_graph"]

    # Create graph for underlying product DFA
    dot_dfa = pygraphviz.AGraph(directed=True)
    for n, d in pdfa["states"].items():
        dot_dfa.add_node(n, **{"label": d['state']})
    dot_dfa.add_node("init", **{"label": "", "shape": "plaintext"})

    for u, d in pdfa["transitions"].items():
        for label, v in d.items():
            dot_dfa.add_edge(u, v, **{"label": label})
    dot_dfa.add_edge("init", pdfa["init_state"], **{"label": ""})

    dot_dfa.layout(prog="dot")

    # preference graph
    dot_pref = pygraphviz.AGraph(directed=True)
    for n in pref_graph["nodes"]:
        dot_pref.add_node(n, **{"label": f"{n}"})

    for u in pref_graph["edges"]:
        for v in pref_graph["edges"][u]:
            dot_pref.add_edge(u, v)

    dot_pref.layout(prog="dot")

    # Generate graphs
    file = pathlib.Path(file)
    parent = file.parent
    stem = file.stem
    suffix = file.suffix

    dot_dfa.draw(os.path.join(parent, f"{stem}_dfa{suffix}"))
    dot_pref.draw(os.path.join(parent, f"{stem}_pref_graph{suffix}"))


def dfa_to_png(dfa, file):
    # Create graph for underlying product DFA
    dot_dfa = pygraphviz.AGraph(directed=True)
    for n in dfa["states"]:
        dot_dfa.add_node(n, **{"label": str(n)})
    dot_dfa.add_node("init", **{"label": "", "shape": "plaintext"})

    for u, d in dfa["transitions"].items():
        for label, v in d.items():
            dot_dfa.add_edge(u, v, **{"label": label})
    dot_dfa.add_edge("init", dfa["init_state"], **{"label": ""})
    dot_dfa.layout(prog="dot")

    # Generate graphs
    dot_dfa.draw(file)


# =================================================================================== #
# TEST FUNCTIONS
# =================================================================================== #

def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    formula = parse_prefltlf(os.path.join(curr_dir, "example", "sample2.prefltlf"))
    model = build_prefltlf_model(formula)
    model = index_model(model)
    save_prefltlf_model(model, os.path.join(curr_dir, "example", "sample2.model"))
    print(model)
    model_ = load_prefltlf_model(os.path.join(curr_dir, "example", "sample2.model"))
    prettystring_prefltlf_model(model_)
    pdfa = translate(model_)
    pdfa_to_png(pdfa, os.path.join(curr_dir, "example", "sample2.png"))


if __name__ == '__main__':
    main()
