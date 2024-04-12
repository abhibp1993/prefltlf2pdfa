import os
import pathlib
import pygraphviz

from translate2 import PrefAutomaton

def dfa2png(dfa, fpath, **kwargs):
    # Create graph for underlying product DFA
    dot_dfa = pygraphviz.AGraph(directed=True)
    for n in dfa["states"]:
        if n in dfa["final_states"]:
            dot_dfa.add_node(n, **{"label": str(n), "shape": "doublecircle"})
        else:
            dot_dfa.add_node(n, **{"label": str(n)})
    dot_dfa.add_node("init", **{"label": "", "shape": "plaintext"})

    for u, d in dfa["transitions"].items():
        for label, v in d.items():
            dot_dfa.add_edge(u, v, **{"label": label})
    dot_dfa.add_edge("init", dfa["init_state"], **{"label": ""})

    # Set drawing engine
    dot_dfa.layout(prog=kwargs.get("engine", "dot"))

    # Generate graphs
    dot_dfa.draw(fpath)


def spec2png(prefltlf, fpath, **kwargs):
    # Create graph for underlying preference model
    dot_model = pygraphviz.AGraph(directed=True)
    for i in range(len(prefltlf.phi)):
        if kwargs.get("show_formula", True):
            dot_model.add_node(i, **{"label": str(prefltlf.phi[i])})
        else:
            dot_model.add_node(i, **{"label": str(i)})

    for u, v in prefltlf.relation:
        dot_model.add_edge(u, v)

    # Set drawing engine
    dot_model.layout(prog=kwargs.get("engine", "dot"))

    # Generate graphs
    dot_model.draw(fpath)


def pdfa2png(pdfa: PrefAutomaton, fpath, **kwargs):
    pref_graph = pdfa.pref_graph

    # Create graph for underlying product DFA
    dot_dfa = pygraphviz.AGraph(directed=True)
    for st, name in pdfa.get_states(name=True):
        if kwargs.get("show_state_name", True):
            dot_dfa.add_node(st, **{"label": name})
        else:
            dot_dfa.add_node(st, **{"label": st})

    dot_dfa.add_node("init", **{"label": "", "shape": "plaintext"})

    for u, d in pdfa.transitions.items():
        for label, v in d.items():
            dot_dfa.add_edge(u, v, **{"label": label})
    dot_dfa.add_edge("init", pdfa.init_state, **{"label": ""})

    # Set drawing engine
    dot_dfa.layout(prog=kwargs.get("engine", "dot"))

    # Preference graph
    dot_pref = pygraphviz.AGraph(directed=True)
    for n, data in pref_graph.nodes(data=True):
        if kwargs.get("show_node_class", True):
            dot_pref.add_node(n, **{"label": data['name']})
        else:
            dot_pref.add_node(n, **{"label": n})

    for u, v in pref_graph.edges():
        dot_pref.add_edge(u, v)

    dot_pref.layout(prog=kwargs.get("engine", "dot"))

    # Generate graphs
    file = pathlib.Path(fpath)
    parent = file.parent
    stem = file.stem
    suffix = file.suffix

    dot_dfa.draw(os.path.join(parent, f"{stem}_dfa{suffix}"))
    dot_pref.draw(os.path.join(parent, f"{stem}_pref_graph{suffix}"))
