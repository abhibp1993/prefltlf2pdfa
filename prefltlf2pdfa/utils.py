from networkx.drawing import nx_agraph
from itertools import chain, combinations
import pygraphviz


def ltlf2dfa(ltlf_formula):
    # Use LTLf2DFA to convert LTLf formula to DFA.
    dot = ltlf_formula.to_dfa()
    # logger.info(f"{ltlf_formula=}, dot={dot}")

    # Convert dot to networkx MultiDiGraph.
    dot_graph = nx_agraph.from_agraph(pygraphviz.AGraph(dot))
    # logger.info(f"{ltlf_formula=}, dot={dot_graph}")

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

        u = int(float(u))
        dfa["states"].add(u)
        dfa["transitions"][u] = dict()
        if d.get('shape', None) == 'doublecircle':
            dfa["final_states"].add(u)

    for u, v, d in dot_graph.edges(data=True):
        if u == "init":
            dfa["init_state"] = int(v)
            continue

        u = int(float(u))
        v = int(float(v))

        dfa["transitions"][u][d['label']] = v

    # logger.info(f"ltlf_formula={ltlf_formula}, dfa={dfa}")
    return dfa


def outcomes(phi, dfa, q):
    return set(phi[i] for i in range(len(q)) if q[i] in dfa[i]["final_states"])


def maximal_outcomes(relation, outcomes):
    # No formula in (sat - f) is preferred to f
    return {f for f in outcomes if not any((t, f) in relation for t in outcomes - {f})}


def vectorize(phi, dfa, outcomes):
    return tuple(1 if phi[i] in outcomes else 0 for i in range(len(dfa)))


def powerset(iterable):
    s = list(iterable)
    assert len(s) < 20, "Too many elements to compute powerset. Limit set to 20."
    return list(map(set, chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))))


if __name__ == '__main__':
    print(powerset([1, 2, 3]))
