"""
Render preference automaton using graphviz.
"""

import base64
import pygraphviz
import os
import seaborn as sns
from loguru import logger
from prefltlf2pdfa.prefltlf import PrefAutomaton


def _color_palette(n):
    if n < 10:
        colors = sns.color_palette("pastel", n_colors=n)
    elif n < 20:
        colors = sns.color_palette("viridis", n_colors=n)
    else:
        colors = [(1, 1, 1) for _ in range(n)]

    return colors


def _create_dot_semi_automaton(paut, node2color, **kwargs):
    # Extract options
    sa_state = kwargs.get("show_sa_state", False)
    sa_class = kwargs.get("show_class", False)
    sa_color = kwargs.get("show_color", False)

    if sa_color:
        assert node2color is not None, "Coloring requested but no color map provided (it is None)."

    # Create graph to display semi-automaton
    dot_semi_aut = pygraphviz.AGraph(directed=True)

    # Add nodes to semi-automaton
    for sid, data in paut.get_states(data=True):
        # Determine state name
        st_label = data["name"] if sa_state else sid

        # Append state class if option enabled
        st_label = f"{st_label}\n[{data['partition']}]" if sa_class else st_label

        # Add node
        if sa_color:
            color = node2color[data['partition']]
            color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            dot_semi_aut.add_node(sid, **{"label": st_label, "fillcolor": color, "style": "filled"})
        else:
            dot_semi_aut.add_node(sid, **{"label": st_label})

    # Add initial state to semi-automaton
    dot_semi_aut.add_node("init", **{"label": "", "shape": "plaintext"})

    # Add edges to semi-automaton
    for u, d in paut.transitions.items():
        for label, v in d.items():
            dot_semi_aut.add_edge(u, v, **{"label": label})
    dot_semi_aut.add_edge("init", paut.init_state, **{"label": ""})

    # Return semi-automaton
    return dot_semi_aut


def _create_dot_pref_graph(paut, node2color, **kwargs):
    # Extract options
    sa_color = kwargs.get("show_color", False)
    pg_state = kwargs.get("show_pg_state", False)

    if sa_color:
        assert node2color is not None, "Coloring requested but no color map provided (it is None)."

    # Preference graph
    dot_pref = pygraphviz.AGraph(directed=True)

    # Add nodes to preference graph
    for n, data in paut.pref_graph.nodes(data=True):
        # n_label = set(phi[i] for i in range(len(phi)) if data['name'][i] == 1) if pg_state else n
        n_label = data['name'] if pg_state else n
        if sa_color:
            color = node2color[n]
            color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        else:
            color = "white"

        dot_pref.add_node(n, **{"label": n_label, "fillcolor": color, "style": "filled"})

    # Add edges to preference graph
    for u, v in paut.pref_graph.edges():
        dot_pref.add_edge(u, v)

    return dot_pref


def paut2dot(paut: PrefAutomaton, **kwargs):
    """
    Generates images for semi-automaton and preference graph.

    :param paut: PrefAutomaton
    :param kwargs: dict of options
    :return: tuple[base64, base64] Two images as base64 encoded strings
    """

    # Extract options
    sa_state = kwargs.get("show_sa_state", False)
    sa_class = kwargs.get("show_class", False)
    sa_color = kwargs.get("show_color", False)
    pg_state = kwargs.get("show_pg_state", False)
    logger.debug(f"Options for paut2dot: {sa_state=}, {sa_class=}, {sa_color=}, {pg_state=}")

    # If partition coloring is requested, then construct a color palette
    node2color = None
    if sa_color:
        colors = _color_palette(n=len(paut.pref_graph.nodes()))
        node2color = {node: colors[i] for i, node in enumerate(paut.pref_graph.nodes())}

    # Construct DOT representation for semi-automaton
    dot_semi_aut = _create_dot_semi_automaton(paut=paut, node2color=node2color, **kwargs)
    dot_pref_graph = _create_dot_pref_graph(paut=paut, node2color=node2color, **kwargs)

    # Set drawing engine
    dot_semi_aut.layout(prog=kwargs.get("engine", "dot"))
    dot_pref_graph.layout(prog=kwargs.get("engine", "dot"))

    return dot_semi_aut, dot_pref_graph


def paut2png(dot_semi_aut, dot_pref_graph, fpath="", fname="out"):
    if ".png" in fname:
        fname = fname[:-4]

    # Generate images (as bytes)
    dot_semi_aut.draw(path=os.path.join(fpath, f"{fname}_sa.png"), format="png")
    dot_pref_graph.draw(path=os.path.join(fpath, f"{fname}_pg.png"), format="png")


def paut2svg(dot_semi_aut, dot_pref_graph, fpath="", fname="out"):
    if ".svg" in fname:
        fname = fname[:-4]

    # Generate images (as bytes)
    dot_semi_aut.draw(path=os.path.join(fpath, f"{fname}_sa.svg"), format="svg")
    dot_pref_graph.draw(path=os.path.join(fpath, f"{fname}_pg.svg"), format="svg")


def paut2base64(dot_semi_aut, dot_pref_graph):
    sa = dot_semi_aut.draw(path=None, format="png")
    pg = dot_pref_graph.draw(path=None, format="png")
    return base64.b64encode(sa), base64.b64encode(pg)
