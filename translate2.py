"""
Translator from PrefLTLf to PDFA.
"""
import itertools
import networkx as nx
import pathlib
import pprint
import pygraphviz
import os
import sympy

from ltlf2dfa.parser.ltlf import LTLfParser
from loguru import logger
from networkx.drawing import nx_agraph

PARSER = LTLfParser()


# =================================================================================== #
# SEMANTICS
# =================================================================================== #

def semantics_forall_exists(preorder, source, target):
    sat_source = {i for i in range(len(source)) if source[i] == 1}
    sat_target = {i for i in range(len(target)) if target[i] == 1}

    # Force empty set to be indifferent to each other. Required for preference graph to be preorder.
    if sat_source == sat_target == set():
        return True

    if sat_target == set():
        return False

    for alpha_to in sat_target:
        if len(sat_source) != 0 and not any((alpha_to, alpha_from) in preorder for alpha_from in sat_source):
            # if len(sat_source) != 0 and not any((alpha_to, alpha_from) in preorder for alpha_from in sat_source - {alpha_to}):
            return False

    return True


def semantics_exists_forall(preorder, source, target):
    sat_source = {i for i in range(len(source)) if source[i] == 1}
    sat_target = {i for i in range(len(target)) if target[i] == 1}

    # Force empty set to be indifferent to each other. Required for preference graph to be preorder.
    if sat_source == sat_target == set():
        return True

    if sat_target == set():
        return False

    for alpha_from in sat_source:
        if not any((alpha_to, alpha_from) in preorder for alpha_to in sat_target):
            return False

    return True


def semantics_forall_forall(preorder, source, target):
    sat_source = {i for i in range(len(source)) if source[i] == 1}
    sat_target = {i for i in range(len(target)) if target[i] == 1}

    # Force empty set to be indifferent to each other. Required for preference graph to be preorder.
    if sat_source == sat_target:
        return True

    if sat_target == set():
        return False

    for alpha_to in sat_target:
        if not all((alpha_to, alpha_from) in preorder for alpha_from in sat_source):
            return False

    return True


def semantics_mp_forall_exists(preorder, source, target):
    return semantics_forall_exists(preorder, source, target)


def semantics_mp_exists_forall(preorder, source, target):
    return semantics_exists_forall(preorder, source, target)


def semantics_mp_forall_forall(preorder, source, target):
    return semantics_forall_forall(preorder, source, target)


# =================================================================================== #
# PARSE FORMULA, BUILD PREFERENCE MODEL
# =================================================================================== #

class PrefLTLf:
    MAXIMAL_SEMANTICS = [semantics_mp_forall_exists, semantics_mp_exists_forall, semantics_mp_forall_forall]

    def __init__(self, f_str, alphabet=None, **kwargs):
        self.f_str = f_str
        self.atoms = set()  # Set of atomic propositions appearing in PrefLTLf specification
        self.alphabet = list(alphabet) if alphabet is not None else None
        self.phi = list()  # List (indexed set) of LTLf formulas appearing in PrefLTLf specification
        self.dfa = list()  # List (indexed set) of LTLf formulas appearing in PrefLTLf specification
        self.relation = set()  # Set of triples (PREF_TYPE, LTLf Formula, LTLf Formula) constructed based on given PrefLTLf spec

        if not kwargs.get("skip_parse", False):
            self.parse()

    def __str__(self):
        return pprint.pformat(self.serialize())

    def __repr__(self):
        return f"<PrefLTLf Formula at {id(self)}>"

    def serialize(self):
        jsonable_dict = {
            "f_str": self.f_str,
            "atoms": list(self.atoms),
            "alphabet": list(self.alphabet) if self.alphabet is not None else None,
            "phi": self.phi,
            "relation": list(self.relation)
        }
        return jsonable_dict

    @classmethod
    def deserialize(cls, obj_dict):
        formula = cls(f_str=obj_dict["f_str"], skip_parse=True)
        formula.atoms = set(obj_dict["atoms"])
        formula.alphabet = set(obj_dict["alphabet"]) if obj_dict["alphabet"] is not None else None
        formula.phi = obj_dict["phi"]
        formula.relation = set(obj_dict["relation"])
        return formula

    @classmethod
    def from_file(cls, fpath, alphabet=None):
        with open(fpath, 'r') as f:
            return cls(f.read(), alphabet=alphabet)

    def parse(self):
        # Separate formula string into lines
        raw_spec = self.f_str.split("\n")
        if len(raw_spec) == 0:
            raise EOFError("Empty PrefLTLf file.")

        # Parse header.
        num_ltlf = self._parse_header(raw_spec)

        # Parse LTLf formulas
        self.atoms, self.phi = self._parse_ltlf(raw_spec[1:num_ltlf + 1])

        # Parse preference relation
        relation_spec = self._parse_relation(raw_spec[num_ltlf + 1:])

        # Build preorder
        self.relation = self._build_preorder(relation_spec)

    def translate(self, semantics="mp_forall_exists"):
        # Define preference automaton and set basic attributes
        aut = PrefAutomaton()
        aut.atoms = self.atoms
        aut.alphabet = self.alphabet

        # Translate LTLf formulas in self.phi to DFAs
        self.dfa = [self._ltlf2dfa(ltlf) for ltlf in self.phi]
        assert len(self.dfa) >= 2, f"PrefLTLf spec must have at least two LTLf formulas."

        # Compute union product of DFAs
        self._construct_underlying_graph(aut)

        # Construct preference graph
        self._construct_preference_graph(aut, semantics)

        # Return preference automaton
        return aut

    def _parse_header(self, raw_spec):
        header = raw_spec[0].split(" ")

        if len(header) != 2:
            raise TypeError(f"PrefLTLf specification must have header of format: `prefltlf <int>`. Received `{header}`.")

        formula_type = header[0].strip().lower()
        if formula_type != "prefltlf":
            raise ValueError(f"Not a PrefLTLf formula. Likely a '{formula_type}' formula.")

        return int(header[1].strip())

    def _parse_ltlf(self, raw_spec):
        atoms = set()
        phi = list()
        for formula in raw_spec:
            ltlf = PARSER(formula.strip())
            phi.append(ltlf)
            atoms.update(set(ltlf.find_labels()))
        return atoms, phi

    def _parse_relation(self, raw_spec):
        relation = set()
        for formula in raw_spec:
            # Split line into operator, left formula, right formula.
            rel = formula.split(",")

            # Determine operator
            pref_type = rel[0].strip()
            assert pref_type in [">", ">=", "~", "<>"], f"The formula is ill-formed. Unrecognized operator `{pref_type}`"

            # Determine formulas
            left = int(rel[1].strip())
            right = int(rel[2].strip())

            # Add relation
            relation.add((pref_type, left, right))

        return relation

    def _build_preorder(self, relation_spec):
        preorder = set()

        # First, process the non-incomparability formulas because they add elements to preorder relation.
        for pref_type, phi1, phi2 in (f for f in relation_spec if f[0] != "<>"):
            assert 0 <= phi1 <= len(self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_1={phi1}."
            assert 0 <= phi2 <= len(self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_2={phi2}."

            if pref_type == ">" or pref_type == ">=":
                preorder.add((phi1, phi2))

            if pref_type == "~":
                preorder.add((phi1, phi2))
                preorder.add((phi2, phi1))

        # Second, process the incomparability formulas because it removes elements to preorder relation.
        for pref_type, phi1, phi2 in (f for f in relation_spec if f[0] == "<>"):
            assert 0 <= phi1 <= len(self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_1={phi1}."
            assert 0 <= phi2 <= len(self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_2={phi2}."

            if (phi1, phi2) in preorder:
                logger.warning(f"{(phi1, phi2)} is removed from preorder.")
                preorder.remove((phi1, phi2))

            if (phi2, phi1) in preorder:
                logger.warning(f"{(phi2, phi1)} is removed from preorder.")
                preorder.remove((phi2, phi1))

        # Reflexive closure
        for i in range(len(self.phi)):
            preorder.add((i, i))

        # Transitive closure
        preorder = self._transitive_closure(preorder)

        # Return preorder
        return preorder

    def _transitive_closure(self, preorder):
        closure = set(preorder)
        while True:
            new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)
            closure_until_now = closure | new_relations
            if closure_until_now == closure:
                break
            closure = closure_until_now
        return closure

    def _construct_underlying_graph(self, aut):
        dfa = self.dfa

        # Initial state
        q0 = tuple([dfa['init_state'] for dfa in dfa])
        aut.init_state = aut.add_state(q0)

        # Visit reachable states
        queue = [q0]
        explored = set()
        transitions = set()
        while queue:
            # Visit next state
            q = queue.pop()
            explored.add(q)

            # Add state to preference automaton
            aut.add_state(q)

            # Add transitions to product dfa
            # Pick one outgoing edge from each of the sub-DFA states in q.
            for condition_on_edges in itertools.product(*[dfa[i]['transitions'][q[i]].keys() for i in range(len(dfa))]):
                # Define condition for synchronous transition of selected edges (AND-ing of all conditions)
                cond = sympy.sympify(("(" + ") & (".join(condition_on_edges) + ")").replace("!", "~")).simplify()

                # If alphabet is provided, require that at least one symbol enables the condition.
                # if self.alphabet is None or all(not evaluate(spot.formula(formula), true_atoms) for true_atoms in alphabet):
                if (
                        self.alphabet is not None and
                        all(
                            not cond.subs({atom: True if atom in true_atoms else False for atom in self.atoms})
                            for true_atoms in self.alphabet
                        )
                ):
                    continue

                # If label is false, then the synchronous transition is not valid.
                if cond == sympy.false:
                    continue

                # Otherwise, add transition
                cond = (str(cond).replace("~", "!").
                        replace("True", "true").
                        replace("False", "false"))

                q_next = tuple([dfa[i]['transitions'][q[i]][condition_on_edges[i]] for i in range(len(dfa))])
                if q_next not in explored:
                    queue.append(q_next)

                # Add transition to preference automaton
                transitions.add((q, q_next, cond))

        for q, p, cond in transitions:
            aut.add_transition(q, p, cond)

    def _construct_preference_graph(self, aut, semantics):
        dfa = self.dfa

        # Create partition and add nodes
        for qid, q in aut.get_states(name=True):
            outcomes = self.outcomes(q)
            if semantics in PrefLTLf.MAXIMAL_SEMANTICS:
                outcomes = self.maximal_outcomes(outcomes)

            cls = self._vectorize(outcomes)
            cls_id = aut.add_class(cls)
            aut.add_state_to_class(cls_id, q)

        # Create edges
        for source_id, target_id in itertools.product(aut.pref_graph.nodes(), aut.pref_graph.nodes()):
            source = aut.get_class_name(source_id)
            target = aut.get_class_name(target_id)
            if semantics(self.relation, source, target):
                aut.add_pref_edge(source_id, target_id)

    def outcomes(self, q):
        return set(i for i in range(len(q)) if q[i] in self.dfa[i]["final_states"])

    def _vectorize(self, outcomes):
        return tuple(1 if i in outcomes else 0 for i in range(len(self.dfa)))

    def maximal_outcomes(self, outcomes):
        # No formula in (sat - f) is preferred to f
        return {f for f in outcomes if not any((t, f) in self.relation for t in outcomes - {f})}

    def _ltlf2dfa(self, ltlf_formula):
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

            u = int(u)
            dfa["states"].add(u)
            dfa["transitions"][u] = dict()
            if d.get('shape', None) == 'doublecircle':
                dfa["final_states"].add(u)

        for u, v, d in dot_graph.edges(data=True):
            if u == "init":
                dfa["init_state"] = int(v)
                continue

            u = int(u)
            v = int(v)

            dfa["transitions"][u][d['label']] = v

        return dfa


class PrefAutomaton:
    def __init__(self):
        # Automaton structure
        self.states = dict()
        self.atoms = set()
        self.alphabet = set()
        self.transitions = dict()
        self.init_state = None
        self.pref_graph = nx.MultiDiGraph()

        # Helper attributes
        self._num_states = 0
        self._num_nodes = 0
        self._inv_state = dict()
        self._inv_nodes = dict()

    def __str__(self):
        return pprint.pformat(self.serialize())

    def serialize(self):
        obj_dict = {
            "states": self.states,
            "atoms": list(self.atoms),
            "alphabet": list(self.alphabet) if self.alphabet is not None else None,
            "transitions": self.transitions,
            "init_state": self.init_state,
            "pref_graph": {
                "nodes": {u: data for u, data in self.pref_graph.nodes(data=True)},
                "edges": {u: v for u, v in self.pref_graph.edges()}
            }
        }
        return obj_dict

    @classmethod
    def deserialize(cls, obj_dict):
        aut = cls()
        aut.states = obj_dict["states"]
        aut.atoms = set(obj_dict["atoms"])
        aut.alphabet = set(obj_dict["alphabet"])
        aut.transitions = obj_dict["transitions"]
        aut.init_state = obj_dict["init_state"]

        for node, data in obj_dict["pref_graph"]["nodes"]:
            aut.pref_graph.add_node(node, **data)

        for u, v in obj_dict["pref_graph"]["edges"]:
            aut.pref_graph.add_edge(u, v)

        aut._num_states = len(aut.states)
        aut._num_nodes = len(aut.pref_graph.number_of_nodes())
        aut._inv_state = {v: k for k, v in aut.states}
        aut._inv_nodes = {data["name"]: k for k, data in aut.pref_graph.nodes(data=True)}

        return aut

    def add_state(self, name):
        if name not in self._inv_state:
            self.states[self._num_states] = name
            self._inv_state[name] = self._num_states
            self.transitions[self._num_states] = dict()
            self._num_states += 1

    def get_states(self, name=False):
        if name:
            return list(self.states.items())
        return list(self.states.keys())

    def add_transition(self, u, v, cond):
        uid = self._inv_state[u]
        vid = self._inv_state[v]
        self.transitions[uid].update({cond: vid})

    def get_state_id(self, name):
        return self._inv_state[name]

    def add_class(self, cls_name):
        cls_id = self._inv_nodes.get(cls_name, None)
        if cls_id is None:
            cls_id = self._num_nodes
            self.pref_graph.add_node(cls_id, name=cls_name, partition=set())
            self._inv_nodes[cls_name] = cls_id
            self._num_nodes += 1
        return cls_id

    def add_state_to_class(self, cls_id, q):
        self.pref_graph.nodes[cls_id]["partition"].add(q)

    def get_class_name(self, cls_id):
        return self.pref_graph.nodes[cls_id]["name"]

    def add_pref_edge(self, source_id, target_id):
        assert self.pref_graph.has_node(source_id), f"Cannot add preference edge. {source_id=} not in preference graph. "
        assert self.pref_graph.has_node(target_id), f"Cannot add preference edge. {target_id=} not in preference graph. "
        self.pref_graph.add_edge(source_id, target_id)


# =================================================================================== #
# PRINTING AND IMAGE GENERATION
# =================================================================================== #
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
        if n in dfa["final_states"]:
            dot_dfa.add_node(n, **{"label": str(n), "shape": "doublecircle"})
        else:
            dot_dfa.add_node(n, **{"label": str(n)})
    dot_dfa.add_node("init", **{"label": "", "shape": "plaintext"})

    for u, d in dfa["transitions"].items():
        for label, v in d.items():
            dot_dfa.add_edge(u, v, **{"label": label})
    dot_dfa.add_edge("init", dfa["init_state"], **{"label": ""})
    dot_dfa.layout(prog="dot")

    # Generate graphs
    dot_dfa.draw(file)


if __name__ == '__main__':
    spec = PrefLTLf.from_file(os.path.join("examples", "icra2023", "icra2023.prefltlf"), alphabet=[
        set(),
        {'t'},
        {'d'},
        {'o'}
    ])
    pdfa = spec.translate(semantics=semantics_forall_exists)
    print(spec)
    print(pdfa)