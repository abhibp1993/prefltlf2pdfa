import itertools
import networkx as nx
import pprint
import pygraphviz
import sympy

from ltlf2dfa.parser.ltlf import LTLfParser
from loguru import logger
from networkx.drawing import nx_agraph
from semantics import *

PARSER = LTLfParser()


class PrefLTLf:
    MAXIMAL_SEMANTICS = [semantics_mp_forall_exists, semantics_mp_exists_forall, semantics_mp_forall_forall]

    def __init__(self, spec, alphabet=None, **kwargs):
        self.raw_spec = spec
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
            "f_str": self.raw_spec,
            "atoms": list(self.atoms),
            "alphabet": [list(symbol) for symbol in self.alphabet] if self.alphabet is not None else None,
            "phi": [str(f) for f in self.phi],
            "relation": list(self.relation)
        }
        logger.debug(pprint.pformat(jsonable_dict))
        return jsonable_dict

    @classmethod
    def deserialize(cls, obj_dict):
        formula = cls(spec=obj_dict["f_str"], skip_parse=True)
        formula.atoms = set(obj_dict["atoms"])
        formula.alphabet = set(obj_dict["alphabet"]) if obj_dict["alphabet"] is not None else None
        formula.phi = obj_dict["phi"]
        formula.relation = set(obj_dict["relation"])
        return formula

    @classmethod
    def from_file(cls, fpath, alphabet=None):
        with open(fpath, 'r') as f:
            return cls(f.read(), alphabet=alphabet)

    def parse2(self):
        # Separate formula string into lines
        raw_spec = self.raw_spec.split("\n")
        if len(raw_spec) == 0:
            raise EOFError("Empty PrefLTLf file.")

            # Parse header.
        num_ltlf = self._parse_header(raw_spec)
        logger.info(f"num_ltlf={num_ltlf}")

        # Parse LTLf formulas
        self.atoms, self.phi = self._parse_ltlf(raw_spec[1:num_ltlf + 1])
        logger.info(f"atoms={self.atoms}")
        logger.info(f"phi={self.phi}")

        # Parse preference relation
        relation_spec = self._parse_relation(raw_spec[num_ltlf + 1:])
        logger.info(f"relation_spec={relation_spec}")

        # Build preorder
        self.relation = self._build_preorder(relation_spec)
        logger.info(f"relation={self.relation}")

    def parse(self):
        # Parse header. Ensure that the spec is well-formed and a PrefLTLf formula.
        header = self._parse_header()
        logger.debug(f"{header=}")

        # Construct intermediate representation of standard spec
        phi, spec_ir = self._construct_spec_ir()
        if not self._is_lang_complete(phi):
            raise ValueError(f"The set of conditions phi = {set(phi.values())} is not complete")
        print(f"{phi=}")
        print(f"{spec_ir=}")

        # Else, we have spec in standard format. Build model. (if user has specified ltlf formulas, add them to model)
        phi, model = self._build_partial_order(phi, spec_ir)
        print(f"{phi=}")
        print(f"{model=}")

        # Construct set of atomic propositions
        atoms = set()
        for varphi in phi.values():
            atoms.update(set(varphi.find_labels()))

        # Return
        self.phi = phi
        self.relation = model
        self.atoms = atoms

        return phi, model, atoms

    def translate(self, semantics="mp_forall_exists"):
        # Define preference automaton and set basic attributes
        aut = PrefAutomaton()
        aut.atoms = self.atoms
        aut.alphabet = self.alphabet

        # Translate LTLf formulas in self.phi to DFAs
        self.dfa = [self._ltlf2dfa(ltlf) for _, ltlf in self.phi.items()]
        assert len(self.dfa) >= 2, f"PrefLTLf spec must have at least two LTLf formulas."
        for dfa in self.dfa:
            logger.info(f"dfa={dfa}")

            # Compute union product of DFAs
        self._construct_underlying_graph(aut)

        # Construct preference graph
        self._construct_preference_graph(aut, semantics)

        # Return preference automaton
        return aut

    def _parse_header(self):
        stmts = (line.strip() for line in self.raw_spec.split("\n"))
        stmts = [line for line in stmts if line and not line.startswith("#")]
        header = stmts[0]
        header = header.split(" ")

        if len(header) != 2:
            raise ValueError(f"Ill-formatted specification file. Header should be `prefltlf <num_formula>`, "
                             f"where specifying <num_formula> is optional.")

        if header[0].strip().lower() != "prefltlf":
            raise ValueError(f"Not a PrefLTLf formula. Likely a '{header[0].strip().lower()}' formula.")

        header[1] = int(header[1].strip())
        return header

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
                        self.alphabet and
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
        logger.info(f"{ltlf_formula=}, dot={dot}")

        # Convert dot to networkx MultiDiGraph.
        dot_graph = nx_agraph.from_agraph(pygraphviz.AGraph(dot))
        logger.info(f"{ltlf_formula=}, dot={dot_graph}")

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

        logger.info(f"ltlf_formula={ltlf_formula}, dfa={dfa}")
        return dfa

    def _construct_spec_ir(self):
        """
        Construct intermediate representation of PrefLTLf specification.

        Steps:
            1. Extract LTLf formulas from the specification.
            2. Eliminate duplicate LTLf formulas (based on language equivalence).

        :return: (phi, spec_ir)
        """
        # Initialize outputs
        phi = dict()

        # Extract non-header lines from spec
        stmts = (line.strip() for line in self.raw_spec.split("\n"))
        stmts = [line for line in stmts if line and not line.startswith("#")]
        stmts = stmts[1:]

        # Extract header
        header = self._parse_header()
        n_phi = header[1]

        # Construct phi: the set of conditions, i.e., LTLf formulas.
        for i in range(n_phi):
            phi[i] = PARSER(stmts[i].strip())

        # Eliminate formulas with equivalent languages
        equiv = self._find_equivalent_ltlf(phi)
        compact_phi, replacement = self._process_equiv_ltlf(phi, equiv)

        # Collect atomic preferences
        spec_ir = set()
        for atomic_pref in stmts[n_phi:]:
            pref_type, l_index, r_index = atomic_pref.split(",")
            l_index = int(l_index.strip())
            r_index = int(r_index.strip())
            assert l_index in phi, f"Index of LTLf formula out of bounds. |Phi|={len(phi)}, l_index={l_index}."
            assert r_index in phi, f"Index of LTLf formula out of bounds. |Phi|={len(phi)}, r_index={l_index}."

            if l_index in replacement:
                l_index = replacement[l_index]
            if r_index in replacement:
                r_index = replacement[r_index]

            spec_ir.add((pref_type, l_index, r_index))

        return compact_phi, list(spec_ir)

    def _is_lang_complete(self, phi):
        formula = " | ".join([f"({varphi})" for varphi in phi.values()])
        equiv_formula = PARSER(formula)
        dfa = self._ltlf2dfa(equiv_formula)

        # Check for universally true DFA
        if len(dfa["states"]) == 1 and len(dfa["final_states"]) == 1:
            return True
        return False

    def _build_partial_order(self, phi, pref_stmts):
        """
        Constructs a partial order on phi using preference statements.

        :param phi: List of LTLf formulas
        :param pref_stmts: List of preference statements of form (op, l_index, r_index).
        :return:
        """
        logger.debug("Building partial order.")

        # Construct P, P', I, J sets
        set_w = set()
        set_p = set()
        set_i = set()
        set_j = set()
        for pref_type, varphi1, varphi2 in pref_stmts:
            if pref_type == ">":
                set_p.add((varphi1, varphi2))

            elif pref_type == ">=":
                set_w.add((varphi1, varphi2))

            elif pref_type == "~":
                set_i.add((varphi1, varphi2))
                set_i.add((varphi2, varphi1))

            elif pref_type == "<>":
                set_j.add((varphi1, varphi2))
                set_j.add((varphi2, varphi1))

        logger.debug(f"Input clauses from raw-spec:\n{set_w=} \n{set_p=} \n{set_i=} \n{set_j=}")

        # Resolve W into P, I, J
        for varphi1, varphi2 in set_w:
            if (varphi1, varphi2) in set_j:
                raise ValueError(f"Inconsistent specification: {varphi1} >= {varphi2} and {varphi1} <> {varphi2}.")

            elif (varphi2, varphi1) in set_p:
                raise ValueError(f"Inconsistent specification: {varphi1} >= {varphi2} and {varphi2} > {varphi1}.")

            elif (varphi1, varphi2) in set_p | set_i:
                pass

            elif (varphi2, varphi1) in set_w:
                set_i.add((varphi1, varphi2))

            else:
                set_p.add((varphi1, varphi2))

        logger.debug(f"Resolving W into PIJ model:\n{set_p=} \n{set_i=} \n{set_j=}")

        # Transitive closure
        set_p = self._transitive_closure(set_p)
        set_i = self._transitive_closure(set_i)
        logger.debug(f"Transitive Closure on P, I:\n{set_p=} \n{set_i=} \n{set_j=}")

        # Consistency check
        if any((varphi1, varphi1) in set_p for varphi1 in phi):
            raise ValueError(f"Inconsistent specification: Reflexivity violated. "
                             f"{[(varphi1, varphi1) in set_p for varphi1 in phi]} in P.")

        if any((varphi2, varphi1) in set_p for varphi1, varphi2 in set_p):
            raise ValueError(f"Inconsistent specification: Strictness violated. "
                             f"{[[(p1, p2), (p2, p1)] for p1, p2 in set_p if (p2, p1) in set_p]} in P.")

        if set.intersection(set_p, set_i):
            raise ValueError(f"Inconsistent specification: {set.intersection(set_p, set_i)} common in P and I.")

        if set.intersection(set_p, set_j):
            raise ValueError(f"Inconsistent specification: {set.intersection(set_p, set_j)} common in P and J.")

        if set.intersection(set_i, set_j):
            raise ValueError(f"Inconsistent specification: {set.intersection(set_i, set_j)} common in I and J.")

        logger.debug(f"Model:\n{phi=} \nmodel={set.union(set_p, set_i)}")

        # # Merge any specifications with same language.
        # equiv = self._find_equivalent_ltlf(phi)
        # phi, model = self._process_equiv_ltlf(phi, set.union(set_p, set_i), equiv)
        # logger.debug(f"Merge language equivalent formulas:\n{phi=} \n{model=}")

        # Apply reflexive closure
        model = set.union(set_p, set_i)
        model.update({(i, i) for i in phi.keys()})
        logger.debug(f"Reflexive closure:\n{model=}")

        # Apply transitive closure
        # model = util.transitive_closure(model)
        # print(phi)
        # print(model)

        # If spec is a preorder, but not a partial order, then construct a partial order. [add option to skip this]
        phi, model = self._pre_to_partial_order(phi, model)
        logger.debug(f"Preorder to Partial Order:\n{phi=}\n{model=}")

        return phi, model

    def _find_equivalent_ltlf(self, phi):
        """
        Returns a set of tuples containing equivalent LTLf formulas.

        :param phi: List of LTLf formulas
        :return: A set of tuples containing equivalent LTLf formulas.

        .. note:: O(n^2) algorithm.
        """
        # Initialize output
        equiv_formulas = set()

        #  Compare every formula with every other formula to identify equivalent LTLf formulas.
        indices = list(phi.keys())
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                # Approach: Construct equivalence LTLf formula.
                #   Check if DFA is universally true, i.e., has a single state which is accepting.

                # Equivalence formula
                equiv_formula = PARSER(f"{phi[indices[i]]} <-> {phi[indices[j]]}")
                dfa = self._ltlf2dfa(equiv_formula)

                # Check for universally true DFA
                if len(dfa["states"]) == 1 and len(dfa["final_states"]) == 1:
                    equiv_formulas.add((indices[i], indices[j]))

                print(equiv_formula, dfa)

        return equiv_formulas

    def _process_equiv_ltlf2(self, phi, model, equiv):
        """
        Approach: Construct an undirected graph with equivalent LTLf formulas as nodes.
            Two nodes have an edge between them <=> they are equivalent.
            SCC's in this graph partitions the LTLf formulas.
            Pick one formula to represent each SCC. Replace all others.

        :param phi:
        :param model:
        :param equiv:
        :return:
        """
        # Initialize output
        n_phi = list()
        n_spec_ir = set()

        # Construct equivalence graph
        equiv_graph = nx.DiGraph()

        for u, v in equiv:
            equiv_graph.add_edge(u, v)
            equiv_graph.add_edge(v, u)

        # equiv_graph.add_edges_from(equiv)
        scc = list(nx.strongly_connected_components(equiv_graph))
        print(scc)

        # Replace any formula index in atomic spec with minimum value from the strongly connected component.
        for l_idx, r_idx in model:
            for component in scc:
                if l_idx in component:
                    l_idx = min(component)
                if r_idx in component:
                    r_idx = min(component)
            n_spec_ir.add((l_idx, r_idx))

        # Construct new phi
        idx_to_remove = set()
        for component in scc:
            idx_to_remove.update(component - {min(component)})

        n_phi = {k: v for k, v in phi.items() if k not in idx_to_remove}

        print(n_phi, n_spec_ir)
        return n_phi, n_spec_ir

    def _process_equiv_ltlf(self, phi, equiv):
        """
        Approach: Construct an undirected graph with equivalent LTLf formulas as nodes.
            Two nodes have an edge between them <=> they are equivalent.
            SCC's in this graph partitions the LTLf formulas.
            Pick one formula to represent each SCC. Replace all others.

        :param phi: Dictionary mapping integer id to LTLf formulas
        :param equiv: Set of tuples containing equivalent LTLf formulas.
        :return: Tuple containing new phi and replacement dictionary.
        """
        # Construct equivalence graph
        equiv_graph = nx.DiGraph()

        for u, v in equiv:
            equiv_graph.add_edge(u, v)
            equiv_graph.add_edge(v, u)

        # equiv_graph.add_edges_from(equiv)
        scc = list(nx.strongly_connected_components(equiv_graph))

        # Construct new phi
        idx_to_remove = set()
        replacement = dict()
        for component in scc:
            keep_f = min(component)
            for f in component - {keep_f}:
                replacement[f] = keep_f
            idx_to_remove.update(component - {min(component)})

        n_phi = {k: v for k, v in phi.items() if k not in idx_to_remove}

        return n_phi, replacement

    def _pre_to_partial_order(self, phi, preorder):
        # Initialize output
        partial_order = set()

        # Construct preorder graph (might contain cycles)
        order_graph = nx.DiGraph()
        order_graph.add_edges_from(preorder)

        # Find strongly connected components
        scc = nx.strongly_connected_components(order_graph)

        # Replace indifferent formulas with their disjunction
        max_index = max(phi.keys())
        replace = dict()

        for component in scc:
            if len(component) >= 2:
                # Construct disjunction of indifferent formulas in the `component`
                formulas = (f"({phi[i]})" for i in component)
                replacement_ltlf = PARSER(" | ".join(formulas))

                # Add new formula to phi
                phi[max_index + 1] = replacement_ltlf

                # Remove indifferent formulas from phi
                max_index = max_index + 1
                for i in component:
                    phi.pop(i)
                    replace[i] = max_index

        # Construct partial order from preorder by replacing indifferent formulas
        for i, j in preorder:
            if i in replace:
                i = replace[i]
            if j in replace:
                j = replace[j]
            partial_order.add((i, j))

        # Return
        return phi, partial_order


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
            "alphabet": [list(symbol) for symbol in self.alphabet] if self.alphabet is not None else None,
            "transitions": self.transitions,
            "init_state": self.init_state,
            "pref_graph": {
                "nodes": {u: {k: list(v) if isinstance(v, set) else v for k, v in data.items()}
                          for u, data in self.pref_graph.nodes(data=True)},
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
        uid = self._inv_state.get(name, None)
        if uid is None:
            uid = self._num_states
            self.states[uid] = name
            self._inv_state[name] = uid
            self.transitions[uid] = dict()
            self._num_states += 1
        return uid

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


if __name__ == '__main__':
    spec = ("""
    # test
    prefltlf 4
    
    
    F a
    G b
    !(F(a) | G(b))
    true U a
    
    # SPec
    >, 0, 1
    >, 0, 2
    >=, 1, 2
    """)

    formula = PrefLTLf(spec)

