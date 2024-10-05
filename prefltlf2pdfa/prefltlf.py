import itertools
import lark.exceptions
import networkx as nx
import pprint
import subprocess
import sympy

from loguru import logger
from ltlf2dfa.parser.ltlf import LTLfParser
from tqdm import tqdm

import prefltlf2pdfa.utils as utils
from prefltlf2pdfa.semantics import *

# logger.remove()
PARSER = LTLfParser()

# Check ltlf2dfa is functioning properly and has access to mona.
command = "mona --help"

# Run the command using subprocess
try:
    result = subprocess.run(command, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if "Usage: mona [options] <filename>" not in result.stdout:
        logger.warning("Mona not found on your system. The tool may not translate LTLf to PDFA properly.")

        # Output and return code
        logger.debug("Return Code:", result.returncode)
        logger.debug("Output:", result.stdout)
        logger.debug("Error (if any):", result.stderr)

except Exception as e:
    print(f"An error occurred: {e}")


class PrefLTLf:
    """
    Represents a PrefLTLf formula.
    """
    MAXIMAL_SEMANTICS = [semantics_mp_forall_exists, semantics_mp_exists_forall, semantics_mp_forall_forall]

    def __init__(self, spec, alphabet=None, **kwargs):
        # Class state variables (will be serialized)
        self.raw_spec = spec
        self.atoms = set()  # Set of atomic propositions appearing in PrefLTLf specification
        self.alphabet = list(alphabet) if alphabet is not None else None
        self.phi = dict()  # Dict (indexed set) of LTLf formulas appearing in PrefLTLf specification
        self.relation = set()  # Set of triples (PREF_TYPE, LTLf, LTLf) constructed based on given PrefLTLf spec
        # self.dfa = list()  # List (indexed set) of LTLf formulas appearing in PrefLTLf specification

        if not kwargs.get("skip_parse", False):
            self.parse(
                auto_complete=kwargs.get("auto_complete", "none"),
            )

    def __eq__(self, other):
        keys = ["f_str", "atoms", "alphabet", "phi", "relation"]
        if isinstance(other, PrefLTLf):
            return ({k: v for k, v in self.serialize().items() if k in keys} ==
                    {k: v for k, v in other.serialize().items() if k in keys})
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.relation)))

    def __str__(self):
        return pprint.pformat(self.serialize())

    def __repr__(self):
        return f"<PrefLTLf Formula at {id(self)}>"

    def __getstate__(self):
        return self.serialize()

    def __setstate__(self, obj_dict):
        self.raw_spec = obj_dict["f_str"]
        self.atoms = set(obj_dict["atoms"])
        self.alphabet = set(obj_dict["alphabet"]) if obj_dict["alphabet"] is not None else None
        self.phi = obj_dict["phi"]
        # self.dfa = obj_dict["dfa"]
        self.relation = set(obj_dict["relation"])
        self.parse(auto_complete="none")

    def serialize(self):
        jsonable_dict = {
            "f_str": self.raw_spec,
            "atoms": list(self.atoms),
            "alphabet": [list(symbol) for symbol in self.alphabet] if self.alphabet is not None else None,
            "phi": {k: str(v) for k, v in self.phi.items()},
            "relation": list(self.relation),
            # "dfa": self.dfa
        }
        return jsonable_dict

    @classmethod
    def deserialize(cls, obj_dict):
        formula = cls(spec=obj_dict["f_str"], skip_parse=True)
        formula.atoms = set(obj_dict["atoms"])
        formula.alphabet = set(obj_dict["alphabet"]) if obj_dict["alphabet"] is not None else None
        formula.phi = {int(k): PARSER(v) for k, v in obj_dict["phi"].items()}
        formula.relation = set(obj_dict["relation"])
        # formula.dfa = obj_dict["dfa"]
        return formula

    @classmethod
    def from_file(cls, fpath, alphabet=None, auto_complete="none"):
        with open(fpath, 'r') as f:
            return cls(f.read(), alphabet=alphabet, auto_complete=auto_complete)

    def parse(self, auto_complete="none"):
        logger.debug(f"Parsing prefltlf formula with {auto_complete=}: \n{self.raw_spec}")

        # Check auto_complete inputs
        if auto_complete not in ["none", "incomparable", "minimal"]:
            logger.error(
                f"Unknown auto_complete value '{auto_complete}' to parse function. "
                f"Accepted values are {['none', 'incomparable', 'minimal']}. "
                f"New auto_complete value is set to 'none'."
            )
            auto_complete = "none"

        # # Parse header. Ensure that the spec is well-formed and a PrefLTLf formula.
        # header = self._parse_header()

        # Construct intermediate representation of standard spec
        phi, spec_ir = self._construct_spec_ir()
        if not self._is_lang_complete(phi):
            if auto_complete != "none":
                phi, spec_ir = self._auto_complete(phi, spec_ir, auto_complete)
            else:
                logger.error(f"The set of conditions phi = {set(phi.values())} is not complete")
                raise ValueError(f"The set of conditions phi = {set(phi.values())} is not complete")

        # Else, we have spec in standard format. Build model. (if user has specified ltlf formulas, add them to model)
        phi, model = self._build_partial_order(phi, spec_ir)
        # print(f"{phi=}")
        # print(f"{model=}")

        # Construct set of atomic propositions
        atoms = set()
        for varphi in phi.values():
            atoms.update(set(varphi.find_labels()))

        # Return
        self.phi = phi
        self.relation = model
        self.atoms = atoms

        return phi, model, atoms

    def translate(self, semantics=semantics_mp_forall_exists, show_progress=False):
        logger.debug(
            f"Translating the formula to preference automaton under semantics=`{semantics.__name__}`:\n{self.raw_spec}"
        )

        # Define preference automaton and set basic attributes
        aut = PrefAutomaton()
        aut.atoms = self.atoms
        aut.alphabet = self.alphabet
        aut.phi = self.phi

        # Translate LTLf formulas in self.phi to DFAs
        sorted_phi = sorted(self.phi.keys())
        aut.dfa = [utils.ltlf2dfa(self.phi[i]) for i in sorted_phi]
        assert len(aut.dfa) >= 2, f"PrefLTLf spec must have at least two LTLf formulas."

        # Log all DFAs
        log_message = f"Constructed Automata: {self.phi} \n"
        for i in range(len(sorted_phi)):
            log_message += f"\n======== {self.phi[sorted_phi[i]]} ========\n{pprint.pformat(aut.dfa[i])}\n"
        logger.debug(log_message)

        # Compute union product of DFAs
        self._construct_semi_automaton(aut, show_progress=show_progress)

        # Construct preference graph
        self._construct_preference_graph(aut, semantics, show_progress=show_progress)

        # Return preference automaton
        return aut

    # noinspection PyMethodMayBeStatic

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

    # noinspection PyMethodMayBeStatic

    def _parse_ltlf(self, raw_spec):
        atoms = set()
        phi = list()
        for formula in raw_spec:
            ltlf = PARSER(formula.strip())
            phi.append(ltlf)
            atoms.update(set(ltlf.find_labels()))
        return atoms, phi

    # noinspection PyMethodMayBeStatic

    def _parse_relation(self, raw_spec):
        relation = set()
        for formula in raw_spec:
            # Split line into operator, left formula, right formula.
            rel = formula.split(",")

            # Determine operator
            pref_type = rel[0].strip()
            assert pref_type in [">", ">=", "~",
                                 "<>"], f"The formula is ill-formed. Unrecognized operator `{pref_type}`"

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
            assert 0 <= phi1 <= len(
                self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_1={phi1}."
            assert 0 <= phi2 <= len(
                self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_2={phi2}."

            if pref_type == ">" or pref_type == ">=":
                preorder.add((phi1, phi2))

            if pref_type == "~":
                preorder.add((phi1, phi2))
                preorder.add((phi2, phi1))

                # Second, process the incomparability formulas because it removes elements to preorder relation.
        for pref_type, phi1, phi2 in (f for f in relation_spec if f[0] == "<>"):
            assert 0 <= phi1 <= len(
                self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_1={phi1}."
            assert 0 <= phi2 <= len(
                self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_2={phi2}."

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

    # noinspection PyMethodMayBeStatic

    def _transitive_closure(self, preorder):
        closure = set(preorder)
        while True:
            new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)
            closure_until_now = closure | new_relations
            if closure_until_now == closure:
                break
            closure = closure_until_now
        return closure

    def _construct_semi_automaton(self, aut, show_progress=False):
        dfa = aut.dfa

        # Initial state
        q0 = tuple([dfa['init_state'] for dfa in dfa])
        aut.init_state = aut.add_state(q0)

        # Visit reachable states
        queue = [q0]
        explored = set()
        transitions = set()
        with tqdm(total=len(queue), desc="Constructing semi-automaton", disable=not show_progress) as pbar:
            while queue:
                # Visit next state
                q = queue.pop()
                explored.add(q)

                # Add state to preference automaton
                aut.add_state(q)

                # Add transitions to product dfa
                # Pick one outgoing edge from each of the sub-DFA states in q.
                for condition_on_edges in itertools.product(
                        *[dfa[i]['transitions'][q[i]].keys() for i in range(len(dfa))]
                ):
                    # Define condition for synchronous transition of selected edges (AND-ing of all conditions)
                    cond = sympy.sympify(("(" + ") & (".join(condition_on_edges) + ")").replace("!", "~")).simplify()

                    # If alphabet is provided, require that at least one symbol enables the condition.
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

                # Update tqdm progress bar
                pbar.update(1)
                pbar.total = len(queue) + pbar.n
                pbar.refresh()

        for q, p, cond in transitions:
            aut.add_transition(q, p, cond)

    def _construct_preference_graph(self, aut, semantics, show_progress=False):
        # dfa = self.dfa
        pg = nx.MultiDiGraph()

        # Create partition and add nodes
        for qid, data in tqdm(
                aut.get_states(data=True),
                desc="Constructing preference graph nodes",
                disable=not show_progress
        ):
            q = data["name"]
            outcomes = utils.outcomes(aut.dfa, q)
            if semantics in PrefLTLf.MAXIMAL_SEMANTICS:
                outcomes = utils.maximal_outcomes(self.relation, outcomes)

            cls = utils.vectorize(aut.dfa, outcomes)
            # cls_id = aut.add_class(cls)

            # Add node to temporary graph
            if not pg.has_node(cls):
                pg.add_node(cls, partition={q})
            else:
                pg.nodes[cls]["partition"].add(q)

            # aut.add_state_to_class(cls_id, q)

        # Create edges
        # for source_id, target_id in itertools.product(aut.pref_graph.nodes(), aut.pref_graph.nodes()):
        for source, target in tqdm(
                itertools.product(pg.nodes(), pg.nodes()),
                total=pg.number_of_nodes() ** 2,
                desc="Constructing preference graph edges",
                disable=not show_progress
        ):
            # source = aut.get_class_name(source_id)
            # target = aut.get_class_name(target_id)
            if semantics(self.relation, source, target):
                # aut.add_pref_edge(source_id, target_id)
                pg.add_edge(source, target)

        # Merge partitions if two nodes are indifferent under constructed edge relation
        scc = nx.strongly_connected_components(pg)

        phi = [self.phi[ltlf_id] for ltlf_id in sorted(self.phi.keys())]
        state2node = dict()
        for component in scc:
            # Define a class name and class id
            cls_name = []
            for cls in component:
                cls_name.append(
                    " & ".join([str(phi[i]) for i in range(len(phi)) if cls[i] == 1])
                )
            cls_name = " | ".join((f"({x})" for x in cls_name))
            cls_name = str(PARSER(cls_name))
            cls_id = aut.add_class(cls_name)

            # Add states to partition
            for cls in component:
                for q in pg.nodes[cls]["partition"]:
                    aut.add_state_to_class(cls_id, q)
                    state2node[cls] = cls_id

        for u, v in pg.edges():
            aut.add_pref_edge(state2node[u], state2node[v])

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

        # Construct phi: the set of LTLf formulas.
        for i in range(n_phi):
            try:
                phi[i] = PARSER(stmts[i].strip())
            except Exception as err:
                if isinstance(err, lark.exceptions.UnexpectedCharacters):
                    # logger.debug(str(err))
                    raise ValueError(
                        f"Could not parse '{stmts[i].strip()}'. Possible reasons:\n"
                        f"1. Did you supply {n_phi} formulas as suggested by the specification header?\n"
                        f"2. LTLf formula is ill-formed"
                    )

        # Eliminate formulas with equivalent languages
        equiv = self._find_equivalent_ltlf(phi)
        compact_phi, replacement = self._process_equiv_ltlf(phi, equiv)

        # Collect atomic preferences
        spec_ir = set()
        for atomic_pref in stmts[n_phi:]:
            pref_type, l_index, r_index = atomic_pref.split(",")
            l_index = int(l_index.strip())
            r_index = int(r_index.strip())
            assert pref_type.strip() in [">", ">=", "~", "<>"], \
                f'{pref_type.strip()} is not a valid. Valid preference operators are {[">", ">=", "~", "<>"]}.'
            assert l_index in phi, f"Index of LTLf formula out of bounds. |Phi|={len(phi)}, l_index={l_index}."
            assert r_index in phi, f"Index of LTLf formula out of bounds. |Phi|={len(phi)}, r_index={l_index}."

            if l_index in replacement:
                l_index = replacement[l_index]
            if r_index in replacement:
                r_index = replacement[r_index]

            spec_ir.add((pref_type, l_index, r_index))

        # Log constructed representation
        log_string = [
            f"{l_index}:{compact_phi[l_index]}  {pref_type}  {r_index}:{compact_phi[r_index]}"
            for pref_type, l_index, r_index in spec_ir
        ]
        logger.debug(f"Intermediate representation based on raw input: \n{pprint.pformat(log_string)}")

        # Return intermediate representation
        return compact_phi, list(spec_ir)

    # noinspection PyMethodMayBeStatic
    def _auto_complete(self, phi, spec_ir, auto_complete):
        # Determine completion formula
        completion_formula = " | ".join([f"({varphi})" for varphi in phi.values()])
        completion_formula = f"!({completion_formula})"
        completion_formula = PARSER(completion_formula)

        # Add completion formula to phi.
        new_id = max(phi.keys()) + 1
        phi.update({new_id: completion_formula})

        # Update spec_ir based on input auto-complete method.
        #   In case auto-complete type is incomparable, no update to spec_ir is necessary.
        if auto_complete == "minimal":
            for varphi in phi.keys():
                spec_ir.append((">=", varphi, new_id))

        # Log constructed representation
        log_string = [
            f"{l_index}:{phi[l_index]}  {pref_type}  {r_index}:{phi[r_index]}"
            for pref_type, l_index, r_index in spec_ir
        ]
        logger.debug(f"Intermediate representation based on raw input: {log_string}")

        # Return intermediate representation
        return phi, spec_ir

    # noinspection PyMethodMayBeStatic
    def _is_lang_complete(self, phi):
        formula = " | ".join([f"({varphi})" for varphi in phi.values()])
        equiv_formula = PARSER(formula)
        dfa = utils.ltlf2dfa(equiv_formula)

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

        logger.debug(f"Clauses from intermediate representation:\n{set_w=} \n{set_p=} \n{set_i=} \n{set_j=}")

        # Resolve W into P, I, J
        for varphi1, varphi2 in set_w:
            if (varphi1, varphi2) in set_j:
                raise ValueError(
                    f"Inconsistent specification: "
                    f"{phi[varphi1]} >= {phi[varphi2]} and {phi[varphi1]} <> {phi[varphi2]}."
                )

            elif (varphi2, varphi1) in set_p:
                raise ValueError(
                    f"Inconsistent specification: "
                    f"{phi[varphi1]} >= {phi[varphi2]} and {phi[varphi2]} > {phi[varphi1]}."
                )

            elif (varphi1, varphi2) in set_p | set_i:
                pass

            elif (varphi2, varphi1) in set_w:
                set_i.add((varphi1, varphi2))

            else:
                set_p.add((varphi1, varphi2))

        # Transitive closure
        set_p = self._transitive_closure(set_p)
        set_i |= {(varphi, varphi) for varphi in phi}
        set_i = self._transitive_closure(set_i)
        logger.debug(f"Clauses after applying transitive and reflexive closures:\n{set_p=} \n{set_i=} \n{set_j=}")

        # Consistency check
        if any((varphi1, varphi1) in set_p for varphi1 in phi):
            graph_ = nx.DiGraph()
            graph_.add_nodes_from(phi.keys())
            graph_.add_edges_from([(j, i) for i, j in set_p])
            cycles = sorted(nx.simple_cycles(graph_))
            cycles = [list(map(phi.get, cycle)) for cycle in cycles if len(cycle) >= 3]
            raise ValueError(
                f"Inconsistent specification: Cyclic preferences detected in P. "
                f"Cycles after transitive closure: \n{pprint.pformat(cycles)}"
            )

        if any((varphi2, varphi1) in set_p for varphi1, varphi2 in set_p):
            raise ValueError(f"Inconsistent specification: Strictness violated. "
                             f"{[[(p1, p2), (p2, p1)] for p1, p2 in set_p if (p2, p1) in set_p]} in P.")

        if set.intersection(set_p, set_i):
            raise ValueError(f"Inconsistent specification: {set.intersection(set_p, set_i)} common in P and I.")

        if set.intersection(set_p, set_j):
            raise ValueError(f"Inconsistent specification: {set.intersection(set_p, set_j)} common in P and J.")

        if set.intersection(set_i, set_j):
            raise ValueError(f"Inconsistent specification: {set.intersection(set_i, set_j)} common in I and J.")

        # # Merge any specifications with same language.
        # equiv = self._find_equivalent_ltlf(phi)
        # phi, model = self._process_equiv_ltlf(phi, set.union(set_p, set_i), equiv)
        # logger.debug(f"Merge language equivalent formulas:\n{phi=} \n{model=}")

        # Apply reflexive closure
        model = set.union(set_p, set_i)
        model.update({(i, i) for i in phi.keys()})

        # Apply transitive closure
        # model = util.transitive_closure(model)
        # print(phi)
        # print(model)

        # If spec is a preorder, but not a partial order, then construct a partial order. [add option to skip this]
        phi, model = self._pre_to_partial_order(phi, model)

        log_model = {f"({phi[l_index]}, {phi[r_index]})" for l_index, r_index in model}
        logger.debug(f"Partial order for input specification:\n{phi=}\n{model=}\nmodel={log_model}")
        return phi, model

    # noinspection PyMethodMayBeStatic
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
                dfa = utils.ltlf2dfa(equiv_formula)

                # Check for universally true DFA
                if len(dfa["states"]) == 1 and len(dfa["final_states"]) == 1:
                    equiv_formulas.add((indices[i], indices[j]))

                # print(equiv_formula, dfa)

        return equiv_formulas

    # noinspection PyMethodMayBeStatic
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

    # noinspection PyMethodMayBeStatic
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
        self.phi = dict()
        self.dfa = list()

        # Helper attributes
        self._num_states = 0
        self._num_nodes = 0
        self._inv_state = dict()
        self._inv_nodes = dict()

    def __str__(self):
        return pprint.pformat(self.serialize())

    def serialize(self):
        # Construct state serialization
        state2node = {v: k for k, v in self.states.items()}
        state_dict = {k: {"name": v, "partition": None} for k, v in self.states.items()}
        for partition_label, data in self.pref_graph.nodes(data=True):
            for state in data["partition"]:
                sid = state2node[state]
                state_dict[sid]["partition"] = partition_label

        #  Collect edges of preference graph
        pref_nodes = set(self.pref_graph.nodes())
        pref_edges = {u: set() for u in pref_nodes}
        for u, v in self.pref_graph.edges():
            pref_edges[u].add(v)

        obj_dict = {
            "states": state_dict,
            "atoms": list(self.atoms),
            "alphabet": [list(symbol) for symbol in self.alphabet] if self.alphabet is not None else None,
            "transitions": self.transitions,
            "init_state": self.init_state,
            "pref_graph": {
                "nodes": {u: {"name": data["name"]}
                          for u, data in self.pref_graph.nodes(data=True)},
                # "edges": {u: v for u, v in self.pref_graph.edges()}
                "edges": {u: list(v) for u, v in pref_edges.items()}
            }
        }
        return obj_dict

    @classmethod
    def deserialize(cls, obj_dict):
        aut = cls()
        aut.states = obj_dict["states"]
        aut.atoms = set(obj_dict["atoms"])
        aut.alphabet = set(map(tuple, obj_dict["alphabet"]))
        aut.transitions = obj_dict["transitions"]
        aut.init_state = obj_dict["init_state"]

        for node, data in obj_dict["pref_graph"]["nodes"].items():
            aut.pref_graph.add_node(int(node), **data)

        for u, vs in obj_dict["pref_graph"]["edges"].items():
            for v in vs:
                aut.pref_graph.add_edge(int(u), int(v))

        aut._num_states = len(aut.states)
        aut._num_nodes = aut.pref_graph.number_of_nodes()
        aut._inv_state = {tuple(v): k for k, v in aut.states.items()}
        # aut._inv_nodes = {data["name"]: k for k, data in aut.pref_graph.nodes(data=True)}

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

    def get_states(self, data=False):
        if data:
            state2node = {v: k for k, v in self.states.items()}
            state_dict = {k: {"name": v, "partition": None} for k, v in self.states.items()}
            for partition_label, data in self.pref_graph.nodes(data=True):
                for state in data["partition"]:
                    sid = state2node[state]
                    state_dict[sid]["partition"] = partition_label

            return state_dict.items()

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
        assert self.pref_graph.has_node(
            source_id), f"Cannot add preference edge. {source_id=} not in preference graph. "
        assert self.pref_graph.has_node(
            target_id), f"Cannot add preference edge. {target_id=} not in preference graph. "
        self.pref_graph.add_edge(source_id, target_id)
