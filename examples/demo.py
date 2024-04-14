import prefltlf
import semantics
import sys

from pprint import pprint
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

# Define specification
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

if __name__ == '__main__':
    # alphabet = set()
    alphabet = [set(), {"a"}, {"b"}]
    formula = prefltlf.PrefLTLf(spec, alphabet=alphabet)
    pdfa = formula.translate(semantics.semantics_forall_exists)
    pprint(pdfa.serialize())
