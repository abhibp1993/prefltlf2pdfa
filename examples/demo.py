from prefltlf2pdfa import *
from pprint import pprint
# from loguru import logger


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
    formula = PrefLTLf(spec)

    print("====================================")
    print("formula = ")
    pprint(formula.serialize())

    print()
    print("====================================")
    print("aut = ")
    aut = formula.translate(semantics=semantics_mp_forall_exists)
    pprint(aut.serialize())

    sa, pg = paut2dot(aut, show_sa_state=True, show_class=True, show_color=True, show_pg_state=True)
    paut2png(sa, pg, fname="aut")
    paut2svg(sa, pg, fname="aut")
