"""

usage: prefltlf2pdfa.py [-h] [-o OUTPUT] [-p PNG] [--verbosity VERBOSITY] [--loglevel LOGLEVEL] [--logfile LOGFILE]
                        [--ifiles IFILES]
                        formula_file

prefltlf2pdfa: Translate a preference formula in prefltlf format to a PDFA.

positional arguments:
  formula_file          LTLf formula file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        saves the PDFA to given file. If not specified, the PDFA is printed to stdout. Output format
                        is a JSON file.
  -p PNG, --png PNG     saves the PDFA in PNG format. Two PNG files <FILE>_dfa.png and <FILE>_pref_graph.png are
                        generated.
  --verbosity VERBOSITY
                        sets logger level for logging to stdout. Use any loguru levels.
  --loglevel LOGLEVEL   sets logger level for logging to file. Use any loguru levels.
  --logfile LOGFILE     sets log file to store logs.
  --ifiles IFILES       Directory to store intermediate files.
"""

import argparse
import sys
import pprint
import time

import ioutils
from translate import *


def parse_args():
    parser = argparse.ArgumentParser(description="prefltlf2pdfa: Translate a preference formula in prefltlf format to a PDFA.")

    parser.add_argument("formula_file", help="PrefLTLf formula file")
    parser.add_argument("-o", "--output", default=None,
                        help="saves the PDFA to given file. Output format is a JSON file.")
    parser.add_argument("-p", "--png",
                        help="saves the PDFA in PNG format. Two PNG files <FILE>_dfa.png and <FILE>_pref_graph.png are generated.")

    parser.add_argument("--verbosity", default="None", help="sets logger level for logging to stdout. Use any loguru levels.")
    parser.add_argument("--loglevel", default="INFO", help="sets logger level for logging to file. Use any loguru levels.")
    parser.add_argument("--logfile", default=None, help="sets log file to store logs.")
    parser.add_argument("--ifiles", default=None, help="Directory to store intermediate files.")

    # Parse arguments
    args_ = parser.parse_args()
    print(args_)

    # Set up logger
    logger.remove()
    debug = args_.verbosity == "DEBUG"
    if args_.verbosity in ["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]:
        logger.add(sys.stdout, level=args_.verbosity)

    if args_.logfile is not None and args_.loglevel in ["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]:
        logger.add(args_.logfile, level=args_.loglevel)

    # Return other arguments
    return args_.formula_file, args_.output, args_.png, args_.ifiles, debug


def main(args):
    formula_file, output_file, png_file, ifiles, debug = args

    # Load formula
    formula = parse_prefltlf(formula_file)

    logger.info(f"Formula: \n{pprint.pformat({(str(a), str(b), str(c)) for (a, b, c) in formula})}")
    if ifiles:
        with open(os.path.join(ifiles, "formula.txt"), 'w') as f:
            f.write("prefltlf\n")
            for triple in formula:
                f.write(",".join((str(e) for e in triple)) + "\n")

    # Start timer
    start_time = time.time()

    # Build preference model
    model = build_prefltlf_model(formula)
    model = index_model(model)

    logger.info(f"Model: \n{prettystring_prefltlf_model(model)}")
    if ifiles:
        with open(os.path.join(ifiles, "model.txt"), 'w') as f:
            atoms, phi, preorder = model
            f.write("prefltlf model\n")
            f.write("atoms: " + ",".join(atoms) + "\n")
            f.write("phi: ")
            for idx in range(len(phi)):
                if idx > 0:
                    f.write(",")
                f.write(f"{idx}:{phi[idx]}")
            f.write("\n")
            for element in preorder:
                f.write(",".join((str(e) for e in element)) + "\n")

    # Translate to PDFA
    pdfa = translate(model, **{"debug": debug})

    # Stop timer
    end_time = time.time()

    # Save PDFA to intermediate files
    if ifiles:
        ioutils.to_json(os.path.join(ifiles, "pdfa.json"), pdfa)

    # Save files as per flags
    if output_file is not None:
        ioutils.to_json(os.path.join(ifiles, "pdfa.json"), pdfa)

    if png_file is not None:
        pdfa_to_png(pdfa, png_file)

    # Print to stdout
    logger.info(f"====== TRANSLATION COMPLETED IN {round((end_time - start_time) * 10 ** 3, 4)} MILLISECONDS =====")
    logger.info(prettystring_pdfa(pdfa))

    print(f"====== TRANSLATION COMPLETED IN {round((end_time - start_time) * 10 ** 3, 4)} MILLISECONDS =====")
    print(prettystring_pdfa(pdfa))


if __name__ == '__main__':
    args = parse_args()
    main(args)
