# prefltlf2pdfa

Tool to translate PrefLTLf formulas to preference deterministic finite automaton (PDFA). 

There are two tools provided in this repository:
1. `prefltlf2pdfa.py` is a command-line tool to translate a PrefLTLf formula to PDFA.
2. `trimpdfa.py` is a python script to trim a PDFA. It allows users to restrict the alphabet of PDFA. 


## PrefLTLf specification

``` 
prefltlf n
<LTLf formula>
<LTLf formula>
... 
<Type>, <int>, <int>
<Type>, <int>, <int>
...
```

PrefLTLf specification is defined in three parts:
1. *header:* The first line of the file includes the keyword `prefltlf` and specifies the number of LTLf formulas in the file.
2. *LTLf formulas:* The next `n` lines define `n` LTLf formulas. LTLf formulas are indexed from `0` to `n-1`. 
   The formula on `i`-th line of the file has index `i-1`.  
3. *Preference relation:* The remaining lines define preference relation. 
   Each preference relation is a triple of type, and indices of two LTLf formulas. 
   For example, `(>, i, j)` means that the LTLf formula with index `i` is strictly preferred to the LTLf formula with index `j`. 



Notes:
* `<Type>` is either `>, >=, ~` or `<>` representing strict, weak, indifference and incomparable formulas, respectively.
* `<LTLf formula>` is a valid LTLf formula parsable by `ltlf2dfa` tool.


> **_WARNING:_**  
> All LTLf formulas appearing in PrefLTLf formulas are assumed to be distinct. 
> The parser does not enforce this check.



## prefltlf2pdfa: Command line utility

prefltlf2pdfa: Translate a preference formula in prefltlf format to a PDFA.

```
python3 prefltlf2pdfa.py <formula_file> [options]
```

Example:
```
python3 prefltlf2pdfa.py 
    /path/to/spec.prefltlf
    --semantics=<forall_exists/exists_forall/forall_forall/mp_forall_exists/mp_exists_forall/mp_forall_forall>
    --verbosity=<DEBUG/INFO/SUCCESS/WARNING/ERROR>
    --loglevel=<DEBUG/INFO/SUCCESS/WARNING/ERROR>
    --logfile=path/to/save/run.log
    --png=path/to/save/out.png
    --ifiles=dir/to/save/intermediate/files
    --output=path/to/save/pdfa.json
```

**positional arguments:**
*  `formula_file `         LTLf formula file

**optional arguments:**
* `-h, --help`            show this help message and exit
* `-s SEMANTICS, --semantics SEMANTICS`
                        semantics of the PDFA. 
                        Choose from `forall_exists/exists_forall/forall_forall/mp_forall_exists/mp_exists_forall/mp_forall_forall`.
                        Default is `forall_exists`.
* `-o OUTPUT, --output OUTPUT`
                        saves the PDFA to given file. If not specified, the PDFA is printed to stdout. Output format
                        is a JSON file.
* `-p PNG, --png PNG`     saves the PDFA in PNG format. Two PNG files <FILE>_dfa.png and <FILE>_pref_graph.png are
                        generated.
* `--verbosity VERBOSITY`
                        sets logger level for logging to stdout. Use any loguru levels.
* `--loglevel LOGLEVEL`   sets logger level for logging to file. Use any loguru levels.
* `--logfile LOGFILE`     sets log file to store logs.
* `--ifiles IFILES`       Directory to store intermediate files.


## trimpdfa: Restrict alphabet of PDFA

`trimpdfa.py` is a python script to trim a PDFA by removing infeasible transition from PDFA.


In several situations, not all symbols from `2^{AP}` are possible.
For instance, consider a gridworld in which propositions correspond to the present location of the robot.
Since the robot cannot be in two locations at the same time, a symbol `{a, b}` is infeasible!

To use `trimpdfa.py`, first use prefltlf2dfa utility to generate PDFA. 
Then, update the section marked `TO BE MODIFIED BY USER` in `trimpdfa.py`. 
Finally, run `trimpdfa.py` to generate a trimmed PDFA.



