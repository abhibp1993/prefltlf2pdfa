# prefltlf2pdfa

Tool to translate PrefLTLf formulas to preference deterministic finite automaton (PDFA). 

## Usage

### PrefLTLf specification

Currently, we do not have a working parser to parse `PrefLTLf` formulas such as `F(a) > G(b) && G(b) >= F(b)`. 
Instead, define a file with the following format:

``` 
prefltlf 
<Type>, <LTLf formula>, <LTLf formula>
<Type>, <LTLf formula>, <LTLf formula>
...
```

where 
* `<Type>` is either `>, >=, ~` or `<>` representing strict, weak, indifference and incomparable formulas, respectively.
* `<Type>` is either `>, >=, ~` or `<>` representing strict, weak, indifference and incomparable formulas, respectively.
* `<LTLf formula>` is a valid LTLf formula parsable by `ltlf2dfa` tool.


> **_WARNING:_**  
> All LTLf formulas appearing in PrefLTLf formulas are assumed to be distinct. 
> The parser does not enforce this check.


### Command-line tool 

prefltlf2pdfa: Translate a preference formula in prefltlf format to a PDFA.

```
python3 prefltlf2pdfa.py <formula_file> [options]
```

Example:
```
python3 prefltlf2pdfa.py example/sample2.prefltlf 
    --verbosity=DEBUG 
    --loglevel=DEBUG 
    --logfile=example/sample2.log 
    --png=example/sample2.png 
    --ifiles=example/ 
    --output=example/sample2.pdfa
```



**positional arguments:**
*  `formula_file `         LTLf formula file

**optional arguments:**
* `-h, --help`            show this help message and exit
* `-o OUTPUT, --output OUTPUT`
                        saves the PDFA to given file. If not specified, the PDFA is printed to stdout. Output format
                        is a JSON file.
*  `-p PNG, --png PNG`     saves the PDFA in PNG format. Two PNG files <FILE>_dfa.png and <FILE>_pref_graph.png are
                        generated.
*  `--verbosity VERBOSITY`
                        sets logger level for logging to stdout. Use any loguru levels.
*  `--loglevel LOGLEVEL`   sets logger level for logging to file. Use any loguru levels.
*  `--logfile LOGFILE`     sets log file to store logs.
*  `--ifiles IFILES`       Directory to store intermediate files.
