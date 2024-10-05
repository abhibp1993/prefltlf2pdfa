# prefltlf2pdfa

`prefltlf2pdfa` is a tool for expressing PrefLTLf formulas and converting them to preference automata.
PrefLTLf formulas enable expressing preferences over LTLf formulas.

This module utilizes the `ltlf2dfa` library to parse and process LTLf formulas
into preference deterministic finite automata (PDFA). 
It also provides functions for validating and manipulating preference automata.

The online tool is available at http://akulkarni.me/prefltlf2pdfa.html

The documentation is available at http://akulkarni.me/docs/prefltlf2pdfa


## Docker 

A docker image with `prefltlf2pdfa` preinstalled is available at https://hub.docker.com/repository/docker/abhibp1993/prefltlf2pdfa/general.

Pull image using `docker pull abhibp1993/prefltlf2pdfa`.


## Installation

`prefltlf2pdfa` is only tested on ubuntu 22.04.  


### Python Prerequisites 
Install all dependencies listed `requirements.txt`.


### MONA Prerequisite
`LTLf2DFA` relies on the MONA tool for the generation of the DFA. 
Please, make sure you have the MONA tool installed on your system before running LTLf2DFA. 
You can follow the instructions on http://www.brics.dk/mona/download.html to get MONA.


### Installing prefltlf2pdfa 

Install using `python setup.py install`. 