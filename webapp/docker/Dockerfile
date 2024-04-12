FROM debian:bullseye

# The version is the same as ggsolver version. It is updated automatically by rebuild.py
LABEL maintainer="Abhishek N. Kulkarni" \
        email="abhi.bp1993@gmail.com"

# Install python
RUN apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    build-essential cmake \
    python3-pip python3-setuptools python3-dev pkg-config  && \
    apt-get clean

# Graphviz
RUN apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    graphviz graphviz-dev && \
    apt-get clean

# Webserver related
RUN apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    gunicorn nginx && \
    apt-get clean

# Install python packages
RUN pip3 install  \
    numpy  \
    scipy \
    matplotlib  \
    pydot  \
    networkx  \
    pygraphviz  \
    pytest \
    loguru \
    tqdm \
    simplejson \
    ltlf2dfa \
    sympy

RUN pip3 install  \
    dash

