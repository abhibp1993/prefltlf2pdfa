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


# Install sphinx documentation packages
# RUN pip3 install sphinx myst-parser sphinx_rtd_theme recommonmark nbsphinx


# Graphviz
RUN apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    graphviz graphviz-dev && \
    apt-get clean


# Install spot
# Reference: https://gitlab.lrde.epita.fr/spot/spot-web/-/blob/master/docker/Dockerfile
RUN echo 'deb [trusted=true] http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list && \
    apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    spot libspot-dev spot-doc python3-spot && \
    apt-get clean

# Standard Python Packages
RUN pip3 install numpy matplotlib pyzmq pydot networkx pygraphviz pytest
RUN pip3 install scipy pygame pyyaml lark
RUN pip3 install sortedcontainers
# RUN pip3 install jupyterlab
RUN pip3 install loguru tqdm
RUN pip3 install ltlf2dfa
RUN pip3 install simplejson
