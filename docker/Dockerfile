# Use the official Ubuntu 22.04 image as the base
FROM ubuntu:22.04

# The version is the same as ggsolver version. It is updated automatically by rebuild.py
LABEL maintainer="Abhishek N. Kulkarni" \
        email="abhi.bp1993@gmail.com" \
		version="1.0"


# Set environment variables to avoid prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    wget \
    python3-pip \
    python3-setuptools \
    python3-dev \
    pkg-config \
    cmake \
    flex \
    bison \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Graphviz
RUN apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    graphviz graphviz-dev && \
    apt-get clean

# Mona
RUN wget http://www.brics.dk/mona/download/mona-1.4-18.tar.gz
RUN tar xvzf mona-1.4-18.tar.gz
RUN cd mona-1.4 && ./configure && make install

# Update LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# Standard Python Packages
RUN pip3 install numpy matplotlib sympy scipy networkx
RUN pip3 install pydot pygraphviz
RUN pip3 install pytest pygame pyyaml lark
RUN pip3 install jupyterlab
RUN pip3 install tqdm pqdm
RUN pip3 install loguru seaborn
RUN pip3 install dd
RUN pip3 install  ltlf2dfa

# Install prefltlf2pdfa
RUN apt update
RUN git clone https://github.com/abhibp1993/prefltlf2pdfa.git
RUN cd prefltlf2pdfa && python3 setup.py install