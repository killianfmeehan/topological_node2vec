# topological_node2vec
This is a modification of the [Node2vec](https://snap.stanford.edu/node2vec/) machine learning framework which incorporates topological information.

As this code involves the computation of Persistence Diagrams, the computation can be a bit heavy without specialized packages often utilizing GPUs. Even so, on small examples this code can be run with CPU only. Currently, GPUs are utilized using [a modification of the Ripser++ package](https://github.com/killianfmeehan/ripser-plusplus-tn2v), while CPUs are utilized using [Gudhi](https://gudhi.inria.fr/).

Package requirements:
- numpy
- pandas
- matplotlib
- os
- scipy
- time
- sklearn
- one of either our Ripser++ fork or Gudhi

Tested on Mac and Linux.

## usage

There are quite a lot of hyperparameters for this network, most of which have proven important in at least some of our experiments. This is an attempt at a brief summary for users. Alternatively (or in concert), please see the included tn2v_examples.ipynb notebook which repeats the examples shown in the paper.

# hyperparameters

- l,r,p,q: these are the Node2vec neighborhood generation parameter
