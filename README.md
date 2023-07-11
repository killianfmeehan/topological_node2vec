# topological_node2vec
This is a modification of the Node2vec [https://snap.stanford.edu/node2vec/] machine learning framework which incorporates topological information.

As this code involves the computation of Persistence Diagrams, the computation can be a bit heavy without specialized packages often utilizing GPUs. Even so, on small examples this code can be run with CPU only. Currently, GPUs are utilized using a modification of the Ripser++ package, while CPUs are utilized using Gudhi.

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
