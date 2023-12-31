# topological_node2vec
This is a modification of the [Node2vec](https://snap.stanford.edu/node2vec/) machine learning framework which incorporates topological information, as detailed in this [paper](https://arxiv.org/abs/2309.08241).

As this code involves the computation of Persistence Diagrams, the computation can be a bit heavy without specialized packages often utilizing GPUs. Even so, on small examples this code can be run with CPU only. Currently, GPUs are utilized using [a modification of the Ripser++ package](https://github.com/jnclark/ripser-plusplus/tree/return-indices), while CPUs are utilized using [Gudhi](https://gudhi.inria.fr/).

Package requirements:
- numpy
- pandas
- matplotlib
- os
- scipy
- time
- sklearn
- one of either our Ripser++ fork or Gudhi
- [homogeneousUROT](https://github.com/tlacombe/homogeneousUROT)
- jupyter (not strictly necessary, but this code was designed with notebook operation in mind)

Tested on Mac and Linux.

### installation

With apologies for the lack of instruction, getting a CUDA setup with GPUs successfully accessible is something I've only pulled off a handful of times on my precise system. If you can get such a setting with the Ripser++ fork running or you simply want to test this on CPU, then all that's left is to:
- install the other requisite packages above
- put utils.py from the homogeousUROT package in the same folder as this repository's tn2v.py
- see the notebook tn2v_examples.ipynb for how to utilize the main function of TN2V, and modify the data input and hyperparameters from these examples to suit your needs

*In a future update, I would like to adjust the code to receive an arbitrary persistent homology computing function from the outside, allowing users to specify the use of any package/function installed on their system.

### usage

Due to the large amount of hyperparameter adjustment that is necessary for operating this code and the diverse data-types of the input arguments, I have designed this code to be used most naturally from a notebook.

NOTE: Please be aware that this code is functional but very in progress. If you are interested in using it and you have any questions or encounter bugs / crashes, please contact me.

### hyperparameters

There are quite a lot of hyperparameters for this code, most of which have proven important in at least some of our experiments. This is an attempt at a brief summary for users. Alternatively (or in concert), please see the included tn2v_examples.ipynb notebook which repeats the examples shown in our paper.

---

**LEN**: the number of epochs / the length of the learning process.

---

**eta_array**: the array of step sizes; all gradient updates are multiplied by the appropriate eta before being applied.
```python
eta_array = np.linspace(0.005,0.001,LEN+1)
```
Higher values in the beginning moves points quickly / creates the necessary features, but low values in the end allow for sharpening of the embedding's finer details.

---

**lambda***: the scalar multipliers on the node2vec loss function vs. the topological loss functions
```python
lambda0 = 1
lambda1 = 256
lambda2 = 768

L0_array = [lambda0 for i in range(LEN+1)]
L1_array = [lambda1 for i in range(LEN+1)]
L2_array = [lambda2 for i in range(LEN+1)]
```
L0 corresponds to the node2vec loss function; L1 corresponds to the topological loss of dimension 1 (perimeters and disks); L2 corresponds to the topological loss of dimension 2 (surfaces and spheres). Any of these which are set to 0 (or not set at all) for a given epoch will not be measured / applied.

---

**main_directory**: all output will be saved in subfolders in this folder
```python
main_directory = home+'/tn2v_output/'
```

---

**data_\*, reciprocal_gamma, reciprocal_nu**: you must specify at least one of [data_pointcloud, data_distance_matrix, data_correlation_matrix]. If not provided, data_distance_matrix or data_correlation_matrix will be automatically generated from the other given data. Specifically, the conversion between distance/correlation matrices is given by the values reciprocal_gamma and reciprocal_nu (see the paper, beginning of Section 4).
```python
data_pointcloud = pd.DataFrame
data_distance_matrix = None
data_correlation_matrix = None

reciprocal_gamma = 0.001
reciprocal_nu = 1.0
```

---

**l,r,p,q**: these are the Node2vec neighborhood generation parameters (see Section 2 in [the paper](https://arxiv.org/abs/2309.08241))
```python
r = 8
l = 1
L_array = [l for i in range(LEN+1)]
R_array = [r for i in range(LEN+1)]
P_array = [0 for i in range(LEN+1)]
Q_array = [1 for i in range(LEN+1)]
```
In the special case that r*l > the size of the input data, neighborhoods ARE NOT SAMPLED. Instead, the direct distance matrix column vectors are inverted, normalized, and used as probability vectors. (See the paper, Remark 1.)

---

**nbhd_regen**: this value instructs the network on how often it should resample the random neighborhoods around each point used as training data. A value of n means that the neighborhoods will be regenerated every n-th epoch. This value is ignored if the neighborhoods are not randomly generated, as above.
```python
nbhd_regen = 1 # any positive integer, or None
```

---

**mbs**: integer values declaring how many points are to be minibatched at each epoch. Currently, minibatching is ONLY DONE for the topological loss function, while the node2vec loss function always operates on the full dataset in each epoch. I plan to update this code with the ability to input dual arrays of minibatches, but in my testing minibatches on the node2vec loss function dramatically reduced embedding quality.
```python
mbs_array = [int(data.shape[0]*1.00) for i in range(LEN+1)]
```

---

**lift**: this is an important hyperparameter that we added after analysis of initial poor performance of the topological loss function on simple embeddings. This is a float which scales off of the total diameter of the target persistence diagram (the PD of the input), meaning that there should be no need for a lift value > 1.0. Simply, this causes the topological loss function to act on translations (lifts) of the two PDs shifted directly away from the diagonal by lift*diam(target PD). This has proved extremely important for embeddings in which original Node2vec erases the topology so thoroughly that most generators match to the diagonal and not to each other.
```python
lift_array = [0.0 for i in range(LEN+1)]
```
