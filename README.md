# LKLPDA 
In this study, drawing inspiration from Fast Kernel Learning (FKL)  and Low-Rank Representation (LRR) algorithms, we propose a Low-Rank Fast Kernel Learning Fusion (LRFKL) algorithm. The main contribu-tion of the LRFKL algorithm is addressing the potential noise in the label similarity matrix within the FKL algo-rithm. Specifically, when the label information is clean, LRR can exactly recover the row space of the original data; for the label information corrupted by arbitrary errors, LRR can also approximately recover the row space with theoreti-cal guarantees. This reduction subsequently decreases the influence of noise on the label similarity matrix. Using LRFKL, we present the model LKLPDA for predicting the association between piRNAs and diseases. The flow chart of the LKLPDA model, as depicted in Figure 1, comprises three essential components. Firstly, we computed similarity matrices for piRNAs and diseases from multiple data sources to capture information from different perspectives of the same entities. Then, we utilize the LRFKL to inte-grate the multi-source similarities of piRNAs and diseases. This step enables the efficient integration of information from diverse data sources, thereby improving the model's performance. Finally, AutoGluon-Tabular (AGT) is used to predict potential piRNA-disease association. In this article, "label information" refers to the piRNA-disease association matrix.

# Requirements
* Matlab >= 2018
  python 3.8
  numpy  1.24.4
 autogluon  0.7.0
 pandas  1.2.4
 scikit-learm 1.3.0

```
Installation has been tested in a Windows platform.

# Dataset Description
* piRNAandSeq:  piRNA sequence information;
* PiRNA_seq: piRNA Sequence similarity matrix;
* piRNA _GIP: piRNA Gaussian Interaction Profile kernel similarity matrix;
* Disease_GIP: Disease Gaussian Interaction Profile kernel similarity  matrix;
* Disease_mesh:Disease Semantic similarity matrix
* Disease_gene: Disease functional similarity matrix;
* Disease_GIP: Disease functional similarity matrix;
* A: piRNA-disease association matrix.

# Functions Description
* ```LRFKL..m```: this function achieve similarity matrix fusion
* ```Rbf_kernel.m```: this function can implement the GIP function;
* ```lrra.m```: This function generates a low-rank representation matrix
* ```autogluon.py```: this function AutoGluon-Tabular Classifier Prediction
# Instructions

We provide detailed step-by-step instructions for running LKLPDA model.

**Step 1**: add datasets\functions paths
```
addpath('Datasets');
load piRNA_Disease.mat
```
**Step 2**: load datasets with association matirx and similarity matrices
```
load piRNA_Disease.mat

Wpp2 = PiRNA_seq;
Wdd2 = Disease_mesh;
Wdd3 = Disease_gene;
pidi=A;
```
**Step 3**: parameter Settings

The hyper-parameters are fixed.
```
lamda1=2000;
lamda2=2000;
```
**Step 4**: Randomly generate negative samples that are equal to the positive samples
```
**Step 5**: run Gaussian Radial Basis function (Rbf_kernel)
```
  [Wpp1,Wdd1] = Rbf_kernel(P_TMat);
```
**Step  6**: run LRFKL algorithm (LR_FKL)
```
[weight_up] = LR_FKL(Wpp,P_TMat,2,lamda1,lamda2);
PS = combine_kernels(weight_up, Wpp);
[weight_ud] = LR_FKL(Wdd,P_TMat,1,lamda1,lamda2);
DS = combine_kernels(weight_ud, Wdd);
```
**Step  7**: run autogluon.py  (autogluon.py)
autogluon.py


# A Quickstart Guide
Users can immediately start playing with LRFKL algorithm  running ```LRFKL.m``` in matlab.
* ```LRFKL.m```: It demonstrates fusing similarity matrices on the piRDisease v1.0 dataset
Then execute the autogluon.py  file to implement AutoGluon-Tabular to predict potential associations between piRNAs and Diseases.

# Contact
If you have any questions or suggestions with the code, please let us know. 
