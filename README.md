# iCluF
Iterative Cluster Fusion (iCluF) is a Python-based tool that captures and integrates commonalities from multiomic datasets and identifies clusters in an unsupervised manner. This method requires at least two datasets to integrate.  
# Download
The software iCluF can be downloaded from https://github.com/GudaLab/iCluF_core
# Running iCluF 
## Example 1: 
To predict 3 clusters for cancer type “CHOL” with n_iter (Number of iterations) = 7, use the following command-

<img src="iCluF_modules/CHOL_Readme.png" width="450"/>

## Example 2: 
To predict 4 clusters for cancer type “ACC” with n_iter (Number of iterations) = 6, use the following command-

<img src="iCluF_modules/ACC_Readme.png" width="450"/>

The predicted clusters are saved in the output folder .data/output/Clusters

Please choose a cancer type from the following list of 30 cancers-
['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS'
, 'UVM']

# Running iCluF on different dataset
For running the algorithm on a dataset that is not in the list, the user needs to create a folder with the proper name in “./data/input_data/TCGA_data”.  The format of the data types should be the same as given in the different cancer types. 

