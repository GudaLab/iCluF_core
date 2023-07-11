from iCluF_modules import iCluF_Part1
from iCluF_modules import iCluF_Part2
import sys
import os

output_path = './data/output/Clusters'
TCGA_data = './data/input_data/TCGA_data'

## python iCluF_main.py CHOL 3 7
cancer = sys.argv[1]
K = int(sys.argv[2])
n_iterations = int(sys.argv[3])

# adjust hyperparameter
alpha = 7.5
beta = 1.5 # hyperparameter for normalizing cluster-cluster distance matrix
μ = 0.05 # hyperparameter for regulating effect of avg centroid distance on neighbourhood calculation


cancer_list = [name for name in os.listdir(TCGA_data) if os.path.isdir(os.path.join(TCGA_data, name))]
if cancer not in cancer_list:
    print ("The selected cancer type is not in the repository")
    print("**** Please select from the following list **** ")
    print ("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print (cancer_list)
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
else:
    ## implementing iCluF part 1
    Adj_dict, sample_list = iCluF_Part1.cal_adj(cancer, alpha) # returns adj patient similarity matrcies for each omics

    ## implementing iCluF part 2
    df = iCluF_Part2.pred_clus(cancer, sample_list, Adj_dict, K, n_iterations, beta, μ) # returns adj patient similarity matrcies for each omics

    ##Saving results
    # Make all directries empty #####
    all_dir = [output_path]
    for dir in all_dir:
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    output_cl = open(output_path + "/" +cancer +"_predicted_clusters_iCluF.txt", "w")
    df.to_csv(output_cl, header=True, index=True, sep='	', mode='a', line_terminator='\n')
    print ("Predicted clusters are saved in folder .data/output/Clusters")
    print ("----------------------------------- ** END ** -----------------------------------")