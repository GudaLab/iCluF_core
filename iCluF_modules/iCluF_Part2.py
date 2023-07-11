### Part 2 (iCluF)- Calculation of neighbourhood matrices and integration

import numpy as np
import pandas as pd
from iCluF_modules import spec_clust
from iCluF_modules import A_bar_Adj_mat
from sklearn import metrics

output = './data/output/'
Int_clus_path = output + "Clusters"

def pred_clus (cancer, samples_list, Adj_dict, K, n_iterations, beta, μ):
    silhouette_list = []
    ietr_list = []
    best_silhouette = 0
    for n_iter in range(n_iterations):
        ietr_list.append(n_iter + 1)
        print("###----------Iterantion "+ str(n_iter + 1) +" completed --------###  ")
        A_bar_nth = dict()
        A_bar_trans_nth = dict()
        dim = len(list(Adj_dict.items())[0][1])
        A_sum = np.zeros((dim, dim))  # create a zero value matric

        ###### operation on individual datasets ############################
        for datatype_no in Adj_dict.keys():
            A = Adj_dict[datatype_no]
            A_sum = A_sum + A
            ## get class labels
            obj_cluster = spec_clust.cluster(A, K)
            ## call A_bar and A_bar_trans
            obj_A_bar = A_bar_Adj_mat.A_bar(A, K, μ)
            A_bar = obj_A_bar.Adj_A_bar(A, K, μ, beta)
            A_bar_nth[datatype_no] = A_bar
            A_bar_trans_nth[datatype_no] = A_bar.transpose()
        ####################################################################

        ###### operation on integrated datasets ############################
        # plotting after integration
        A_sum = A_sum / len(Adj_dict.keys())
        obj_cluster = spec_clust.cluster(A_sum, K)
        kmeans, cl_labels, Esubset_norm = obj_cluster.spec_clus()
        #pred_label = (cl_labels).tolist()

        ## silhouette_score calculation
        silhouette_score = round(metrics.silhouette_score(Esubset_norm, cl_labels), 3)
        silhouette_list.append(silhouette_score)

        ## find best best_silhouette
        if silhouette_score > best_silhouette:
            best_silhouette = silhouette_score
            best_cl = cl_labels + 1
            best_A = A_sum
            best_iter_no = n_iter + 1
        else:
            None

        # updating A_dict by iterations, adding effects of other datatypes.
        for datatype_no in Adj_dict.keys():
            remain_data = list(Adj_dict.keys() - datatype_no)
            # perform matrix multiplication
            sum_mat = 0
            for d in remain_data:
                sum_mat = sum_mat + np.dot(np.dot(A_bar_nth[datatype_no], Adj_dict[d]), A_bar_trans_nth[datatype_no])
            start = 0
            end = 1
            width = end - start
            sum_mat = (sum_mat - sum_mat.min()) / (sum_mat.max() - sum_mat.min()) * width + start
            Adj_dict[datatype_no] = sum_mat
    #print ("Best cluster achieved at iteration = %s" % best_iter_no + "\n")
    #print ("Silhouette= ", best_silhouette)
    print("Predicted clusters = ", list(best_cl))
    df = pd.DataFrame({'samples': samples_list, 'K': K, 'cluster': best_cl})
    df.index += 1
    # print(df)
    output_cl = open(Int_clus_path + "/predicted_clusters.txt", "w")
    df.to_csv(output_cl, header=True, index=True, sep='	', mode='a', line_terminator='\n')
    return df
