import numpy as np
from iCluF_modules import spec_clust
from sklearn.preprocessing import normalize


## calculating similarity neighborhood adjecancy matrix A-bar
class A_bar():
    def __init__(self, Adj_mat, K, μ):
        self.Adj_mat = Adj_mat
        self.K = K
        self.μ =  μ

    def Adj_A_bar (self, Adj_mat, K, μ, beta):
        obj_cluster = spec_clust.cluster(Adj_mat, K)

        ## cluster information
        kmeans, cl_labels, Esubset_norm = obj_cluster.spec_clus()
        cl_labels = cl_labels + 1
        p_no = []
        for i in range(len(Adj_mat)):
            p_no.append(i + 1)
        cluster_dict = dict(zip(p_no, cl_labels))

        ## similarity between clusters
        cluster_dist_dict = obj_cluster.intr_cl_distance(self.Adj_mat, K, beta)
        cl_dist_frm_cent = obj_cluster.cl_cent_distance(self.Adj_mat, K)  # call function for clalculating avg distance in each clusters from all cluster points to its centroid

        #print (len(Adj_mat))
        A_bar_list = []
        for i in range (len(Adj_mat)):
            p1 = i+1
            ## Calculating Avg (Centroid Distance) for p1's cluster
            cl1_of_p1 = cluster_dict[p1]
            All_p_in_c1 = [k for k, v in cluster_dict.items() if v == cl1_of_p1]  # collect all points in p1's cluster
            av_c1 = []  # avg distance of all points to centroid in c1
            for point in All_p_in_c1:
                av_c1.append(cl_dist_frm_cent[point])
            av_c1 = sum(av_c1) / len(av_c1)

            tmp_list = []
            for j in range (len(Adj_mat)):
                p2 = j + 1
                A_score = Adj_mat[i][j]
                #print (p1, p2, A_score)

                # centroid distance of clusters of points p1 and p2.
                ## Calculating Avg (Centroid Distance) for p2's cluster
                cl2_of_p2 = cluster_dict[p2]
                All_p_in_c2 = [k for k, v in cluster_dict.items() if v == cl2_of_p2]  # collect all points in p2's cluster

                # print("p_in_c2 = ", p_in_c2)
                av_c2 = []  # avg distance of all points to centroid in c2
                for point in All_p_in_c2:
                    av_c2.append(cl_dist_frm_cent[point])
                av_c2 = sum(av_c2) / len(av_c2)

                # Distance between clusters of point p1 and p2
                D_p1_p2 = cluster_dist_dict[str(cl1_of_p1)+"_"+str(cl2_of_p2)]
                A_bar_score = (2 * (A_score **2))/((D_p1_p2 * (μ + av_c1 + av_c2)))

                # print ("------------------------------------------------------")
                tmp_list.append(A_bar_score)
            A_bar_list.append(tmp_list)

        #print (A_bar_list)
        A_bar = np.array(A_bar_list)
        A_bar = normalize(A_bar, axis=0, norm='l1') #normalize columns of matrix
        #print(A_bar)

        return A_bar

