
from sklearn.cluster import KMeans
import numpy as np
import operator
from scipy import linalg
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

class cluster():

    def __init__(self, Adj_mat, K):
        self.Adj_mat = Adj_mat
        self.K = K

    def spec_clus(self):
        ## PART1- cluster identification
        # calculate D and M matrix

        D = self.Adj_mat.sum(axis=1)
        M = np.multiply(D[np.newaxis, :], np.multiply(self.Adj_mat, D[:, np.newaxis]))
        # EVD decomposition
        E, Kappa, _ = linalg.svd(M, full_matrices=False, lapack_driver='gesvd')
        # Apply K-means now
        Esubset = E[:, 0:self.K]
        Esubset_norm = normalize(Esubset)
        #print(Esubset_norm)
        kmeans = KMeans(n_clusters=self.K, init = "k-means++", n_init = 5, max_iter = 1000).fit(Esubset_norm)
        #kmeans = km.fit(Esubset_norm)
        cl_labels = kmeans.labels_
        #print ("cluster_labels --", cl_labels)
        return kmeans, cl_labels, Esubset_norm

    def cl_cent_distance(self, Adj_mat, K): # function for calulating distance of each point to its respective centroid
        obj_cluster = cluster(Adj_mat, K)
        kmeans, cl_labels, Esubset_norm = obj_cluster.spec_clus()
        X_dist = kmeans.transform(Esubset_norm)
        # print (X_dist)
        # print("alldistances = ", alldistances)
        center_dists = np.array(
            [X_dist[i][x] for i, x in enumerate(cl_labels)])  ## To calculate distance of each point to its centroid
        #print ("center_dists ===", center_dists)
        cl_cent_dist_dict = dict()  # dictionary stores dist of all points to its cluster center in each clusters
        cl_no = 1
        for dist in center_dists:
            cl_cent_dist_dict[cl_no] = dist
            cl_no = cl_no + 1
        #print("cl_cent_dist_dict", cl_cent_dist_dict)
        #########################################

        return cl_cent_dist_dict

    def intr_cl_distance(self, Adj_mat, K, beta): # function for calulating similarity/closeness between clusters
        obj_cluster = cluster(Adj_mat, K)
        kmeans, cl_labels, Esubset_norm = obj_cluster.spec_clus()
        cl_labels = cl_labels +1
        clust_dist = euclidean_distances(kmeans.cluster_centers_)

        # normalizing distance matrix ####
        start = 0
        end = 10
        width = end - start
        clust_dist = (clust_dist - clust_dist.min()) / (clust_dist.max() - clust_dist.min()) * width + start
        #print(clust_dist)
        ##################################

        #### Calculating eta to scale distance matrix
        p_no = []
        for i in range (len(Adj_mat)):
            p_no.append(i+1)
        cluster_dict = dict(zip(p_no, cl_labels))
        cl_dist_frm_cent = obj_cluster.cl_cent_distance(self.Adj_mat, K) # call function for clalculating avg distance in each clusters from all cluster points to its centroid
        ####----------
        D_c1_c2 = []
        for i in range(len(clust_dist)):
            c1 = i + 1
            p_in_c1 = [k for k, v in cluster_dict.items() if v == c1]  # collect all points of cluster c1
            #print ("p_in_c1 = ", p_in_c1)
            av_c1 = [] # avg distance of all points to centroid in c1
            for point in p_in_c1:
                av_c1.append(cl_dist_frm_cent[point])
            av_c1 = sum(av_c1)/len(av_c1)
            # print (Adj_mat[i])
            tmp_list = []
            for j in range(len(clust_dist)):
                c2 = j + 1
                p_in_c2 = [k for k, v in cluster_dict.items() if v == c2] # collect all points of cluster c2
                #print("p_in_c2 = ", p_in_c2)
                av_c2 = [] # avg distance of all points to centroid in c2
                for point in p_in_c2:
                    av_c2.append(cl_dist_frm_cent[point])
                av_c2 = sum(av_c2) / len(av_c2)
                d_cl1_cl2 = clust_dist[i][j] # inter cluster distance between c1 and c2
                d_cl1_cl2_sq = d_cl1_cl2**2
                #print ("av_c1 , av_c2, d_cl1_cl2", av_c1, av_c2, d_cl1_cl2)
                Eeta_c1_c2 = (av_c1 + av_c2 + d_cl1_cl2)/3 # calculate eta to normalize cluster distances
                #Eeta_c1_c2 = (av_c1 + av_c2)/2
                d = np.exp((beta) * (d_cl1_cl2_sq)/(Eeta_c1_c2 + 0.001)) # hyperparameter beta,
                tmp_list.append(d)
            #print("Eeta_c1_c2", Eeta_c1_c2)
            D_c1_c2.append(tmp_list)
        #print("matrix D ")
        #print(np.array(D_c1_c2))
        D_c1_c2 = normalize(D_c1_c2, axis=1, norm='l1')
        cl_dist_dict = dict()  # dictionary stores avg dist of all points to its cluster center in each clusters
        cl_size = len(D_c1_c2)
        for i in range (cl_size):
            for j in range(len(D_c1_c2[i])):
                dist = D_c1_c2[i][j]
                #print (i+1, j+1, dist)
                cl_dist_dict [str(i+1) +"_"+ str(j+1)] = dist
        return cl_dist_dict

    def scatterPLot(self, Adj_mat, K):
        obj_cluster = cluster(Adj_mat, K)
        self.Adj_mat = Adj_mat
        self.K = K

        kmeans, cl_labels, Esubset_norm = obj_cluster.spec_clus()
        #print ("Esubset_norm = ", Esubset_norm)
        cl_centers = kmeans.cluster_centers_
        #print("cl_centers --", cl_centers)
        cl_labels = cl_labels + 1  # just to convert default class 0, 1,.. into 1, 2, ...iCluF identifies clusters starting from 1 not 0.
        #print("cl_labels --", cl_labels)
        # scatter plot of clusters and centroids
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        #print ("----> ", cl_centers[:, 0], cl_centers[:, 1])
        sns.scatterplot(cl_centers[:, 0], cl_centers[:, 1], marker='+', color='black', s=100)

        LABEL_COLOR_MAP = {1: "black",
                           2: "blue",
                           3: "red",
                           4: "green",
                           5: "orange",
                           6: "purple"
        }

        label_color = [LABEL_COLOR_MAP[l] for l in cl_labels]
        #print ("cl_labels = ", cl_labels)

        plt.scatter(Esubset_norm[:, 0], Esubset_norm[:, 1],
                    marker='.', s=150, linewidths=0.1, c=label_color)

        #ax.set(title='K-Means Clustering')
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)

        return plt
        ########################################