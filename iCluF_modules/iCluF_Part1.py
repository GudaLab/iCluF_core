### Part 1 (iCluF)- Calculation of adjacancy matrixes using raw patient data
import os
import pandas as pd
from sklearn.metrics import pairwise_distances
import numpy as np


def Adj_A_Euc (alpha, df_data):
    #print(df_data[1:4, 1:6])
    dist = pairwise_distances(df_data, metric='sqeuclidean')

    # Normalizing distance matrix before calculating Adjacancy
    start = 0
    end = 10
    width = end - start
    dist = (dist - dist.min()) / (dist.max() - dist.min()) * width + start
    A = np.exp(-alpha * dist)
    return A


def cal_adj(cancer, alpha):
    input_path = './data/input_data'  # the folder of input datasets
    folder_path = input_path + "/TCGA_data/" + cancer

    # input_files_lst = os.listdir(folder_path)
    input_files_lst = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    n_datasets = len(input_files_lst)
    print("Input data files =", input_files_lst)
    print("Total datatypes = ", n_datasets)
    print("===================== Datafiles imported ======================" + "\n")

    adj_dict = dict()
    for i in range(n_datasets):
        file_name = input_files_lst[i]
        #print("===================== Datatype", file_name, "======================")
        file_path = folder_path + "/" + file_name
        dataset_df = pd.read_csv(file_path, index_col=0, delimiter='	')
        sample_list = list(dataset_df.columns)
        dataset_df = dataset_df.T

        ## log normalization of dataset #####################
        if (dataset_df.max().max() <= 1) and (dataset_df.min().min() <= 1):
           None
        else:
           dataset_df = np.log2(dataset_df+1)
        ####################################################
        ## get A (adjacancy matrix)
        Adj = Adj_A_Euc(alpha, dataset_df)
        adj_dict[file_name] = Adj
    return adj_dict, sample_list
