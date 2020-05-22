import csv
import itertools
import math
import time
import logging
import sys
import os
import random
import warnings
import pandas as pd
import numpy as np
import scipy
import optuna
import sklearn.preprocessing as pp

from tqdm import tqdm_notebook as tqdm
from collections import Counter, defaultdict

from pathlib import Path
from sklearn import random_projection
from sklearn.preprocessing import normalize, scale, MultiLabelBinarizer
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags, spdiags, vstack, hstack

# projection method: choose from Gaussian and Sparse
# input matrix: choose from adjacency and transition matrix
# alpha adjusts the weighting of nodes according to their degree

def adj_matrix_weight_merge(A, adj_weight):

    N = A[0][0].shape[0]
    temp = csr_matrix((N,N))
    for i in range(len(adj_weight)):
        
        try:
            temp = temp + adj_weight[i]*A[i][0].tocsr()
            # temp = temp + adj_weight[i]*(A[i][0]+csc_matrix(np.eye(N)))
        except:
            temp = temp + adj_weight[i]*A[0][i].tocsr()
            # temp = temp + adj_weight[i]*(A[0][i]+csc_matrix(np.eye(N)))
    return temp+temp.transpose()

def fastrp_projection(train, feature, final_adj_matrix, edge_type, q=3, dim=128, projection_method='gaussian', input_matrix='adj', alpha=None, s=1, threshold=0.95, gama=1, feature_similarity=False):
    assert input_matrix == 'adj' or input_matrix == 'trans'
    assert projection_method == 'gaussian' or projection_method == 'sparse'
    
    num_edge = len(edge_type)
    M = final_adj_matrix

    if feature_similarity == True:
        feature = pp.normalize(feature, axis=1).T
    

    # Gaussian projection matrix
    if projection_method == 'gaussian':
        transformer = random_projection.GaussianRandomProjection(n_components=dim, random_state=7)
    # Sparse projection matrix
    else:
        transformer = random_projection.SparseRandomProjection(n_components=dim, random_state=7)
    Y = transformer.fit(feature)
    

    # Construct the inverse of the degree matrix
    if input_matrix != 'adj':
        rowsum = M.sum(axis=1)
        colsum = M.sum(axis=0).T
        rowsum = np.squeeze(np.asarray(rowsum+colsum))**-1
        rowsum[np.isinf(rowsum)]=1
        D_inv = diags(rowsum)

    cur_U = transformer.transform(feature)
    if feature_similarity == True:
        cur_U = feature.T @ cur_U
    cur_U = M @ cur_U

    if input_matrix != 'adj':
        # normalization
        cur_U = D_inv @ cur_U
    U_list = [cur_U]

    for j in range(1, q):
        # cur_U = M @ cur_U
        cur_U = M.dot(cur_U)
        if input_matrix != 'adj':
            # normalization
            cur_U = D_inv @ cur_U

        U_list.append(cur_U)

    return U_list

# When weights is None, concatenate instead of linearly combines the embeddings from different powers of A
def fastrp_merge(U_list, weights, edge_types, normalization=False, q=3):

    print('merge')
    num_edge = len(edge_types)

    if weights is None:
        # return np.concatenate(_U_list, axis=1)
        return hstack(U_list)
    
    U = np.zeros_like(U_list[0])
    for cur_U, weight in zip(U_list, weights):
        U += cur_U * weight
    try:
        U = U.todense()
    except:
        pass
    U = np.squeeze(np.asarray(U)) # convert numpy matrix to array
    return U.todense() if type(U) == csr_matrix else U

# A is always the adjacency matrix
# the choice between adj matrix and trans matrix is decided in the conf
def fastrp_wrapper(A, feature, motifs, conf):
    final_adj_matrix = adj_matrix_weight_merge(A, adj_weight = conf['adj_weight'])
    U_list = fastrp_projection(A,
                               feature,
                               final_adj_matrix,
                               q=conf['q'],
                               dim=conf['dim'],
                               projection_method=conf['projection_method'],
                               input_matrix=conf['input_matrix'],
                               edge_type = conf['edge_type'],
                               s=conf['s'],
                               feature_similarity=conf['feature_similarity']
    )
    U = fastrp_merge(U_list, conf['weights'], conf['edge_type'], conf['normalization'], conf['q'])
    return U

def get_emb_filename(prefix, conf):
    return prefix + '-dim=' + str(conf['dim']) + ',projection_method=' + conf['projection_method'] \
        + ',input_matrix=' + conf['input_matrix'] + ',normalization=' + str(conf['normalization']) \
        + ',weights=' + (','.join(map(str, conf['weights'])) if conf['weights'] is not None else 'None') \
        + ',alpha=' + (str(conf['alpha']) if 'alpha' in conf else '') \
        + ',C=' + (str(conf['C']) if 'alpha' in conf else '1.0') \
        + '.mat'

