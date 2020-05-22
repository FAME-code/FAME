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

from pathlib import Path
from sklearn import random_projection
from sklearn.manifold import TSNE
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, eye
from scipy.io import loadmat, savemat
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, MultiLabelBinarizer
import warnings

from fastrp import *
from utils import *
from evaluate import *

# to ignore sklearn warning
def warn(*args, **kwargs):
    pass

def objective(trial):

    weights = [trial.suggest_loguniform('weight'+ str(order),1e-6,1e6) for order in range(order_range)]
    adj_weight = [trial.suggest_loguniform('adj_weight'+ str(order),1e-6,1e6) for order in range(number_edge_type)]
    conf['adj_weight'] = adj_weight
    conf['weights'] = weights
    eval_type = conf['eval_type']
    H = fastrp_wrapper(train, feature, motifs, conf)
    print('eval')
    average_auc, average_f1, average_pr, t_, t_, t_ = predict_model(file_name, H, file_name, eval_type, node_matching)
    return -(average_f1+average_auc+average_pr)

if __name__ == "__main__":
    warnings.warn = warn
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    node_matching = False

    # Aminer
    # net_path = r'data/Aminer/Aminer_10k_4class.mat'
    # savepath = r'embedding/LP_aminer_embedding_'
    # file_name = 'data/Aminer'
    # order_range = 3
    # number_edge_type = 4
    # feature_similarity = False
    # eval_type = 'all'

    # IMDB
    net_path = r"data/IMDB/data.mat"
    savepath = r"embedding/imdb_embedding_"
    eval_name = r'imdb'
    file_name = r'data/IMDB'
    order_range = 3
    number_edge_type = 4
    feature_similarity = False
    eval_type = 'all'

    # load data
    mat = loadmat(net_path)
    try:
        train = mat['train']
    except:
        print('load network error!')
    try:
        feature = mat['feature']
    except:
        print('load feature error!')
    try:
        feature = csc_matrix(feature)
    except:
        pass

    row = train[0][0].shape[0]
    col = train[0][0].shape[1]

    motifs = csr_matrix((row,col))
    edge_types = len(train)

    epochs = 1  # epoch for overall algorithm
    
    conf = {
        'projection_method': 'sparse', # sparse gaussian
        'input_matrix': 'trans',
        'adj_weight': [1] * number_edge_type,
        'normalization': True,
        'dim': 200,
        'q': order_range,
        'weights': [0] * order_range,
        'edge_type': [0,1,2,3],
        's': 1, 
        'feature_similarity': feature_similarity,
        'eval_type': eval_type,
        'trials': 100
    }

    times = []
    average_auc=[0]*epochs
    average_f1=[0]*epochs
    average_pr=[0]*epochs
    # tuning weights
    for _ in range(epochs):
        study = optuna.create_study()
        study.optimize(objective, n_trials=conf['trials'])
        conf['weights']=[0]*order_range
        for i in range(order_range):
            conf['weights'][i]=study.best_params['weight'+str(i)]
        conf['adj_weight']=[0]*number_edge_type
        for i in range(number_edge_type):
            conf['adj_weight'][i]=study.best_params['adj_weight'+str(i)]

        t_start = time.time()

        H = fastrp_wrapper(train, feature, motifs, conf)

        t_end = time.time()-t_start
        times.append(t_end)
        savemat(savepath+str(i)+'.mat',{'H':H})

        # evaluate
        times.append(t_end)
        t_, t_, t_, average_auc[_], average_f1[_], average_pr[_] = predict_model(file_name, H, file_name, eval_type, node_matching)
    
    print('AUC:', average_auc)
    print('PR:', average_pr)
    print('F1:', average_f1)
    print('time:', times)
    print('Overall ROC-AUC:', np.mean(average_auc))
    print('Overall PR-AUC', np.mean(average_pr))
    print('Overall F1:', np.mean(average_f1))
    print('mean time: {:.2f} s'.format(np.mean(times)))