from scipy import spatial
import sklearn as sk
import csv
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# from confusion_metrics import tf_confusion_metrics
from collections import Counter
from collections import defaultdict
import scipy.io as sio
import numpy as np
import os


'''Read mat'''
filename = 'eventall_featlabel_clean_sym05-Mar-201719-58.mat'
timeflag = '05-Mar-201719-58'
dir = '../../../../AllData/Fei-DrugData/ddi_prep/matlab_results/' + timeflag + '/'

mat_contents = sio.loadmat(dir + filename)

def compute_model_similarity(target_taskid):
    """
    Compute the cosin similarity of model vectors get from Logistic regression.
    :return:
    """
    index_auc = mat_contents['results'][target_taskid][0][0][5]
    Ws = mat_contents['results'][target_taskid][0][0][2]
    W_target = Ws[index_auc][0][0][0]

    W_corrs = np.zeros((1318, 1))
    for ii in range(1318):
        index_auc = mat_contents['results'][ii][0][0][5]
        Ws = mat_contents['results'][ii][0][0][2]
        W = Ws[index_auc][0][0][0]
        W_corrs[ii] = 1 - spatial.distance.cosine(W.T, W_target.T)

    # print W_corrs[:10]
    W_corrs = np.reshape(W_corrs, W_corrs.shape[0] * W_corrs.shape[1])
    sorted_inds = np.argsort(W_corrs)[::-1]
    W_corrs_sorted = W_corrs[sorted_inds]
    # print W_corrs_sorted[:10], sorted_inds[:10]

    return W_corrs_sorted, sorted_inds  # note that sorted_inds includes the task it self.


def compute_model_similarity_run():
    """ run all tasks
    :return:
    """

    sorted_index_all = []
    sorted_simis_all = []
    for ii in range(1318):
        sorted_simis, sorted_index = compute_model_similarity(ii)
        sorted_index_all.append(sorted_index)
        sorted_simis_all.append(sorted_simis)

    train_dir = "../../../../AllData/Fei-DrugData/ddi_prep/" + timeflag + '/'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    np.save(train_dir + "tasks_lrmodel_similarity.npy", [sorted_index_all, sorted_simis_all])
    return sorted_index_all



if __name__ == "__main__":
    # compute_model_similarity(400)
    compute_model_similarity_run()