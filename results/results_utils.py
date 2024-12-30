#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:48:57 2021

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union, List
import math

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics.cluster import contingency_matrix

import src.results.binary_prediction_utils as bin_utils
from tqdm import tqdm


def get_clus_outc_numbers(y_true: np.ndarray, clus_pred: np.ndarray):
    """
    Compute contingency matrix: entry (i,j) denotes the number of patients with true sample i and predicted clus j.

    Params:
    - y_true: array-like of true outcome one-hot assignments.
    - clus_pred: array-like of cluster label assignments.

    Returns:
        - cont_matrix: numpy ndarray of shape (num_outcs, num_clus) where entry (i,j) denotes the number of patients
        with outcome i and cluster j.
    """

    # Convert to categorical
    labels_true = np.argmax(y_true, axis=1)
    labels_pred = clus_pred

    # Compute contingency matrix
    cont_matrix = contingency_matrix(labels_true, labels_pred)

    return cont_matrix


def _convert_to_one_hot_from_probs(array_pred: Union[np.ndarray, pd.DataFrame]):
    """
    Convert array of predicted class/cluster probability assignments to one-hot encoding of the most common class/clus.

    Params: - array_pred: array-like of shape (N, K), where K is the number of target classes, with probability class
    assignments.

    Returns:
    - Output: array-like of shape (N, K) with one-hot encoded most likely class assignments.
    """

    # Convert to array if necessary
    if isinstance(array_pred, pd.DataFrame):
        array_pred = array_pred.values

    # Compute dimensionality
    if len(array_pred.shape) == 2:
        _, K = array_pred.shape

        # Convert to categorical
        class_pred = np.eye(K)[np.argmax(array_pred, axis=1)]

    else:
        # Array_pred already categorical
        K = array_pred.size

        # Convert to categorical
        class_pred = np.eye(K)[array_pred]

    return class_pred


def purity(y_true: np.ndarray, clus_pred: np.ndarray) -> float:
    """
    Computes Purity Score from predicted and true outcome labels. Purity Score is an external cluster validation tool
    which computes the largest number of individuals from a given class in a given cluster, and consequently averages
    this values over the number of clusters.

    Params:
    - y_true: array-like of shape (N, num_outcs) of true outcome labels in one-hot encoded format.
    - clus_pred: array-like of shape (N, num_clus) of predicted outcome cluster assignments.

    Returns:
    - purity_score: float indicating purity score.
    """

    # Convert clus_pred to categorical cluster assignments
    cm = get_clus_outc_numbers(y_true, clus_pred)  # shape (num_outcs, num_clus)

    # Number of most common class in each cluster
    max_class_numbers = np.amax(cm, axis=0)

    # Compute average
    purity_score = np.sum(max_class_numbers) / np.sum(cm)

    return purity_score


def compute_supervised_scores(y_true: np.ndarray, y_pred: np.ndarray, avg=None, outc_names=None):
    """
    Compute set of supervised classification scores between y_true and y_pred. List of metrics includes:
    a) AUROC, b) Recall, c) F1, d) Precision, e) Adjusted Rand Index and f) Normalised Mutual Information Score.

    Params:
    - y_true: array-like of shape (N, num_outcs) of one-hot encoded true class membership.
    - y_pred: array-like of shape (N, num_outcs) of predicted outcome probability assignments.
    - avg: parameter for a), b), c) and d) computation indicating whether class scores should be averaged, and how.
    (default = None, all scores reported).
    - outc_names: List or None, name of outcome dimensions.

    Returns:
        - Dictionary of performance scores:
            - "ROC-AUC": list of AUROC One vs Rest values.
            - "Recall": List of Recall One vs Rest values.
            - "F1": List of F1 score One vs Rest values.
            - "Precision": List of Precision One vs Rest values.
            - "ARI": Float value indicating Adjusted Rand Index performance.
            - "NMI": Float value indicating Normalised Mutual Information Score performance.
    """
    num = 10000

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values


    # # Compute AUROC and AUPRC, both custom and standard
    # auroc, auprc = bin_utils.custom_auc_auprc(y_true, y_pred, mode="OvR", num=num).values()
    # auroc_custom, auprc_custom = bin_utils.custom_auc_auprc(y_true, y_pred, mode="custom", num=num).values()

    # # GET ROC AND PRC CURVES
    # roc_prc_curves = {
    #     "OvR": bin_utils.plot_auc_auprc(y_true, y_pred, mode="OvR", outc_names=outc_names, num=num),
    #     "Custom": bin_utils.plot_auc_auprc(y_true, y_pred, mode="custom", outc_names=outc_names, num=num)
    # }
    print(" shape of y_true,",np.shape(y_true))
    print("y_true",y_true[0])
    print(" shape of y_pred,", np.shape(y_pred))
    print("y_pred", y_pred[0])



    auroc = []
    f1 = []
    rec = []
    prec = []
    ari = []
    nmi= []
    cm = []

    f1_custom = []
    rec_custom = []
    prec_custom = []
    ari_custom = []
    nmi_custom = []
    custom_thredsholds = []

    #######TODO: custom thredsholds: find the best one
    print("custom thredsholds: find the best one")
    for j in range(y_true.shape[1]):
        print("j:", j)
        temp_y_true = y_true[:, j]
        temp_y_pred = np.squeeze(y_pred[:, j])
        custom_threshold = np.arange(0, 1, 0.001)
        labels_true = np.squeeze(temp_y_true)
        scores = [ f1_score(labels_true, (temp_y_pred >= temp_threshold).astype(int), average=avg) \
                        for temp_threshold in custom_threshold]
        ix = np.argmax(scores)
        print("Best threshold,",custom_threshold[ix])
        print("Best f1_score,", scores[ix])
        custom_thredsholds.append(custom_threshold[ix])
        labels_pred = (temp_y_pred >= custom_threshold[ix]).astype(int)
        # Compute F1
        f1_custom.append(f1_score(labels_true, labels_pred, average=avg))
        # Compute Recall
        rec_custom.append(recall_score(labels_true, labels_pred, average=avg))
        # Compute Precision
        prec_custom.append( precision_score(labels_true, labels_pred, average=avg))
        # Compute ARI
        ari_custom.append( adjusted_rand_score(labels_true, labels_pred))
        # Compute NMI
        nmi_custom.append( normalized_mutual_info_score(labels_true, labels_pred))

    for j in range(y_true.shape[1]):
        temp_y_true = y_true[:,j]
        temp_y_pred = np.squeeze(y_pred[:,j])
        print("Enter into results evaluation part...")
        print("j:",j)
        print("np.mean(temp_y_pred)",np.mean(temp_y_pred))
        print("np.median(temp_y_pred)", np.median(temp_y_pred))
        auroc.append(roc_auc_score(temp_y_true, temp_y_pred, average=None))
        # Convert input arrays to categorical labels
        # labels_true, labels_pred = np.argmax(temp_y_true, axis=1), np.argmax(temp_y_pred, axis=1)
        labels_true = np.squeeze(temp_y_true)
        labels_pred = (temp_y_pred >= 0.5).astype(int)
        # Compute F1
        f1.append(f1_score(labels_true, labels_pred, average=avg))
        # Compute Recall
        rec.append(recall_score(labels_true, labels_pred, average=avg))
        # Compute Precision
        prec.append( precision_score(labels_true, labels_pred, average=avg))
        # Compute ARI
        ari.append( adjusted_rand_score(labels_true, labels_pred))
        # Compute NMI
        nmi.append( normalized_mutual_info_score(labels_true, labels_pred))
        # Compute Confusion matrix
        cm.append( confusion_matrix(y_true=labels_true, y_pred=labels_pred, labels=None, sample_weight=None, normalize=None))

    # Return Dictionary
    scores_dic = {
        "ROC-AUC": auroc,
        # "ROC-PRC": auprc,
        # # "ROC-AUC-custom": auroc_custom,
        # # "ROC-PRC-custom": auprc_custom,
        "F1": f1,
        "Recall": rec,
        "Precision": prec,
        "ARI": ari,
        "NMI": nmi,
        "custom_thredsholds":custom_thredsholds,
        "f1_custom": f1_custom,
        "rec_custom": rec_custom,
        "prec_custom": prec_custom,
        "ari_custom": ari_custom,
        "nmi_custom": nmi_custom,
    }
    print("scores_dic,",scores_dic)

    # return scores_dic, cm, roc_prc_curves
    return scores_dic, cm, None


def compute_from_eas_scores(y_true: np.ndarray, scores: np.ndarray, outc_names: np.ndarray = None, **kwargs) -> dict:
    """
    Compute supervised performance metrics given input array scores.


    Params:
    - y_true: array-like of shape (N, num_outcs).
    - scores: array-like of shape (N, ).
    - outc_names: array-like of shape (num_outcs, ) with particular outcome names.
    - kwargs: any other arguments. They are kept for coherence.

    Returns:
    - dict with scores ROC-AUC, F1, Recall, Precision per class
    """

    # Useful info
    num_outcs = y_true.shape[-1]

    if outc_names is None:
        outc_names = range(num_outcs)

    # Useful info and initialise output
    SCORE_NAMES = {"ROC-AUC": roc_auc_score, "F1": f1_score, "Recall": recall_score, "Precision": precision_score}
    output_dic = {}

    # Convert to useful format
    if isinstance(scores, pd.Series) or isinstance(scores, pd.DataFrame):
        scores = scores.values.reshape(-1)

    # Convert scores to probability thresholds
    scores_max = np.max(scores)
    scores = scores / scores_max

    # Iterate through the 4 binary scores
    for score_name, score_fn in SCORE_NAMES.items():

        # Get scoring fn
        scoring_fn = SCORE_NAMES[score_name]
        output_dic[score_name] = []

        # Iterate over outcomes
        for outc_id, outc in enumerate(outc_names):

            # Compute score for this particular outcome
            outc_labels_true = y_true[:, outc_id] == 1
            output_dic[score_name].append(scoring_fn(outc_labels_true.astype(int), scores))

    # Return object
    return output_dic


def compute_cluster_performance(X, clus_pred, y_true):
    """
    Compute cluster performance metrics given input data X and predicted cluster probability assignments clus_pred.
    Metrics computed include a) Silhouette Score, b) Davies Bouldin Score, c) Calinski Harabasz Score and d) Purity.
    Performance is computed averaged over features.

    Params:
    - X: array-like of shape (N, T, D_f) where N is the number of patients, T the number of time steps and D_f the
    number of features (+2, the id col and the time col).
    - clus_pred: array-like of shape (N, ) with predicted label assignments.
    - y_true: array-like of shape (N, num_outcs) with one-hot encoding of true class assignments.

    Returns:
        - Dictionary of output cluster performance metrics. Includes:
            - "Silhouette": Silhouette Score computation.
            - "DBI": Davies-Bouldin Index computation.
            - "VRI": Variance-Ratio Criterion (also known as Calinski Harabasz Index).
            - "Purity": Purity Score computation.
    """

    # If not converted to categorical, then convert
    if len(clus_pred.shape) == 2:
        clus_pred = np.argmax(clus_pred, axis=1)

    # Compute the same taking average over each feature dimension
    sil_avg, dbi_avg, vri_avg = 0, 0, 0
    print("Starting computing Silhouette average over each feature dimension")

    print("silhouette_score(X, clus_pred, metric=dtw)",silhouette_score(X, clus_pred, metric="dtw"))
    print("euclidean",silhouette_score(X, clus_pred, metric="euclidean"))
    print("softdtw",silhouette_score(X, clus_pred, metric="softdtw"))

    # for feat in tqdm(range(X.shape[-1])):
    #     sil_avg += silhouette_score(X[:, :, feat], clus_pred, metric="euclidean")
    #     dbi_avg += davies_bouldin_score(X[:, :, feat], clus_pred)
    #     vri_avg += calinski_harabasz_score(X[:, :, feat], clus_pred)
    # Compute Purity Score
    purity_score = purity(y_true, clus_pred)

    # Compute average factor
    num_feats = X.shape[-1]

    # Return Dictionary
    # clus_perf_dic = {
    #     "Silhouette": sil_avg / num_feats,
    #     "DBI": dbi_avg / num_feats,
    #     "VRI": vri_avg / num_feats,
    #     "Purity": purity_score / num_feats
    # }
    clus_perf_dic = {
        "Silhouette": sil_avg / num_feats,
        "Purity": purity_score / num_feats
    }
    print("clus_perf_dic",clus_perf_dic)

    return clus_perf_dic

