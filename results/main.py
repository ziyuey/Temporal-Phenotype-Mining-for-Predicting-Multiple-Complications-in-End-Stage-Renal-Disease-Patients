#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to main.py results performance.

Includes evaluation based on predicted outcomes and cluster scores, if available.

@author: Henrique Aguiar, Department of Engineering Science
email: henrique.aguiar@eng.ox.ac.uk
"""
from csv import writer

import pandas as pd

import src.results.results_utils as utils
import numpy as np
from lifelines.utils import concordance_index


def evaluate(y_true=None, y_pred=None, clus_pred=None, data_info=None, save_fd=None, avg=None, scores=None,
             **kwargs):
    """
    Evaluate function to print result information given results and/or experiment ids. Returns a dictionary of scores
    given the outputs of the model (for e.g. if the model does not do clustering, then no cluster scores are returned).

    Params:
    - y_true: array-like of shape (N, num_outcs) with one-hot encodings of true class membership. defaults to None.
    - y_pred: array-like of shape (N, num_outcs) with predicted outcome likelihood assignments. defaults to None.
    - clus_pred: array-like of shape (N, num_clus) with predicted cluster membership probabilities. default to None
    - data_info: dict of input data information and objects.
    - save_fd: str, folder where to write scores to.
    - age: str, useful how to average class individual scores (defaults to None, which returns no average).
    - scores: array-like of shape (N, ) of scores. Only relevant for score-based benchmarks, such as NEWS2 and/or ESI.
    - **kwargs: other parameters given to scoring supervised scores.


    Returns:
        - list of supervised performance scores and cluster performance scores (where valid).
        - prints information for each of the associated score.
        - saves score information in relevant folder.
    """
    # Checks for instances Df vs array and loads data properties
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    print("y_true.shape",np.shape(y_true))
    # y_true = y_true[:,0:4]
    y_true = y_true[:, 0:-2]
    print("y_true.shape", np.shape(y_true))
    print("y_true", y_true[0:50])

    if "news" in save_fd.lower() or "esi" in save_fd.lower():

        # Compute scores
        scores = utils.compute_from_eas_scores(y_true=y_true, scores=scores, **kwargs)

        # Definitions for completeness
        cm = None
        clus_metrics = {}

    else:

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values

        ####评估生存分析效果
        y_survival_t = y_true[:, -2]
        y_survival_e = y_true[:, -1]
        ci_value = concordance_index(y_survival_t, np.squeeze(-y_pred[:,-1]), y_survival_e)
        print("C-index",ci_value)
        ci_value_dic = {
            "ci_value": ci_value,
        }

        ####TODO:评估分类效果
        y_pred = y_pred[:,:-1]

        # Load data relevant properties
        data_properties = data_info["data_properties"]
        outc_names = data_properties["outc_names"]
        # outc_names = ['type1','type2','type3','type4']
        # outc_names = ['type1', 'type2']

        # Compute scores and confusion matrix
        scores, cm, Roc_curves = utils.compute_supervised_scores(y_true, y_pred, avg=avg, outc_names=outc_names)

        print("Convert Confusion Matrix to pdDataFrame... ")

        # Convert Confusion Matrix to pdDataFrame
        for ii in range(len(cm)):
            cm_item = cm[ii]
            cm_item = pd.DataFrame(cm_item, index=pd.Index(data=['0','1'], name="True Class"),
                          columns=pd.Index(data=['0','1'], name="Predicted Class"))
            # Save Confusion matrix
            cm_item.to_csv(save_fd + "confusion_matrix"+str(ii)+".csv", index=True, header=True)

        # If clustering results exist, output cluster performance scores
        print("If clustering results exist, output cluster performance scores... ")
        clus_metrics = {}
        if clus_pred is not None:

            if isinstance(clus_pred, pd.DataFrame):
                clus_pred = clus_pred.values

            # Compute X_test in 3 dimensional format
            min_, max_ = data_properties["norm_min"], data_properties["norm_max"]
            x_test_3d = data_info["X"][-1] * min_ + max_
                        # (max_ - min_) + min_

            # Compute metrics
            try:
                clus_metrics = utils.compute_cluster_performance(x_test_3d, clus_pred=clus_pred, y_true=y_true)
            except ValueError:
                print("Too little predicted labels. Can't compute clustering metrics.")
                clus_metrics = {}



    # Jointly compute scores
    print("Jointly compute scores...")
    scores = {**scores, **clus_metrics,**ci_value_dic}
    print("scores:",scores)

    # Save
    # for key, value in Roc_curves.items():
    
    #     # Get fig, ax and save
    #     fig, _ = value
    #     fig.savefig(save_fd + key)


    with open(save_fd + "scores.csv", "w+", newline="\n") as f:
        csv_writer = writer(f, delimiter=",")

        # Iterate through score key and score value(s)
        for key, value in scores.items():

            # Define row to save
            if isinstance(value, list):
                row = tuple([key, *value])
            else:
                row = tuple([key, value])

            csv_writer.writerow(row)

    # Print information
    print("\nScoring information for this experiment\n")
    for key, value in scores.items():
        print(f"{key} value: {value}")

    print("\nConfusion Matrix for predicting results", cm, sep="\n")

    return scores


def evaluate_without_survival(y_true=None, y_pred=None, clus_pred=None, data_info=None, save_fd=None, avg=None, scores=None,
             **kwargs):
    """
    Evaluate function to print result information given results and/or experiment ids. Returns a dictionary of scores
    given the outputs of the model (for e.g. if the model does not do clustering, then no cluster scores are returned).

    Params:
    - y_true: array-like of shape (N, num_outcs) with one-hot encodings of true class membership. defaults to None.
    - y_pred: array-like of shape (N, num_outcs) with predicted outcome likelihood assignments. defaults to None.
    - clus_pred: array-like of shape (N, num_clus) with predicted cluster membership probabilities. default to None
    - data_info: dict of input data information and objects.
    - save_fd: str, folder where to write scores to.
    - age: str, useful how to average class individual scores (defaults to None, which returns no average).
    - scores: array-like of shape (N, ) of scores. Only relevant for score-based benchmarks, such as NEWS2 and/or ESI.
    - **kwargs: other parameters given to scoring supervised scores.


    Returns:
        - list of supervised performance scores and cluster performance scores (where valid).
        - prints information for each of the associated score.
        - saves score information in relevant folder.
    """
    # Checks for instances Df vs array and loads data properties
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    print("y_true.shape", np.shape(y_true))
    # y_true = y_true[:,0:4]
    y_true = y_true[:, 0:-2]
    print("y_true.shape", np.shape(y_true))

    if "news" in save_fd.lower() or "esi" in save_fd.lower():

        # Compute scores
        scores = utils.compute_from_eas_scores(y_true=y_true, scores=scores, **kwargs)

        # Definitions for completeness
        cm = None
        clus_metrics = {}

    else:

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values


        ####TODO:评估分类效果

        # Load data relevant properties
        data_properties = data_info["data_properties"]
        outc_names = data_properties["outc_names"]
        # outc_names = ['type1','type2','type3','type4']
        # outc_names = ['type1', 'type2']

        # Compute scores and confusion matrix
        scores, cm, Roc_curves = utils.compute_supervised_scores(y_true, y_pred, avg=avg, outc_names=outc_names)

        print("Convert Confusion Matrix to pdDataFrame... ")

        # Convert Confusion Matrix to pdDataFrame
        for ii in range(len(cm)):
            cm_item = cm[ii]
            cm_item = pd.DataFrame(cm_item, index=pd.Index(data=['0', '1'], name="True Class"),
                                   columns=pd.Index(data=['0', '1'], name="Predicted Class"))
            # Save Confusion matrix
            cm_item.to_csv(save_fd + "confusion_matrix" + str(ii) + ".csv", index=True, header=True)

        # If clustering results exist, output cluster performance scores
        print("If clustering results exist, output cluster performance scores... ")
        clus_metrics = {}
        if clus_pred is not None:

            if isinstance(clus_pred, pd.DataFrame):
                clus_pred = clus_pred.values

            # Compute X_test in 3 dimensional format
            min_, max_ = data_properties["norm_min"], data_properties["norm_max"]
            x_test_3d = data_info["X"][-1] * min_ + max_
            # (max_ - min_) + min_

            # Compute metrics
            try:
                clus_metrics = utils.compute_cluster_performance(x_test_3d, clus_pred=clus_pred, y_true=y_true)
            except ValueError:
                print("Too little predicted labels. Can't compute clustering metrics.")
                clus_metrics = {}

    # Jointly compute scores
    print("Jointly compute scores...")
    scores = {**scores, **clus_metrics}
    print("scores:", scores)

    # Save
    # for key, value in Roc_curves.items():

    #     # Get fig, ax and save
    #     fig, _ = value
    #     fig.savefig(save_fd + key)

    with open(save_fd + "scores.csv", "w+", newline="\n") as f:
        csv_writer = writer(f, delimiter=",")

        # Iterate through score key and score value(s)
        for key, value in scores.items():

            # Define row to save
            if isinstance(value, list):
                row = tuple([key, *value])
            else:
                row = tuple([key, value])

            csv_writer.writerow(row)

    # Print information
    print("\nScoring information for this experiment\n")
    for key, value in scores.items():
        print(f"{key} value: {value}")

    print("\nConfusion Matrix for predicting results", cm, sep="\n")

    return scores
