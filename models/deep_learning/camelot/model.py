#!/usr/bin/env python3

"""
Model definition for Camelot data

Date Last updated: 24 Jan 2022
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.cluster import KMeans

import os
import json
from typing import Union

import src.models.deep_learning.camelot.model_utils as model_utils
from src.models.deep_learning.model_blocks import MLP, AttentionRNNEncoder, LSTMEncoder, multi_task_MLP

tf.compat.v1.enable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay


class CAMELOT(tf.keras.Model):
    """
    Model Class for CAMELOT architecture.

    Params:

        (General)
    - num_clusters: number of clusters. (default = 10)
    - latent_dim: dimensionality of latent space. (default = 32)
    - output_dim: dimensionality of output space. (default = 4)
    - seed: Seed to run analysis on. (default = 4347)
    - name: Name to give the model. (default = "CAMELOT")

        (Loss functions)
    - alpha: weighting in cluster entropy. (default = 0.01)
    - beta: weighting in clustering representation separation. (default = 0.01)

        (Regularisation Params)
    - regulariser_params: tuple of l1_l2 float weights. (default = (0.01, 0.01))
    - dropout: float corresponding to dropout value. (default = 0.6)

        (Encoder Params)
    - encoder_params: Dictionary indicating parameters for Encoder architecture, as follows:
            - activation: activation function of custom feature projection component (default = "linear")
            - hidden_layers: Number of "hidden"/intermediate LSTM layers.  (default = 1)
            - hidden_nodes: Dimensionality of the intermediate state computation. (default = 20)
            - state_fn: The activation function to use on cell state/output. (default = 'tanh')
            - recurrent_activation: The activation function to use on F/I/O gates. (default = 'sigmoid')
            - recurrent_dropout: dropout rate to be used on forget/input/output gates. (default = 0.0)
    Default value is {}, which resets to default parameters.

        (Identifier Params)
    - identifier_params: Dictionary indicating parameters for Identifier block, as follows:
        - hidden_layers: int, Number of "hidden" feedforward layers. (default = 2)
        - hidden_nodes: int, For hidden feedforward layers, the dimensionality of the output space. (default = 30)
        - activation_fn: str/fn, The activation function to use. (default = 'sigmoid')
    Default value is {"name": "Identifier"}, which resets to default parameters.

        (Predictor Params)
    - predictor_params: Dictionary indicating parameters for Predictor block, as follows:
        - hidden_layers: int, Number of "hidden" feedforward layers. (default = 2)
        - hidden_nodes: int, For hidden feedforward layers, the dimensionality of the output space. (default = 30)
        - activation_fn: str/fn, The activation function to use. (default = 'sigmoid')
    Default value is {"name": "Predictor"}, which resets to default parameters.
    """

    def __init__(self, num_clusters=10, latent_dim=32, seed=4347, output_dim=4, name="CAMELOT",
                 alpha_1=0.01, alpha_2=0.01, alpha_3=0.01, beta=0.01, survival_coef=0, regulariser_params=(0.01, 0.01),
                 dropout=0.6,
                 encoder_params=None, identifier_params=None, predictor_params=None):

        super().__init__(name=name)

        # General params
        self.K = num_clusters
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seed = seed

        # Loss function params
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.beta = beta
        self.survival_coef = survival_coef

        # Common to all Networks
        self.regulariser = regulariser_params
        self.dropout = dropout

        # Build Networks
        self.encoder_params = encoder_params if encoder_params is not None else {}
        self.identifier_params = identifier_params if identifier_params is not None else {}
        self.predictor_params = predictor_params if predictor_params is not None else {}

        self.Encoder = AttentionRNNEncoder(units=self.latent_dim, dropout=self.dropout,
                                           regulariser_params=self.regulariser, name="Encoder",
                                           **self.encoder_params)
        # self.Encoder = LSTMEncoder(latent_dim=self.latent_dim, dropout=self.dropout,
        #                             regulariser_params=self.regulariser, name="Encoder",
        #                             **self.encoder_params)
        self.Identifier = MLP(output_dim=self.K, dropout=self.dropout, output_fn="softmax",
                              regulariser_params=self.regulariser, seed=self.seed, name="Identifier",
                              **self.identifier_params)
        self.Predictor = multi_task_MLP(output_dim=self.output_dim, dropout=self.dropout, output_fn="sigmoid",
                                        regulariser_params=self.regulariser, seed=self.seed, name="Predictor",
                                        **self.predictor_params)

        # Cluster Representation params
        self.cluster_rep_set = tf.Variable(initial_value=tf.zeros(shape=[self.K, self.latent_dim], dtype='float32'),
                                           trainable=True, name='cluster_rep')
        # self.cluster_opt = optimizers.Adam(learning_rate=self.cluster_rep_lr)

        # Initialisation loss trackers
        self.enc_pred_loss_tracker = None
        self.iden_loss_tracker = None

    # Build and Call Methods
    def build(self, input_shape):
        """Build method to serialise layers."""
        self.Encoder.build(input_shape)
        self.Encoder.feat_time_attention_layer.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Call method for model.

        Params:
        - inputs: array-like of shape (bs, T, D_f)

        Returns: tuple of arrays:
            - y_pred: array-like of shape (bs, outcome_dim) with probability assignments.
            - pi: array-like of shape (bs, K) of cluster probability assignments.
        """
        y_pred, pi = self.forward_pass(inputs)

        return y_pred

    # TRAINING RELATED METHODS
    def forward_pass(self, inputs):
        """
        Single forward pass given input data. Inputs are encoded. Encoded vectors pass through Identifier and
        assigned to clusters. Cluster representations are used to predict outcome.

        Params:
        - inputs: array-like of shape (bs, T, D_f) of input data.

        Returns: tuple of arrays:
            - y_pred: array-like of shape (bs, outcome_dim) with probability assignments.
            - pi: array-like of shape (bs, K) of cluster probability assignments.
        """

        z = self.Encoder(inputs)
        pi = self.Identifier(z)

        # Sample from cluster assignments and assign corresponding cluster representations
        clus_phens = self.Predictor(self.cluster_rep_set)
        y_pred = tf.linalg.matmul(pi, clus_phens)

        return y_pred, pi

    def _sample_from_probs(self, clus_probs):
        """
        Method to sample cluster given cluster assignment probabilities and categorical sampling.

        Params:
        - clus_probs: array-like of shape (bs, K) of cluster assignment probabilities.

        Returns:
            - output: array-like of shape (bs,) with the corresponding sampled cluster.
        """

        # Convert to logits
        logits = tf.math.log(tf.reshape(clus_probs, shape=[-1, self.K]))

        # Categorical sampling
        sample = tf.random.categorical(logits, num_samples=1, seed=self.seed)

        return tf.squeeze(sample)

    def _select_representations_from_sample(self, clus_samp):
        """
        Method to select cluster representation vector given cluster assignment.

        Params:
        - clus_samp: array-like of shape (bs, ) with the corresponding cluster.

        Returns:
            - output: array-like of shape (bs, latent_dim) with the corresponding cluster representation vector.
        """

        # Convert to one-hot encoding
        idx_mask = tf.one_hot(clus_samp, depth=self.K)

        # Obtain representation
        clus_rep = tf.linalg.matmul(idx_mask, self.cluster_rep_set)

        return clus_rep

    def train_step(self, inputs):
        """
        Method to train model.

        Params:
            - inputs: tuple of input (X, y) data:
                - X: array-like of shape (bs, T, D_f) of input time-series feature data.
                - y: array-like of shape (bs, num_outcs) of input outcome class data.

        Method updates model weights according to loss backpropagation.
        """

        # Unpack inputs
        # x, y = inputs
        x, y_raw = inputs
        y = y_raw[:, :-2]
        y_survival_t = y_raw[:, -2]
        y_survival_e = y_raw[:, -1]

        # Define variables for each network
        pred_vars = [var for var in self.trainable_variables if 'predictor' in var.name.lower()]
        enc_id_vars = [var for var in self.trainable_variables if 'encoder' in var.name.lower() or
                       'identifier' in var.name.lower()]
        rep_vars = self.cluster_rep_set

        # ------------------------------------------ OPTIMISE ALL ----------------------------------------
        all_vars = pred_vars + enc_id_vars + [rep_vars]

        # Initialise GradientTape to compute gradients
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(all_vars)

            # Make forward pass
            y_pred, pi = self.forward_pass(x)
            clus_phens = self.Predictor(rep_vars)

            # Compute Loss Weights
            loss_weights = model_utils.class_weighting(y)

            # compute losses
            l_crit = model_utils.l_crit(y, y_pred, y_survival_t, y_survival_e, weights=None)
            l_dist = model_utils.l_dist(y_pred=y_pred, true_dist=loss_weights)
            l_pat_entr = model_utils.l_pat_dist(clusters_prob=pi)
            l_clus_entr = model_utils.l_clus_dist(clusters_prob=pi)
            l_clus = model_utils.l_clus(clus_phens)
            l_survival = model_utils.l_survival(Y_label_T=y_survival_t, Y_label_E=y_survival_e, y_pred=y_pred)

            # Get loss
            loss = l_crit + self.alpha_1 * l_dist + self.alpha_2 * l_pat_entr + self.alpha_3 * l_clus_entr + \
                   self.beta * l_clus + self.survival_coef * l_survival

            # loss = l_crit + self.alpha_1 * l_dist + self.alpha_2 * l_pat_entr + self.alpha_3 * l_clus_entr + \
            #        self.beta * l_clus

        # Compute gradients and train model
        all_grad = tape.gradient(target=loss, sources=all_vars)
        clipped_grad = [(tf.clip_by_value(grad, clip_value_min=-5.0, clip_value_max=5.0))
                     for grad in all_grad]
        self.optimizer.apply_gradients(zip(clipped_grad, all_vars))

        # Return losses
        return {"Loss": loss, "L_pred": l_crit, "l_dist": l_dist, "l_pat_entr": l_pat_entr, "l_clus_entr": l_clus_entr,
                "L_clus": l_clus, "L_survival": l_survival}

        # return {"Loss": loss, "L_pred": l_crit, "l_dist": l_dist, "l_pat_entr": l_pat_entr, "l_clus_entr": l_clus_entr,
        #         "L_clus": l_clus}

    def test_step(self, inputs):
        """
        Method to compute test step model.

        Params:
            - inputs: tuple of input (X, y) data:
                - X: array-like of shape (bs, T, D_f) of input time-series feature data.
                - y: array-like of shape (bs, num_outcs) of input outcome class data.
        """

        # Unpack input data
        # x, y = inputs
        x, y_raw = inputs
        # y = y_raw[:,0:4]
        # y_survival_t = y_raw[:,4]
        # y_survival_e = y_raw[:, 5]
        y = y_raw[:, :-2]
        y_survival_t = y_raw[:, -2]
        y_survival_e = y_raw[:, -1]

        # Make forward pass
        y_pred, pi = self.forward_pass(x)
        clus_phens = self.Predictor(self.cluster_rep_set)

        # Compute Weights
        loss_weights = model_utils.class_weighting(y)

        # compute losses
        l_crit = model_utils.l_crit(y, y_pred, y_survival_t, y_survival_e, weights=loss_weights)
        l_dist = model_utils.l_dist(y_pred=y_pred, true_dist=loss_weights)
        l_pat_entr = model_utils.l_pat_dist(clusters_prob=pi)
        l_clus_entr = model_utils.l_clus_dist(clusters_prob=pi)
        l_clus = model_utils.l_clus(clus_phens)
        l_survival = model_utils.l_survival(Y_label_T=y_survival_t, Y_label_E=y_survival_e, y_pred=y_pred)

        # Get loss
        loss = l_crit + self.alpha_1 * l_dist + self.alpha_2 * l_pat_entr + self.alpha_3 * l_clus_entr + \
               self.beta * l_clus + self.survival_coef * l_survival
        # loss = l_crit + self.alpha_1 * l_dist + self.alpha_2 * l_pat_entr + self.alpha_3 * l_clus_entr + \
        #        self.beta * l_clus
        # # Return losses
        return {"Loss": loss, "L_pred": l_crit, "l_dist": l_dist, "l_pat_entr": l_pat_entr, "l_clus_entr": l_clus_entr,
                "L_clus": l_clus, "L_survival": l_survival}

        ######初始化后的模型预测表现：
        # print("------Test step: Enter into results evaluation part...")

        threadhold = [0.5]*y.shape[1]
        f1_sc = []
        recall_sc = []
        precision_sc = []
        ari_sc = []
        nmi_sc = []

        for j in range(y.shape[1]):
            temp_y_true = y[:, j]
            temp_y_pred = np.squeeze(y_pred[:, j])
            # print("AUROC",roc_auc_score(temp_y_true, temp_y_pred, average=None))
            # Convert input arrays to categorical labels
            labels_true = np.squeeze(temp_y_true)
            labels_pred = (temp_y_pred > threadhold[j]).astype(int)
            # Compute F1
            f1_sc.append(f1_score(labels_true, labels_pred))
            # Compute Recall
            recall_sc.append(recall_score(labels_true, labels_pred))
            # Compute Precision
            precision_sc.append(precision_score(labels_true, labels_pred))
            # Compute ARI
            ari_sc.append(adjusted_rand_score(labels_true, labels_pred))
            # Compute NMI
            nmi_sc.append(normalized_mutual_info_score(labels_true, labels_pred))

        return {"Loss": loss, "L_pred": l_crit, "l_dist": l_dist, "l_pat_entr": l_pat_entr, "l_clus_entr": l_clus_entr,
                "L_clus": l_clus}

    # Initialisation Methods for Model Training
    def initialise_model(self, data: tuple, val_data: tuple, epochs_1: int = 100, epochs_2: int = 100,
                         batch_size: int = 64, patience_epochs: int = 200, training_rate: float = 0.001, **kwargs):
        """
        Initialisation Method for Model.

        Params:
            - data: tuple (X, y) of data to train the model.
                - X: array-like of shape (N, T, D_f)
                - y: array-like of shape (N, num_outcs)
            - val_data: tuple (X, y) of validation data to see loss performance.
                - X: array-like of shape (N', T, D_f)
                - y: array-like of shape (N', num_outcs)
            - epochs: int, number of epochs for training. (default = 100)
            - learning_rate: float, starting learning rate for initialisation training
            - batch_size: int, size of individual batches.
            - patience_epochs: int, number of epochs to wait until improvement is seen.
            - kwargs: dictionary arguments for KMeans initialisation.

        Updates model according to initialisation procedure. Initialisation consists of 3 steps:
        - Outcome prediction by applying Predictor network directly on input data representation.
        - Cluster representation initialisation through K-Means on input data representation.
        - Identifier initialisation by minimising loss on clusters as predicted by KMeans.
        """

        # Unpack data
        # x, y = data
        x, y_raw = data
        y = y_raw[:, 0:-2]

        # val_x, val_y = val_data
        val_x, val_y_raw = val_data
        val_y = val_y_raw[:, 0:-2]

        # Go through initialisation steps
        self._initialise_enc_pred(data=data, val_data=val_data, epochs=epochs_1, batch_size=batch_size,
                                  patience_epochs=patience_epochs, training_rate=training_rate)

        clus_train_y, clus_val_y = self._initialise_clus(x, val_x)

        # Initialise Identifier
        iden_train_data = (x, clus_train_y)
        iden_val_data = val_x, clus_val_y
        self._initialise_iden(data=iden_train_data, val_data=iden_val_data,
                              epochs=epochs_2, batch_size=batch_size, patience_epochs=patience_epochs)

    def _initialise_enc_pred(self, data, val_data, epochs=100, batch_size=64, patience_epochs=200, training_rate=0.001):
        """
          Initialisation Method for Encoder and Predictor blocks.

          Params:
              - data: tuple (X, y) of data to train the model.
                  - X: array-like of shape (N, T, D_f)
                  - y: array-like of shape (N, num_outcs)
              - val_data: tuple (X, y) of validation data to see loss performance.
                  - X: array-like of shape (N', T, D_f)
                  - y: array-like of shape (N', num_outcs)
              - epochs: int, number of epochs for training. (default = 100)
              - batch_size: int, size of individual batches.
              - training_rate: float, learning rate of initialise training (default=0.001)

        Input data passes through Encoder network to obtain data representations. Predictor then outputs a predicted
        class. This is matched against the true class.
        """

        # Get optimiser information
        optimiser = optimizers.Adam(learning_rate=training_rate)

        # Load into data dataset
        def filter_batches(x_batch, y_batch):
            return tf.math.reduce_sum(y_batch[:,-1])!=0
        input_dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(1000, seed=self.seed).batch(batch_size).filter(filter_batches)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data).shuffle(1000, seed=self.seed).batch(batch_size).filter(filter_batches)

        # Initialise loss tracker
        self.enc_pred_loss_tracker = pd.DataFrame(data=np.nan, index=[], columns=["train_loss", "val_loss"])
        enc_pred_vars = [var for var in self.trainable_variables if "Encoder" in var.name or "Predictor" in var.name]

        # Iterate through epochs and batches
        print("-" * 20, "\n", "Initialising encoder-predictor training.")
        for epoch in range(epochs):

            epoch_loss, step_ = 0, 0
            for step_, (x_batch, y_batch) in enumerate(input_dataset):
                # One Training Step
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(enc_pred_vars)

                    # Prediction and loss
                    y_pred = self.Predictor(self.Encoder(x_batch))
                    # print("x_batch,",x_batch)
                    # print("self.Encoder(x_batch)",self.Encoder(x_batch))
                    # print("y_pred", y_pred)
                    # print("y_batch[:,0:-2]",np.shape(y_batch[:,0:-2]))
                    # print("y_pred", np.shape(y_pred))
                    loss_batch = model_utils.l_crit_init(y_batch[:, 0:-2], y_pred, y_batch[:, -2], y_batch[:, -1],
                                                         del_flag=True)
                    # print("loss_batch",loss_batch)

                # Update gradients
                enc_pred_grad = tape.gradient(loss_batch, enc_pred_vars)
                clipped_enc_pred_grad = [(tf.clip_by_value(grad, clip_value_min=-5.0, clip_value_max=5.0))
                                for grad in enc_pred_grad]
                optimiser.apply_gradients(zip(clipped_enc_pred_grad, enc_pred_vars))

                # Update loss
                epoch_loss += loss_batch

                # Print current batch loss - clears line and re-writes
                print("Batch Loss %.4f" % loss_batch, end="\r", flush=True)

            # Take mean over whole data
            epoch_loss = epoch_loss / step_

            # Compute validation loss on validation data
            val_loss, val_step_ = 0, 0
            for val_step_, (x_val, y_val) in enumerate(val_dataset):
                # Get forward pass
                y_val_pred = self.Predictor(self.Encoder(x_val))
                # print("x_val,",x_val)
                # print("self.Encoder(x_val)",self.Encoder(x_val))
                # print("y_val_pred", y_val_pred)
                # print("y_val_pred[:,0:-2]",np.shape(y_val_pred[:,0:-2]))
                # print("y_val_pred", np.shape(y_val_pred))
                loss_val_batch = model_utils.l_crit_init(y_val[:, 0:-2], y_val_pred, y_val[:, -2], y_val[:, -1],
                                                         del_flag=True)
                print("Batch Val Loss %.4f" % loss_val_batch, end="\r", flush=True)

                # Update loss
                val_loss += loss_val_batch

            # Take mean over whole dataset
            val_loss = val_loss / val_step_

            # Print result and update tracker
            self.enc_pred_loss_tracker.loc[epoch + 1, :] = [epoch_loss, val_loss]
            print("End of epoch %d - \n Training loss: %.4f  Validation loss %.4f" % (epoch, epoch_loss, val_loss))

            # Check if result hasn't improved for 2 epochs
            # ###临时修改！
            # if epoch >2:
            if epoch > patience_epochs and epochs > 20 and val_loss >= self.enc_pred_loss_tracker.iloc[
                                                                       -patience_epochs - 20:-20, -1].min():
                break

    def _initialise_clus(self, x, val_x, **kwargs):
        """
          Initialisation Method for cluster representation

          Params:
              - x: array-like of shape (N, T, D_f)
              - val_x: array-like of shape (N', T, D_f)
              - kwargs: other arguments relevant to KMeans method.

        Cluster representations are initialised through KMeans on the set of data representations.

        Returns:
            - tuple:
                - clus_train_y: array-like of shape (N, num_clus) of cluster one-hot assignments.
                - clus_val_y: array-like of shape (N', num_clus) of cluster one-hot assignments.
        """

        # Compute Latent Projections
        print("-" * 20, "\n", "Initialising cluster representations.")
        print("shape of x,", np.shape(x))
        print("shape of val_x,", np.shape(val_x))

        # z = self.Encoder(x).numpy()
        # y_pred = self.Predictor(self.Encoder(x)).numpy()
        #######################更改成分批预测：开始
        # z = self.Encoder(x).numpy()
        # y_pred = self.Predictor(self.Encoder(x)).numpy()
        def BatchNext(x_train, batch_size, iters, idx):
            # idx = idx[:int(batch_size)]
            start_idx = batch_size * iters
            idx = idx[start_idx:start_idx + int(batch_size)]
            batch_x = [x_train[i] for i in idx]
            batch_x = np.array(batch_x)
            return batch_x

        z = []
        y_pred = []
        val_idx = np.arange(0, x.shape[0])
        for iter in range(int(len(x) / 128) + 1):
            batch_x = BatchNext(x, 128, iter, val_idx)
            z_batch = self.Encoder(batch_x).numpy()
            y_pred_batch = self.Predictor(self.Encoder(batch_x)).numpy()
            z.extend(z_batch)
            y_pred.extend(y_pred_batch)
        print("shape of z,", np.shape(z))
        print("shape of y_pred,", np.shape(y_pred))
        print("shape of y_pred,", np.mean(y_pred, axis=0))
        #######################更改成分批预测：结束

        # Fit KMeans
        km = KMeans(n_clusters=self.K, init="k-means++", random_state=self.seed, **kwargs)
        km.fit(y_pred)
        print("KMeans fit has completed.")

        # Make predictions and get predicted centers
        cluster_pred = km.predict(y_pred)
        _one_hot_clus_assign = np.eye(self.K)[cluster_pred]

        # Compute average of cluster assignments
        centers = np.matmul(np.transpose(_one_hot_clus_assign), z) / np.sum(_one_hot_clus_assign, axis=0).reshape(-1, 1)

        # Compute Initialised Estimates
        print("\nInitialised Phenotypes: ", self.Predictor(centers))
        print("\nEstimated Cluster distribution: ", np.unique(cluster_pred, return_counts=True))

        # Initialise embeddings and convert to one-hot encoding for Identifier
        self.cluster_rep_set.assign(tf.convert_to_tensor(centers, name='cluster_rep', dtype='float32'))
        clus_train_y = np.eye(self.K)[cluster_pred]

        # Make predictions on validation data
        # y_val = self.Predictor(self.Encoder(val_x)).numpy()
        #######################更改成分批预测：开始
        # y_val = self.Predictor(self.Encoder(val_x)).numpy()
        y_val = []
        val_idx1 = np.arange(0, val_x.shape[0])
        for iter1 in range(int(len(val_x) / 128) + 1):
            batch_val_x = BatchNext(val_x, 128, iter1, val_idx1)
            z_val_batch = self.Predictor(self.Encoder(batch_val_x)).numpy()
            y_val.extend(z_val_batch)
        print("shape of y_val,", np.shape(y_val))
        #######################更改成分批预测：结束

        clus_val_y = np.eye(self.K)[km.predict(y_val)]

        return clus_train_y.astype(np.float32), clus_val_y.astype(np.float32)

    def _initialise_iden(self, data, val_data, epochs=100, batch_size=64, patience_epochs=200, training_rate=0.001):
        """
          Initialisation Method for Identifier Network

          Params:
              - data: tuple (X, clus_train_y) of data to train the model.
                  - X_train: array-like of shape (N, T, D_f)
                  - clus_train_y: array-like of shape (N, K)
              - val_data: tuple (X, clus_val_y) of validation data to see loss performance.
                  - X_train: array-like of shape (N', T, D_f)
                  - clus_val_y: array-like of shape (N', K)
              - epochs: int, number of epochs for training. (default = 100)
              - batch_size: int, size of individual batches.
              - training_rate: float, learning rate of initialise training (default=0.001)

        Input data passes through Encoder network to obtain data representations. Predictor then outputs a predicted
        class. This is matched against the true class.
        """

        # Get optimiser
        optimiser = optimizers.Adam(learning_rate=training_rate)

        # Convert to data Dataset
        def filter_batches(x_batch, y_batch):
            return tf.math.reduce_sum(y_batch[:,-1])!=0
        input_dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(1000, seed=self.seed).batch(batch_size).filter(filter_batches)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data).shuffle(1000, seed=self.seed).batch(batch_size).filter(filter_batches)

        # Initialise loss tracker
        self.iden_loss_tracker = pd.DataFrame(data=np.nan, index=[], columns=["train_loss", "val_loss"])
        iden_vars = [var for var in self.trainable_variables if "identifier" in var.name.lower()]

        # Forward Identifier pass and train
        print("-" * 20, "\nInitialising Identifier training.")

        for epoch in range(epochs):  # Iterate through epochs
            epoch_loss, step_ = 0, 0
            for step_, (x_batch, clus_batch) in enumerate(input_dataset):  # Iterate through batch

                # One Training Step
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(iden_vars)

                    # Prediction and loss
                    clus_pred = self.Identifier(self.Encoder(x_batch))
                    loss_batch = model_utils.l_crit_clus(clus_batch, clus_pred)

                # Update gradients
                iden_grad = tape.gradient(loss_batch, iden_vars)

                optimiser.apply_gradients(zip(iden_grad, iden_vars))

                # Update loss
                epoch_loss += loss_batch

                # Print current batch loss - clears line and re-writes
                print("Batch Loss %.4f" % loss_batch, end="\r", flush=True)

            # Get average over data
            epoch_loss = epoch_loss / step_

            # Compute validation loss on validation data
            val_loss, val_step_ = 0, 0
            for val_step_, (x_val, clus_val) in enumerate(val_dataset):
                # Get forward pass
                clus_pred = self.Identifier(self.Encoder(x_val))
                loss_val_batch = model_utils.l_crit_clus(clus_val, clus_pred)

                # Update loss
                val_loss += loss_val_batch

            # Get average over data
            val_loss = val_loss / val_step_

            # Print result and update tracker
            print("End of epoch %d - \n Training loss: %.4f  Validation loss %.4f" % (epoch, epoch_loss, val_loss))
            self.iden_loss_tracker.loc[epoch + 1, :] = [epoch_loss, val_loss]

            # Check if result hasn't improved for 2 epochs
            # ###临时修改！
            # if epoch >2:
            if epoch > patience_epochs and epoch > 20 and val_loss >= self.iden_loss_tracker.iloc[
                                                                      -patience_epochs - 20:-21, -1].min():
                break

    # Useful Methods to compute attributes and model properties.
    def compute_unnorm_attention_weights(self, inputs):
        """
        Computes unnormalised attention weights alpha, beta, gamma.

        Params:
        - inputs: array-like of shape (N, T, D_f).

        Return: tuple of unnormalised attention weights.
            - alpha: array-like of shape (N, T, D_f)
            - beta: array-like of shape (1, T, 1)
            - gamma: array-like of shape (N, K, D_f)
        """
        scores = self.Encoder.compute_unnorm_scores(inputs, cluster_reps=self.cluster_rep_set)

        return scores

    def compute_norm_attention_weights(self, inputs):
        """
        Computes normalised attention weights alpha, beta, gamma.

        Params:
        - inputs: array-like of shape (N, T, D_f).

        Return: tuple of normalised attention weights.
            - alpha: array-like of shape (N, T, D_f)
            - beta: array-like of shape (1, T, 1)
            - gamma: array-like of shape (N, K, D_f)
        """
        scores = self.Encoder.compute_norm_scores(inputs, cluster_reps=self.cluster_rep_set)

        return scores

    def compute_cluster_phenotypes(self):
        """
        Compute Cluster Phenotypes given cluster representations.
        """
        phens = self.Predictor(self.cluster_rep_set).numpy()

        return phens

    def clus_assign(self, x):
        """
        Compute cluster assignments given input data X.

        Params:
        - X: array-like of shape (bs, T, D_f)

        Returns:
        - clus_pred: array-like of shape (bs, ) with corresponding cluster assignment.
            """

        # pi = self.Identifier(self.Encoder(x)).numpy()
        #######################更改成分批预测：开始
        # pi = self.Identifier(self.Encoder(x)).numpy()
        def BatchNext(x_train, batch_size, iters, idx):
            # idx = idx[:int(batch_size)]
            start_idx = batch_size * iters
            idx = idx[start_idx:start_idx + int(batch_size)]
            batch_x = [x_train[i] for i in idx]
            batch_x = np.array(batch_x)
            return batch_x

        pi = []
        val_idx = np.arange(0, x.shape[0])
        for iter in range(int(len(x) / 128) + 1):
            batch_x = BatchNext(x, 128, iter, val_idx)
            pi_batch = self.Identifier(self.Encoder(batch_x)).numpy()
            pi.extend(pi_batch)
        print("shape of y_val,", np.shape(pi))
        #######################更改成分批预测：结束

        clus_pred = np.argmax(pi, axis=1)

        return clus_pred

    def get_cluster_reps(self):
        return self.cluster_rep_set.numpy()

    def compute_pis(self, x):
        """Obtain cluster assignment probabilities."""

        # pis = self.Identifier(self.Encoder(x)).numpy()
        #######################更改成分批预测：开始
        #         pis = self.Identifier(self.Encoder(x)).numpy()
        def BatchNext(x_train, batch_size, iters, idx):
            # idx = idx[:int(batch_size)]
            start_idx = batch_size * iters
            idx = idx[start_idx:start_idx + int(batch_size)]
            batch_x = [x_train[i] for i in idx]
            batch_x = np.array(batch_x)
            return batch_x

        pis = []
        val_idx = np.arange(0, x.shape[0])
        for iter in range(int(len(x) / 128) + 1):
            batch_x = BatchNext(x, 128, iter, val_idx)
            pis_batch = self.Identifier(self.Encoder(batch_x)).numpy()
            pis.extend(pis_batch)
        print("shape of y_val,", np.shape(pis))
        #######################更改成分批预测：结束
        return pis

    def get_config(self):
        """Update configuration for layer."""
        config = {}

        # Update configuration
        config.update({f"{self.name}-num_clusters": self.K,
                       f"{self.name}-latent_dim": self.latent_dim,
                       f"{self.name}-output_dim": self.output_dim,
                       f"{self.name}-seed": self.seed,
                       f"{self.name}-alpha_1": self.alpha_1,
                       f"{self.name}-alpha_2": self.alpha_2,
                       f"{self.name}-alpha_3": self.alpha_3,
                       f"{self.name}-beta": self.beta,
                       f"{self.name}-regulariser": self.regulariser,
                       f"{self.name}-dropout": self.dropout})

        # Update configuration for each model
        config.update(self.Encoder.get_config())
        config.update(self.Predictor.get_config())
        config.update(self.Identifier.get_config())

        return config


CAMELOT_INPUT_PARAMS = ["num_clusters", "latent_dim", "seed", "output_dim", "name", "alpha_1", "alpha_2", "alpha_3",
                        "beta",
                        "regulariser_params", "dropout", "encoder_params", "identifier_params",
                        "predictor_params"]


class Model(CAMELOT):
    """
    Model Class Wrapper for CAMELOT with train and analyse methods.
    """

    def __init__(self, data_info: dict, model_config: dict, training_config: dict):
        """
        Initialise object with model configuration.

        Params:
        - data_info: dict, contains basic data information, objects and properties
        - kwargs: other model configuration parameters
        """
        # Get output dimension
        output_dim = data_info["y"][-1].shape[-1] - 2

        # Useful information
        relevant_params = {key: value for key, value in model_config.items() if key in CAMELOT_INPUT_PARAMS}
        self.model_config = {"output_dim": output_dim, **relevant_params}
        self.callback_lst = None
        self.run_num = 1
        self.model_name = "CAMELOT"

        # Initialise training parameters
        self.training_params = training_config

        super().__init__(**self.model_config)

        self.build_model(data_info)


    def build_model(self, data_info):
        """
        Build all model parameters.

        Params:
        - data_info: dictionary with data information and parameters.

        Returns:
            if applicable, model built.
        """

        # Get data useful info
        data_name = data_info["data_load_config"]["data_name"]
        cbck_str, patience_epochs = self.training_params["cbck_str"], self.training_params["patience_epochs"]
        lr = self.training_params["lr"]

        # Unpack relevant data information
        X_train, X_val, X_test = data_info["X"]
        y_train, y_val, y_test = data_info["y"]

        # Initialise model
        self.build(X_train.shape)

        # Load optimizer
        optimizer = optimizers.Adam(learning_rate=lr)
        self.compile(optimizer=optimizer, run_eagerly=True)

        # Load Checkpoint
        callbacks, run_num = model_utils.get_callbacks((X_val, y_val), data_name=data_name, track_loss="L_pred",
                                                       other_cbcks=cbck_str, patience_epochs=patience_epochs,
                                                       early_stop=True, lr_scheduler=True, tensorboard=True)

        # Update run num
        self.run_num = run_num
        self.callback_lst = callbacks

        return None

    def train(self, data_info, lr_init: float = 0.001, lr: float = 0.001, epochs_init_1: int = 100,
              epochs_init_2: int = 100, epochs: int = 100,
              bs: int = 32, patience_epochs: int = 200, gpu: Union[str, None] = None, **kwargs):
        """
        Fit method for training CAMELOT model.

        Params:
        - data_info: dictionary with data information and parameters.
        - lr_init": str, learning rate for initialisation training (default=0.001)
        - "lr": learning rate for training (default = 0.001)
        - "epochs_init": number of epochs to train initialisation (default = 100)
        - "epochs": number of epochs for main.py training (default = 100)
        - "bs": batch size (default = 32)
        - "patience_epochs": int, maximum number of epochs to wait for improvement. (default=200)
        - gpu: str or None indicating how to use gpu training. If None, then no gpu is used. If "strategy", then
        parallelise. Otherwise, regular GPU training.
        """
        self.training_params.update({
            "lr_init": lr_init,
            "lr": lr,
            "epochs_init_1": epochs_init_1,
            "epochs_init_2": epochs_init_2,
            "epochs": epochs,
            "bs": bs,
            "patience_epochs": patience_epochs,
            "gpu": gpu
        })

        # Unpack relevant data information
        X_train, X_val, X_test = data_info["X"]
        y_train, y_val, y_test = data_info["y"]

        # Train model on initialisation procedure
        train_data = X_train, y_train
        val_data = X_val, y_val

        print("-" * 20, "\n", "Initialising Model", sep="\n")
        self.initialise_model(data=train_data, val_data=val_data, epochs_1=epochs_init_1, epochs_2=epochs_init_2,
                              learning_rate=lr_init, batch_size=bs, patience_epochs=patience_epochs, training_rate=lr)

        ######初始化后的模型预测表现：
        print("After Initialise : Enter into results evaluation part...")

        def BatchNext(x_train, batch_size, iters, idx):
            # idx = idx[:int(batch_size)]
            start_idx = batch_size * iters
            idx = idx[start_idx:start_idx + int(batch_size)]
            batch_x = [x_train[i] for i in idx]
            batch_x = np.array(batch_x)
            return batch_x

        output_test = []
        val_idx = np.arange(0, X_test.shape[0])
        for iter in range(int(len(X_test) / 128) + 1):
            batch_x = BatchNext(X_test, 128, iter, val_idx)
            output_test_batch = self.Predictor(self.Encoder(batch_x)).numpy()
            output_test.extend(output_test_batch[:, :-1])
        output_test = np.stack(output_test)
        threadhold = [0.5] * output_test.shape[1]
        for j in range(output_test.shape[1]):
            temp_y_true = y_test[:, j]
            temp_y_pred = np.squeeze(output_test[:, j])
            print("j:", j)
            print("np.mean(temp_y_pred)", np.mean(temp_y_pred))
            print("np.median(temp_y_pred)", np.median(temp_y_pred))
            print("AUROC", roc_auc_score(temp_y_true, temp_y_pred, average=None))
            # Convert input arrays to categorical labels
            labels_true = np.squeeze(temp_y_true)
            labels_pred = (temp_y_pred > threadhold[j]).astype(int)
            # Compute F1
            print("f1,", f1_score(labels_true, labels_pred))
            # Compute Recall
            print("rec,", recall_score(labels_true, labels_pred))
            # Compute Precision
            print("prec,", precision_score(labels_true, labels_pred))
            # Compute ARI
            print("ari,", adjusted_rand_score(labels_true, labels_pred))
            # Compute NMI
            print("nmi,", normalized_mutual_info_score(labels_true, labels_pred))

        # Main Training phase
        print("-" * 20, "\n", "STARTING MAIN TRAINING PHASE")
        def filter_batches(x_batch, y_batch):
            return tf.math.reduce_sum(y_batch[:,-1])!=0
        train_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(1000).batch(bs).filter(filter_batches)
        val_data = tf.data.Dataset.from_tensor_slices(val_data).shuffle(1000).batch(bs).filter(filter_batches)

        # Disable Autoshard
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data, val_data = train_data.with_options(options), val_data.with_options(options)

        # Fit model
        history = self.fit(train_data, validation_data=val_data, epochs=epochs,
                           verbose=2, callbacks=self.callback_lst)
        #####保存模型
        model_fd = f"{self.model_name}/run{self.run_num}/saved_model/"
        if not os.path.exists(model_fd):
            os.makedirs(model_fd)
        self.save_weights(model_fd)

        return history

    def analyse(self, data_info):
        """
        Evaluation method for computing output results.

        Params:
        - data_info: dictionary with data information.

        Returns:
            - y_pred: dataframe of shape (N, output_dim) with outcome probability prediction.
            - outc_pred: Series of shape (N, ) with predicted outcome based on most likely outcome prediction.
            - y_true: dataframe of shape (N, output_dim) ith one-hot encoded true outcome.
            - pis_pred: dataframe of shape (N, K) of cluster probability assignment.
            - clus_pred: Series of shape (N, ) with cluster assignment based on most likely cluster probability.
            - clus_representations: Numpy array of shape (K, latent_dim) with corresponding luster representation
            vectors.
            - clus_phenotypes: DataFrame of shape (K, output_dim) with predicted cluster outcome probability.

        Saves a variety of model information, as well.
        """

        # Unpack test data
        _, _, X_test = data_info["X"]
        _, _, y_test = data_info["y"]
        data_properties = data_info["data_properties"]
        data_load_config = data_info["data_load_config"]

        # Source outcome names and patient id info
        id_info = data_info["ids"][-1]
        # pat_ids = id_info[:, 0, 0]
        pat_ids = id_info
        outc_dims = data_properties["outc_names"]
        data_name = data_load_config["data_name"]

        # Define save_fd
        # save_fd = f"run{self.run_num}/results/{self.model_name}/{data_name}/"
        # track_fd = f"run{self.run_num}/experiments/{self.model_name}/{data_name}/"
        save_fd = f"{self.model_name}/run{self.run_num}/results/"
        track_fd = f"{self.model_name}/run{self.run_num}/experiments/"


        if not os.path.exists(save_fd):
            os.makedirs(save_fd)

        if not os.path.exists(track_fd):
            os.makedirs(track_fd)


        # Other useful definitions
        K = self.K
        cluster_names = list(range(1, K + 1))

        # output_test = self.Predictor(self.Encoder(X_test)).numpy()

        #######################更改成分批预测：开始
        # output_test = self.Predictor(self.Encoder(X_test)).numpy()
        def BatchNext(x_train, batch_size, iters, idx):
            # idx = idx[:int(batch_size)]
            start_idx = batch_size * iters
            idx = idx[start_idx:start_idx + int(batch_size)]
            batch_x = [x_train[i] for i in idx]
            batch_x = np.array(batch_x)
            return batch_x

        output_test = []
        val_idx = np.arange(0, X_test.shape[0])
        for iter in range(int(len(X_test) / 128) + 1):
            batch_x = BatchNext(X_test, 128, iter, val_idx)
            output_test_batch = self.Predictor(self.Encoder(batch_x)).numpy()
            output_test.extend(output_test_batch)
        print("shape of output_test,", np.shape(output_test))
        #######################更改成分批预测：结束

        # # Firstly, compute predicted y estimates
        # y_pred = pd.DataFrame(output_test, index=pat_ids, columns=outc_dims)
        # outc_pred = pd.Series(np.argmax(output_test, axis=-1), index=pat_ids)
        # y_true = pd.DataFrame(y_test, index=pat_ids, columns=outc_dims)
        #
        # # Secondly, compute predicted cluster assignments
        # pis_pred = pd.DataFrame(self.compute_pis(X_test), index=pat_ids, columns=cluster_names)
        # clus_pred = pd.Series(self.clus_assign(X_test), index=pat_ids)

        # Firstly, compute predicted y estimates
        y_pred = pd.DataFrame(output_test, index=pat_ids, columns=outc_dims + ['Surv'])
        outc_pred = pd.Series(np.argmax(output_test, axis=-1), index=pat_ids)
        # y_true = pd.DataFrame(y_test, columns=['type1','type2','type3','type4','T','E'])
        y_true = pd.DataFrame(y_test, index=pat_ids, columns=outc_dims + ['T', 'E'])

        # Secondly, compute predicted cluster assignments
        pis_pred = pd.DataFrame(self.compute_pis(X_test), columns=cluster_names)
        clus_pred = pd.Series(self.clus_assign(X_test))

        # Thirdly, compute cluster phenotype information
        clus_phenotypes = pd.DataFrame(self.compute_cluster_phenotypes(), index=cluster_names,
                                       columns=outc_dims + ['Surv'])
        cluster_rep_set = self.get_cluster_reps()

        # Fourth, save model init losses
        init_loss_1 = self.enc_pred_loss_tracker
        init_loss_2 = self.iden_loss_tracker
        init_loss_1.index.name, init_loss_2.index.name = "epoch", "epoch"

        # Fifth, compute attention scores
        # alpha, beta, gamma = self.compute_unnorm_attention_weights(X_test)
        # alpha_norm, beta_norm, gamma_norm = self.compute_norm_attention_weights(X_test)

        # Sixth, get configuration
        all_model_config = self.get_config()

        # ----------------------------- Save Output Data --------------------------------
        # Useful objects
        y_pred.to_csv(save_fd + "y_pred.csv", index=True, header=True)
        outc_pred.to_csv(save_fd + "outc_pred.csv", index=True, header=True)
        y_true.to_csv(save_fd + "y_true.csv", index=True, header=True)
        pis_pred.to_csv(save_fd + "pis_pred.csv", index=True, header=True)
        clus_pred.to_csv(save_fd + "clus_pred.csv", index=True, header=True)
        clus_phenotypes.to_csv(save_fd + "clus_phenotypes.csv", index=True, header=True)
        np.save(save_fd + "cluster_representations", cluster_rep_set, allow_pickle=True)

        # save init losses
        init_loss_1.to_csv(save_fd + "enc_pred_init_loss.csv", index=True, header=True)
        init_loss_2.to_csv(save_fd + "iden_init_loss.csv", index=True, header=True)

        # # Save attention weights
        # np.savez(save_fd + "unnorm_weights", alpha=alpha, beta=beta, gamma=gamma)
        # np.savez(save_fd + "norm_weights", alpha=alpha_norm, beta=beta_norm, gamma=gamma_norm)

        # save model parameters
        save_params = {**data_info["data_load_config"], **self.model_config, **self.training_params}
        with open(save_fd + "config.json", "w+") as f:
            json.dump(save_params, f, indent=4)

        with open(track_fd + "config.json", "w+") as f:
            json.dump(save_params, f, indent=4)

        with open(save_fd + "model_config.json", "w+") as f:
            json.dump(all_model_config, f, indent=4)

        with open(track_fd + "model_config.json", "w+") as f:
            json.dump(all_model_config, f, indent=4)



        # Return objects
        outputs_dic = {
            "y_pred": y_pred, "class_pred": outc_pred, "y_true": y_true, "pis_pred": pis_pred, "clus_pred": clus_pred,
            "clus_representations": cluster_rep_set, "clus_phenotypes": clus_phenotypes,
            "init_loss_enc_pred": init_loss_1, "init_loss_iden": init_loss_2,
            # "attention_unnorm": (alpha, beta, gamma),
            # "attention_norm": (alpha_norm, beta_norm, gamma_norm),
            "logs": track_fd + "logs",
            "save_fd": save_fd, "model_config": self.model_config
        }

        # Print Data
        print(f"\n\n Experiments saved under {track_fd} and {save_fd}")

        return outputs_dic
