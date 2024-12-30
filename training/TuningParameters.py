import json
import matplotlib.pyplot as plt

from src.data_processing.data_loader import data_loader
import src.models.model_utils as model_utils
from src.results.main import evaluate
import src.visualisation.main as vis_main
import warnings
import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore")
import tensorflow as tf
import optuna

tf.compat.v1.enable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)
print(tf.test.is_gpu_available())
warnings.filterwarnings('ignore')

#
# cmd:
# CUDA_VISIBLE_DEVICES=1 python -m src.training.TuningParameters

def objective(trial):
    ##############################
    model_config_model_name = 'camelot'
    # classificationOutcomes = trial.suggest_categorical("classificationOutcomes", [
    #     ['class_hypertension', 'class_anemia', 'class_heart', 'class_diabetes', 'class_infect', 'class_k', 'class_bone'],
    #     ['class_hypertension', 'class_anemia', 'class_heart', 'class_diabetes', 'class_infect', ],
    #     ['class_hypertension', 'class_anemia', 'class_heart',]
    # ])
    classificationOutcomes = ['class_hypertension', 'class_anemia', 'class_heart', 'class_diabetes', 'class_infect']

    '''
    ['patient_ID', 'time_to_start', 'death_label_t', 'death_label_e',
     'class_hypertension', 'class_anemia', 'class_heart', 'class_diabetes',
     'class_infect', 'class_vascular', 'class_kidney_failure', 'class_liver',
     'class_k', 'class_digestive', 'class_bone', 'class_breath',
     'class_nerves', 'class_thrombocytopenia', 'class_malnutrition',
     'class_renal_amyloidosis', 'train_test_flag']
     '''
    model_config_num_clusters = trial.suggest_int("model_config_num_clusters", 6, 25)
    # model_config_num_clusters = 14
    model_config_latent_dim = trial.suggest_int("model_config_latent_dim", 128, 512)
    # model_config_latent_dim = 181
    # model_config_seed = trial.suggest_int("model_config_seed", 1, 50000)
    model_config_seed = 1111
    model_config_name = "ABL1"
    # model_config_alpha_1 = trial.suggest_uniform('model_config_alpha_1', 0.0, 0.01)
    # model_config_alpha_1 = trial.suggest_categorical("model_config_alpha_1",[0, 1])
    model_config_alpha_1 = 1
    model_config_alpha_2 = 0.01
    # model_config_alpha_3 = trial.suggest_uniform('model_config_alpha_3', 0.0, 0.01)
    model_config_alpha_3 = 0
    # model_config_beta = trial.suggest_loguniform('model_config_beta',
    #                                                    1e-2, 1e-1)
    model_config_beta= 0.01
    # loss = l_crit + self.alpha_1 * l_dist + self.alpha_2 * l_pat_entr + self.alpha_3 * l_clus_entr + \
    #        self.beta * l_clus

    # model_config_regulariser_params_value = trial.suggest_loguniform('model_config_regulariser_params_value',
    #                                                                  1e-4, 1e-2)
    # model_config_regulariser_params = [model_config_regulariser_params_value,model_config_regulariser_params_value]
    model_config_regulariser_params = [0.01, 0.01]

    model_config_dropout = trial.suggest_uniform('model_config_dropout', 0.0, 0.5)
    # model_config_dropout = 0.1

    model_config_encoder_params_hidden_layers = trial.suggest_int("model_config_encoder_params_hidden_layers", \
                                                                  2, 4)

    model_config_encoder_params_hidden_nodes = trial.suggest_categorical("model_config_encoder_params_hidden_nodes",[32,64,128])
    model_config_encoder_params = {"hidden_layers": model_config_encoder_params_hidden_layers,\
                                   "hidden_nodes": model_config_encoder_params_hidden_nodes}
    # model_config_encoder_params = {
    #     "hidden_layers": 4,
    #     "hidden_nodes": 64
    # }

    model_config_identifier_params_hidden_layers = trial.suggest_int("model_config_identifier_params_hidden_layers", \
                                                                  2, 4)
    model_config_identifier_params_hidden_nodes = trial.suggest_categorical("model_config_identifier_params_hidden_nodes",[32,64,128])
    model_config_identifier_params = {"hidden_layers": model_config_identifier_params_hidden_layers, \
                                   "hidden_nodes": model_config_identifier_params_hidden_nodes}
    # model_config_identifier_params = {
    #     "hidden_layers": 2,
    #     "hidden_nodes": 32
    # }

    model_config_predictor_params_hidden_layers = trial.suggest_int("model_config_predictor_params_hidden_layers", \
                                                                     2, 4)
    model_config_predictor_params_hidden_nodes = trial.suggest_categorical("model_config_predictor_params_hidden_nodes",[32,64,128])
    model_config_predictor_params = {"hidden_layers": model_config_predictor_params_hidden_layers, \
                                      "hidden_nodes": model_config_predictor_params_hidden_nodes}
    # model_config_predictor_params = {
    #     "hidden_layers": 3,
    #     "hidden_nodes": 128
    # }
    training_config_lr_init = trial.suggest_loguniform('training_config_lr_init',
                                                                     1e-7, 1e-4)
    # training_config_lr_init = 1.2964512201227502e-05
    # training_config_lr = trial.suggest_loguniform('training_config_lr',
    #                                                    1e-7, 1e-2)
    training_config_lr = trial.suggest_loguniform('training_config_lr_init', 1e-8, 1e-5)
    # training_config_lr = 6.728968591078265e-05
    training_config_epochs_init_1 = 45
    training_config_epochs_init_2 = 45
    training_config_epochs = 45
    # training_config_bs = trial.suggest_int("training_config_bs", 64, 128)
    training_config_bs = 128
    training_config_patience_epochs = 20
    training_config_gpu = 0

    model_config = {
        "model_name": model_config_model_name,
        "num_clusters": model_config_num_clusters,
        "latent_dim": model_config_latent_dim,
        "seed": model_config_seed,
        "name": model_config_name,
        "alpha_1": model_config_alpha_1,
        "alpha_2": model_config_alpha_2,
        "alpha_3": model_config_alpha_3,
        "beta": model_config_beta,
        "regulariser_params": model_config_regulariser_params,
        "dropout": model_config_dropout,
        "encoder_params": model_config_encoder_params,
        "identifier_params": model_config_identifier_params,
        "predictor_params": model_config_predictor_params
    }
    training_config = {
        "lr_init": training_config_lr_init,
        "lr": training_config_lr,
        "epochs_init_1": training_config_epochs_init_1,
        "epochs_init_2": training_config_epochs_init_2,
        "epochs": training_config_epochs,
        "bs": training_config_bs,
        "cbck_str": "",
        "patience_epochs": training_config_patience_epochs,
        "gpu": training_config_gpu
    }
    print("model_config",model_config)
    print("training_config", training_config)
    data_info = data_loader(seed = 1024,classificationOutcomes=classificationOutcomes)

    # -------------------------- Loading and Training Model -----------------------------

    "Load model and fit"
    print("\n\n\n\n")
    model = model_utils.get_model_from_str(data_info=data_info, model_config=model_config,
                                           training_config=training_config)

    # Train model
    history = model.train(data_info=data_info, **training_config)
    "Compute results on test data"
    outputs_dic = model.analyse(data_info)
    print(outputs_dic.keys())
    # -------------------------------------- Evaluate Scores --------------------------------------

    "Evaluate scores on the resulting models. Note X_test is converted back to input dimensions."
    scores = evaluate(**outputs_dic, data_info=data_info, avg='weighted')
    print("Finished Evaluate scores on the resulting models")

    # ------------------------ Results Visualisations --------------------------
    "Learnt Group averages"

    # Cluster Groups understanding where relevant
    vis_main.visualise_cluster_groups(**outputs_dic, data_info=data_info)
    print("Finished Cluster Groups understanding where relevant")

    # "Losses where relevant"
    vis_main.plot_losses(history=history, **outputs_dic, data_info=data_info)
    print("Finished Losses where relevant")

    # # "Clus assignments where relevant"
    # vis_main.visualise_cluster_assignment(**outputs_dic, data_info=data_info)
    # print("Finished Clus assignments where relevant")

    # "Attention maps where relevant"
    # vis_main.visualise_attention_maps(**outputs_dic, data_info=data_info)
    # print("Finished Attention maps where relevant")

    print("Analysis Complete.")

    return np.mean(scores["ROC-AUC"])



if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))