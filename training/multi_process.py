"""
Single Run Training

Date Last updated: 24 Jan 2022
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import subprocess

import multiprocessing
import time
import json
import matplotlib.pyplot as plt

from src.data_processing.data_loader import data_loader
import src.models.model_utils as model_utils
from src.results.main import evaluate
import src.visualisation.main as vis_main
import warnings
from multiprocessing.pool import Pool

warnings.filterwarnings("ignore")
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)
print(tf.test.is_gpu_available())


def task(data_info, model_config, training_config):
    print('Starting a new process with new parameters...')
    print("Confirm model_config", model_config)
    print("Confirm training_config", training_config)

    "Visualise Data Properties"
    # vis_main.visualise_data_groups(data_info)

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

    # "Clus assignments where relevant"
    vis_main.visualise_cluster_assignment(**outputs_dic, data_info=data_info)
    print("Finished Clus assignments where relevant")

    # "Attention maps where relevant"
    vis_main.visualise_attention_maps(**outputs_dic, data_info=data_info)
    print("Finished Attention maps where relevant")

    # Load tensorboard if exists
    vis_main.load_tensorboard(**outputs_dic, data_info=data_info)
    print("Finished Load tensorboard if exists")

    # # Show Figures
    # plt.show(block=False)

    print("Analysis Complete.")
    print('Finished sleeping')


if __name__ == "__main__":
    start_time = time.perf_counter()
    import multiprocess as multiprocessing

    multiprocessing.set_start_method('spawn')
    processes = []

    with open("src/training/data_config.json", "r") as f:
        data_config = json.load(f)
        f.close()
    # "Data Loading."
    data_info = data_loader(**data_config)
    # ---------------------------- Load Configurations --------------------------------------
    model_name = 'camelot'
    latent_dim = 128
    model_config_name = "ABL1"
    model_config_alpha_1 = 1
    model_config_alpha_2 = 0.1
    model_config_alpha_3 = 1
    model_config_beta = 1
    model_config_regulariser_params = [0.01, 0.01]
    model_config_dropout = 0.1
    model_config_encoder_params = {"hidden_layers": 2, "hidden_nodes": 128}
    model_config_identifier_params = {
        "hidden_layers": 2,
        "hidden_nodes": 128
    }
    model_config_predictor_params = {
        "hidden_layers": 2,
        "hidden_nodes": 128
    }
    training_config_lr_init = 0.000001
    training_config_epochs_init_1 = 30
    training_config_epochs_init_2 = 30
    training_config_epochs = 50
    training_config_bs = 128
    training_config_patience_epochs = 10

    input_model_config = []
    input_training_config = []
    for num_clusters in [2, 3, 4, 5, 6]:
        for seed in [1111, 3618, 9253]:
            for training_config_lr in [0.001, 0.0001, 0.00001, 0.000001]:
                input_model_config.append({
                    "model_name": 'camelot',
                    "num_clusters": num_clusters,
                    "latent_dim": latent_dim,
                    "seed": seed,
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
                })
                input_training_config.append(
                    {
                        "lr_init": training_config_lr_init,
                        "lr": training_config_lr,
                        "epochs_init_1": training_config_epochs_init_1,
                        "epochs_init_2": training_config_epochs_init_2,
                        "epochs": training_config_epochs,
                        "bs": training_config_bs,
                        "cbck_str": "",
                        "patience_epochs": training_config_patience_epochs,
                        "gpu": 0
                    })

    ######constuct parameter pairs
    start_time = time.perf_counter()
    processes = []

    # Creates 10 processes then starts them
    for i in range(len(input_model_config)):
        temp_model_config = input_model_config[i]
        temp_training_config = input_training_config[i]
        print("temp_model_config",temp_model_config)
        print("temp_training_config", temp_training_config)
        p = multiprocessing.Process(target=task,args=(data_info, temp_model_config, temp_training_config,))
        p.start()
        processes.append(p)

    # Joins all the processes
    for p in processes:
        p.join()

    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time - start_time} seconds")

###parameters
#
# model_config_num_clusters
#
# model_config_latent_dim=128
#
# model_config_alpha_1
# model_config_alpha_2
# model_config_alpha_3
# model_config_beta
# model_config_regulariser_params = [0.01, 0.01]
# model_config_dropout = 0.1
#
# model_config_encoder_params = {"hidden_layers": 2,"hidden_nodes": 128}
# model_config_identifier_params = {
#     "hidden_layers": 2,
#     "hidden_nodes": 128
# }
# model_config_predictor_params = {
#     "hidden_layers": 2,
#     "hidden_nodes": 128
# }
#
# training_config_lr_init = 0.000005
# training_config_lr = 0.000005
#
# training_config_bs
#
# cmd = "git --version"
#
# returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
# print('returned value:', returned_value)
