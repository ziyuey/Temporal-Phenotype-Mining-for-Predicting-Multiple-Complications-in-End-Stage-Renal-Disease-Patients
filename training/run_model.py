"""
Single Run Training

Date Last updated: 24 Jan 2022
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import json
import matplotlib.pyplot as plt

from src.data_processing.data_loader import data_loader
import src.models.model_utils as model_utils
from src.results.main import evaluate,evaluate_without_survival
import src.visualisation.main as vis_main
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)
print(tf.test.is_gpu_available())
print("tf version",tf.__version__)

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_config_model_name", help="display a name of a training config", type=str,
                    default='CAMELOT')
# ACTPC
parser.add_argument("--model_config_num_clusters", help="display a name of a model config", type=int, default=14)
parser.add_argument("--model_config_latent_dim", help="display a name of a model config", type=int, default=181)
parser.add_argument("--model_config_seed", help="display a name of a model config", type=int, default=1111)
parser.add_argument("--model_config_name", help="display a name of a model config", type=str, default="ABL1")
parser.add_argument("--model_config_alpha_1", help="display a name of a model config", type=float, default=1)
parser.add_argument("--model_config_alpha_2", help="display a name of a model config", type=float, default=0)
parser.add_argument("--model_config_alpha_3", help="display a name of a model config", type=float, default=0)
parser.add_argument("--model_config_beta", help="display a name of a model config", type=float, default=0.1)
# loss = l_crit + self.alpha_1 * l_dist + self.alpha_2 * l_pat_entr + self.alpha_3 * l_clus_entr + \
#        self.beta * l_clus
parser.add_argument("--model_config_regulariser_params", help="display a name of a model config", type=list,
                    default=[0.01, 0.01])
parser.add_argument("--model_config_dropout", help="display a name of a model config", type=float, default=0.1)
parser.add_argument("--model_config_encoder_params", help="display a name of a model config", default={"hidden_layers":4,\
                                                                                                       "hidden_nodes": 64})
parser.add_argument("--model_config_identifier_params", help="display a name of a model config", type=list, default={
    "hidden_layers": 2,
    "hidden_nodes": 32
})
parser.add_argument("--model_config_predictor_params", help="display a name of a model config", type=list, default={
    "hidden_layers": 3,
    "hidden_nodes": 128
})

parser.add_argument("--training_config_lr_init", help="display a name of a model config", type=float, default=0.001)
parser.add_argument("--training_config_lr", help="display a name of a model config", type=float, default=0.0000001)
parser.add_argument("--training_config_epochs_init_1", help="display a name of a model config", type=int, default=0)
parser.add_argument("--training_config_epochs_init_2", help="display a name of a model config", type=int, default=0)
parser.add_argument("--training_config_epochs", help="display a name of a model config", type=int, default=10)
parser.add_argument("--training_config_bs", help="display a name of a model config", type=int, default=128)
parser.add_argument("--training_config_patience_epochs", help="display a name of a model config", type=int, default=20)
parser.add_argument("--training_config_gpu", help="display a name of a model config", default=None)

args = parser.parse_args()


def main():
    # ---------------------------- Load Configurations --------------------------------------
    model_config = {
        "model_name": args.model_config_model_name,
        "num_clusters": args.model_config_num_clusters,
        "latent_dim": args.model_config_latent_dim,
        "seed": args.model_config_seed,
        "name": args.model_config_name,
        "alpha_1": args.model_config_alpha_1,
        "alpha_2": args.model_config_alpha_2,
        "alpha_3": args.model_config_alpha_3,
        "beta": args.model_config_beta,
        "regulariser_params": args.model_config_regulariser_params,
        "dropout": args.model_config_dropout,
        "encoder_params": args.model_config_encoder_params,
        "identifier_params": args.model_config_identifier_params,
        "predictor_params": args.model_config_predictor_params
    }
    training_config = {
        "lr_init": args.training_config_lr_init,
        "lr": args.training_config_lr,
        "epochs_init_1": args.training_config_epochs_init_1,
        "epochs_init_2": args.training_config_epochs_init_2,
        "epochs": args.training_config_epochs,
        "bs": args.training_config_bs,
        "cbck_str": "",
        "patience_epochs": args.training_config_patience_epochs,
        "gpu": args.training_config_gpu
    }
    print("model_config",model_config)
    print("training_config", training_config)


    # with open("src/training/model_config.json", "r") as f:
    #     model_config = json.load(f)
    #     f.close()
    #
    # with open("src/training/training_config.json", "r") as f:
    #     training_config = json.load(f)
    #     f.close()

    # ----------------------------- Load Data and Plot summary statistics -------------------------------

    "Data Loading."
    classificationOutcomes = ['class_hypertension', 'class_anemia', 'class_heart', 'class_diabetes', 'class_infect']
    data_info = data_loader(seed = 9940,classificationOutcomes=classificationOutcomes)

    "Visualise Data Properties"
    # vis_main.visualise_data_groups(data_info)

    # -------------------------- Loading and Training Model -----------------------------

    "Load model and fit"
    print("\n\n\n\n")
    model = model_utils.get_model_from_str(data_info=data_info, model_config=model_config,
                                           training_config=training_config)


    model_fd = f'/data/data_yangzy/BP_paper/Codes/Model/my_camelot_complication/run29/saved_model/'
    model.load_weights(model_fd)

    # Train model
    history = model.train(data_info=data_info, **training_config)

    "Compute results on test data"
    outputs_dic = model.analyse(data_info)
    print(outputs_dic.keys())




    # -------------------------------------- Evaluate Scores --------------------------------------

    "Evaluate scores on the resulting models. Note X_test is converted back to input dimensions."
    scores = evaluate_without_survival(**outputs_dic, data_info=data_info, avg='binary')
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
    # vis_main.visualise_cluster_assignment(**outputs_dic, data_info=data_info)
    # print("Finished Clus assignments where relevant")

    # "Attention maps where relevant"
    # vis_main.visualise_attention_maps(**outputs_dic, data_info=data_info)
    # print("Finished Attention maps where relevant")

    # # Load tensorboard if exists
    # vis_main.load_tensorboard(**outputs_dic, data_info=data_info)
    # print("Finished Load tensorboard if exists")

    # Show Figures
    # plt.show(block=False)

    print("Analysis Complete.")


if __name__ == "__main__":
    main()
