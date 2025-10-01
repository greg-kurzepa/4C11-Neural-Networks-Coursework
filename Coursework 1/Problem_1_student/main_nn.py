import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.adamax
import torch.utils.data as Data
import numpy as np
import h5py
import wandb
import time
import itertools
import os
import warnings
from torchinfo import summary

import _train

def product_dict2(**kwargs):
    """Finds the Cartesian product of the given 2-nested dictionary (dictionary inside a dictionary)"""
    keys = kwargs.keys()
    for instance in itertools.product(*[product_dict(**value) for value in kwargs.values()]):
        yield dict(zip(keys, instance))

def product_dict(**kwargs):
    """Finds the Cartesian product of the given dictionary.
    see https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists"""
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

if torch.cuda.is_available():
     dev = torch.device("cuda")
     print("CUDA available. Using CUDA.")
else:
     dev = torch.device("cpu")
     print("CUDA not available. Using CPU.")

def run_experiment(config, hyperparams):
    # Initialise neural network architecture and output a summary
    model = config.model(**hyperparams["model_hyperparams"]).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)
    # summary(model, input_size=(batch_size, flattened_size))

    # Loss function and optimiser
    optimiser = torch.optim.AdamW(model.parameters(), **hyperparams["optimiser_hyperparams"])

    # current time in string form to use in trained parameters filename and to save to Weights&Biases
    start_time = time.strftime('%Y%m%d-%H%M%S')

    epochs = hyperparams["hyperparams"]["epochs"]
    batch_size = hyperparams["hyperparams"]["batch_size"]
    clip_value = hyperparams["hyperparams"]["clip_value"]
    train_dl = config.train_dl_initialiser.initialise(batch_size, dev)
    valid_dl = config.valid_dl_initialiser.initialise(batch_size, dev)

    # output model ONNX file to view in Netron
    try:
        batch = next(iter(train_dl))[0][0].unsqueeze(0)
        with warnings.catch_warnings(): # ignore warning that TrainingMode is deprecated
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model,                  # model to export
                batch,               # inputs of the model,
                f"onnx\\model-{config.name}-{start_time}.onnx",        # filename of the ONNX model
                input_names=["input"],  # Rename inputs for the ONNX model
                output_names=["output"],
                training=torch.onnx.TrainingMode.TRAINING
            )
    except:
        print("ONNX export failed")

    # initialise Weights&Biases
    wandb_config = {
            "config_name": config.name,
            "time" : start_time,
    }
    wandb_config.update(hyperparams["model_hyperparams"])
    wandb_config.update(hyperparams["optimiser_hyperparams"])
    wandb_config.update(hyperparams["hyperparams"])
    
    if config.log_wandb: wandb.init(
        # set the wandb project where this run will be logged
        project = config.wandb_project_name,
        # track hyperparameters and metadata
        config = wandb_config
    )

    # record gradients & parameters
    if config.log_wandb: wandb.watch(
        model,
        criterion = config.loss_function,
        log = "all",
        log_freq = 1,
    )

    # train model, save its trained parameters, and finish Weights&Biases
    _train.train(model, config.loss_function, optimiser, train_dl, valid_dl, epochs, batch_size, dev, show_plot=False, log_wandb=config.log_wandb, clip_value=clip_value)
    torch.save(model.state_dict(), os.path.join(config.params_folder, f"{start_time}.torchparams"))
    if config.log_wandb: wandb.finish()

# Select config to use
import configs._config_q1 as _config
config = _config.CONFIGS["cw1_q1_lstm"]()

# Iterate over all possible hypermarameter combinations
hyperparams_iterator = product_dict2(**config.HYPERPARAMS)
for hyperparams in hyperparams_iterator:
    run_experiment(config, hyperparams)