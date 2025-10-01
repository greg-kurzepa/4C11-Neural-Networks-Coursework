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

# if torch.cuda.is_available():
#      dev = torch.device("cuda")
#      print("CUDA available. Using CUDA.")
# else:
#      dev = torch.device("cpu")
#      print("CUDA not available. Using CPU.")
dev = torch.device("cpu")

def product_dict2(**kwargs):
    """Finds the Cartesian product of the given 2-nested dictionary (dictionary inside a dictionary). Used to generate hyperparameter combinations"""
    keys = kwargs.keys()
    for instance in itertools.product(*[product_dict(**value) for value in kwargs.values()]):
        yield dict(zip(keys, instance))

def product_dict(**kwargs):
    """Finds the Cartesian product of the given dictionary.
    see https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists"""
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

def run_experiment(config, hyperparams):
    # current time in string form to use in filenames and to save to Weights&Biases
    start_time = time.strftime('%Y%m%d-%H%M%S')

    # Initialise neural network architecture
    model = config.model(**hyperparams["model_hyperparams"]).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)

    # initialise datalaoders
    train_dl = config.train_dl_initialiser.initialise(hyperparams["hyperparams"]["batch_size"], dev)
    valid_dl = config.valid_dl_initialiser.initialise(hyperparams["hyperparams"]["batch_size"], dev)
    
    # output model ONNX file to view in Netron
    try:
        batch = model.example_input_array.to(dev)
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
    except torch.onnx.errors.UnsupportedOperatorError:
        print("ONNX export failed. Unsupported operator.")

    # Optimiser and scheduler
    optimiser = torch.optim.AdamW(model.parameters(), **hyperparams["optimiser_hyperparams"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=600, gamma=0.6)

    # initialise Weights&Biases
    wandb_config = {
            "config_name": config.name,
            "time" : start_time,
            "n_params" : n_params,
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
    _train.train(model=model, loss_function=config.loss_function, optimiser=optimiser, train_dl=train_dl, valid_dl=valid_dl, dev=dev, scheduler=scheduler, **hyperparams["hyperparams"], show_plot=False, log_wandb=config.log_wandb)
    torch.save(model.state_dict(), os.path.join(config.params_folder, f"{start_time}.torchparams"))
    if config.log_wandb: wandb.finish()

# Select config to use
import _config_q2
config = _config_q2.CONFIGS["cw2_q2_fno"]()

# Iterate over all possible hypermarameter combinations
hyperparams_iterator = product_dict2(**config.HYPERPARAMS)
for hyperparams in hyperparams_iterator:
    run_experiment(config, hyperparams)