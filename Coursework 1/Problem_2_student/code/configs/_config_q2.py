# to try: triplet learning.
# https://www.ibm.com/think/topics/few-shot-learning
# https://stats.stackexchange.com/questions/475655/in-training-a-triplet-network-i-first-have-a-solid-drop-in-loss-but-eventually

import torch
import h5py
import numpy as np
import _dataloader
import _model
from tqdm import tqdm

np.random.seed(0)

class InitialiseDataloader():
    # it will contain a function that creates the dataloader.
    # this function will be set by the user and will have 2 types of parameters that go in the dataloader:
    # 1. (variable length) are fixed and will be class members
    # 2. parameters that can vary in an experiment, for example batch size (rn just batch_size and dev)
    # For each experiment, this function will be called with the different experiment parameters.

    def __init__(self, init_fn, fixed_params):
        self.init_fn = init_fn
        self.fixed_params = fixed_params

    def initialise(self, batch_size, dev):
        return self.init_fn(batch_size, dev, **self.fixed_params)

class ExperimentConfig():
    def __init__(self, wandb_project_name, name,
                 train_dl_initialiser, valid_dl_initialiser,
                 model, loss_function,
                 OPTIMISER_HYPERPARAMS, MODEL_HYPERPARAMS, HYPERPARAMS,
                 test_dl_initialiser=None, log_wandb=True, params_folder="params"):
        
        self.train_dl_initialiser = train_dl_initialiser
        self.valid_dl_initialiser = valid_dl_initialiser
        self.test_dl_initialiser = test_dl_initialiser

        # Dictionaries containing the hyperparameters {names : [values]} to search through
        self.HYPERPARAMS = {
            "model_hyperparams" : MODEL_HYPERPARAMS,
            "optimiser_hyperparams" : OPTIMISER_HYPERPARAMS,
            "hyperparams" : HYPERPARAMS
        }

        self.model = model
        self.loss_function = loss_function

        self.log_wandb = log_wandb
        self.wandb_project_name = wandb_project_name
        self.name = name

        self.params_folder = params_folder

# This reads the matlab data from the .mat file provided
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

        self.load_apply = np.array(self.data['load_apply']).T
        self.result = np.array(self.data['result']).T

        self.load_apply = torch.tensor(self.load_apply, dtype=torch.float32)
        self.result = torch.tensor(self.result, dtype=torch.float32)

        p = torch.randperm(self.load_apply.shape[0]) # create indices to shuffle input and output in the same way
        self.load_apply = self.load_apply[p]
        self.result = self.result[p]

    def read_data(self):
        return self.load_apply, self.result
    
class NormaliseMeanStd():
    def __init__(self, data):
        self.mean = data.mean()
        self.std = data.std()
    
    def normalise(self, data):
        return (data - self.mean) / self.std
    
    def denormalise(self, data):
        return data * self.std + self.mean

def cw1_q2_fcnn():
    # Read data from .mat file
    path = "Data\\Eiffel_data.mat"
    data_reader = MatRead(path)
    load_apply, result = data_reader.read_data()
    assert load_apply.shape[0] == result.shape[0], "Number of samples in input and output data do not match"
    inputs, outputs = load_apply, result

    # normalise inputs
    normaliser = NormaliseMeanStd(inputs)
    inputs = normaliser.normalise(inputs)

    # Split data into train and test
    ntrain = round(inputs.shape[0]*0.8) # Specify the training data
    train_inputs = inputs[:ntrain]
    train_outputs = outputs[:ntrain]
    test_inputs = inputs[ntrain:]
    test_outputs = outputs[ntrain:]

    # Create data loaders
    fixed_params_train = {
        "in" : train_inputs,
        "out" : train_outputs,
        "drop_last" : True,
    }
    fixed_params_valid = {
        "in" : test_inputs,
        "out" : test_outputs,
        "drop_last" : False,
    }

    # drop_last means that the last batch will be dropped if it is not full. avoids large updates from small samples
    def init_dl(batch_size, dev, **fixed_params):
        set = torch.utils.data.TensorDataset(fixed_params["in"], fixed_params["out"])
        dl = _dataloader.WrappedDataLoader(
            torch.utils.data.DataLoader(set, batch_size, shuffle=True, drop_last=fixed_params["drop_last"]),
            lambda x, y: (x.to(dev), y.to(dev))
        )
        return dl
    
    train_dl_initialiser = InitialiseDataloader(init_dl, fixed_params_train)
    valid_dl_initialiser = InitialiseDataloader(init_dl, fixed_params_valid)

    loss_function = torch.nn.BCEWithLogitsLoss() # initialise it here. so torch.nn.MSELoss() for example.
    model = _model.FCNN

    # Optimiser hyperparameters to try. The model will be trained for all possible combinations of these.
    OPTIMISER_HYPERPARAMS = {
        "lr" : [0.00005],
        # "lr" : [0.0001, 0.00005],
    }

    # Define parameters for FCNN model
    MODEL_HYPERPARAMS = {
        "layer_sizes" : [[20, 500, 500, 1],],
        "do_batchnorm" : [False],
        # "dropout_p" : [0.5],
        # "dropout_layers" : [[1, 2]]
    }

    # Other hyperparameters to try. The model will be trained for all possible combinations of these.
    HYPERPARAMS = {
        "epochs" : [500],
        "batch_size" : [100],
        "clip_value" : [1.0],
        "track_accuracy" : [True],
        "accuracy_fn" : [lambda result, target: torch.sum((torch.nn.functional.sigmoid(result) >= 0.5) == target).item()],
    }

    return ExperimentConfig(
        "Coursework 1 Part 1 Deep FCNN", "cw1_q2_fcnn",
        train_dl_initialiser, valid_dl_initialiser,
        model, loss_function,
        OPTIMISER_HYPERPARAMS, MODEL_HYPERPARAMS, HYPERPARAMS,
        log_wandb=True, params_folder="code\\params",
    )

def cw1_q2_resnet():
    # Read data from .mat file
    path = "Data\\Eiffel_data.mat"
    data_reader = MatRead(path)
    load_apply, result = data_reader.read_data()
    assert load_apply.shape[0] == result.shape[0], "Number of samples in input and output data do not match"
    inputs, outputs = load_apply, result
    assert inputs.shape == (1000, 20), "Input data shape is not as expected"
    assert outputs.shape == (1000, 1), "Output data shape is not as expected"

    # normalise inputs
    normaliser = NormaliseMeanStd(inputs)
    inputs = normaliser.normalise(inputs)

    # pick out the test set and the training set
    # Split data into train and test
    ntrain = round(inputs.shape[0]*0.8) # Specify the training data
    train_inputs = inputs[:ntrain]
    train_outputs = outputs[:ntrain]
    test_inputs = inputs[ntrain:]
    test_outputs = outputs[ntrain:]

    # Create data loaders
    fixed_params_train = {
        "in" : train_inputs,
        "out" : train_outputs,
        "drop_last" : True,
    }
    fixed_params_valid = {
        "in" : test_inputs,
        "out" : test_outputs,
        "drop_last" : False,
    }

    # drop_last means that the last batch will be dropped if it is not full. avoids large updates from small samples
    def init_dl(batch_size, dev, **fixed_params):
        set = torch.utils.data.TensorDataset(fixed_params["in"], fixed_params["out"])
        dl = _dataloader.WrappedDataLoader(
            torch.utils.data.DataLoader(set, batch_size, shuffle=True, drop_last=fixed_params["drop_last"]),
            lambda x, y: (x.unsqueeze(1).to(dev), y.to(dev))
        )
        return dl

    train_dl_initialiser = InitialiseDataloader(init_dl, fixed_params_train)
    valid_dl_initialiser = InitialiseDataloader(init_dl, fixed_params_valid)

    loss_function = torch.nn.BCEWithLogitsLoss() # initialise it here. so torch.nn.MSELoss() for example.
    model = _model.ResNet

    # Optimiser hyperparameters to try. The model will be trained for all possible combinations of these.
    OPTIMISER_HYPERPARAMS = {
        "lr" : [0.0001, 0.00005],
        # "lr" : [0.0001, 0.00005],
        # "weight_decay" : [0.1],
    }

    # Define parameters for FCNN model
    MODEL_HYPERPARAMS = {
        "classes_num" : [1],
        "in_channels" : [1],
        "input_size" : [20],
    }

    # Other hyperparameters to try. The model will be trained for all possible combinations of these.
    HYPERPARAMS = {
        "epochs" : [125],
        "batch_size" : [100],
        "clip_value" : [1.0],
        "track_accuracy" : [True],
        "accuracy_fn" : [lambda result, target: torch.sum((torch.nn.functional.sigmoid(result) >= 0.5) == target).item()],
    }

    return ExperimentConfig(
        "Coursework 1 Part 1 Deep FCNN", "q2_small_resnet",
        train_dl_initialiser, valid_dl_initialiser,
        model, loss_function,
        OPTIMISER_HYPERPARAMS, MODEL_HYPERPARAMS, HYPERPARAMS,
        log_wandb=True, params_folder="code\\params",
    )

CONFIGS = {
    "cw1_q2_fcnn" : cw1_q2_fcnn,
    "cw1_q2_resnet" : cw1_q2_resnet
}