import torch
import h5py
import numpy as np
import _dataloader
import unet, fno
from tqdm import tqdm

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

def cw2_q2_unet():
    ############################# Data processing #############################
    # Read data from mat
    train_path = r"C:\Users\gregk\Documents\MyDocuments\IIB\4C11\cw2\Coursework2_Problem_2\Darcy_2D_data_train.mat"
    test_path = r"C:\Users\gregk\Documents\MyDocuments\IIB\4C11\cw2\Coursework2_Problem_2\Darcy_2D_data_test.mat"

    data_reader = _dataloader.MatRead(train_path)
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    data_reader = _dataloader.MatRead(test_path)
    a_test = data_reader.get_a()
    u_test = data_reader.get_u()

    # Normalize data
    a_normalizer = _dataloader.UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)
    a_test = a_normalizer.encode(a_test)

    u_normalizer = _dataloader.UnitGaussianNormalizer(u_train)
    u_train = u_normalizer.encode(u_train)
    u_test = u_normalizer.encode(u_test)

    print(a_train.shape)
    print(a_test.shape)
    print(u_train.shape)
    print(u_test.shape)

    fixed_params_train = {
        "in" : a_train,
        "out" : u_train,
        "drop_last" : True,
    }
    fixed_params_test = {
        "in" : a_test,
        "out" : u_test,
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

    train_dl_initialiser = _dataloader.InitialiseDataloader(init_dl, fixed_params_train)
    test_dl_initialiser = _dataloader.InitialiseDataloader(init_dl, fixed_params_test)

    loss_function = _dataloader.LpLoss()
    model = unet.UNet

    # Optimiser hyperparameters to try. The model will be trained for all possible combinations of these.
    OPTIMISER_HYPERPARAMS = {
        "lr" : [0.0001],
        "weight_decay" : [0],
    }

    # Define parameters for FCNN model
    MODEL_HYPERPARAMS = {
    }

    # Other hyperparameters to try. The model will be trained for all possible combinations of these.
    HYPERPARAMS = {
        "epochs" : [100],
        "batch_size" : [100],
        "clip_value" : [1.0],
    }

    return ExperimentConfig(
        "Coursework 2 Part 2", "q2_unet_first",
        train_dl_initialiser, test_dl_initialiser,
        model, loss_function,
        OPTIMISER_HYPERPARAMS, MODEL_HYPERPARAMS, HYPERPARAMS,
        log_wandb=True, params_folder="params",
    )

def cw2_q2_fno():
    ############################# Data processing #############################
    # Read data from mat
    train_path = r"C:\Users\gregk\Documents\MyDocuments\IIB\4C11\cw2\Coursework2_Problem_2\Darcy_2D_data_train.mat"
    test_path = r"C:\Users\gregk\Documents\MyDocuments\IIB\4C11\cw2\Coursework2_Problem_2\Darcy_2D_data_test.mat"

    data_reader = _dataloader.MatRead(train_path)
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    data_reader = _dataloader.MatRead(test_path)
    a_test = data_reader.get_a()
    u_test = data_reader.get_u()

    # Normalize data
    a_normalizer = _dataloader.UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)
    a_test = a_normalizer.encode(a_test)

    u_normalizer = _dataloader.UnitGaussianNormalizer(u_train)
    u_train = u_normalizer.encode(u_train)
    u_test = u_normalizer.encode(u_test)

    print(a_train.shape)
    print(a_test.shape)
    print(u_train.shape)
    print(u_test.shape)

    fixed_params_train = {
        "in" : a_train,
        "out" : u_train,
        "drop_last" : True,
    }
    fixed_params_test = {
        "in" : a_test,
        "out" : u_test,
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

    train_dl_initialiser = _dataloader.InitialiseDataloader(init_dl, fixed_params_train)
    test_dl_initialiser = _dataloader.InitialiseDataloader(init_dl, fixed_params_test)

    loss_function = _dataloader.LpLoss()
    model = fno.FNO

    # Optimiser hyperparameters to try. The model will be trained for all possible combinations of these.
    OPTIMISER_HYPERPARAMS = {
        "lr" : [0.0005],
        "weight_decay" : [0],
    }

    # Define parameters for FCNN model
    MODEL_HYPERPARAMS = {
        "modes1" : [12],
        "modes2" : [12],
        "width" : [32],
    }

    # Other hyperparameters to try. The model will be trained for all possible combinations of these.
    HYPERPARAMS = {
        "epochs" : [50],
        "batch_size" : [25],
        # clip value cannot be set since clamp is not supported for complex types
    }

    return ExperimentConfig(
        "Coursework 2 Part 2", "q2_fno_first_with_scheduler",
        train_dl_initialiser, test_dl_initialiser,
        model, loss_function,
        OPTIMISER_HYPERPARAMS, MODEL_HYPERPARAMS, HYPERPARAMS,
        log_wandb=True, params_folder="params",
    )

CONFIGS = {
    "cw2_q2_unet" : cw2_q2_unet,
    "cw2_q2_fno" : cw2_q2_fno,
}