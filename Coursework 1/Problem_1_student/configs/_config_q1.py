import torch
import h5py
import numpy as np
import _dataloader
import _model

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
class MatRead():
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_strain(self):
        strain = np.array(self.data['strain']).transpose(2,0,1)
        return torch.tensor(strain, dtype=torch.float32)

    def get_stress(self):
        stress = np.array(self.data['stress']).transpose(2,0,1)
        return torch.tensor(stress, dtype=torch.float32)
    
class NormaliseMeanStd():
    def __init__(self, data):
        self.mean = data.mean()
        self.std = data.std()
    
    def normalise(self, data):
        return (data - self.mean) / self.std
    
    def denormalise(self, data):
        return data * self.std + self.mean

def cw1_q1_markov():
    # Read data from .mat file
    path = "Data\\Material_B.mat"
    data_reader = MatRead(path)
    strain = data_reader.get_strain()
    stress = data_reader.get_stress()
    assert stress.shape[0] == strain.shape[0], "Number of samples in stress and strain data do not match"

    # Normalise the data
    strain_normaliser = NormaliseMeanStd(strain)
    stress_normaliser = NormaliseMeanStd(stress)
    stress = stress_normaliser.normalise(stress)
    strain = strain_normaliser.normalise(strain)

    # Process data to split it into input (s_t, e_t, s_t+1) and output (e_t+1)
    input_samples = []
    output_samples = []
    for i in range(strain.shape[0]):
        for t in range(strain.shape[1]-1):
            input_samples.append(torch.cat((stress[i][t], strain[i][t], stress[i][t+1])))
            output_samples.append(strain[i][t+1])
    inputs = torch.stack(input_samples)
    outputs = torch.stack(output_samples)

    # Split data into train and test
    ntrain = round(inputs.shape[0]*0.8) # Specify the training data
    ntest = round(inputs.shape[0]*0.2)  # Specify the test data
    assert ntrain + ntest == inputs.shape[0], "Train and test data do not add up to total data"
    train_inputs = inputs[:ntrain]
    train_outputs = outputs[:ntrain]
    test_inputs = inputs[-ntest:]
    test_outputs = outputs[-ntest:]

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

    loss_function = torch.nn.MSELoss()
    model = _model.FCNN

    # Optimiser hyperparameters to try. The model will be trained for all possible combinations of these.
    OPTIMISER_HYPERPARAMS = {
        "lr" : [0.001, 0.0005],
        # "betas" : [(0.9, 0.999)],
        # "weight_decay" : [1e-2],
    }

    # Define parameters for FCNN model
    MODEL_HYPERPARAMS = {
        "layer_sizes" : [[18, 1000, 6]]
    }

    # Other hyperparameters to try. The model will be trained for all possible combinations of these.
    HYPERPARAMS = {
        "epochs" : [50],
        "batch_size" : [1024, 128],
        "clip_value" : [1.0],
    }

    return ExperimentConfig(
        "Coursework 1 Part 1 Deep FCNN", "cw1_q1_markov_matB",
        train_dl_initialiser, valid_dl_initialiser,
        model, loss_function,
        OPTIMISER_HYPERPARAMS, MODEL_HYPERPARAMS, HYPERPARAMS,
        log_wandb=True, params_folder="params",
    )

def cw1_q1_symmetric():
    """strain_t exclusively predicts stress_t
    """

    # Read data from .mat file
    path = "Data\\Material_C.mat"
    data_reader = MatRead(path)
    strain = data_reader.get_strain()
    stress = data_reader.get_stress()
    assert stress.shape[0] == strain.shape[0], "Number of samples in stress and strain data do not match"

    # FLatten data into input (e_t) and output (s_t)
    strain = strain.reshape(-1,1)
    stress = stress.reshape(-1,1)

    # Normalise the data
    strain_normaliser = NormaliseMeanStd(strain)
    stress_normaliser = NormaliseMeanStd(stress)
    stress = stress_normaliser.normalise(stress)
    strain = strain_normaliser.normalise(strain)

    # Split data into train and test
    ntrain = round(strain.shape[0]*0.8) # Specify the training data
    ntest = round(strain.shape[0]*0.2)  # Specify the test data
    assert ntrain + ntest == strain.shape[0], "Train and test data do not add up to total data"
    train_inputs = strain[:ntrain]
    train_outputs = stress[:ntrain]
    test_inputs = strain[-ntest:]
    test_outputs = stress[-ntest:]

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

    loss_function = torch.nn.MSELoss()
    model = _model.FCNN

    # Optimiser hyperparameters to try. The model will be trained for all possible combinations of these.
    OPTIMISER_HYPERPARAMS = {
        "lr" : [0.001, 0.0005],
        # "betas" : [(0.9, 0.999)],
        # "weight_decay" : [1e-2],
    }

    # Define parameters for FCNN model
    MODEL_HYPERPARAMS = {
        "layer_sizes" : [[1, 100, 100, 1]]
    }

    # Other hyperparameters to try. The model will be trained for all possible combinations of these.
    HYPERPARAMS = {
        "epochs" : [30],
        "batch_size" : [256],
        "clip_value" : [1.0],
    }

    return ExperimentConfig(
        "Coursework 1 Part 1 Deep FCNN", "cw1_q1_symmetric_matC",
        train_dl_initialiser, valid_dl_initialiser,
        model, loss_function,
        OPTIMISER_HYPERPARAMS, MODEL_HYPERPARAMS, HYPERPARAMS,
        log_wandb=True, params_folder="params",
    )

def cw1_q1_memory_fcnn():
    """all previous strains predict current stress
    this config just flattens the data to be 50 inputs and 50 outputs and does an FCNN in between.
    """

    # Read data from .mat file
    path = "Data\\Material_C.mat"
    data_reader = MatRead(path)
    strain = data_reader.get_strain()
    stress = data_reader.get_stress()
    assert stress.shape[0] == strain.shape[0], "Number of samples in stress and strain data do not match"

    # FLatten time dimension of data
    strain = strain.reshape(-1,50)
    stress = stress.reshape(-1,50)

    # Normalise the data
    strain_normaliser = NormaliseMeanStd(strain)
    stress_normaliser = NormaliseMeanStd(stress)
    stress = stress_normaliser.normalise(stress)
    strain = strain_normaliser.normalise(strain)

    # Split data into train and test
    ntrain = round(strain.shape[0]*0.8) # Specify the training data
    ntest = round(strain.shape[0]*0.2)  # Specify the test data
    assert ntrain + ntest == strain.shape[0], "Train and test data do not add up to total data"
    train_inputs = strain[:ntrain]
    train_outputs = stress[:ntrain]
    test_inputs = strain[-ntest:]
    test_outputs = stress[-ntest:]

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

    loss_function = torch.nn.MSELoss()
    model = _model.FCNN

    # Optimiser hyperparameters to try. The model will be trained for all possible combinations of these.
    OPTIMISER_HYPERPARAMS = {
        "lr" : [0.0005],
        # "betas" : [(0.9, 0.999)],
        # "weight_decay" : [1e-2],
    }

    # Define parameters for FCNN model
    MODEL_HYPERPARAMS = {
        "layer_sizes" : [[50, 1000, 1000, 50]],
        # "dropout_p" : [0.5],
        # "dropout_layers" : [[2], [1]],
    }

    # Other hyperparameters to try. The model will be trained for all possible combinations of these.
    HYPERPARAMS = {
        "epochs" : [100],
        "batch_size" : [256],
        "clip_value" : [1.0],
    }

    return ExperimentConfig(
        "Coursework 1 Part 1 Deep FCNN", "cw1_q1_symmetric_memory_fcnn_matC",
        train_dl_initialiser, valid_dl_initialiser,
        model, loss_function,
        OPTIMISER_HYPERPARAMS, MODEL_HYPERPARAMS, HYPERPARAMS,
        log_wandb=True, params_folder="params",
    )

def cw1_q1_lstm():
    """all previous strains predict all stresses.
    """

    # Read data from .mat file
    path = "Data\\Material_C.mat"
    data_reader = MatRead(path)
    strain = data_reader.get_strain()
    stress = data_reader.get_stress()
    assert stress.shape[0] == strain.shape[0], "Number of samples in stress and strain data do not match"

    # shift all the stresses 1 to the right (throw out the last stress)
    stress = torch.roll(stress, 1, dims=1)
    stress[:,0] = 0

    # Normalise the data
    strain_normaliser = NormaliseMeanStd(strain)
    stress_normaliser = NormaliseMeanStd(stress)
    strain = strain_normaliser.normalise(strain)
    stress = stress_normaliser.normalise(stress)

    # Select input and output
    input = strain
    output = stress

    # Split data into train and test
    ntrain = round(strain.shape[0]*0.8) # Specify the training data
    ntest = round(strain.shape[0]*0.2)  # Specify the test data
    assert ntrain + ntest == strain.shape[0], "Train and test data do not add up to total data"
    train_inputs = input[:ntrain]
    train_outputs = output[:ntrain]
    test_inputs = input[-ntest:]
    test_outputs = output[-ntest:]

    # def pad_series(series_batches):
    #     expanded_batches = []
    #     sequence_lengths = []
    #     for t in reversed(range(series_batches.shape[1])):
    #         for b in range(series_batches.shape[0]):
    #             expanded_batches.append(series_batches[b,:t+1])
    #             sequence_lengths.append(t+1)
    #     pad_batches = torch.nn.utils.rnn.pad_sequence(expanded_batches, batch_first=True, padding_side="left")
    #     return pad_batches

    # # make many series out of each series.
    # train_inputs = pad_series(train_inputs)
    # train_outputs = pad_series(train_outputs)[:,-1,:]
    # test_inputs = pad_series(test_inputs)
    # test_outputs = pad_series(test_outputs)[:,-1,:]

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

    loss_function = torch.nn.MSELoss()
    # model = _model.FCNN
    # model = _model.LSTMWithLinear
    model = _model.LSTMWithLinear

    # Optimiser hyperparameters to try. The model will be trained for all possible combinations of these.
    OPTIMISER_HYPERPARAMS = {
        "lr" : [0.001, 0.0005],
        # "betas" : [(0.9, 0.999)],
        "weight_decay" : [0],
    }

    # Define parameters for FCNN model
    MODEL_HYPERPARAMS = {
        "hidden_size" : [100],
        "num_layers" : [5],
        "output_size" : [1],
        "dropout" : [0.2]
        # "output_size" : [1] # NOT the sequence length.
        # "layer_sizes" : [[50, 1000, 1000, 50]],
        # "dropout_p" : [0.5],
        # "dropout_layers" : [[2], [1]],
    }

    # Other hyperparameters to try. The model will be trained for all possible combinations of these.
    HYPERPARAMS = {
        "epochs" : [100],
        "batch_size" : [110],
        "clip_value" : [1.0],
    }

    return ExperimentConfig(
        "Coursework 1 Part 1 Deep FCNN", "cw1_q1_lstm_shifted",
        train_dl_initialiser, valid_dl_initialiser,
        model, loss_function,
        OPTIMISER_HYPERPARAMS, MODEL_HYPERPARAMS, HYPERPARAMS,
        log_wandb=True, params_folder="params",
    )

CONFIGS = {
    "cw1_q1_markov" : cw1_q1_markov,
    "cw1_q1_symmetric" : cw1_q1_symmetric,
    "cw1_q1_memory_fcnn" : cw1_q1_memory_fcnn,
    "cw1_q1_lstm" : cw1_q1_lstm,
}