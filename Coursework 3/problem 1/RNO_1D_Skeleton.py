#%%
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import numpy as np
import scipy.io
import h5py

import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import wandb
import os

import utils

# if torch.cuda.is_available():
#      dev = torch.device("cuda")
#      print("CUDA available. Using CUDA.")
# else:
#      dev = torch.device("cpu")
#      print("CUDA not available. Using CPU.")
dev = torch.device("cpu")

#Define your data path
TRAIN_PATH = r'viscodata_3mat.mat'

# define train and test data
Ntotal     = 400
train_size = 320
test_start = train_size

N_test = Ntotal-test_start

# Read data from the .mat file
F_FIELD = 'epsi_tol'

SIG_FIELD = 'sigma_tol'

# Define loss function
loss_func = nn.MSELoss()
######### Preprocessing data ####################
temp = torch.zeros(Ntotal,1)

data_loader = utils.MatReader(TRAIN_PATH)
data_input  = data_loader.read_field(F_FIELD).contiguous().view(Ntotal, -1)
data_output  = data_loader.read_field(SIG_FIELD).contiguous().view(Ntotal, -1)

# We down sample the data to a coarser grid in time. This is to help saving the training time
s = 4

data_input  = data_input[:,0::s]
data_output = data_output[:,0::s]

inputsize   = data_input.size()[1]

#%%

# Normalize your data using the min-max normalizer
input_normaliser = utils.MinMaxNormaliser(data_input)
output_normaliser = utils.MinMaxNormaliser(data_output)

data_input  = input_normaliser.normalise(data_input)
data_output = output_normaliser.normalise(data_output)

# define train and test data
x_train = data_input[0:train_size,:]
y_train = data_output[0:train_size,:]

# define the time increment dt in the RNN
dt = 1.0/(y_train.shape[1]-1)

x_test = data_input[test_start:Ntotal,:].to(dev)
y_test  = data_output[test_start:Ntotal,:]
testsize = x_test.shape[0]


# Define number of hidden variables to use
n_hidden = 1
# Define the RNN architecture
input_dim     = 1

# Define RNN
inner_layers = [500, 500, 500]
layer_input = [2*input_dim+n_hidden] + inner_layers + [input_dim]
layer_hidden = [input_dim+n_hidden] + inner_layers + [n_hidden]
net = utils.RNN(input_dim, n_hidden, input_dim, layer_input, layer_hidden).to(dev)

# Number of training epochs
epochs = 100

# Optimizer and learning drate scheduler
learning_rate = 0.001
weight_decay = 0
optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = None

# Batch size
b_size = 40

# Wrap training data in loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=b_size,
                                           shuffle=True)

params_folder = "params"
start_time = time.strftime('%Y%m%d-%H%M%S')

# Set up wandb
log_wandb = True
if log_wandb:
    wandb_config = {
        "time" : start_time,
        "epochs" : epochs,
        "batch_size" : b_size,
        "learning_rate" : learning_rate,
        "weight_decay" : weight_decay,
        "n_hidden" : n_hidden,
        "inner_layers" : inner_layers,
    }
    wandb.init(
        project = "Coursework 3",
        config = wandb_config
    )
    wandb.watch(
        net,
        criterion = loss_func,
        # log = "all",
        log_freq = 1,
    )

# Train neural net
T = inputsize
train_err = np.zeros((epochs,))
test_err = np.zeros((epochs,))
y_test_approx = torch.zeros(testsize, inputsize)

for ep in tqdm(range(epochs)):
    if scheduler is not None:
        scheduler.step()
    train_loss = 0.0
    test_loss  = 0.0
    for x, y in train_loader:
        x, y = x.to(dev), y.to(dev)
        hidden = net.initHidden(b_size).to(dev)
        optimizer.zero_grad()
        y_approx = torch.zeros(b_size,T).to(dev)
        y_true  = y
        y_approx[:,0] = y_true[:,0]
        for i in range(1,T):
            y_approx[:,i], hidden = net(x[:,i].unsqueeze(1), x[:,i-1].unsqueeze(1), hidden,dt)

        loss = loss_func(y_approx,y_true)
        loss.backward()
        train_loss = train_loss + loss.item()

        optimizer.step()
    with torch.no_grad():
        hidden_test = net.initHidden(testsize).to(dev)
        y_test_approx[:,0] = y_test[:,0]
        for j in range(1,T):
           y_test_approx[:, j], hidden_test = net(x_test[:, j].unsqueeze(1), x_test[:, j-1].unsqueeze(1), hidden_test,dt)
        t_loss = loss_func(y_test_approx,y_test)
        test_loss = t_loss.item()

    train_err[ep] = train_loss/len(train_loader)
    test_err[ep]  = test_loss
    print(f"train loss: {train_err[ep]:.4e}, test loss: {test_err[ep]:.4e}")
    
    if log_wandb:
        wandb.log({
            "train_loss": train_err[ep],
            "test_loss": test_err[ep],
        })

    if (ep % 50 == 0 and ep != 0) or (ep == epochs - 1):
        torch.save(
            {
                "epoch" : ep,
                "model_state_dict" : net.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
            },
            os.path.join(params_folder, f"{start_time}.torchparams")
        )
        print("Saved model and optimizer state")

if log_wandb:
    wandb.finish()
# %%
""