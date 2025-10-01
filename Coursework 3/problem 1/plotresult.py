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

# Load the trained model
params_path = os.path.join("selected_params", "hidden1.torchparams")
n_hidden = 1
input_dim = 1
inner_layers = [100, 100, 100]
layer_input = [2*input_dim+n_hidden] + inner_layers + [input_dim]
layer_hidden = [input_dim+n_hidden] + inner_layers + [n_hidden]
net = utils.RNN(input_dim, n_hidden, input_dim, layer_input, layer_hidden)
state = torch.load(params_path)
net.load_state_dict(state["model_state_dict"], weights_only=True)

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

x_test = data_input[test_start:Ntotal,:]
y_test  = data_output[test_start:Ntotal,:]
testsize = x_test.shape[0]

# Evaluate!
T = inputsize
y_test_approx = torch.zeros(testsize, inputsize)
hidden_test = net.initHidden(testsize)
y_test_approx[:,0] = y_test[:,0]
for j in range(1,T):
    y_test_approx[:, j], hidden_test = net(x_test[:, j].unsqueeze(1), x_test[:, j-1].unsqueeze(1), hidden_test,dt)
t_loss = loss_func(y_test_approx,y_test)
test_loss = t_loss.item()

#%% plot it

plot_id = 0
this_loss = loss_func(y_test_approx[plot_id], y_test[plot_id]).item()
fig, ax = plt.subplots(1, 2, figsize=(8,4), sharex=True, dpi=300)
plt.tight_layout()
plt.subplots_adjust(top=0.89)

plt.suptitle(f"Test Set Sample {plot_id}")

ax[0].set_title("Normalised Input Strain History")
ax[0].plot(x_test[plot_id])
ax[0].set_xlim(left=0, right=inputsize)
ax[0].set_xlabel("t")
ax[0].set_ylabel("Normalised Strain")

ax[1].set_title("Normalised Stress History")
ax[1].plot(y_test[plot_id], label="Ground Truth Stress History")
ax[1].plot(y_test_approx[plot_id].detach().numpy(), label=f"Predicted Stress History, Loss {this_loss:.2e}", linestyle='--')
ax[1].legend()
ax[1].set_xlim(left=0, right=inputsize)
ax[1].set_xlabel("t")
ax[1].set_ylabel("Normalised Stress")

plt.show()
#%%