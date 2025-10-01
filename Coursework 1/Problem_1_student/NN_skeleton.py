#%%

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.adamax
import torch.utils.data as Data
import numpy as np
import h5py
import wandb
import time

from torchinfo import summary

if torch.cuda.is_available():
     dev = torch.device("cuda")
     print("CUDA available. Using CUDA.")
else:
     dev = torch.device("cpu")
     print("CUDA not available. Using CPU.")

log_wandb = True

# Define your loss function here
class Lossfunc(object):
    pass

# This reads the matlab data from the .mat file provided
class MatRead(object):
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

# Define data normalizer
class DataNormalizer(object):
    pass

# Define network your neural network for the constitutive model below
class Const_Net(nn.Module):
    pass

######################### Data processing #############################
# Read data from .mat file
path = "Data\\Material_A.mat" #Define your data path here
data_reader = MatRead(path)
strain = data_reader.get_strain()
stress = data_reader.get_stress()

#%%

stress_labels = ["$\\sigma_{11}$", "$\\sigma_{22}$", "$\\sigma_{33}$", "$\\sigma_{12}$", "$\\sigma_{23}$", "$\\sigma_{13}$"]
strain_labels = ["$\\varepsilon_{11}$", "$\\varepsilon_{22}$", "$\\varepsilon_{33}$", "$\\varepsilon_{12}$", "$\\varepsilon_{23}$", "$\\varepsilon_{13}$"]
sample_idx = 2
fig, ax = plt.subplots(2, 1)
for i in range(6):
    ax[0].plot(stress[sample_idx][:,i], label=stress_labels[i])
for i in range(6):
    ax[1].plot(strain[sample_idx][:,i], label=strain_labels[i])
ax[0].set_title("Stress")
ax[1].set_title("Strain")
ax[0].grid()
ax[1].grid()

ax[0].legend()
ax[1].legend()
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(strain[0])

#%%

# Split data into train and test
ntrain =  # Specify the training data
ntest =   # Specify the test data
train_strain =
train_stress =
test_strain =
test_stress =

# Normalize your data
strain_normalizer   =
train_strain_encode =   # this should be the data after normalization
test_strain_encode  =

stress_normalizer   =
train_stress_encode =
test_stress_encode  =

ndim = strain.shape[2]  # Number of components
nstep = strain.shape[1] # Number of time steps
dt = 1/(nstep-1)

# Create data loader
batch_size = 20
train_set = Data.TensorDataset(train_strain, train_stress)
train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

############################# Define and train network #############################
# Create Nueral network, define loss function and optimizer
net = Const_Net()   # specify your neural network architecture

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) #Calculate the number of training parameters
print('Number of parameters: %d' % n_params)
summary(net, input_size=(batch_size, 50, 6))

loss_func = torch.nn.MSELoss() #define loss function
optimizer = torch.optim.Adam(net.parameters(), lr=0.001) # define optimizer
# scheduler =  # define scheduler, not strictly necessary with Adam.

# Train network
epochs =    # define number of training epochs

# ----------------------------
# current time in string form to use in trained parameters filename and to save to Weights&Biases
start_time = time.strftime('%Y%m%d-%H%M%S')
# initialise Weights&Biases
if log_wandb: wandb.init(
    # set the wandb project where this run will be logged
    project="Coursework 1 Part 1",

    # track hyperparameters and metadata
    config={
        "architecture": "DeepresUNet with Layernorm",
        "optimiser" : optimiser_name,
        **optimiser_args,
        "loss_function" : loss_function_name,
        "batch_size" : batch_size,
        "epochs": epochs,
        "transform" : transform_name,
        "do_batchnorm" : do_batchnorm,
        "do_layernorm" : do_layernorm,
        "clip_value" : "None" if clip_value is None else clip_value,
        "time" : start_time,
    }
)

# record gradients & parameters
if log_wandb: wandb.watch(
    my_model,
    criterion = loss_function,
    log = "all",
    log_freq = 1,
)

# train model, save its trained parameters, and finish Weights&Biases
_train.train(my_model, loss_function, optimiser, train_dl, valid_dl, epochs, batch_size, dev, show_plot=False, log_wandb=log_wandb, clip_value=clip_value, log_standard_loss=True)
torch.save(my_model.state_dict(), os.path.join(params_folder, f"{start_time}.torchparams"))
if log_wandb: wandb.finish()
# ----------------------------

print("Start training for {} epochs...".format(epochs))

loss_train_list = []
loss_test_list = []

for epoch in range(epochs):
    net.train(True)
    trainloss = 0


    for i, data in enumerate(train_loader):
        input, target = data
        #define forward neural network evaluation below
        output_encode = # Forward
        output        = # Decode output
        loss = loss_func() # Calculate loss

                             # Clear gradients
                             # Backward
                             # Update parameters
                             # Update learning rate

        trainloss +=         # update your train loss here

    # Compute your test loss below
    net.eval()
    with torch.no_grad():

    # Print train loss every 10 epochs
    if epoch % 10 == 0:
        print("epoch:{}, train loss:{}, test loss:{}".format(epoch, trainloss/len(train_loader), testloss))

    # Save loss
    loss_train_list.append(trainloss/len(train_loader))
    loss_test_list.append(testloss)


print("Train loss:{}".format(trainloss/len(train_loader)))
print("Test loss:{}".format(testloss))

############################# Plot your result below using Matplotlib #############################
plt.figure(1)
plt.title('Train and Test Losses')

plt.figure(2)
plt.title('Truth Stresses vs Approximate Stresses for Sample {}')