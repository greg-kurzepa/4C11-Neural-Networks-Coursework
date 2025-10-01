#%%
import numpy as np
import matplotlib.pyplot as plt
import torch

import _dataloader
import _model
import configs._config_q1 as _config

# Read data from .mat file
path = "Data\\Material_C.mat"
data_reader = _config.MatRead(path)
# they are loaded in as torch tensors
strain = data_reader.get_strain()
stress = data_reader.get_stress()

initial_strain = strain[:,0,0]
initial_stresses = stress[:,0,0]

# plot two histograms (in 2 subplots), on for the initial strain values and one for the initial stress values
fig, axs = plt.subplots(2)
plt.tight_layout()
axs[0].hist(initial_strain.numpy(), bins=50)
axs[0].set_title("Initial strain values")
axs[1].hist(initial_stresses.numpy(), bins=50)
axs[1].set_title("Initial stress values")
plt.show()
# %%
