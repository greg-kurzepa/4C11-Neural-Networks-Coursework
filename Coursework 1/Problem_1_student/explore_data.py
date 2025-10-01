#%%

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

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

######################### Data processing #############################
# Read data from .mat file
path = "Data\\Material_B.mat" #Define your data path here
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
# %%
