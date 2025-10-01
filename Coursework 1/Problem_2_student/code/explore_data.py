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

        self.load_apply = np.array(self.data['load_apply'])
        # np.random.shuffle(self.load_apply)
        self.load_apply_out = torch.tensor(self.load_apply.T, dtype=torch.float32)

        self.result = np.array(self.data['result'])
        # np.random.shuffle(self.result)
        self.result_out = torch.tensor(self.result.T, dtype=torch.float32)

    def read_data(self):
        return self.load_apply_out, self.result_out

######################### Data processing #############################
# Read data from .mat file
path = "C:\\Users\\gregk\\Documents\\MyDocuments\\IIB\\4C11\\cw1\\Problem_2_student\\Data\\Eiffel_data.mat" #Define your data path here
data_reader = MatRead(path)
load_apply, result = data_reader.read_data()

#%%