import numpy as np
import matplotlib.pyplot as plt
import torch

import _dataloader
import _model
import configs._config_q1 as _config

dev = torch.device("cpu")
print("Using CPU.")

# Read data from .mat file
path = "Data\\Material_A.mat"
data_reader = _config.MatRead(path)
strain = data_reader.get_strain()
stress = data_reader.get_stress()

strain_normaliser = _config.NormaliseMeanStd(strain)
stress_normaliser = _config.NormaliseMeanStd(stress)

nonzero_id = np.array([0,1,3]) # plane strain conditions
sample_id = -1 # must be in test dataset

# Import trained model for flattened experiment
flattened_shape = strain.shape[1]*strain.shape[2]
model = _model.FCNN([flattened_shape, 1000, flattened_shape]).to(dev)
params_dir = "selected_params\\best_flattened.torchparams"
model.load_state_dict(torch.load(params_dir, weights_only=True))
prepared_stress = stress_normaliser.normalise(stress[sample_id].reshape(1, flattened_shape))
model.eval()
with torch.no_grad():
    result = model(prepared_stress).detach()
    strain_path_flattened = strain_normaliser.denormalise(result).numpy()
    strain_pred_flattened = strain_path_flattened.reshape(strain.shape[1], strain.shape[2])
# Find mean square error
loss_function = torch.nn.functional.mse_loss
loss_flattened = loss_function(torch.tensor(strain_path_flattened), torch.tensor(strain[sample_id].flatten()))

# Import trained model for markov experiment
model2 = _model.FCNN([18,1000,6]).to(dev)
params_dir = "selected_params\\best_markov.torchparams"
model2.load_state_dict(torch.load(params_dir, weights_only=True))
prepared_stress = stress_normaliser.normalise(stress[sample_id])
model2.eval()
strain_path_markov = [torch.tensor([0,0,0,0,0,0])]
with torch.no_grad():
    for t in range(1, prepared_stress.shape[0]):
        result = model2(torch.unsqueeze(torch.cat((prepared_stress[t-1], strain_path_markov[-1], prepared_stress[t])), 0)).detach()
        strain_path_markov.append(strain_normaliser.denormalise(result)[0])
strain_pred_markov = np.array([x.numpy() for x in strain_path_markov])
loss_markov = loss_function(torch.tensor(strain_pred_markov), torch.tensor(strain[sample_id]))

strain_path_true = strain[sample_id, :, nonzero_id]
strain_pred_flattened = strain_pred_flattened[:, nonzero_id]
strain_pred_markov = strain_pred_markov[:, nonzero_id]

# make a 3d plot of strain_path_true
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(strain_path_true[:,0], strain_path_true[:,1], strain_path_true[:,2], label="true strain path")
ax.plot(strain_pred_flattened[:,0], strain_pred_flattened[:,1], strain_pred_flattened[:,2], label="predicted strain path, loss = %.4f" % loss_flattened)
ax.plot(strain_pred_markov[:,0], strain_pred_markov[:,1], strain_pred_markov[:,2], label="predicted strain path (markov), loss = %.6f" % loss_markov)
ax.set_xlabel("$\\varepsilon_{xx}$")
ax.set_ylabel("$\\varepsilon_{yy}$")
ax.set_zlabel("$\\varepsilon_{xy}$")
plt.legend()
plt.show()

# make a 3d line plot
