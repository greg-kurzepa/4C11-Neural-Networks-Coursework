#%%

import numpy as np
import matplotlib.pyplot as plt
import torch

import _dataloader
import _model
import configs._config_q1 as _config

dev = torch.device("cpu")
print("Using CPU.")

# Read data from .mat file
path = "Data\\Material_C.mat"
data_reader = _config.MatRead(path)
# they are loaded in as torch tensors
strain = data_reader.get_strain()
stress = data_reader.get_stress()

strain_normaliser = _config.NormaliseMeanStd(strain)
stress_normaliser = _config.NormaliseMeanStd(stress)

sample_id = 1099 # must be in test dataset
ndims = 1 # number of stress dimensions. 6 for mat_A, mat_B but 1 for mat_C.

#%%

# Import trained model for no memory experiment
# Here we were predicting stress from strain
model = _model.FCNN([ndims,100,ndims]).to(dev)
params_dir = "selected_params\\best_nomemory_matC.torchparams"
model.load_state_dict(torch.load(params_dir, weights_only=True))
true_stress = stress[sample_id]
prepared_strain = strain_normaliser.normalise(strain[sample_id])
prepared_stress = stress_normaliser.normalise(true_stress)
model.eval()
# first just get prediction for the chosen sample in the test dataset
# no memory, so each time step can be treated as a batch: input to model is 50x6
with torch.no_grad():
    result = model(prepared_strain)
    mse = torch.nn.functional.mse_loss(prepared_stress, result)

#%%
# Import trained model for fcnn memory experiment (material C only!)
model = _model.FCNN([50,1000,1000,50]).to(dev)
params_dir = "selected_params\\best_memory_fcnn_matC.torchparams"
model.load_state_dict(torch.load(params_dir, weights_only=True))
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % n_params)

true_stress = stress[sample_id]
prepared_strain = strain_normaliser.normalise(strain[sample_id].flatten())
prepared_stress = stress_normaliser.normalise(true_stress).flatten()
model.eval()
# first just get prediction for the chosen sample in the test dataset
with torch.no_grad():
    result = model(prepared_strain)
    mse = torch.nn.functional.mse_loss(prepared_stress, result)

    # reshape for plotting
    prepared_stress = prepared_stress.reshape(50, ndims).numpy()
    result = result.reshape(50, ndims).numpy()

    # Hysteresis test: bring strain to 1 and back to zero and see what happens to stress.
    hysteresis_strain = torch.cat([torch.linspace(0, 0.5, 25), torch.linspace(0.5, 0, 25)]).unsqueeze(0)
    with torch.no_grad():
        h_result = model(hysteresis_strain)

#%%
# Import trained model for lstm memory experiment (material C only!)
model = _model.LSTMWithLinear(100, 1, num_layers=5).to(dev)
params_dir = "selected_params\\best_memory_lstm_matC.torchparams"
model.load_state_dict(torch.load(params_dir, weights_only=True))
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % n_params)

true_stress = stress[sample_id]
prepared_strain = strain_normaliser.normalise(strain[sample_id])
prepared_stress = stress_normaliser.normalise(true_stress)
model.eval()
# first just get prediction for the chosen sample in the test dataset
with torch.no_grad():
    result = model(prepared_strain)
    mse = torch.nn.functional.mse_loss(prepared_stress, result)

#%%
# Import trained model for lstm memory shifted data experiment (material C only!)
model = _model.LSTMWithLinear(100, 1, num_layers=5).to(dev)
params_dir = "selected_params\\best_memory_lstm_shift_matC.torchparams"
model.load_state_dict(torch.load(params_dir, weights_only=True))
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % n_params)

stress_shifted = torch.roll(stress, 1, dims=1)
stress_shifted[:,0] = 0

true_stress = stress_shifted[sample_id]
prepared_strain = strain_normaliser.normalise(strain[sample_id])
prepared_stress = stress_normaliser.normalise(true_stress)
model.eval()
# first just get prediction for the chosen sample in the test dataset
with torch.no_grad():
    result = model(prepared_strain)
    mse = torch.nn.functional.mse_loss(prepared_stress, result)

# Hysteresis test: bring strain to 1 and back to zero and see what happens to stress.
# also investigate rate dependency.
n = 10
n2 = 2*n
hysteresis_strain1 = torch.cat([torch.linspace(0, 1, n), torch.linspace(1, -1, 2*n), torch.linspace(-1, 0, n)]).unsqueeze(0).unsqueeze(2)
hysteresis_strain2 = torch.cat([torch.linspace(0, 1, n2), torch.linspace(1, -1, 2*n2), torch.linspace(-1, 0, n2)]).unsqueeze(0).unsqueeze(2)
with torch.no_grad():
    h_result1 = model(hysteresis_strain1)
    h_result2 = model(hysteresis_strain2)

#%%

# plot the strain path.
fig = plt.figure()
fig.set_size_inches(9, 3)
plt.plot(prepared_strain)
plt.title(f"Material C, Input Strain Path Sample nr. {sample_id}")
plt.xlabel("$t$")
plt.ylabel("Normalised Strain")
plt.show()

#%%

# plot the true and predicted stress.
fig, ax = plt.subplots()
fig.set_size_inches(9, 3)
# first 6 default matplotlib colours
cols = plt.rcParams['axes.prop_cycle'].by_key()['color'][:6]
for i in range(ndims):
    # input true_stress is a 2d tensor, in this case the columns are separate datasets.
    ax.plot(prepared_stress[:,i], color=cols[i], alpha=0.5)
    ax.plot(result[:,i], color=cols[i], linestyle="--")
# make custom legend and show
ax.plot(0, 0, label="true stress", color="k", alpha=0.5)
ax.plot(0, 0, label="predicted stress, loss = %.2E" % mse, color="k", linestyle="--")
ax.legend()
plt.title(f"Material C, LSTM, Loading Path Sample nr. {sample_id}")
plt.xlim(left=0, right=49)
plt.xlabel("$t$")
plt.ylabel("Normalised Stresses")
plt.show()
# %%

# plot the hysteresis test, strain on x-axis and stress on y-axis.
plt.plot(hysteresis_strain.flatten(), h_result.flatten(), label="$n=40$")
plt.scatter(hysteresis_strain.flatten(), h_result.flatten(), marker="x")
plt.plot(hysteresis_strain2.flatten(), h_result2.flatten(), label="$n=80$")
plt.scatter(hysteresis_strain2.flatten(), h_result2.flatten(), marker="x")

plt.title("Hysteresis and Rate Dependency Test")
plt.xlabel("Normalised Strain")
plt.ylabel("Normalised Predicted Stress")
plt.legend()
plt.show()

# %%
