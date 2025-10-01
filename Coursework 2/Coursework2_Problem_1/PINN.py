#%%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import scipy
from tqdm import tqdm
import time
import wandb
import os

model_dtype = torch.float32

if torch.cuda.is_available():
     dev = torch.device("cuda")
     print("CUDA available. Using CUDA.")
else:
     dev = torch.device("cpu")
     print("CUDA not available. Using CPU.")

# Define Neural Network
class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1], dtype=model_dtype))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

############################# Data processing #############################
# Read data from mat
# Specify your data path here
path = 'plate_data.mat'
data = scipy.io.loadmat(path)
torch.set_default_tensor_type(torch.DoubleTensor)
L_boundary = torch.tensor(data['L_boundary'], dtype=model_dtype).to(dev)
R_boundary = torch.tensor(data['R_boundary'], dtype=model_dtype).to(dev)
T_boundary = torch.tensor(data['T_boundary'], dtype=model_dtype).to(dev)
B_boundary = torch.tensor(data['B_boundary'], dtype=model_dtype).to(dev)
C_boundary = torch.tensor(data['C_boundary'], dtype=model_dtype).to(dev)
Boundary   = torch.tensor(data['Boundary'], dtype=model_dtype, requires_grad=True).to(dev)

# truth solution from FEM
disp_truth = torch.tensor(data['disp_data'], dtype=model_dtype).to(dev)

# connectivity matrix - this helps you to plot the figure but we do not need it for PINN
t_connect  = torch.tensor(data['t'].astype(float), dtype=model_dtype).to(dev)

# all collocation points
x_full = torch.tensor(data['p_full'], dtype=model_dtype,requires_grad=True).to(dev)

# collocation points excluding the boundary
x = torch.tensor(data['p'], dtype=model_dtype, requires_grad=True).to(dev)

# This chooses 50 fixed points from the truth solution, which we will use for part (e)
rand_index = torch.randint(0, len(x_full), (50,))
disp_fix = disp_truth[rand_index,:]

# We will use two neural networks for the problem:
# NN1: to map the coordinates [x,y] to displacement u
# NN2: to map the coordinates [x,y] to the stresses [sigma_11, sigma_22, sigma_12]
# What we will do later is to first compute strain by differentiate the output of NN1
# And then we compute a augment stress using Hook's law to find an augmented stress sigma_a
# And we will require the output of NN2 to match sigma_a  - we shall do this by adding a term in the loss function
# This will help us to avoid differentiating NN1 twice (why?)
#   !My Answer:
#   Because the loss function involves first differentiating the displacement to get stress (via strain),
#   and then differentiating the stress to get the equilibrium equation.
#   if we did this it would introduce numerical instability.
#   Instead, one NN learns displacement, then another learns displacement to stress, which can be used with the full equilibrium loss fn.
# As it is well known that PINN suffers from higher order derivatives

Disp_layer = [2, 300, 300, 2] # Architecture of displacement net - you may change as you wish
Stress_layer = [2,400,400,3] # Architecture of stress net - you may change as you wish

stress_net = DenseNet(Stress_layer,nn.Tanh).to(dev) # Note we choose hyperbolic tangent as an activation function here
disp_net =  DenseNet(Disp_layer,nn.Tanh).to(dev)

# # load pre-trained weights
params_dir = r"C:\Users\gregk\Documents\MyDocuments\IIB\4C11\cw2\Coursework2_Problem_1\params\20250318-115801"
stress_net.load_state_dict(torch.load(os.path.join(params_dir, "stress_net_it10k.torchparams"), weights_only=True))
disp_net.load_state_dict(torch.load(os.path.join(params_dir, "disp_net_it10k.torchparams"), weights_only=True))

# Define material properties
E = 10
mu = 0.3

# Define stress values at top and right boundary
SIGMA_T = 0
SIGMA_R = 0.1

stiff = E/(1-mu**2)*torch.tensor([[1,mu,0],[mu,1,0],[0,0,(1-mu)/2]], dtype=model_dtype).to(dev) # Hooke's law for plane stress
stiff = stiff.unsqueeze(0)

# Define loss function
loss_func = torch.nn.MSELoss()

# Broadcast stiffness for batch multiplication later
stiff_bc = stiff
stiff = torch.broadcast_to(stiff, (len(x),3,3))

stiff_bc = torch.broadcast_to(stiff_bc, (len(Boundary),3,3))

params = list(stress_net.parameters()) + list(disp_net.parameters())

# Define optimizer and scheduler
# optimizer = torch.optim.AdamW(params, weight_decay=0)
# # Since we have no batches, LBFGS is a better choice than SGD.
optimizer = torch.optim.LBFGS(params, max_iter=10, line_search_fn="strong_wolfe",
                              tolerance_change=1e-100, tolerance_grad=1e-100)
# # load pre-trained optimizer state
# optimizer.load_state_dict(torch.load(os.path.join(params_dir, "lbfgs_it9k.torchparams"), weights_only=True))
# # set the tolerance to be what I want (it was overriden by the load_state_dict)
# optimizer.param_groups[0]["tolerance_change"] = 1e-100
# optimizer.param_groups[0]["tolerance_grad"] = 1e-100

#%%

def get_loss(do_data):
    # To compute stress from stress net
    sigma = stress_net(x)
    # To compute displacement from disp net
    disp     = disp_net(x)

    # displacement in x direction
    u = disp[:,0]
    # displacement in y direction
    v = disp[:,1]

    # find the derivatives
    dudx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    dvdx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),create_graph=True)[0]

    # Define strain
    e_11 = dudx[:,0].unsqueeze(1)
    e_22 = dvdx[:,1].unsqueeze(1)
    e_12 = 0.5*(dudx[:,1] + dvdx[:,0]).unsqueeze(1)

    e = torch.cat((e_11,e_22,e_12), 1)
    e = e.unsqueeze(2)

    # Define augment stress
    sig_aug = torch.bmm(stiff, e).squeeze(2)

    # Define constitutive loss - forcing the augment stress to be equal to the neural network stress
    loss_cons = loss_func(sig_aug, sigma)

    # find displacement and stress at the boundaries
    disp_bc = disp_net(Boundary)
    sigma_bc = stress_net(Boundary)
    u_bc = disp_bc[:,0]
    v_bc = disp_bc[:,1]

    # Compute the strain and stresses at the boundary
    dudx_bc = torch.autograd.grad(u_bc, Boundary, grad_outputs=torch.ones_like(u_bc),create_graph=True)[0]
    dvdx_bc = torch.autograd.grad(v_bc, Boundary, grad_outputs=torch.ones_like(v_bc),create_graph=True)[0]

    e_11_bc = dudx_bc[:,0].unsqueeze(1)
    e_22_bc = dvdx_bc[:,1].unsqueeze(1)
    e_12_bc = 0.5*(dudx_bc[:,1] + dvdx_bc[:,0]).unsqueeze(1)

    e_bc = torch.cat((e_11_bc,e_22_bc,e_12_bc), 1)
    e_bc = e_bc.unsqueeze(2)

    sig_aug_bc = torch.bmm(stiff_bc, e_bc).squeeze(2)

    # force the augment stress to agree with the NN stress at the boundary
    loss_cons_bc = loss_func(sig_aug_bc, sigma_bc)

    #============= equilibrium ===================#

    sig_11 = sigma[:,0]
    sig_22 = sigma[:,1]
    sig_12 = sigma[:,2]

    # stress equilibrium in x and y direction
    dsig11dx = torch.autograd.grad(sig_11, x, grad_outputs=torch.ones_like(sig_11),create_graph=True)[0]
    dsig22dx = torch.autograd.grad(sig_22, x, grad_outputs=torch.ones_like(sig_22),create_graph=True)[0]
    dsig12dx = torch.autograd.grad(sig_12, x, grad_outputs=torch.ones_like(sig_12),create_graph=True)[0]

    eq_x1 = dsig11dx[:,0]+dsig12dx[:,1]
    eq_x2 = dsig12dx[:,0]+dsig22dx[:,1]

    # zero body forces
    f_x1 = torch.zeros_like(eq_x1)
    f_x2 = torch.zeros_like(eq_x2)

    loss_eq1 = loss_func(eq_x1, f_x1)
    loss_eq2 = loss_func(eq_x2, f_x2)
    #========= boundary ========================#

    # specify the boundary condition
    tau_R = SIGMA_R
    tau_T = SIGMA_T
    #
    u_L= disp_net(L_boundary)
    u_B = disp_net(B_boundary)

    sig_R = stress_net(R_boundary)
    sig_T = stress_net(T_boundary)
    sig_C = stress_net(C_boundary)

    # Symmetry boundary condition left
    loss_BC_L = loss_func(u_L[:,0], torch.zeros_like(u_L[:,0]))
    # Symmetry boundary condition bottom
    loss_BC_B = loss_func(u_B[:, 1], torch.zeros_like(u_B[:, 1]))
    # Force boundary condition right
    loss_BC_R = loss_func(sig_R[:, 0], tau_R*torch.ones_like(sig_R[:, 0])) \
                + loss_func(sig_R[:, 2],  torch.zeros_like(sig_R[:, 2]))

    loss_BC_T = loss_func(sig_T[:, 1], tau_T*torch.ones_like(sig_T[:, 1]))   \
                + loss_func(sig_T[:, 2],  torch.zeros_like(sig_T[:, 2]))

    # traction free on circle
    loss_BC_C = loss_func(sig_C[:,0]*C_boundary[:,0]+sig_C[:,2]*C_boundary[:,1], torch.zeros_like(sig_C[:, 0]))  \
                + loss_func(sig_C[:,2]*C_boundary[:,0]+sig_C[:,1]*C_boundary[:,1], torch.zeros_like(sig_C[:, 0]))

    # ======= uncomment below for part (e) =======================
    # data_loss_fix
    if do_data:
        x_fix = x_full[rand_index, :]
        u_fix = disp_net(x_fix)
        loss_fix = loss_func(u_fix,disp_fix)
        loss = loss_eq1+loss_eq2+loss_cons+loss_BC_L+loss_BC_B+loss_BC_R+loss_BC_T+loss_BC_C+loss_cons_bc + 100*loss_fix
    else:
        loss = loss_eq1+loss_eq2+loss_cons+loss_BC_L+loss_BC_B+loss_BC_R+loss_BC_T+loss_BC_C+loss_cons_bc

    # with torch.autograd.profiler.profile() as prof:
    # loss.backward()
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    # input("pause!")

    return loss

def closure():
    # scheduler.step()
    optimizer.zero_grad() #! test the other method which should be faster (it was slower.)

    loss = get_loss(do_data=False)
    loss.backward()
    return loss

def train():
    start_time = time.strftime('%Y%m%d-%H%M%S')

    # initialise Weights&Biases
    wandb_config = {
            "config_name": "PINN_LBFGS_WithData",
            "model dtype" : str(model_dtype),
            "time" : start_time,
            "opt": optimizer.param_groups[0],
    }

    if log_wandb: wandb.init(
        # set the wandb project where this run will be logged
        project = "Coursework 2 Part 1",
        config = wandb_config
    )

    # record gradients & parameters
    if log_wandb: wandb.watch(
        stress_net,
        criterion = loss_func,
        log = "all",
        log_freq = log_wandb_frequency,
    )
    if log_wandb: wandb.watch(
        disp_net,
        criterion = loss_func,
        log = "all",
        log_freq = log_wandb_frequency,
    )

    params_folder = r"C:\Users\gregk\Documents\MyDocuments\IIB\4C11\cw2\Coursework2_Problem_1\params"
    if not os.path.exists(params_folder):
        os.mkdir(params_folder)
    os.mkdir(os.path.join(params_folder, start_time))

    prev_loss = torch.inf
    for epoch in tqdm(range(1, iterations+1)): # +1 to include last iteration (and exclude first) in multiples of 100 for easy wandb tracking

        optimizer.step(closure)

        loss = get_loss(do_data=False).item()
        no_data_loss = get_loss(do_data=False).item()

        if epoch % print_loss_frequency == 0 or epoch == iterations - 1:
            print(f"loss: {loss}")
        
        if epoch % save_params_frequency == 0 or epoch == iterations - 1:

            save_str = f"it{epoch//save_params_frequency}k" if epoch != iterations - 1 else "final"
            torch.save(stress_net.state_dict(), os.path.join(params_folder, start_time, f"stress_net_{save_str}.torchparams"))
            torch.save(disp_net.state_dict(), os.path.join(params_folder, start_time, f"disp_net_{save_str}.torchparams"))
            
            # save lbfgs state
            torch.save(optimizer.state_dict(), os.path.join(params_folder, start_time, f"lbfgs_{save_str}.torchparams"))
            print("Saved params and LBFGS state.")

            # if lbfgs has terminated, the loss will stop decreasing
            if loss >= prev_loss:
                print("LBFGS has terminated. Stopping training.")
                break
            prev_loss = loss

        if log_wandb and epoch % log_wandb_frequency == 0 or epoch == iterations - 1:
            wandb.log({"training_loss" : loss, "no_data_loss" : no_data_loss})

    if log_wandb: wandb.finish()

log_wandb_frequency = 1
save_params_frequency = 50
print_loss_frequency = 1
log_wandb = True

# PINN requires super large number of iterations to converge (on the order of 50e^3-100e^3)
# Different for LBFGS! Each LBFGS has mini-batch size 20 so use 50k/20 = 2500 epochs
iterations = int(2500)

#%% profile the train_epoch function

# %load_ext line_profiler
# %lprun -f train train()

#%% TRAIN!

train()