#%%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import scipy
from tqdm import tqdm
import torch.jit as jit
import matplotlib.tri as mtri
import typing

# Define Neural Network
class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x.float()).double()

        return x
    
class JITModule(jit.ScriptModule):
    def __init__(self):
        super(JITModule, self).__init__()

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

        self.stress_net = DenseNet(Stress_layer,nn.Tanh) # Note we choose hyperbolic tangent as an activation function here
        self.disp_net =  DenseNet(Disp_layer,nn.Tanh)

        ############################# Data processing #############################
        # Read data from mat
        # Specify your data path here
        path = 'plate_data.mat'
        data = scipy.io.loadmat(path)
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.L_boundary = torch.tensor(data['L_boundary'], dtype=torch.float64)
        self.R_boundary = torch.tensor(data['R_boundary'], dtype=torch.float64)
        self.T_boundary = torch.tensor(data['T_boundary'], dtype=torch.float64)
        self.B_boundary = torch.tensor(data['B_boundary'], dtype=torch.float64)
        self.C_boundary = torch.tensor(data['C_boundary'], dtype=torch.float64)
        self.Boundary   = torch.tensor(data['Boundary'], dtype=torch.float64, requires_grad=True)

        # truth solution from FEM
        self.disp_truth = torch.tensor(data['disp_data'], dtype=torch.float64)

        # connectivity matrix - this helps you to plot the figure but we do not need it for PINN
        self.t_connect  = torch.tensor(data['t'].astype(float), dtype=torch.float64)

        # all collocation points
        self.x_full = torch.tensor(data['p_full'], dtype=torch.float64,requires_grad=True)

        # collocation points excluding the boundary
        self.x = torch.tensor(data['p'], dtype=torch.float64, requires_grad=True)

        # This chooses 50 fixed points from the truth solution, which we will use for part (e)
        self.rand_index = torch.randint(0, len(self.x_full), (50,))
        self.disp_fix = self.disp_truth[self.rand_index,:]

        # Define material properties
        self.E = 10
        self.mu = 0.3

        # Define stress values at top and right boundary
        self.SIGMA_T = 0
        self.SIGMA_R = 0.1

        self.stiff = self.E/(1-self.mu**2)*torch.tensor([[1,self.mu,0],[self.mu,1,0],[0,0,(1-self.mu)/2]]) # Hooke's law for plane stress
        self.stiff = self.stiff.unsqueeze(0)

        # Broadcast stiffness for batch multiplication later
        self.stiff_bc = self.stiff
        self.stiff = torch.broadcast_to(self.stiff, (len(self.x),3,3))
        self.stiff_bc = torch.broadcast_to(self.stiff_bc, (len(self.Boundary),3,3))

    @jit.script_method
    def get_loss(self):
        # To compute stress from stress net
        sigma = self.stress_net(self.x)
        # To compute displacement from disp net
        disp     = self.disp_net(self.x)

        # displacement in x direction
        u = disp[:,0]
        # displacement in y direction
        v = disp[:,1]

        # find the derivatives
        grad_outputs_u : typing.List[typing.Optional[torch.Tensor]] = [torch.ones_like(u),]
        grad_outputs_v : typing.List[typing.Optional[torch.Tensor]] = [torch.ones_like(v),]
        dudx = torch.autograd.grad([u,], [self.x,], grad_outputs=grad_outputs_u, create_graph=True)[0]
        dvdx = torch.autograd.grad([v,], [self.x,], grad_outputs=grad_outputs_v, create_graph=True)[0]
        if dudx is None: dudx = torch.full_like(u, float("nan"), dtype=torch.float64)
        if dvdx is None: dvdx = torch.full_like(v, float("nan"), dtype=torch.float64)

        # Define strain
        e_11 = dudx[:,0].unsqueeze(1)
        e_22 = dvdx[:,1].unsqueeze(1)
        e_12 = 0.5*(dudx[:,1] + dvdx[:,0]).unsqueeze(1)

        e = torch.cat((e_11,e_22,e_12), 1)
        e = e.unsqueeze(2)

        # Define augment stress
        sig_aug = torch.bmm(self.stiff, e).squeeze(2)

        # Define constitutive loss - forcing the augment stress to be equal to the neural network stress
        loss_cons = loss_func(sig_aug, sigma)

        # find displacement and stress at the boundaries
        disp_bc = self.disp_net(self.Boundary)
        sigma_bc = self.stress_net(self.Boundary)
        u_bc = disp_bc[:,0]
        v_bc = disp_bc[:,1]

        # Compute the strain and stresses at the boundary
        grad_outputs_u_bc : typing.List[typing.Optional[torch.Tensor]] = [torch.ones_like(u_bc),]
        grad_outputs_v_bc : typing.List[typing.Optional[torch.Tensor]] = [torch.ones_like(v_bc),]
        dudx_bc = torch.autograd.grad([u_bc,], [self.Boundary,], grad_outputs=grad_outputs_u_bc, create_graph=True)[0]
        dvdx_bc = torch.autograd.grad([v_bc,], [self.Boundary,], grad_outputs=grad_outputs_v_bc, create_graph=True)[0]
        if dudx_bc is None: dudx_bc = torch.full_like(u_bc, float("nan"))
        if dvdx_bc is None: dvdx_bc = torch.full_like(v_bc, float("nan"))

        e_11_bc = dudx_bc[:,0].unsqueeze(1)
        e_22_bc = dvdx_bc[:,1].unsqueeze(1)
        e_12_bc = 0.5*(dudx_bc[:,1] + dvdx_bc[:,0]).unsqueeze(1)

        e_bc = torch.cat((e_11_bc,e_22_bc,e_12_bc), 1)
        e_bc = e_bc.unsqueeze(2)

        sig_aug_bc = torch.bmm(self.stiff_bc, e_bc).squeeze(2)

        # force the augment stress to agree with the NN stress at the boundary
        loss_cons_bc = loss_func(sig_aug_bc, sigma_bc)

        #============= equilibrium ===================#

        sig_11 = sigma[:,0]
        sig_22 = sigma[:,1]
        sig_12 = sigma[:,2]

        # stress equilibrium in x and y direction
        grad_outputs_dsig11dx : typing.List[typing.Optional[torch.Tensor]] = [torch.ones_like(sig_11),]
        grad_outputs_dsig22dx : typing.List[typing.Optional[torch.Tensor]] = [torch.ones_like(sig_22),]
        grad_outputs_dsig12dx : typing.List[typing.Optional[torch.Tensor]] = [torch.ones_like(sig_12),]
        dsig11dx = torch.autograd.grad([sig_11,], [self.x,], grad_outputs=grad_outputs_dsig11dx, create_graph=True)[0]
        dsig22dx = torch.autograd.grad([sig_22,], [self.x,], grad_outputs=grad_outputs_dsig22dx, create_graph=True)[0]
        dsig12dx = torch.autograd.grad([sig_12,], [self.x,], grad_outputs=grad_outputs_dsig12dx, create_graph=True)[0]
        if dsig11dx is None: dsig11dx = torch.full_like(sig_11, float("nan"))
        if dsig22dx is None: dsig22dx = torch.full_like(sig_22, float("nan"))
        if dsig12dx is None: dsig12dx = torch.full_like(sig_12, float("nan"))

        eq_x1 = dsig11dx[:,0]+dsig12dx[:,1]
        eq_x2 = dsig12dx[:,0]+dsig22dx[:,1]

        # zero body forces
        f_x1 = torch.zeros_like(eq_x1)
        f_x2 = torch.zeros_like(eq_x2)

        loss_eq1 = loss_func(eq_x1, f_x1)
        loss_eq2 = loss_func(eq_x2, f_x2)
        #========= boundary ========================#

        # specify the boundary condition
        tau_R = self.SIGMA_R
        tau_T = self.SIGMA_T
        #
        u_L= self.disp_net(self.L_boundary)
        u_B = self.disp_net(self.B_boundary)

        sig_R = self.stress_net(self.R_boundary)
        sig_T = self.stress_net(self.T_boundary)
        sig_C = self.stress_net(self.C_boundary)

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
        loss_BC_C = loss_func(sig_C[:,0]*self.C_boundary[:,0]+sig_C[:,2]*self.C_boundary[:,1], torch.zeros_like(sig_C[:, 0]))  \
                    + loss_func(sig_C[:,2]*self.C_boundary[:,0]+sig_C[:,1]*self.C_boundary[:,1], torch.zeros_like(sig_C[:, 0]))

        # ======= uncomment below for part (e) =======================
            # data_loss_fix
            #x_fix = x_full[rand_index, :]
            #u_fix = disp_net(x_fix)
            #loss_fix = loss_func(u_fix,disp_fix)
            #loss = loss_eq1+loss_eq2+loss_cons+loss_BC_L+loss_BC_B+loss_BC_R+loss_BC_T+loss_BC_C+loss_cons_bc + 100*loss_fix

        # Define loss function:
        loss = loss_eq1+loss_eq2+loss_cons+loss_BC_L+loss_BC_B+loss_BC_R+loss_BC_T+loss_BC_C+loss_cons_bc
        return loss
    
    def plot(self):
        # Plot the stress

        stiff = self.E / (1 - self.mu ** 2) * torch.tensor([[1, self.mu, 0], [self.mu, 1, 0], [0, 0, (1 - self.mu) / 2]])
        stiff = stiff.unsqueeze(0)

        stiff_bc = stiff
        stiff_full = stiff
        stiff = torch.broadcast_to(stiff, (len(self.x), 3, 3))

        stiff_bc = torch.broadcast_to(stiff_bc, (len(self.Boundary), 3, 3))
        stiff_full = torch.broadcast_to(stiff_full, (len(self.x_full), 3, 3))

        u_full = model.disp_net(self.x_full)
        stress_full = model.stress_net(self.x_full)

        xx = self.x_full[:,0].detach().numpy()
        yy = self.x_full[:,1].detach().numpy()
        sig11 = stress_full[:,1].detach().numpy()

        connect =(self.t_connect -1).detach().numpy()

        triang = mtri.Triangulation(xx, yy, connect)

        u_11 = u_full[:,0].detach().numpy()

        u = u_full[:, 0]
        v = u_full[:, 1]

        dudx = torch.autograd.grad(u, self.x_full, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        dvdx = torch.autograd.grad(v, self.x_full, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        e_11 = dudx[:, 0].unsqueeze(1)
        e_22 = dvdx[:, 1].unsqueeze(1)
        e_12 = 0.5 * (dudx[:, 1] + dvdx[:, 0]).unsqueeze(1)

        e = torch.cat((e_11, e_22, e_12), 1)
        e = e.unsqueeze(2)

        sigma = torch.bmm(stiff_full, e).squeeze(2)

        plt.figure(2)
        plt.clf()
        plt.tricontourf(triang,sigma[:,0].detach().numpy())
        plt.colorbar()
        plt.show()

# PINN requires super large number of iterations to converge (on the order of 50e^3-100e^3)
iterations = int(100e3)

loss_func = torch.nn.MSELoss()
model = JITModule()
params = list(model.stress_net.parameters()) + list(model.disp_net.parameters())
optimizer = torch.optim.AdamW(params)

def train():
    for epoch in tqdm(range(iterations)):
        optimizer.zero_grad() #! test the other method which should be faster
        loss = model.get_loss()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}")

train()

model.plot()