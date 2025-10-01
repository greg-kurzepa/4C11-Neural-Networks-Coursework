#%%
import _dataloader
import unet, fno
import torch
import numpy as np
import matplotlib.pyplot as plt

############################# Models, Loss Fn #############################
params_path_unet = r"C:\Users\gregk\Documents\MyDocuments\IIB\4C11\cw2\Coursework2_Problem_2\code\params\20250317-174829.torchparams"
model_unet = unet.UNet()
model_unet.load_state_dict(torch.load(params_path_unet))
params_path_fno = r"C:\Users\gregk\Documents\MyDocuments\IIB\4C11\cw2\Coursework2_Problem_2\code\params\20250317-225306.torchparams"
model_fno = fno.FNO(12, 12, 32)
model_fno.load_state_dict(torch.load(params_path_fno))
loss_function = _dataloader.LpLoss()

############################# Data processing #############################
# Read data from mat
train_path = r"C:\Users\gregk\Documents\MyDocuments\IIB\4C11\cw2\Coursework2_Problem_2\Darcy_2D_data_train.mat"
test_path = r"C:\Users\gregk\Documents\MyDocuments\IIB\4C11\cw2\Coursework2_Problem_2\Darcy_2D_data_test.mat"

data_reader = _dataloader.MatRead(train_path)
a_train = data_reader.get_a()
u_train = data_reader.get_u()

data_reader = _dataloader.MatRead(test_path)
a_test = data_reader.get_a()
u_test = data_reader.get_u()

# Normalize data
a_normalizer = _dataloader.UnitGaussianNormalizer(a_train)
a_train = a_normalizer.encode(a_train)
a_test = a_normalizer.encode(a_test)

u_normalizer = _dataloader.UnitGaussianNormalizer(u_train)
u_train = u_normalizer.encode(u_train)
u_test = u_normalizer.encode(u_test)

#%%############################ Evaluate #############################

for i in range(a_test.shape[0]):
    # get a and the prediction of u (normalised)
    a = a_test[i]
    u = u_test[i]
    a_in = a.unsqueeze(0).float()
    model_unet.eval()
    model_fno.eval()
    with torch.no_grad():
        out_unet = model_unet(a_in.unsqueeze(0))
        out_fno = model_fno(a_in)
    pred_u_unet = out_unet.squeeze().detach().numpy()
    pred_u_fno = out_fno.squeeze().detach().numpy()
    loss_unet = loss_function(torch.tensor(u), torch.tensor(pred_u_unet))
    loss_fno = loss_function(torch.tensor(u), torch.tensor(pred_u_fno))

    # plot it, two subplots, first for u, second for predicted u (normalised)
    fig, axs = plt.subplots(1, 3)
    plt.tight_layout()
    axs[0].imshow(u, cmap="plasma", interpolation="none")
    axs[0].set_title('True u')
    axs[0].axis('off')
    axs[1].imshow(pred_u_unet, cmap="plasma", interpolation="none")
    axs[1].set_title(f'Predicted u, UNet\nLoss: {loss_unet.item():.4f}')
    axs[1].axis('off')
    axs[2].imshow(pred_u_fno, cmap="plasma", interpolation="none")
    axs[2].set_title(f'Predicted u, FNO\nLoss: {loss_fno.item():.4f}')
    axs[2].axis('off')
    plt.show()
# %%
