import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

def train(model, loss_function, optimiser, train_dl, valid_dl, epochs, batch_size, dev, show_plot, log_wandb, clip_value=None):

    print(f"Start training for {epochs} epochs...")

    loss_train_list = []
    loss_valid_list = [] 

    # Iterate over epochs
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_train_loss = 0

        # Iterate over batches
        # for idx, (input, target) in (pbar := tqdm(enumerate(train_dl), total=len(train_dl))):
        for idx, (input, target) in enumerate(train_dl):
            result = model(input)
            
            loss = loss_function(result, target)
            epoch_train_loss += loss.item()

            # pbar.set_description(f"TR LOSS: {loss.item():.4f}")
            # pbar.refresh()

            loss.backward()
            # clip gradients
            if clip_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
                # The below approach is more correct (but is slower). It clips each gradient before it backpropagates instead of at its enpoint.
                # for p in model.parameters():
                #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

            optimiser.step()
            optimiser.zero_grad()
            # for p in model.parameters(): p.grad = None # this line replaces optimiser.zero_grad(), is slightly more efficient

            # can be used to view the output of the model at this epoch. Displays the feature, label and model output for that feature.
            # if show_plot and <>:

        loss_train_list.append(epoch_train_loss/len(train_dl))

        # Compute your test loss below
        model.eval()
        epoch_valid_loss = 0
        with torch.no_grad():
            # for idx, (input, target) in (pbar := tqdm(enumerate(valid_dl), total=len(valid_dl))):
            for idx, (input, target) in enumerate(valid_dl):
                input, target = input.to(dev), target.to(dev)

                result = model(input)
                loss = loss_function(result, target)
                epoch_valid_loss += loss.item()

                # pbar.set_description(f"VL LOSS: {loss.item():.4f}")
                # pbar.refresh()

        loss_valid_list.append(epoch_valid_loss/len(valid_dl))
        print(f"Epoch: {epoch}, train loss: {loss_train_list[-1]}, valid loss: {loss_valid_list[-1]}")

        # Save losses to Weights&Biases
        if log_wandb:
            print("Logging wandb")
            wandb.log({"training_loss": loss_train_list[-1], "validation_loss": loss_valid_list[-1]})