#%%
import torch

result = torch.tensor([0.1,0.2,0.6,0.7], dtype=torch.float32)
target = torch.tensor([1, 1, 0, 0], dtype=torch.float32)

print(torch.sum((result >= 0.5) == target).item())

loss = torch.nn.BCELoss()
print(loss(result, target).item())
# %%
