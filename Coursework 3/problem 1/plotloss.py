#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_df = pd.read_csv("test_loss.csv")
train_df = pd.read_csv("train_loss.csv")

fig, ax = plt.subplots(1, 2, figsize=(8,4), sharex=True, sharey=True, dpi=300)
plt.tight_layout()
plt.subplots_adjust(top=0.89)

ax[0].plot(test_df["hidden1"], label="1 hidden variable")
ax[0].plot(test_df["hidden2"], label="2 hidden variables")
ax[0].plot(test_df["hidden100"], label="100 hidden variables")
ax[0].set_xlim(left=0, right=100)
ax[0].set_ylim(bottom=0, top=6e-5)
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
# ax[0].legend()
ax[0].set_title("Test Loss")

ax[1].plot(train_df["hidden1"], label="1 hidden variable")
ax[1].plot(train_df["hidden2"], label="2 hidden variables")
ax[1].plot(train_df["hidden100"], label="100 hidden variables")
ax[1].set_xlabel("Epochs")
ax[1].legend()
ax[1].set_title("Train Loss")

plt.show()
# %%
