import pandas as pd
import matplotlib.pyplot as plt

# Load the data from loss_q2.xlsx
data = pd.read_excel("loss_q2.xlsx")

# Plot
names = ["vanilla ", "scheduler ", "more_width ", "more_modes "]
types = ["training_loss", "validation_loss"]
fig, ax = plt.subplots(1,2)
plt.tight_layout()

for i in range(2):
    for j in range(4):
        ax[i].plot(data[names[j] + types[i]], label="FNO "+names[j], alpha=0.75)
    ax[i].set_title(types[i])
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('L2 Loss')
    ax[i].set_ylim(bottom=0, top=0.2)
    ax[i].set_xlim(left=0, right=99)
ax[0].plot(data["unet training_loss"], label="U-Net", alpha=0.75)
ax[1].plot(data["unet validation_loss"], label="U-Net", alpha=0.75)
ax[1].legend()
plt.show()