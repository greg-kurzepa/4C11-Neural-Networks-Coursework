#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_mlp = pd.read_excel("code\\exported_traces\\balmy-deluge-171 best mlp.xlsx")
df_mlp_dropout = pd.read_excel("code\\exported_traces\\likely-hill-195 best mlp dropout.xlsx")
df_rnn = pd.read_excel("code\\exported_traces\\dropout-0.0-179 best resnet.xlsx")
df_rnn_dropout = pd.read_excel("code\\exported_traces\\dropout-0.5-187 best resnet dropout.xlsx")

# %%

# plot training and validation loss of df_mlp and df_mlp_dropout in one subplot
# plot training and validation accuracy of df_mlp and df_mlp_dropout in the second

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(2, 2, figsize=(10,10))

ax[0,0].plot(df_mlp["training_loss"][:500], label="train loss, no dropout", c=colors[0], alpha=0.5)
ax[0,0].plot(df_mlp["validation_loss"][:500], label="test loss, no dropout", c=colors[0])
ax[0,0].plot(df_mlp_dropout["training_loss"], label="train loss, dropout", c=colors[1], alpha=0.5)
ax[0,0].plot(df_mlp_dropout["validation_loss"], label="test loss, dropout", c=colors[1])
ax[0,0].set_xlim(0,500)
ax[0,0].set_ylim(bottom=0, top=0.2)
ax[0,0].set_title("MLP Loss")
ax[0,0].legend()

ax[0,1].plot(df_mlp["training_accuracy"][:500], label="train accuracy, no dropout", c=colors[0], alpha=0.5)
ax[0,1].plot(df_mlp["validation_accuracy"][:500], label="test accuracy, no dropout", c=colors[0])
ax[0,1].plot(df_mlp_dropout["training_accuracy"], label="train accuracy, dropout", c=colors[1], alpha=0.5)
ax[0,1].plot(df_mlp_dropout["validation_accuracy"], label="test accuracy, dropout", c=colors[1])
ax[0,1].set_xlim(0,500)
ax[0,1].set_title("MLP Accuracy")
ax[0,1].set_ylim(bottom=0.9, top=1)
# ax[0,1].legend()

ax[1,0].plot(df_rnn["training_loss"], label="train loss, no dropout", c=colors[0], alpha=0.5)
ax[1,0].plot(df_rnn["validation_loss"], label="test loss, no dropout", c=colors[0])
ax[1,0].plot(df_rnn_dropout["training_loss"], label="train loss, dropout", c=colors[1], alpha=0.5)
ax[1,0].plot(df_rnn_dropout["validation_loss"], label="test loss, dropout", c=colors[1])
ax[1,0].set_xlim(0, 124)
ax[1,0].set_title("ResNet Loss")
ax[1,0].set_ylim(bottom=0, top=0.2)
# ax[1,0].legend()

ax[1,1].plot(df_rnn["training_accuracy"], label="train accuracy, no dropout", c=colors[0], alpha=0.5)
ax[1,1].plot(df_rnn["validation_accuracy"], label="test accuracy, no dropout", c=colors[0])
ax[1,1].plot(df_rnn_dropout["training_accuracy"], label="train accuracy, dropout", c=colors[1], alpha=0.5)
ax[1,1].plot(df_rnn_dropout["validation_accuracy"], label="test accuracy, dropout", c=colors[1])
ax[1,1].set_xlim(0, 124)
ax[1,1].set_title("ResNet Accuracy")
ax[1,1].set_ylim(bottom=0.9, top=1)
# ax[1,1].legend()

# fig.suptitle("Title centered above all subplots", fontsize=14)
# fig.subplots_adjust(wspace=0.1)
# fig.subplots_adjust(hspace=0.1)
plt.tight_layout()
plt.show()
# %%
