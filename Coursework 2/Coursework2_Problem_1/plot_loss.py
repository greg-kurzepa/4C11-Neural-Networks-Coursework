#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("loss_q1.xlsx")
# %%
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(1,2)
ax[0].plot(df["Step"], df["pde"], label="Pure PDE Training, PDE Loss", c=cols[0])
ax[1].plot(df["Step"], df["pde_data_dataloss"], label="PDE+Data Training, Data Loss", c=cols[1], linestyle="--")
ax[1].plot(df["Step"], df["pde_data_nodataloss"], label="PDE+Data Training, PDE Loss", c=cols[1])

ax[0].set_xlim(left=0, right=500)
ax[1].set_xlim(left=0, right=500)
ax[0].ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax[1].ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax[0].set_ylim(bottom=0, top=0.0001)
ax[1].set_ylim(bottom=0, top=0.001)
ax[0].set_xlabel("Epoch")
ax[1].set_xlabel("Epoch")
ax[0].legend()
ax[1].legend()

plt.show()
# %%
