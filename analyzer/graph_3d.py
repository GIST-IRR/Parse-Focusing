import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from utils import *

# Create a figure
fig = plt.figure()

# Create a 3D axes object
ax = fig.add_subplot(111, projection='3d')

NT = np.arange(0, 1, 0.01)
TN = np.arange(0, 1, 0.01)
NT, TN = np.meshgrid(NT, TN)

# Dataset
dataset = {
    # 1: 131,
    2: 261,
    3: 297,
    4: 401,
    5: 491,
    6: 598,
    7: 726,
    8: 905, 
    9 :978, 
    10 :1219, 
    11 :1168, 
    12 :1319, 
    13 :1439, 
    14 :1494, 
    15 :1453, 
    16 :1570, 
    17 :1526, 
    18 :1633, 
    19 :1470, 
    20 :1562, 
    21 :1524, 
    22 :1489, 
    23 :1449, 
    24 :1387, 
    25 :1393, 
    26 :1257, 
    27 :1213, 
    28 :1104, 
    29 :999, 
    30 :934, 
    31 :790, 
    32 :703, 
    33 :599, 
    34 :552, 
    35 :516, 
    36 :451, 
    37 :418, 
    38 :364, 
    39 :313, 
    40 :248,
}

optimum = np.inf
opt_point = ()
nn, nt, tn, tt = 0.25, 0.25, 0.25, 0.25

def update(TT):
    global NT, TN, optimum, opt_point
    tt_org = np.full_like(NT, TT)
    NN = (1 - NT - TN - TT).clip(1e-9, 1)

    nn, nt, tn, tt = NN, NT.copy(), TN.copy(), tt_org

    # global NN, TT
    # NT = (1 - NN - TT) / 2
    # TN = NT

    nn, nt, tn, tt = map(
        lambda x: torch.tensor(x).float().log().clamp(-1e9, 1e9),
        (nn, nt, tn, tt)
    )
    t = torch.logsumexp(torch.stack([nn, nt, tn, tt]), dim=0)
    nn, nt, tn, tt = map(
        lambda x: x - t,
        (nn, nt, tn, tt)
    )
    
    Z = torch.full_like(nn, -np.inf)
    for k, v in dataset.items():
        l = loss_function(k, nn, nt, tn, tt, log=True) \
            + torch.log(torch.tensor(v, dtype=torch.float32))
        Z = torch.logsumexp(torch.stack([Z, l]), dim=0)

    # nt = np.where(TT > 1e-8, NT, 0)
    # tn = np.where(TT > 1e-8, TN, 0)
    # Z = np.where(TT > 1e-8, Z, 0)
    # tt = np.where(TT > 1e-8, TT, 0)

    # Z = np.where(Z > 1e-8, -np.log(Z), np.inf)
    nn, nt, tn, tt, Z = \
        nn.numpy(), nt.numpy(), tn.numpy(), tt.numpy(), Z.numpy()
    
    new_optimum = Z.min()
    if new_optimum > -1e9:
        if new_optimum < optimum:
            optimum = new_optimum
            idx = np.unravel_index(Z.argmin(), Z.shape)
            opt_point = (NN[idx], NT[idx], TN[idx], tt_org[idx])

    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(2.5, 10)

    ax.set_title(f'TT = {TT:.2f}, Opt = {optimum:.2f}, Pt = {opt_point}')
    ax.set_xlabel(r'$N\rightarrow N T$')
    ax.set_ylabel(r'$N\rightarrow T N$')
    ax.set_zlabel(r'Loss')
    ax.plot_surface(NT, TN, Z, cmap=plt.cm.YlGnBu_r, alpha=0.9)
    # c = plt.cm.YlGnBu_r(np.random.rand(len(X), len(Y), len(Z)))

    # # Plot the surface.
    # ax.plot_surface(NT, TN, Z, cmap=plt.cm.YlGnBu_r)
    # ax.plot_surface(
    #     NT, TN, Z, facecolors=c, rstride=1, cstride=1, linewidth=0, alpha=0.5
    # )

anim = anim.FuncAnimation(
    fig, update, frames=np.arange(0, 1, 0.01), interval=500)

# plt.savefig('graph_3d.png')
anim.save('graph_3d_len6_indNNTT.gif', fps=15)