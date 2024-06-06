import numpy as np
import matplotlib.pyplot as plt
import os

thisdir = os.path.abspath('')

# path to the scripts directory of the repository
scriptsdir = os.path.split(thisdir)[0]

# path to the main directory of the repository
maindir = os.path.split(scriptsdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(maindir, 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')

# path to the output_source subsubdirectory
output_source_dir = os.path.join(datadir, 'output_source')
# 定义参数范围和步长
k = np.linspace(0.1, 1, 20)
l = np.linspace(0, 1, 20)
K, L = np.meshgrid(k, l)

# 定义参数和计算R0的函数
p = 0.5
w = 0.5
beta_gammas = [1.3, 1.2, 1.0, 0.8]


def calc_R0(beta_gamma):
    return (beta_gamma / K) * (p * K + (1 - p) * w * (1 - L + K * L))


# 生成四幅子图，分别对应不同的beta_gamma
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle=(r'$p=0.5$,w=$0.5$')

for i, ax in enumerate(axes.flat):
    beta_gamma = beta_gammas[i]
    R0 = calc_R0(beta_gamma)
    im = ax.imshow(R0, cmap='Blues', extent=[0.1, 1,0, 1], origin='lower')
    cs = ax.contour(K, L, R0, levels=[1], colors='red')
    ax.set_title(r'$\beta/\gamma=$' + str(beta_gamma))
    ax.set_xlabel('k')
    ax.set_ylabel('l')
    ax.set_xticks(np.linspace(0.1, 1, 5))
    ax.set_yticks(np.linspace(0.1, 1, 5))
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)


plt.tight_layout()
plt.show()
file_name='R0_with_betagamma_blue.pdf'
fig_path = os.path.join(resultsdir, 'R0', file_name)
fig.savefig(fig_path, format='pdf')
