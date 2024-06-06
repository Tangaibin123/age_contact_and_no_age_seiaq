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
# 定义数据和参数
p = 0.5
w = 0.5
beta_gamma_list = [1.3, 1.2, 1.0, 0.8]
extent = [0.1, 1, 0, 1]

# 创建子图和画布
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# 遍历每个子图
for i in range(2):
    for j in range(2):
        # 计算数据
        K, L = np.meshgrid(np.linspace(0.1, 1, 20), np.linspace(0, 1, 20))
        data = (beta_gamma_list[i*2+j] / K) * (p * K + (1 - p) * w * (1 - L + K * L))
        # 绘制热力图
        im = axs[i, j].imshow(data, cmap='YlOrRd', extent=extent, origin='lower', vmin=0.5, vmax=4)

        # 添加等高线
        axs[i, j].contour(K, L, data, levels=[1], colors='black')
        # 设置子图标题
        axs[i, j].set_title(f'$\\beta/\gamma={beta_gamma_list[i*2+j]}$')
        # 设置横纵坐标标签
        axs[i, j].set_xlabel('$k$', fontsize=12)
        axs[i, j].set_ylabel('$l$', fontsize=12)


# 添加整个图的颜色条
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8,location='bottom')
cbar.ax.set_position([0.2, 0.1, 0.6, 0.03])


# 调整子图间距和整个图的标题
fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.9, bottom=0.2)
cbar.set_label('$R_0$', fontsize=16)
fig.suptitle('$R_0$ under different $\\beta/\gamma$', fontsize=20)

plt.show()


file_name='R0_with_betagamma.pdf'
fig_path = os.path.join(resultsdir, 'R0', file_name)
fig.savefig(fig_path, format='pdf')
