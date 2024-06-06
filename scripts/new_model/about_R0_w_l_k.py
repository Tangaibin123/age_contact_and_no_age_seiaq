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
w_values = [0.2, 0.4, 0.6, 0.8]
extent = [0.1, 1, 0, 1]

# 创建一个大图形
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# 用于存储所有数据的列表
all_data = []

for i in range(2):
    for j in range(2):
        K, L = np.meshgrid(np.linspace(0.1, 1, 20), np.linspace(0, 1, 20))
        data = K - w_values[i*2+j] + w_values[i*2+j]*L - w_values[i*2+j]*K*L
        all_data.extend(data.flatten())
# 找到所有数据的最小和最大值
min_value = min(all_data)
max_value = max(all_data)

# 计算数据和绘制子图
for i in range(2):
    for j in range(2):
        K, L = np.meshgrid(np.linspace(0.1, 1, 20), np.linspace(0, 1, 20))
        data = K - w_values[i*2+j] + w_values[i*2+j]*L - w_values[i*2+j]*K*L

        #存放每个值
        all_data.extend(data.flatten())
        #绘图
        im = axs[i, j].contourf(data, cmap='YlOrRd', extent=extent, origin='lower', levels=10,vmin=min_value, vmax=max_value)
        # 翻转颜色映射
        im.set_cmap(plt.cm.YlOrRd_r)

        #等高线
        axs[i, j].contour(K, L, data, levels=[0],colors='royalblue', linewidths=2.5)
        axs[i, j].set_title(f'$w={w_values[i*2+j]}$')
        axs[i, j].set_xlabel('$k$', fontsize=12)
        axs[i, j].set_ylabel('$l$', fontsize=12)


# 添加整个图的颜色条，并设置范围
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, location='bottom', ticks=np.linspace(min_value, max_value, 11))
cbar.ax.set_position([0.2, 0.1, 0.6, 0.03])

# 调整子图间距和整个图的标题
fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.9, bottom=0.2)
fig.suptitle('Effect under different $w$', fontsize=15)
file_name = f"R0_w_k_l.png"
fig_path = os.path.join(resultsdir, 'Master_graduation', file_name)
fig.savefig(fig_path, format='png',dpi=300)
plt.show()
