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

# 设置 k 和 l 的范围
k = np.linspace(0.1, 1, 20)
l = np.linspace(0, 1, 20)

# 生成二维数组
K, L = np.meshgrid(k, l)

# 计算函数值
p=0.5
w = 0.5
beta = 1
gamma_inverse= 1.2
gamma = 1/gamma_inverse
R0 = (beta/gamma/k)*(p*K+(1-p)*w*(1-L+K*L))
data = K - w + w*L - w*K*L

# 绘制热力图
fig, ax = plt.subplots()
im = ax.contourf(data, cmap='YlOrRd', extent=[0.1, 1,0, 1], levels=10)

# 翻转颜色映射
im.set_cmap(plt.cm.YlOrRd_r)

# 添加等高线
cs = ax.contour(K, L, data, levels=[0], colors='royalblue',linewidths=2.5)
# ax.clabel(cs, inline=False, fontsize=10)

ax.set_title('Effect of vaccination on transmission')
# 添加横纵坐标标签
ax.set_xlabel('$k$')
ax.set_ylabel('$l$')
# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)

file_name = f"R0_k_l.png"
fig_path = os.path.join(resultsdir, 'Master_graduation', file_name)
fig.savefig(fig_path, format='png',dpi=300)

# 显示图形
plt.show()
