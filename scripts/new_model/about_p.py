import numpy as np
import matplotlib.pyplot as plt

# 设置 k 和 l 的范围
k = np.linspace(0, 1, 100)
l = np.linspace(0, 1, 100)

# 生成二维数组
K, L = np.meshgrid(k, l)

# 计算函数值
w = 0.5
data = K - w + w*L - w*K*L

# 绘制热力图
fig, ax = plt.subplots()
im = ax.imshow(data, cmap='Reds_r', extent=[0, 1,1, 0])

# 翻转 y 轴
plt.gca().invert_yaxis()

# 添加等高线
cs = ax.contour(K, L, data, levels=[0], colors='blue')
# ax.clabel(cs, inline=False, fontsize=10)

ax.set_title('Effect of vaccination on transmission')
# 添加横纵坐标标签
ax.set_xlabel('$k$')
ax.set_ylabel('$l$')
# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)

# 显示图形
plt.show()
