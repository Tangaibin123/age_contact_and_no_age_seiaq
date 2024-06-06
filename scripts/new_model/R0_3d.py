import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义一个函数
def get_R0(beta, p, w, q, l, k, gamma_inverse):
    """
    Get the basic reproduction number R0 for a SIR compartmental model with an age dependent susceptibility drop factor and the age specific contact matrix.

    Args:
        beta (float)                              : the transmissibility
        p (float)                                 : 接种的比例（the probability of vaccining）
        w (float)                                 :
        q (float)                                 :
        l (float)                                 :
        k (float)                                 :
        gamma_inverse (float)                     : the mean recovery period

    Returns:
        float: The basic reproduction number for a SEAIQ compartmental model with an age dependent susceptibility drop factor and age specific contact patterns.
    """

    gamma = 1.0 / gamma_inverse
    R0 = beta * ((p + (1 - p) * l * w) * ((1 - q) + k * q))/(k * gamma)
    return R0

p = 0.5  # percent of unvaccinated
w = 0.5
beta = 0.5  #
q = 0.95 #
l = 0.8  #
sigma_inverse = 6.0  # mean latent period
k = 0.5  #
gamma_inverse = 7.0  # mean quarantined period
# 生成网格
l = np.linspace(0, 1, 10)
k = np.linspace(0, 1, 10)
X, Y = np.meshgrid(l, k)
Z = get_R0(beta, p, w, q, X, Y, gamma_inverse)

# 绘制等高线图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.set_xlabel('l Label')
ax.set_ylabel('k Label')
ax.set_zlabel('R0 Label')
# ax.view_init(elev=20, azim=160)
plt.show()
