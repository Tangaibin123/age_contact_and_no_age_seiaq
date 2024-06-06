import csv

import numpy as np
import copy
import os
import pandas as pd
import datetime
import seaborn as sbn
import matplotlib.pyplot as plt

# set some initial paths

# path to the directory where this script lives
thisdir = os.path.abspath('')
print(thisdir)
# path to the scripts directory of the repository
scriptsdir = os.path.split(thisdir)[0]
print(scriptsdir)
# path to the main directory of the repository
maindir = os.path.split(scriptsdir)[0]
print(maindir)
# path to the results subdirectory
resultsdir = os.path.join(maindir, 'results')
print(resultsdir)

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')

# path to the output_source subsubdirectory
output_source_dir = os.path.join(datadir, 'output_source')


def swap(matrix):
    len = matrix.shape
    a = np.ones(len)

    for i in range(len[0]):
        a[i, :] = matrix[len[0] - 1 - i, :]
    return a


# %% 计算R0
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
l = 0.5  #
sigma_inverse = 6.0  # mean latent period
k = 0.5  #
gamma_inverse = 4.0  # mean quarantined period


x_label = [i / 10.0 for i in range(11)]
y_label = [i / 10.0 for i in range(10, -1, -1)]

# fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(4, 4))
# plt.subplots_adjust(left=0.09, right=0.95, bottom=0.2, top=0.85, wspace=0.1, hspace=0.4)
# plt.suptitle('R0', fontsize=20)
fig = plt.figure()
plt.title('R0 p=0.5 w=0.5',fontsize=20)

m_R0 = np.zeros((11, 11))
print(m_R0)

for l in range(11):
    for k in range(11):
        m_R0[l][k] = get_R0(beta, p, w, q, l/10, (k+1) / 10.0, gamma_inverse)
# m_R0 = swap(m_R0)
sbn.heatmap(m_R0, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label)
plt.xlabel('k')
plt.ylabel('l')
plt.show()


m_R0 = np.zeros((11, 11))
# l = 0.5
# plt.title('R0 w=0.5 l=0.5',fontsize=20)
# for p in range(11):
#     for k in range(11):
#         m_R0[p][k] = get_R0(beta, p/10, w, q, l, (k+1) / 10.0, gamma_inverse)
# m_R0 = swap(m_R0)
# sbn.heatmap(m_R0, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label)
# plt.xlabel('k')
# plt.ylabel('p')
# plt.show()


# # fig = plt.figure()
# plt.title('R0 k=0.5 l=0.5',fontsize=20)
# m_R0 = np.zeros((11, 11))
# for p in range(11):
#     for w in range(11):
#         m_R0[p][w] = get_R0(beta, p / 10.0, w / 10.0, q, l, k, gamma_inverse)
# # m_R0 = swap(m_R0)
# sbn.heatmap(m_R0, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label)
# plt.xlabel('p')
# plt.ylabel('w')
# plt.show()



# m_R0 = np.zeros((11, 11))
# w = 0.8
# for p in range(11):
#     for l in range(11):
#         m_R0[p][l] = get_R0(beta, p / 10.0, w, q, l / 10.0, k, gamma_inverse)
# m_R0 = swap(m_R0)
# sbn.heatmap(m_R0, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label, ax=axes[1])
#
# axes[1].set_xlabel('l', labelpad=15, fontsize=15)




# fig_name = 'R0_change_with_param.pdf'
# fig_path = os.path.join(resultsdir, 'no_age-structured_result', fig_name)
# fig.savefig(fig_path, format='pdf')
#
# fig_name = 'R0_change_with_param.eps'
# fig_path = os.path.join(resultsdir, 'no_age-structured_result', fig_name)
# fig.savefig(fig_path, format='eps')