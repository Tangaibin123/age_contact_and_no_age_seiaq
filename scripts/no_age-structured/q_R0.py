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

p = 0.2  # percent of unvaccinated
w = 0.8
beta = 0.6  #
q = 0.6  #
l = 0.8  #
sigma_inverse = 6.0  # mean latent period
k = 0.5  #
gamma_inverse = 4.0  # mean quarantined period

x_label = [i / 10.0 for i in range(11)]
y_label = [i / 10.0 for i in range(10, -1, -1)]

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 4))
plt.subplots_adjust(left=0.09, right=0.95, bottom=0.2, top=0.85, wspace=0.1, hspace=0.4)
plt.suptitle('R0', fontsize=20)

m_R0 = np.zeros(11)
for k in range(10):
    m_R0[k] = get_R0(beta, p, w, q, l, (k+1) / 10.0, gamma_inverse)

plt.title('R0')
plt.show(m_R0)