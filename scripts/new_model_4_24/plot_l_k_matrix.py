import csv

import numpy as np
import copy
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sbn
import matplotlib as mpl
# set some initial paths
# mpl.use('pdf')

# path to the directory where this script lives
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


def plot_l_k_matrix(country,location, p, p_v, l, l_v, k, k_v,num_agebrackets=18):

    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8}
    matrix = np.zeros((11,11))
    matrix_day = np.zeros((11,11))
    for i in range(11):
        l_v = i / 10
        for j in range(11):
            k_v = j / 10

            file_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'Per_day_new_case_lk', file_name)
            print(file_path + ' was read')
            per_day_new_cases = np.loadtxt(file_path, skiprows=1, delimiter=',')
            per_day_new_cases = per_day_new_cases[:100].T
            #找出每日新增所有感染者最大值
            max = np.max(per_day_new_cases[indices['asymptomaticV']] + per_day_new_cases[indices['asymptomaticN']] +
                         per_day_new_cases[indices['infectedN']] + per_day_new_cases[indices['infectedV']])
            #找出每日新增最多感染者的时间（天）
            day = np.argmax(np.array(per_day_new_cases[indices['asymptomaticV']] + per_day_new_cases[indices['asymptomaticN']] +
                         per_day_new_cases[indices['infectedN']] + per_day_new_cases[indices['infectedV']]))
            matrix[i][j]= max
            matrix_day[i][j] = day

    x_label = [i / 10.0 for i in range(11)]
    y_label = [i / 10.0 for i in range(10, -1, -1)]
    matrix = np.flipud(matrix)
    sbn.heatmap(matrix, cmap='hot_r', yticklabels=y_label, xticklabels=x_label)
    # plt.imshow(matrix, cmap='hot_r', interpolation='nearest')
    # plt.colorbar()
    plt.xlabel('$k$')
    plt.ylabel('$l$')
    # plt.title('Highest number of new infections per day')
    plt.show()
    fig_name=f'{country}_{location}_max_new_case_{num_agebrackets}_{p}.pdf'
    fig_path=os.path.join(resultsdir, 'new_paper', fig_name)
    # plt.savefig(fig_path, format='pdf')

    ##day
    plt.figure()
    x_label = [i / 10.0 for i in range(11)]
    y_label = [i / 10.0 for i in range(10, -1, -1)]
    matrix_day = np.flipud(matrix_day)
    sbn.heatmap(matrix_day, cmap='GnBu_r', yticklabels=y_label, xticklabels=x_label)
    # plt.imshow(matrix, cmap='hot_r', interpolation='nearest')
    # plt.colorbar()
    plt.xlabel('$k$')
    plt.ylabel('$l$')
    # plt.title('Time(day) of highest number of new infections')
    plt.show()
    fig_name = f'{country}_{location}_max_day_new_case_{num_agebrackets}_{p}.pdf'
    fig_path = os.path.join(resultsdir, 'new_paper', fig_name)
    # plt.savefig(fig_path, format='pdf')
    # print(matrix)
    return 0



p = 'p'
w = 'w'
# beta = 'beta'
q = 'q'
l = 'l'
# param = 'sigma_inverse'
k = 'k'
q_v = 1
# param = 'gamma_inverse'
p_v = 0.5  # 0.1 0.5 0.9  # percent of unvaccinated
w_v = 0.5
beta = 0.5  #
l_v = 0.5 #接种后有症状的比例
sigma_inverse = 5.0  # mean incubation period
k_v = 0.5
gamma_inverse = 1.6  # mean quarantined period

location = 'Shanghai'
country = 'China'
level = 'subnational'
state = "all_states"


plot_l_k_matrix(country,location, p, p_v, l, l_v, k, k_v)