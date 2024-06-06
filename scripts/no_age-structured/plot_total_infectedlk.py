import csv

import numpy as np
import copy
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sbn

# set some initial paths

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


def plot_lk_states(state,l,l_v,k,k_v,isSave = True):
    file_name1 = f"{state}_numbers_no_age_p=0.10_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path1 = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name1)
    print(file_path1 + ' was read')

    file_name5 = f"{state}_numbers_no_age_p=0.50_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path5 = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name5)
    print(file_path5 + ' was read')

    file_name9 = f"{state}_numbers_no_age_p=0.90_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path9 = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name9)
    print(file_path1 + ' was read')

    infected1 = np.loadtxt(file_path1, skiprows=1, delimiter=',')
    infected1 = infected1[:160].T

    infected5 = np.loadtxt(file_path5, skiprows=1, delimiter=',')
    infected5 = infected5[:160].T

    infected9 = np.loadtxt(file_path9, skiprows=1, delimiter=',')
    infected9 = infected9[:160].T

    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8}

    fig, (ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(15, 5))  # 调整figsize宽度
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)


    ax1[0].set_title("总感染者")
    ax1[0].set_ylabel('感染人数')
    ax1[0].plot(infected1[indices['infectedN']]+infected1[indices['asymptomaticV']]+infected1[indices['infectedV']],linewidth=3)
    ax1[0].plot(infected5[indices['infectedN']]+infected5[indices['asymptomaticV']]+infected5[indices['infectedV']],color='coral',linewidth=3)
    ax1[0].plot(infected9[indices['infectedN']]+infected9[indices['asymptomaticV']]+infected9[indices['infectedV']],color='indigo',linewidth=3)
    ax1[0].set_xlabel("时间（天）")

    ax1[1].set_title("无症状感染者")
    ax1[1].plot(infected1[indices['asymptomaticV']],label='p=0.1',linewidth=3)
    ax1[1].plot(infected5[indices['asymptomaticV']],label='p=0.5',color='coral',linewidth=3)
    ax1[1].plot(infected9[indices['asymptomaticV']],label='p=0.9',color='indigo',linewidth=3)
    ax1[1].set_xlabel("时间（天）")

    handles, labels = ax1[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.show()

    if isSave:
        file_name = f"Total&Asymptomatic_{l}={l_v:.2f}_{k}={k_v:.2f}.png"
        fig_path = os.path.join(resultsdir, 'Master_graduation', file_name)
        fig.savefig(fig_path, format='png',dpi=300)
        # file_name = f"{country}_{location}_per_day_new_cases_of_infectious_{num_agebrackets}.eps"
        # fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        # fig.savefig(fig_path, format='eps')
        pass


def plot_cumulative_incidence(state,l,l_v,k,k_v,isSave = True):
    file_name1 = f"cumulative_incidence_p=0.1_{l}_{l_v:.1f}_{k}.csv"
    file_path1 = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name1)
    print(file_path1 + ' was read')

    file_name5 = f"cumulative_incidence_p=0.5_{l}_{l_v:.1f}_{k}.csv"
    file_path5 = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name5)
    print(file_path5 + ' was read')

    file_name9 = f"cumulative_incidence_p=0.9_{l}_{l_v:.1f}_{k}.csv"
    file_path9 = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name9)
    print(file_path1 + ' was read')

    infected1 = np.loadtxt(file_path1, skiprows=1, delimiter=',')
    infected1 = infected1[:300].T

    infected5 = np.loadtxt(file_path5, skiprows=1, delimiter=',')
    infected5 = infected5[:300].T

    infected9 = np.loadtxt(file_path9, skiprows=1, delimiter=',')
    infected9 = infected9[:300].T

    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8}

    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False,figsize=(12,5))
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)


    ax1.set_title("总感染概率")
    ax1.set_ylabel('感染率（%）')
    ax1.plot(infected1[indices['quarantined']], label='p=0.1',linewidth=3)
    ax1.plot(infected5[indices['quarantined']], label='p=0.5',color='coral',linewidth=3)
    ax1.plot(infected9[indices['quarantined']], label='p=0.9',color='indigo',linewidth=3)
    ax1.set_xlabel("时间（天）")


    handles, labels = ax1.get_legend_handles_labels()
    fig.legend()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.show()

    if isSave:
        file_name = f"Total&Asymptomatic_{l}={l_v:.2f}_{k}={k_v:.2f}.png"
        fig_path = os.path.join(resultsdir, 'Master_graduation', file_name)
        fig.savefig(fig_path, format='png',dpi=300)
        # file_name = f"{country}_{location}_per_day_new_cases_of_infectious_{num_agebrackets}.eps"
        # fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        # fig.savefig(fig_path, format='eps')
        pass



k_v = 0.2  # 0-1
l_v = 0.2  # 0-1
k = 'k'
l = 'l'
state = "all_states"
plot_lk_states(state,l, l_v, k, k_v)

plot_cumulative_incidence(state,l, l_v, k, k_v)