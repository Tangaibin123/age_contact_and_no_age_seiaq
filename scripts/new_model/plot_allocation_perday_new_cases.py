import csv

import numpy as np
import copy
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sbn
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为
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


def plot_per_day_new_cases(country, location, p, p_v, num_agebrackets, isSave=True):
    # file_name = f"{country}_{location}_per_day_new_increase_infectious_numbers_{num_agebrackets}.csv"
    # file_name = f"China_Shanghai_new_case_per_day_18_p=0.50_INP.csv"
    # file_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_NPI.csv"
    file_name = f"{country}_{location}_new_case_per_day_allocation"
    file_name1 = f"{country}_{location}_new_case_per_day_18_p=0.50_baseline.csv"
    # file_name = f"{country}_{location}_new_case_per_day_18_p=0.50_baseline.csv"
    # file_path = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name)
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'vaccine_allocation', file_name)
    file_path1 = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name1)
    print(file_path + ' was read')
    print(file_path1 + 'was read')
    per_day_new_cases = np.loadtxt(file_path, skiprows=1, delimiter=',')
    per_day_new_cases1 = np.loadtxt(file_path1, skiprows=1, delimiter=',')
    per_day_new_cases = per_day_new_cases[:100].T
    per_day_new_cases1 = per_day_new_cases1[:100].T
    # indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
    #            'infectedV': 7, 'quarantined': 8, 'asymptomatic': 9, 'infected': 10, 'infectious': 11}

    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8}

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True)
    # plt.suptitle('Per day new increase with age-structured')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.5)

    ax0[0].set_title('每日新增总感染者人数')
    ax0[0].set_xlabel('时间（天）')
    ax0[0].plot(per_day_new_cases[indices['asymptomaticV']] + per_day_new_cases[indices['asymptomaticN']] +
                per_day_new_cases[indices['infectedN']] + per_day_new_cases[indices['infectedV']],
                label='optimal allocation')
    ax0[0].plot(per_day_new_cases1[indices['asymptomaticV']] + per_day_new_cases1[indices['asymptomaticN']] +
                per_day_new_cases1[indices['infectedN']] + per_day_new_cases1[indices['infectedV']],
                label='avg allocation')
    max = np.max(per_day_new_cases[indices['asymptomaticV']] + per_day_new_cases[indices['asymptomaticN']] +
                 per_day_new_cases[indices['infectedN']] + per_day_new_cases[indices['infectedV']])
    max1 = np.max(per_day_new_cases1[indices['asymptomaticV']] + per_day_new_cases1[indices['asymptomaticN']] +
                  per_day_new_cases1[indices['infectedN']] + per_day_new_cases1[indices['infectedV']])
    ax0[0].axhline(y=max, color='b', linestyle='--')
    ax0[0].axhline(y=max1, color='coral', linestyle='--')
    ax0[0].text(60, max - 5000, str(max), color='b')
    ax0[0].text(60, max1 - 5000, str(max1), color='coral')

    ax0[1].set_title('每日新增的未接种有症状感染者人数')
    # ax1[1].set_ylabel('count')
    ax0[1].plot(per_day_new_cases[indices['infectedN']])
    ax0[1].plot(per_day_new_cases1[indices['infectedN']])
    max = np.max(per_day_new_cases[indices['infectedN']])
    max1 = np.max(per_day_new_cases1[indices['infectedN']])
    ax0[1].axhline(y=max, color='b', linestyle='--')
    ax0[1].axhline(y=max1, color='r', linestyle='--')
    ax0[1].text(60, max + 500, str(max), color='b')
    ax0[1].text(60, max1 + 500, str(max1), color='coral')
    ax0[1].set_xlabel('时间（天）')

    ax1[0].set_title('每日新增无症状感染者人数')
    ax1[0].plot(per_day_new_cases[indices['asymptomaticV']])
    ax1[0].plot(per_day_new_cases1[indices['asymptomaticV']])
    max = np.max(per_day_new_cases[indices['asymptomaticV']])
    max1 = np.max(per_day_new_cases1[indices['asymptomaticV']])
    ax1[0].axhline(y=max, color='b', linestyle='--')
    ax1[0].axhline(y=max1, color='r', linestyle='--')
    ax1[0].text(60, max + 500, str(max) ,color='blue')
    ax1[0].text(60, max1 - 4000, str(max1), color='coral')
    ax1[0].set_xlabel('时间（天）')

    ax1[1].set_title('每日新增接种疫苗的有症状感染者人数')
    # ax2[1].set_ylabel('count'1
    ax1[1].plot(per_day_new_cases[indices['infectedV']],label='最高权重分配')
    ax1[1].plot(per_day_new_cases1[indices['infectedV']],label = '平均分配')
    max = np.max(per_day_new_cases[indices['infectedV']])
    max1 = np.max(per_day_new_cases1[indices['infectedV']])
    ax1[1].axhline(y=max, color='b', linestyle='--',linewidth=0.8)
    ax1[1].axhline(y=max1, color='coral', linestyle='--',linewidth=0.8)
    ax1[1].text(40, max + 500, str(max), color='b')
    ax1[1].text(70, max1 + 500, str(max1), color='coral')
    ax1[1].set_xlabel('时间（天）')

    handles, labels = ax1[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels))
    plt.subplots_adjust(top=0.85)

    plt.show()

    if isSave:
        # file_name = f"optimal&avg_per_day_new_cases_of_infectious.pdf"
        # fig_path = os.path.join(resultsdir, 'new_paper', file_name)
        # fig.savefig(fig_path, format='pdf')
        #
        # file_name = f"optimal&avg_per_day_new_cases_of_infectious.eps"
        # fig_path = os.path.join(resultsdir, 'new_paper', file_name)
        # fig.savefig(fig_path, format='eps')

        file_name = f"optimal&avg_perday_new_cases.png"
        fig_path = os.path.join(resultsdir, 'Master_graduation', file_name)
        fig.savefig(fig_path, format='png',dpi=300)



p = 'p'
p_v = 0.5
location = 'Shanghai'
country = 'China'
num_agebrackets = 18
plot_per_day_new_cases(country, location, p, p_v, num_agebrackets)
