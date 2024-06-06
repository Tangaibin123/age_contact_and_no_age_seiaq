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


def plot_per_day_new_cases(country, location, p, p_v, num_agebrackets, isSave=False):
    # file_name = f"{country}_{location}_per_day_new_increase_infectious_numbers_{num_agebrackets}.csv"
    # file_name = f"China_Shanghai_new_case_per_day_18_p=0.50_INP.csv"
    # file_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_NPI.csv"
    file_name = f"{country}_{location}_new_case_per_day_allocation"
    # file_name = f"{country}_{location}_new_case_per_day_18_p=0.50_baseline.csv"
    # file_path = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name)
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'vaccine_allocation', file_name)
    print(file_path + ' was read')
    per_day_new_cases = np.loadtxt(file_path, skiprows=1, delimiter=',')
    per_day_new_cases = per_day_new_cases[:100].T
    # indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
    #            'infectedV': 7, 'quarantined': 8, 'asymptomatic': 9, 'infected': 10, 'infectious': 11}

    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8}

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    plt.suptitle('optimal allocation  new increase with age-structured')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.4)

    ax1[0].set_title('asymptomatic and no-vaccine')
    ax1[0].plot(per_day_new_cases[indices['asymptomaticN']])
    max = np.max(per_day_new_cases[indices['asymptomaticN']])
    ax1[0].axhline(y=max, color='r', linestyle='--')
    ax1[0].text(0, max + 500, str(max), color='r')

    ax1[1].set_title('symptomatic and no-vaccine')
    # ax1[1].set_ylabel('count')
    ax1[1].plot(per_day_new_cases[indices['infectedN']])
    max = np.max(per_day_new_cases[indices['infectedN']])
    ax1[1].axhline(y=max, color='r', linestyle='--')
    ax1[1].text(0, max + 500, str(max), color='r')

    ax2[0].set_title('asymptomatic and vaccine')
    ax2[0].plot(per_day_new_cases[indices['asymptomaticV']])
    max = np.max(per_day_new_cases[indices['asymptomaticV']])
    ax2[0].axhline(y=max, color='r', linestyle='--')
    ax2[0].text(0, max + 500, str(max), color='r')

    ax2[1].set_title('symptomatic and vaccine')
    # ax2[1].set_ylabel('count')
    ax2[1].plot(per_day_new_cases[indices['infectedV']])
    max = np.max(per_day_new_cases[indices['infectedV']])
    ax2[1].axhline(y=max, color='r', linestyle='--')
    ax2[1].text(0, max + 500, str(max), color='r')

    ax3[0].set_title('total asymptomatic')
    ax3[0].set_xlabel('day')
    ax3[0].plot(per_day_new_cases[indices['asymptomaticV']] + per_day_new_cases[indices['asymptomaticN']])
    max = np.max(per_day_new_cases[indices['asymptomaticV']] + per_day_new_cases[indices['asymptomaticN']])
    ax3[0].axhline(y=max, color='r', linestyle='--')
    ax3[0].text(0, max - 5000, str(max), color='r')

    ax3[1].set_title('total symptomatic')
    # ax3[1].set_ylabel('count')
    ax3[1].set_xlabel('day')
    ax3[1].plot(per_day_new_cases[indices['infectedN']] + per_day_new_cases[indices['infectedV']])
    max = np.max(per_day_new_cases[indices['infectedN']] + per_day_new_cases[indices['infectedV']])
    ax3[1].axhline(y=max, color='r', linestyle='--')
    ax3[1].text(0, max - 5000, str(max), color='r')

    plt.show()

    if isSave:
        file_name = f"{country}_{location}_per_day_new_cases_of_infectious_{num_agebrackets}.pdf"
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')
        file_name = f"{country}_{location}_per_day_new_cases_of_infectious_{num_agebrackets}.eps"
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')


p = 'p'
p_v = 0.5
location = 'Shanghai'
country = 'China'
num_agebrackets = 18
plot_per_day_new_cases(country, location, p, p_v, num_agebrackets)
