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
    # file_name = f"{country}_{location}_new_case_per_day_allocation"
    file_name = f"{country}_{location}_new_case_per_day_18_p=0.50_baseline.csv"
    file_name1 = f"{country}_{location}_new_case_per_day_18_p=0.10_baseline.csv"
    file_name9 = f"{country}_{location}_new_case_per_day_18_p=0.90_baseline.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name)
    file_path1 = os.path.join(output_source_dir, 'age-structured_output_source', 'p_0.1', file_name1)
    file_path9 = os.path.join(output_source_dir, 'age-structured_output_source', 'p_0.9', file_name9)
    # file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'vaccine_allocation', file_name)
    print(file_path + ' was read')
    print(file_path1 + ' was read')
    print(file_path9 + ' was read')
    per_day_new_cases = np.loadtxt(file_path, skiprows=1, delimiter=',')
    per_day_new_cases = per_day_new_cases[:100].T

    per_day_new_cases_1 = np.loadtxt(file_path1, skiprows=1, delimiter=',')
    per_day_new_cases_1 = per_day_new_cases_1[:100].T

    per_day_new_cases_9 = np.loadtxt(file_path9, skiprows=1, delimiter=',')
    per_day_new_cases_9 = per_day_new_cases_9[:100].T

    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8}

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(9, 8))
    plt.suptitle("symptomatic and no-vaccine new increase", fontsize=20)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.4)
    axes.set_xlabel('Day', labelpad=15, fontsize=15)
    axes.plot(per_day_new_cases[indices['infectedN']],label = 'p=0.5')
    plt.plot(per_day_new_cases_1[indices['infectedN']],label = 'p=0.1')
    plt.plot(per_day_new_cases_9[indices['infectedN']],label = 'p=0.9')
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(9, 8))
    plt.suptitle("asymptomatic and vaccinated new increase", fontsize=20)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.4)
    axes.set_xlabel('Day', labelpad=15, fontsize=15)
    axes.plot(per_day_new_cases[indices['asymptomaticV']], label='p=0.5')
    plt.plot(per_day_new_cases_1[indices['asymptomaticV']], label='p=0.1')
    plt.plot(per_day_new_cases_9[indices['asymptomaticV']], label='p=0.9')
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(9, 8))
    plt.suptitle("symptomatic and vaccinated new increase", fontsize=20)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.4)
    axes.set_xlabel('Day', labelpad=15, fontsize=15)
    axes.plot(per_day_new_cases[indices['infectedV']], label='p=0.5')
    plt.plot(per_day_new_cases_1[indices['infectedV']], label='p=0.1')
    plt.plot(per_day_new_cases_9[indices['infectedV']], label='p=0.9')
    plt.legend()
    plt.show()

    # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    # plt.suptitle("total symptomatic new increase", fontsize=20)
    # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.4)
    # axes.set_xlabel('Day', labelpad=15, fontsize=15)
    # axes.plot(per_day_new_cases[indices['infectedV']]+per_day_new_cases[indices['infectedN']], label='p=0.5')
    # plt.plot(per_day_new_cases_1[indices['infectedV']]+per_day_new_cases_1[indices['infectedN']], label='p=0.1')
    # plt.plot(per_day_new_cases_9[indices['infectedV']]+per_day_new_cases_9[indices['infectedN']], label='p=0.9')
    # plt.legend()
    # plt.show()

    # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    # plt.suptitle("total new increase", fontsize=20)
    # # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.4)
    # axes.set_xlabel('Day', labelpad=15, fontsize=15)
    # axes.plot(per_day_new_cases[indices['infectedV']] + per_day_new_cases[indices['infectedN']]+
    #           per_day_new_cases[indices['asymptomaticV']], label='p=0.5')
    # plt.plot(per_day_new_cases_1[indices['infectedV']] + per_day_new_cases_1[indices['infectedN']]+
    #          per_day_new_cases_1[indices['asymptomaticV']], label='p=0.1')
    # plt.plot(per_day_new_cases_9[indices['infectedV']] + per_day_new_cases_9[indices['infectedN']]+
    #          per_day_new_cases_9[indices['asymptomaticV']], label='p=0.9')
    # plt.legend()
    # plt.show()

    # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    # plt.suptitle("perday new quarantined", fontsize=20)
    # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.4)
    # axes.set_xlabel('Day', labelpad=15, fontsize=15)
    # axes.plot(per_day_new_cases[indices['quarantined']], label='p=0.5')
    # plt.plot(per_day_new_cases_1[indices['quarantined']], label='p=0.1')
    # plt.plot(per_day_new_cases_9[indices['quarantined']], label='p=0.9')
    # plt.legend()
    # plt.show()




    # if isSave:
    #     file_name = f"{country}_{location}_per_day_new_cases_of_infectious_{num_agebrackets}.pdf"
    #     fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
    #     fig.savefig(fig_path, format='pdf')
    #     file_name = f"{country}_{location}_per_day_new_cases_of_infectious_{num_agebrackets}.eps"
    #     fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
    #     fig.savefig(fig_path, format='eps')


p = 'p'
p_v = 0.5
location = 'Shanghai'
country = 'China'
num_agebrackets = 18
plot_per_day_new_cases(country, location, p, p_v, num_agebrackets)
