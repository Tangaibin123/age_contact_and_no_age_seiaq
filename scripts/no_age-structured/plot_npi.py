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



def plot_npi(isSave = True):
    file_name1 = f"Contact_limitation&Early_identification.csv"
    file_path1 = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name1)
    print(file_path1 + ' was read')

    file_name5 = f"Contact_limitation.csv"
    file_path5 = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name5)
    print(file_path5 + ' was read')

    file_name9 = f"Early_identification.csv"
    file_path9 = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name9)
    print(file_path9 + ' was read')

    file_name_base = f"base.csv"
    file_path_base = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name_base)
    print(file_path_base + ' was read')

    infected1 = np.loadtxt(file_path1, skiprows=1, delimiter=',')
    infected1 = infected1[:200].T

    infected5 = np.loadtxt(file_path5, skiprows=1, delimiter=',')
    infected5 = infected5[:200].T

    infected9 = np.loadtxt(file_path9, skiprows=1, delimiter=',')
    infected9 = infected9[:200].T

    infected_base = np.loadtxt(file_path_base, skiprows=1, delimiter=',')
    infected_base = infected_base[:200].T

    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8}

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)


    ax1.set_title('Total infected')
    ax1.plot(infected1[indices['infectedN']]+infected1[indices['asymptomaticV']]+infected1[indices['infectedV']],label='Both')
    ax1.plot(infected5[indices['infectedN']]+infected5[indices['asymptomaticV']]+infected5[indices['infectedV']],label='Contact_limitation')
    ax1.plot(infected9[indices['infectedN']]+infected9[indices['asymptomaticV']]+infected9[indices['infectedV']],label='Early_identification')
    ax1.plot(infected_base[indices['infectedN']]+infected_base[indices['asymptomaticV']]+infected_base[indices['infectedV']],label='base')
    ax1.set_xlabel('day')

    # ax1.set_title('Asymptomatic infected')
    # ax1.plot(infected1[indices['asymptomaticV']], label='Both')
    # ax1.plot(infected5[indices['asymptomaticV']], label='Contact_limitation')
    # ax1.plot(infected9[indices['asymptomaticV']], label='Early_identification')
    # ax1.plot(infected_base[indices['asymptomaticV']],label='base')
    # handles, labels = ax1[1].get_legend_handles_labels()
    fig.legend()
    plt.show()

    if isSave:
        file_name = f"NPI.pdf"
        fig_path = os.path.join(resultsdir, 'new_paper', file_name)
        fig.savefig(fig_path, format='pdf')
        # file_name = f"{country}_{location}_per_day_new_cases_of_infectious_{num_agebrackets}.eps"
        # fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        # fig.savefig(fig_path, format='eps')

plot_npi()