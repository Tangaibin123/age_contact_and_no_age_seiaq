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

def plot_states(country, location,state, p, p_v, num_agebrackets, isSave=True):
    file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets}_{p}={p_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'p', file_name)

    print(file_path + ' was read')
    states = np.loadtxt(file_path, skiprows=1, delimiter=',')
    states = states[:100].T
    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8}
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(9, 8))
    plt.suptitle(f"states,p={p_v}")
    axes.set_xlabel('time(day)', labelpad=15, fontsize=15)
    axes.set_ylabel('Number', labelpad=10, fontsize=15)
    plt.plot(states[indices['infectedN']],label='infected without vaccination')
    plt.plot(states[indices['infectedV']],label='infected with vaccination')
    plt.plot(states[indices['asymptomaticV']],label='asymptomatic')
    plt.plot(states[indices['infectedN']]+states[indices['infectedV']]+states[indices['asymptomaticV']],label='Total infected')
    plt.legend()
    plt.show()
def plot_states_p(country, location,state, p, p_v, num_agebrackets, isSave=True):
    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8}
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10.5, 8))
    plt.suptitle(f"Total infections about p, l=0.5,k=0.5")
    axes.set_xlabel('time(day)', labelpad=15, fontsize=15)
    axes.set_ylabel('Number', labelpad=10, fontsize=15)
    for i in [1,5,9]:
        p_v= i/10
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets}_{p}={p_v:.2f}.csv"
        file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'p', file_name)
        print(file_path + ' was read')
        states = np.loadtxt(file_path, skiprows=1, delimiter=',')
        states = states[:100].T
        plt.plot(states[indices['infectedN']]+states[indices['infectedV']]+states[indices['asymptomaticV']],label=f'p={p_v}')
        plt.legend()
    plt.show()

    if isSave:
        file_name = f'{location}_states_{num_agebrackets}.pdf'
        fig_path = os.path.join(resultsdir, 'new_paper', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_states_{num_agebrackets}.eps'
        fig_path = os.path.join(resultsdir, 'new_paper', file_name)
        fig.savefig(fig_path, format='eps')
        print('fig is save')

def plot_new_increase_about_p(country, location,state, p, p_v, num_agebrackets, isSave=True):
    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8}
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10.5, 8))
    # plt.suptitle(f"Per day new infections about p, l=0.5,k=0.5")
    axes.set_xlabel('time(day)', labelpad=15, fontsize=15)
    axes.set_ylabel('Number', labelpad=10, fontsize=15)
    for i in [1,5,9]:
        p_v= i/10
        file_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_baseline.csv"
        file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'p', file_name)
        print(file_path + ' was read')
        states = np.loadtxt(file_path, skiprows=1, delimiter=',')
        states = states[:100].T
        plt.plot(states[indices['infectedN']]+states[indices['infectedV']]+states[indices['asymptomaticV']],label=f'p={p_v}')
        plt.legend(fontsize=25)
    plt.show()

    if isSave:
        file_name = f'{location}_new_increase_{num_agebrackets}.pdf'
        fig_path = os.path.join(resultsdir, 'new_paper', file_name)
        fig.savefig(fig_path, format='pdf')

        # file_name = f'{location}_new_increase_{num_agebrackets}.eps'
        # fig_path = os.path.join(resultsdir, 'new_paper', file_name)
        # fig.savefig(fig_path, format='eps')
        # print('fig is save')
p='p'
location = 'Shanghai'
country = 'China'
num_agebrackets = 18
p_v=0.5
state_type = 'all_states'
plot_states(country,location,state_type,p,p_v,num_agebrackets)
# plot_states_p(country,location,state_type,p,p_v,num_agebrackets,isSave=True)
# plot_new_increase_about_p(country,location,state_type,p,p_v,num_agebrackets,isSave=True)














