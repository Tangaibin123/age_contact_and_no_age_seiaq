import matplotlib.pyplot as plt
import numpy as np
import os
import time

# set some initial paths

# path to the directory where this script lives
thisdir = os.path.abspath('')

# path to the main directory of the repository
maindir = os.path.split(thisdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(os.path.split(thisdir)[0], 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')
# path to the output_source subsubdirectory
output_source_dir = os.path.join(datadir, 'output_source')


def read_states_numbers(location, country, state, allocation_capacity, num_agebrackets, mode):
    """
    Read in the contact for each setting.

    Args:
        location (str)        : name of the location
        country (str)         : name of the country
        state (str)           : name of state
        num_agebrackets (int) : the number of age brackets for the matrix
        mode (str)            : the mode for allocating resource
        allocation_capacity   : the capacity for allocating resource

    Returns:
        np.ndarray: A numpy matrix of contact.
    """
    if(mode == 'baseline'):
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets}_{mode}.csv"
    else :
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets}_{mode}_{allocation_capacity}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)
    print(file_path)
    M = np.loadtxt(file_path, delimiter=',')
    return M


location = 'Shanghai'
country = 'China'
level = 'subnational'
state = 'all_states'
num_agebrackets = 85
mode = 'baseline'
# mode = 'case_based'
# allocation_capacity = 20000
# allocation_capacity = 40000
allocation_capacity = 60000
all_states = read_states_numbers(country=country, location=location, state=state, allocation_capacity=allocation_capacity,num_agebrackets=num_agebrackets,mode=mode)


# 画图
fontsizes = {'colorbar': 30, 'colorbarlabels': 22, 'title': 44, 'ylabel': 28, 'xlabel': 28, 'xticks': 24, 'yticks': 24}
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
plt.plot(all_states[:, 0], all_states[:, 1])
plt.plot(all_states[:, 0], all_states[:, 2])
plt.plot(all_states[:, 0], all_states[:, 3])
plt.plot(all_states[:, 0], all_states[:, 4])
plt.plot(all_states[:, 0], all_states[:, 5])
plt.plot(all_states[:, 0], all_states[:, 6])
plt.plot(all_states[:, 0], all_states[:, 7])
plt.plot(all_states[:, 0], all_states[:, 8])
plt.legend(['S', 'En', 'Ev', 'An', 'Av', 'In', 'Iv', 'Q'])
title = location.replace('_', ' ').replace('-ken', '').replace('-to', '').replace('-fu', '').replace('-', ' ')
if mode == 'baseline':
    title += '---' + mode
else:
    title += '---' + str(allocation_capacity)
ax.set_title(title, fontsize=fontsizes['title'])
ax.set_ylabel('number of ' + state, fontsize=fontsizes['ylabel'])
ax.set_xlabel('time', fontsize=fontsizes['xlabel'])
# ax.set_xticks(np.arange(0, 81, 10))
# ax.set_yticks(np.arange(0, 81, 10))
# ax.tick_params(labelsize=fontsizes['xticks'])
plt.show()
if mode == 'baseline':
    file_name = country + '_' + level + '_' + location + '_' + '%i' % num_agebrackets + '_' + mode + '.pdf'
else:
    file_name = country + '_' + level + '_' + location + '_' + '%i' % num_agebrackets + '_' + mode + '_' + '%i' % allocation_capacity + '.pdf'
fig_path = os.path.join(resultsdir, file_name)
fig.savefig(fig_path, format='pdf')
