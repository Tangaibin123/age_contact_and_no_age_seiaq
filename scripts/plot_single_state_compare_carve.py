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


def read_compare_state_numbers(location, country, state, indices, allocation_capacity, num_agebrackets, mode, param,
                               param_value):
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
    if (mode == 'baseline'):
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets}_{mode}_{param}={param_value:.4f}.csv"
    else:
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets}_{mode}_{allocation_capacity}_{param}={param_value:.4f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)
    print(file_path + '已读取')
    M = np.loadtxt(file_path, delimiter=',', usecols=indices)
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

# param = 'a'
# param = 'b1'
# param = 'b2'
# param = 'sigmaN1_inverse'
# param = 'sigmaN2_inverse'
# param = 'sigmaV1_inverse'
param = 'sigmaV2_inverse'
# param = 'alpha_inverse'

# single_state = 'susceptible'
# single_state = 'exposedN'
# single_state = 'exposedV'
single_state = 'asymptomaticN'
# single_state = 'asymptomaticV'
# single_state = 'infectedN'
# single_state = 'infectedV'
# single_state = 'quarantined'

# indices
indices = {'susceptible': 0, 'exposedN': 1, 'exposedV': 2, 'asymptomaticN': 3, 'asymptomaticV': 4, 'infectedN': 5,
           'infectedV': 6, 'quarantined': 7}
compare_states = []
for i in range(5):
    param_value = sigmaV1_inverse = i + 2 +1
    compare_state = read_compare_state_numbers(country=country, location=location, state=state, indices=indices[single_state],
                                               allocation_capacity=allocation_capacity, num_agebrackets=num_agebrackets,
                                               mode=mode, param=param, param_value=param_value)
    compare_states.append(compare_state)

compare_states = np.array(compare_states).T
time = np.arange(351)
# 画图
fontsizes = {'colorbar': 30, 'colorbarlabels': 22, 'title': 44, 'ylabel': 28, 'xlabel': 28, 'xticks': 24, 'yticks': 24}
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

plt.plot(time, compare_states[:, 0])
plt.plot(time, compare_states[:, 1])
plt.plot(time, compare_states[:, 2])
plt.plot(time, compare_states[:, 3])
plt.plot(time, compare_states[:, 4])
# plt.plot(time, compare_states[:, 5])
# plt.plot(time, compare_states[:, 6])
# plt.plot(time, compare_states[:, 7])
# plt.plot(time, compare_states[:, 8])
# plt.plot(time, compare_states[:, 9])
# plt.plot(time, compare_states[:, 10])
plt.legend([(param + '=' + str(a+2+1)) for a in range(5)])

title = location.replace('_', ' ').replace('-ken', '').replace('-to', '').replace('-fu', '').replace('-', ' ')
ax.set_title(title, fontsize=fontsizes['title'])
ax.set_ylabel('number of ' + single_state, fontsize=fontsizes['ylabel'])
ax.set_xlabel('time', fontsize=fontsizes['xlabel'])

plt.show()
if mode == 'baseline':
    file_name = country + '_' + level + '_' + location + '_' + '%i' % num_agebrackets + '_' + mode + '_' + param + '_' + single_state + '.pdf'
else:
    file_name = country + '_' + level + '_' + location + '_' + '%i' % num_agebrackets + '_' + mode + '_' + '%i' % allocation_capacity + '_' + param + '_' + single_state + '.pdf'
fig_path = os.path.join(resultsdir, '', file_name)
fig.savefig(fig_path, format='pdf')
