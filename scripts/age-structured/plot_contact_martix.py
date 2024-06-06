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

# path to the results subdirectory`
resultsdir = os.path.join(maindir, 'results')

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

# %% 读取接触矩阵
def read_contact_matrix(location, country, level, setting, num_agebrackets=85):
    """
    Read in the contact for each setting.

    Args:
        location (str)        : name of the location
        country (str)         : name of the country
        level (str)           : name of level (country or subnational)
        setting (str)         : name of the contact setting
        num_agebrackets (int) : the number of age brackets for the matrix

    Returns:
        A numpy matrix of contact.
    """
    setting_type, setting_suffix = 'F', 'setting'
    if setting == 'overall':
        setting_type, setting_suffix = 'M', 'contact_matrix'

    if country == 'Europe':
        country = location
        level = 'country'
    if level == 'country':
        file_name = country + '_' + level + '_level_' + setting_type + '_' + setting + '_' + setting_suffix + '_' + '%i' % num_agebrackets + '.csv'
    else:
        file_name = country + '_' + level + '_' + location + '_' + setting_type + '_' + setting + '_' + setting_suffix + '_' + '%i' % num_agebrackets + '.csv'
    file_path = os.path.join(datadir, 'origin_resource', file_name)
    return np.loadtxt(file_path, delimiter=',')


# %% 获取接触矩阵字典
def get_contact_matrix_dic(location, country, level, num_agebrackets=85):
    """
    Get a dictionary of the setting contact matrices for a location.

    Args:
        location (str)        : name of the location
        country (str)         : name of the country
        level (str)           : name of level (country or subnational)
        num_agebrackets (int) : the number of age brackets for the matrix

    Returns:
        dict: A dictionary of the setting contact matrices for the location.
    """
    settings = ['household', 'school', 'workplace', 'community']
    matrix_dic = {}
    for setting in settings:
        matrix_dic[setting] = read_contact_matrix(location, country, level, setting, num_agebrackets)
    return matrix_dic


# %% 合成综合矩阵
def combine_synthetic_matrices(contact_matrix_dic, weights, num_agebrackets=85):
    """
    A linear combination of contact matrices for different settings to create an overall contact matrix given the weights for each setting.

    Args:
        contact_matrix_dic (dict) : a dictionary of contact matrices for different settings of contact. All setting contact matrices must be square and have the dimensions (num_agebrackets, num_agebrackets).
        weights (dict)            : a dictionary of weights for each setting of contact
        num_agebrackets (int)     : the number of age brackets of the contact matrix

    Returns:
        np.ndarray: A linearly combined overall contact matrix.
    """
    contact_matrix = np.zeros((num_agebrackets, num_agebrackets))
    for setting in weights:
        contact_matrix += contact_matrix_dic[setting] * weights[setting]
    return contact_matrix

def get_ages(location, country, level, num_agebrackets=18):
    """
    Get the age count for the synthetic population of the location.

    Args:
        location (str)        : name of the location
        country (str)         : name of the country
        level (str)           : name of level (country or subnational)
        num_agebrackets (int) : the number of age brackets

    Returns:
        dict: A dictionary of the age count.
    """

    if country == 'Europe':
        country = location
        level = 'country'
    if level == 'country':
        file_name = country + '_' + level + '_level_age_distribution_' + '%i' % num_agebrackets + '.csv'
    else:
        file_name = country + '_' + level + '_' + location + '_age_distribution_' + '%i' % num_agebrackets + '.csv'
    file_path = os.path.join(datadir, 'origin_resource', file_name)
    df = pd.read_csv(file_path, delimiter=',', header=None)
    df.columns = ['age', 'age_count']
    return dict(zip(df.age.values.astype(int), df.age_count.values))


def calculate_new_matrix(matrix,ages,num_agebrackets):
    for i in range(num_agebrackets):
        for j in range(num_agebrackets):
            matrix[i][j]= matrix[i][j]*ages[i]/ages[j]
    return matrix

location = 'Shanghai'
country = 'China'
level = 'subnational'
num_agebrackets = 18  # number of age brackets for the contact matrices

ages = get_ages(location, country, level, num_agebrackets)

weights = {'household': 4.11, 'school': 11.41, 'workplace': 8.07,
           'community': 2.79}  # effective setting weights as found in our study
contact_matrix_dic = get_contact_matrix_dic(location, country, level, num_agebrackets)
combine_synthetic_matric = combine_synthetic_matrices(contact_matrix_dic, weights, num_agebrackets)
contact_matrix = combine_synthetic_matric

#%%加入干预措施后的综合接触矩阵
weights = {'household': 4.11, 'school': 0, 'workplace': 1,
           'community': 1}  # effective setting weights as found in our study
contact_matrix_dic = get_contact_matrix_dic(location, country, level, num_agebrackets)
combine_synthetic_matric1 = combine_synthetic_matrices(contact_matrix_dic, weights, num_agebrackets)
contact_matrix1 = combine_synthetic_matric1
#%%
# contact_matrix = np.array(contact_matrix)
# contact_matrix1 = np.array(contact_matrix1)
difference = contact_matrix - contact_matrix1

#
# for i in range(len(contact_matrix)):
#     for j in range(len(contact_matrix[0])):
#         difference[i][j]= contact_matrix[i][j] - contact_matrix1[i][j]
#         print(difference[i][j])


# print(contact_matrix1)


# np.savetxt("contact_matrix.txt", contact_matrix, fmt="%s", delimiter=",")
# np.savetxt("difference.txt", difference, fmt="%s", delimiter=",")


if num_agebrackets == 85:
    age_brackets = [str(i) + '-' + str((i + 1)) for i in range(0, 85)] + ['85+']
elif num_agebrackets == 18:
    age_brackets = [str(5 * i) + '-' + str(5 * (i + 1) - 1) for i in range(18)] + ['85+']

fig = plt.figure(figsize=(6, 5))
plt.title('Synthetic', fontsize=20)

y_label = np.flipud(age_brackets)
x_label = age_brackets

sbn.heatmap(swap(contact_matrix), cmap='GnBu', yticklabels=y_label, xticklabels=x_label)

plt.xlabel('Age', labelpad=15)
plt.ylabel('Age of contact', labelpad=20)
plt.show()
fig_name = 'synthetic_matrix_18_Ni.pdf'
fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
fig.savefig(fig_path, format='pdf')

# #%
# fig = plt.figure(figsize=(5, 5))
# plt.title(' Synthetic', fontsize=20)
# y_label = np.flipud(age_brackets)
# x_label = age_brackets
# sbn.heatmap(swap(contact_matrix1), cmap='GnBu', yticklabels=y_label, xticklabels=x_label)
# plt.xlabel('Age', labelpad=15)
# plt.ylabel('Age of contact', labelpad=20)
# plt.show()

# fig = plt.figure(figsize=(5, 5))
# plt.title('Synthetic', fontsize=20)
# y_label = np.flipud(age_brackets)
# x_label = age_brackets
# contact_matrix_N_i = calculate_new_matrix(contact_matrix,ages,num_agebrackets)
# sbn.heatmap(swap(contact_matrix_N_i), cmap='GnBu', yticklabels=y_label, xticklabels=x_label)
# plt.xlabel('Age', labelpad=15)
# plt.ylabel('Age of contact', labelpad=20)
# plt.show()
# #%%
#
# fig_name = 'synthetic_matrix_18.pdf'
# fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
# fig.savefig(fig_path, format='pdf')
# fig_name = 'combine_synthetic_matrix_18.eps'
# fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
# fig.savefig(fig_path, format='eps')
#
settings = ['household', 'school', 'workplace', 'community']
#
for setting in settings:
    matrix = contact_matrix_dic[setting]
    fig = plt.figure(figsize=(6, 5))
    if setting == "community":
        setting = 'Others'
    plt.title(setting.capitalize(), fontsize=20)
    y_label = np.flipud(age_brackets)
    x_label = age_brackets
    # sbn.heatmap(swap(matrix), cmap='GnBu', yticklabels=y_label, xticklabels=x_label)
    sbn.heatmap(swap(calculate_new_matrix(matrix,ages,num_agebrackets)), cmap='GnBu', yticklabels=y_label, xticklabels=x_label)

    plt.xlabel('Age', labelpad=15)
    plt.ylabel('Age of contact', labelpad=50)
    plt.show()
    fig_name = setting.capitalize() + '_contact_matrix_18_Ni.pdf'
    fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
    fig.savefig(fig_path, format='pdf')
    # fig_name = setting.capitalize() + '_contact_matrix_18.eps'
    # fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
    # fig.savefig(fig_path, format='eps')


