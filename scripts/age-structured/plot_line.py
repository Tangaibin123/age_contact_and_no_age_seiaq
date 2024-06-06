import csv
import numpy as np
import copy
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# %%path to the directory where this script lives
thisdir = os.path.abspath('')

# path to the scripts directory of the repository
scriptsdir = os.path.split(thisdir)[0]  # script目录

# path to the main directory of the repository
maindir = os.path.split(scriptsdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(os.path.split(maindir)[0], 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')

# path to the output_source subsubdirectory
output_source_dir = os.path.join(datadir, 'output_source')


# %%
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


# %%%读取接触矩阵
def read_contact_matrix(location, country, level, setting, num_agebrackets=18):
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


# %%获取接触矩阵字典
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


# %%合成综合接触矩阵
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


# %%获取R0
def get_R0(beta, p, w, q, l, k, gamma_inverse):
    """
    Get the basic reproduction number R0 for a SIR compartmental model with an age dependent susceptibility drop factor and the age specific contact matrix.

    Args:
        beta (float)                              : the transmissibility
        p (float)                                 : 接种的比例（the probability of vaccining）
        w (float)                                 :
        q (float)                                 :
        l (float)                                 :
        k (float)                                 :
        gamma_inverse (float)                     : the mean recovery period

    Returns:
        float: The basic reproduction number for a SEAIQ compartmental model with an age dependent susceptibility drop factor and age specific contact patterns.
    """

    gamma = 1.0 / gamma_inverse
    R0 = beta * ((p + (1 - p) * l * w) * ((1 - q) + k * q)) / (k * gamma)
    return R0


location = 'Shanghai'
country = 'China'
level = 'subnational'
state = "all_states"
num_agebrackets = 18

ages = get_ages(location, country, level, num_agebrackets)
# {0: 17982.0, 1: 14121.0, 2: 12745.0, 3: 24738.0, 4: 53462.0, 5: 56045.0, 6: 46302.0, 7: 42597.0, 8: 41824.0, 9: 39930.0, 10: 39005.0, 11: 37168.0, 12: 24612.0, 13: 14098.0, 14: 10728.0, 15: 10765.0, 16: 6542.0, 17: 6840.0
# for i in ages:
#     print(i,ages[i])
Baseline_weights = {'household': 4.11, 'school': 11.41, 'workplace': 8.07,
           'community': 2.79}  # effective setting weights as found in our study
NPI_weights = {'household': 3.5, 'school': 0, 'workplace': 0.5,
           'community': 0.5}  # effective setting weights as found in our study
contact_matrix_dic = get_contact_matrix_dic(location, country, level, num_agebrackets)
Baseline_matrix = combine_synthetic_matrices(contact_matrix_dic, Baseline_weights, num_agebrackets)
NPI_matrix = combine_synthetic_matrices(contact_matrix_dic,NPI_weights,num_agebrackets)

#分为baseline和after NPI
age_brackets = [str(5 * i) + '-' + str(5 * (i + 1) - 1) for i in range(18)]
Baseline_group_incidence = np.zeros(18)
NPI_group_incidence = np.zeros(18)
for i in range(18):
    Baseline_group_incidence[i] = (Baseline_matrix[i].sum())/20
    NPI_group_incidence[i] = NPI_matrix[i].sum()/20


for i in range(num_agebrackets):
    plt.bar(age_brackets[i],Baseline_group_incidence[i],color='coral')
    plt.bar(age_brackets[i],NPI_group_incidence[i],color='green')
plt.bar(age_brackets,Baseline_group_incidence,label='Baseline')
plt.bar(age_brackets,NPI_group_incidence,label='NPI')
plt.title('Infection probability of each group')
plt.xticks(rotation = 90)#让坐标横着写出来
plt.xlabel('Age group'),plt.ylabel('Incidence')
plt.legend()
plt.show()
# print(plt.bar().witdh)


# def vaccine_allocatin(vaccine_num, ages, p, contactmatrix_dic, beta, k, gamma, q, w, l, units=10):
#     R = np.zeros((18, 18))
#     RI = np.zeros(18)
#     for i in ages:
#         dp = vaccine_num / ages[i]
#         p_new = p[i] - dp
#         for j in ages:
#             R[i][j] = (beta * k / gamma) * \
#                       ((p[i] * (1 - q) + k * q) + (1 - p[i]) * w * ((1 - q * l) + k * q * l))
