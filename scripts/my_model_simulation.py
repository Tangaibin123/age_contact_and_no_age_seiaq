import csv

import numpy as np
import copy
import os
import pandas as pd
import datetime

# set some initial paths

# path to the directory where this script lives
thisdir = os.path.abspath('')

# path to the main directory of the repository
maindir = os.path.split(thisdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(os.path.split(thisdir)[0], 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')


# %% 将得到的结果读入csv文件中
def write_compare_data(country, location, state, num_agebrackets, mode, param, param_value, allocation_capacity, value,
                       overwrite=True):
    if mode == 'baseline':
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets:.0f}_{mode}_{param}={param_value:.4f}.csv"
    else:
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets:.0f}_{mode}_{allocation_capacity:.0f}_{param}={param_value:.4f}.csv"
    file_path = os.path.join(datadir, 'output_source', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{v:.16f},{value[v][0]:.16f},{value[v][1]:.16f},{value[v][2]:.16f},{value[v][3]:.16f},{value[v][4]:.16f},{value[v][5]:.16f},{value[v][6]:.16f},{value[v][7]:.16f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{v:.16f},{value[v][0]:.16f},{value[v][1]:.16f},{value[v][2]:.16f},{value[v][3]:.16f},{value[v][4]:.16f},{value[v][5]:.16f},{value[v][6]:.16f},{value[v][7]:.16f}\n")
        f.close()
    return file_path


# %% 将得到的结果读入csv文件中
def write_data(country, location, state, num_agebrackets, mode, allocation_capacity, value, overwrite=True):
    if mode == 'baseline':
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets:.0f}_{mode}.csv"
    else:
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets:.0f}_{mode}_{allocation_capacity:.0f}.csv"
    file_path = os.path.join(datadir, 'output_source', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{v:.16f},{value[v][0]:.16f},{value[v][1]:.16f},{value[v][2]:.16f},{value[v][3]:.16f},{value[v][4]:.16f},{value[v][5]:.16f},{value[v][6]:.16f},{value[v][7]:.16f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{v:.16f},{value[v][0]:.16f},{value[v][1]:.16f},{value[v][2]:.16f},{value[v][3]:.16f},{value[v][4]:.16f},{value[v][5]:.16f},{value[v][6]:.16f},{value[v][7]:.16f}\n")
        f.close()
    return file_path


# %% 模型的模拟
def write_param_value(a, b1, b2, betaN, betaV, sigmaN1_inverse, sigmaN2_inverse, sigmaV1_inverse, sigmaV2_inverse,
                      alpha_inverse, C, result_file_path):
    date = datetime.datetime.now()
    a = str(a)
    b1 = str(b1)
    b2 = str(b2)
    betaN = str(betaN)
    betaV = str(betaV)
    sigmaN1_inverse = str(sigmaN1_inverse)
    sigmaN2_inverse = str(sigmaN2_inverse)
    sigmaV1_inverse = str(sigmaV1_inverse)
    sigmaV2_inverse = str(sigmaV2_inverse)
    alpha_inverse = str(alpha_inverse)
    C = str(C)
    file_path = os.path.join(datadir, 'param_run_log.csv')
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists, a new line of data is written')
        file = open(file_path, mode='a+', encoding='utf-8', newline='')
        sWrite = csv.writer(file)
        sWrite.writerow(
            [date, a, b1, b2, betaN, betaV, sigmaN1_inverse, sigmaN2_inverse, sigmaV1_inverse, sigmaV2_inverse,
             alpha_inverse, C, result_file_path])
        file.close()
    else:
        print(f'{file_path} is created, a row of data is written')
        file = open(file_path, mode='w', encoding='utf-8', newline='')
        sWrite = csv.writer(file)
        sWrite.writerow(
            [date, a, b1, b2, betaN, betaV, sigmaN1_inverse, sigmaN2_inverse, sigmaV1_inverse, sigmaV2_inverse,
             alpha_inverse, C, result_file_path])
        file.close()
    return


def dou_seaiq_with_age_specific_contact_martix(contact_matrix_arr, ages, a, betaN, betaV, susceptibility_factor_vector,
                                               b1, b2, sigmaN1_inverse, sigmaN2_inverse, sigmaV1_inverse,
                                               sigmaV2_inverse, alpha_inverse, allocation_capacity, mode,
                                               initial_infected_age,
                                               percent_of_initial_infected_seeds, num_agebrackets, timesteps):
    """
    Args:
        contact_matrix_arr (np.ndarray)           : a dictionary of contact matrices for different settings of contact. All setting contact matrices must be square and have the dimensions (num_agebrackets, num_agebrackets).
        ages (dict)                               : a dictionary of the age count of people in the population
        a (float)                                 : percent of the population with unvaccinated
        betaN (float)                             : the transmissibility for unvaccinated
        betaV (float)                             : the transmissibility for vaccinated
        susceptibility_factor_vector (np.ndarray) : vector of age specific susceptibility, where the value 1 means fully susceptibility and 0 means unsusceptible.
        b1 (float)                                : percentage of unvaccinated exposed becoming symptomatic infections
        b2 (float)                                : percentage of vaccinated exposed becoming symptomatic infections
        sigmaN1_inverse (float)                   : the incubation period for unvaccinated exposed
        sigmaN2_inverse (float)                   : the latent period for unvaccinated exposed
        sigmaV1_inverse (float)                   : the incubation period for vaccinated exposed
        sigmaV2_inverse (float)                   : the latent period for vaccinated exposed
        alpha_inverse (float)                     : the quarantined period for asyptomatic infections
        allocation_capacity (float)               : the capacity for allocating resource
        mode (str)                                : the mode for allocating resource
        initial_infected_age (int)                : the initial age seeded with infections
        percent_of_initial_infected_seeds (float) : percent of the population initially seeded with infections.
        num_agebrackets (int)                     : the number of age brackets of the contact matrix
        timesteps (int)                           : the number of timesteps

    Returns:
        A numpy array with the number of people in each disease state (Dou S-E-A-I-Q) by age for each timestep,
        a numpy array with the incidence by age for each timestep and the disease state indices.
    """
    allocation_rate = 0
    sigmaN1 = 1. / sigmaN1_inverse
    sigmaN2 = 1. / sigmaN2_inverse
    sigmaV1 = 1. / sigmaV1_inverse
    sigmaV2 = 1. / sigmaV2_inverse
    alpha = 1. / alpha_inverse

    total_population = sum(ages.values())  # 总人口数：各个年龄段的人数总和
    print('总人数为：', total_population)
    # 初始化起初的感染人数
    initial_infected_number = min(total_population * percent_of_initial_infected_seeds, ages[initial_infected_age])
    print('初始感染的人数为：', initial_infected_number)
    # 总共有多少个年龄段
    num_agebrackets = len(ages)

    # simulation output
    states = np.zeros((8, num_agebrackets, timesteps + 1))
    states_increase = np.zeros((8, num_agebrackets, timesteps + 1))
    states_all_increase = np.zeros(8)
    incidenceN = np.zeros((num_agebrackets, timesteps))
    incidenceV = np.zeros((num_agebrackets, timesteps))

    # indices
    indices = {'susceptible': 0, 'exposedN': 1, 'exposedV': 2, 'asymptomaticN': 3, 'asymptomaticV': 4, 'infectedN': 5,
               'infectedV': 6, 'quarantined': 7}

    # setting indices
    setting_indices = {'household': 0, 'school': 1, 'work': 2, 'community': 3}

    settings = ['household', 'school', 'work', 'community']

    # initial conditions
    states[indices['infectedN']][initial_infected_age][0] = initial_infected_number
    for age in range(num_agebrackets):
        states[indices['susceptible']][age][0] = copy.deepcopy(ages[age]) - states[indices['infectedN']][age][0]

    age_effective_contact_matrix_arr = get_age_effective_contact_matrix_with_factor_vector(contact_matrix_arr,
                                                                                           susceptibility_factor_vector)

    for t in range(timesteps):
        # if t < 30:
        #     allocation_rate = 0
        # else:
        allocation_rate = get_allocation_method(states_all_increase, indices, total_population, t, allocation_capacity,
                                                mode)
        for i in range(num_agebrackets):
            for j in range(num_agebrackets):
                incidenceN[i][t] += a * betaN * age_effective_contact_matrix_arr[i][j] * states[indices['susceptible']][i][t] * (
                        states[indices['infectedN']][j][t] + states[indices['infectedV']][j][t] + states[indices['asymptomaticN']][j][t] +
                        states[indices['asymptomaticV']][j][t]) / ages[j]
                incidenceV[i][t] += (1 - a) * betaV * age_effective_contact_matrix_arr[i][j] * states[indices['susceptible']][i][
                    t] * (states[indices['infectedN']][j][t] + states[indices['infectedV']][j][t] + states[indices['asymptomaticN']][j][t] +
                          states[indices['asymptomaticV']][j][t]) / ages[j]

            states[indices['susceptible']][i][t + 1] = states[indices['susceptible']][i][t] - incidenceN[i, t] - incidenceV[i, t]
            states[indices['exposedN']][i][t + 1] = states[indices['exposedN']][i][t] + incidenceN[i, t] - b1 * sigmaN2 * \
                                         states[indices['exposedN']][i][t] - (1 - b1) * sigmaN1 * states[indices['exposedN']][i][t]
            states[indices['exposedV']][i][t + 1] = states[indices['exposedV']][i][t] + incidenceV[i, t] - b2 * sigmaV2 * \
                                         states[indices['exposedV']][i][t] - (1 - b2) * sigmaV1 * states[indices['exposedV']][i][t]
            states[indices['asymptomaticN']][i][t + 1] = states[indices['asymptomaticN']][i][t] + (1 - b1) * sigmaN1 * states[indices['exposedN']][i][
                t] - allocation_rate * states[indices['asymptomaticN']][i][t]
            states[indices['asymptomaticV']][i][t + 1] = states[indices['asymptomaticV']][i][t] + (1 - b2) * sigmaV1 * states[indices['exposedV']][i][
                t] - allocation_rate * states[indices['asymptomaticV']][i][t]
            states[indices['infectedN']][i][t + 1] = states[indices['infectedN']][i][t] + b1 * sigmaN2 * states[indices['exposedN']][i][t] - alpha * \
                                          states[indices['infectedN']][i][t]
            states[indices['infectedV']][i][t + 1] = states[indices['infectedV']][i][t] + b2 * sigmaV2 * states[indices['exposedV']][i][t] - alpha * \
                                          states[indices['infectedV']][i][t]
            states[indices['quarantined']][i][t + 1] = states[indices['quarantined']][i][t] + alpha * (
                    states[indices['infectedV']][i][t] + states[indices['infectedN']][i][t]) + allocation_rate * (
                                                    states[indices['asymptomaticN']][i][t] + states[indices['asymptomaticV']][i][t])

            states_increase[indices['asymptomaticN']][i][t] = (1 - b1) * sigmaN1 * states[indices['exposedN']][i][t]
            states_increase[indices['asymptomaticV']][i][t] = (1 - b2) * sigmaV1 * states[indices['exposedV']][i][t]
            states_increase[indices['infectedN']][i][t] = b1 * sigmaN2 * states[indices['exposedN']][i][t]
            states_increase[indices['infectedV']][i][t] = b2 * sigmaV2 * states[indices['exposedV']][i][t]
        states_all_increase = np.sum(states_increase[:, :, t], axis=1)
    return states, incidenceN, incidenceV, indices, allocation_rate


# %% 获得有效的年龄接触矩阵

def get_age_effective_contact_matrix_with_factor_vector(contact_matrix, susceptibility_factor_vector):
    """
    Get an effective age specific contact matrix with an age dependent susceptibility drop factor.

    Args:
        contact_matrix (np.ndarray)        : the contact matrix
        susceptibility_factor_vector (int): vector of age specific susceptibility, where the value 1 means fully susceptibility and 0 means unsusceptible.


    Returns:
        np.ndarray: A numpy square matrix that gives the effective contact matrix given an age dependent susceptibility drop factor.
    """
    effective_matrix = contact_matrix * susceptibility_factor_vector
    return effective_matrix


# %% 获得C的值，资源分配的方式

def get_allocation_method(states_all_increase, indices, population_total, t, allocation_capacity, mode='baseline'):
    if (mode == 'baseline'):
        res = 0.01
    if (mode == 'case_based'):
        asymptomatic_newly = states_all_increase[indices['asymptomaticN']] + states_all_increase[
            indices['asymptomaticV']]
        all_newly = states_all_increase[indices['asymptomaticN']] + states_all_increase[indices['asymptomaticV']] + \
                    states_all_increase[indices['infectedN']] + states_all_increase[indices['asymptomaticV']]
        esp = np.finfo(all_newly).eps
        all_newly = np.maximum(all_newly, esp)
        res = (asymptomatic_newly / all_newly) * (allocation_capacity / population_total)
        res = np.minimum(res, 1)
    return res


# %% 年龄计数，合成的人口分布

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


# %% 计算最大特征值

def get_eigenvalue(matrix):
    """
    Get the real component of the leading eigenvalue of a square matrix.

    Args:
        matrix (np.ndarray): square matrix

    Returns:
        float: Real component of the leading eigenvalue of the matrix.
    """
    eigenvalue = max(np.linalg.eigvals(matrix)).real
    return eigenvalue


# %% 计算R0

def get_R0_with_factor_vector(betaN, betaV, a, b1, b2, sigmaN1_inverse, sigmaN2_inverse, sigmaV1_inverse,
                              sigmaV2_inverse, alpha_inverse, C, susceptibility_factor_vector, num_agebrackets,
                              contact_matrix):
    """
    Get the basic reproduction number R0 for a SIR compartmental model with an age dependent susceptibility drop factor and the age specific contact matrix.

    Args:
        beta (float)                              : the transmissibility
        susceptibility_factor_vector (np.ndarray) : vector of age specific susceptibility, where the value 1 means fully susceptibility and 0 means unsusceptible.
        num_agebrackets (int)                     : the number of age brackets of the contact matrix
        gamma_inverse (float)                     : the mean recovery period
        contact_matrix (np.ndarray)               : the contact matrix

    Returns:
        float: The basic reproduction number for a SEAIQ compartmental model with an age dependent susceptibility drop factor and age specific contact patterns.
    """
    sigmaN1 = 1.0 / sigmaN1_inverse
    sigmaN2 = 1.0 / sigmaN2_inverse
    sigmaV1 = 1.0 / sigmaV1_inverse
    sigmaV2 = 1.0 / sigmaV2_inverse
    alpha = 1.0 / alpha_inverse
    effective_matrix = get_age_effective_contact_matrix_with_factor_vector(contact_matrix.T,
                                                                           susceptibility_factor_vector)
    eigenvalue = get_eigenvalue(effective_matrix)
    print('接触矩阵的最大特征值为：', eigenvalue)
    # R0 = ((betaV * sigmaV2 * (1 - a))((1 - b2) * alpha + b2 * C) / (C * alpha * (sigmaV1 - b2 * sigmaV1 + b2 * sigmaV2)) + ((a * betaN) * (b1 * sigmaN2 * C + (1 - b1) * sigmaN1 * alpha)) / (C * alpha * (sigmaN1 - b1 * sigmaN1 + b1 * sigmaN2))) * eigenvalue
    R0 = (eigenvalue * betaV * sigmaV2 * (a - 1) * (b2 - 1)) / (C * (sigmaV1 - b2 * sigmaV1 + b2 * sigmaV2)) - (
            eigenvalue * b2 * betaV * sigmaV2 * (a - 1)) / (alpha * (sigmaV1 - b2 * sigmaV1 + b2 * sigmaV2)) + (
                 eigenvalue * a * b1 * betaN * sigmaN2) / (alpha * (sigmaN1 - b1 * sigmaN1 + b1 * sigmaN2)) - (
                 eigenvalue * a * betaN * sigmaN1 * (b1 - 1)) / (C * (sigmaN1 - b1 * sigmaN1 + b1 * sigmaN2))
    return R0


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


# %% 定义易感染因子  define the vector of susceptibility by age
def get_susceptibility_factor_vector(num_agebrackets, example):
    """
        :param num_agebrackets: the total age brackets
        :param example:
        :return: susceptibility_factor_vector

        Example 1: past 18 the suseptibility of individuals drops to a relative factor of 0.6 of those under 18
        susceptibility_factor_vector = np.ones((num_agebrackets, 1))
        for age in range(18, num_agebrackets):
            susceptibility_factor_vector[age, :] *= 0.6

        Example 2: under 18 the susceptibility of individuals drops to 0.2, from 18 to 65 the susceptibility is 0.8,
        and those 65 and over are fully susceptible to the disease.

        """
    susceptibility_factor_vector = np.ones((num_agebrackets, 1))
    if num_agebrackets == 85:
        if (example == 1):
            for age in range(18, num_agebrackets):
                susceptibility_factor_vector[age, :] *= 0.6
        if (example == 2):
            for age in range(0, 18):
                susceptibility_factor_vector[age, :] *= 0.2
            for age in range(18, 65):
                susceptibility_factor_vector[age, :] *= 0.8
    if (num_agebrackets == 18):
        if (example == 1):
            for age in range(4, num_agebrackets):
                susceptibility_factor_vector[age, :] *= 0.6
        if (example == 2):
            for age in range(0, 4):
                susceptibility_factor_vector[age, :] *= 0.2
            for age in range(4, 15):
                susceptibility_factor_vector[age, :] *= 0.8
    return susceptibility_factor_vector


# %%
# Dou-SEAIQ model parameters
num_agebrackets = 85  # number of age brackets for the contact matrices
a = 0.2  # percent of unvaccinated
betaN = 0.5  #
betaV = 0.3
b1 = 0.8
b2 = 0.2
sigmaN1_inverse = 5.0  # mean incubation period for unvaccinated
sigmaN2_inverse = 4.0  # mean latent period for unvaccinated
sigmaV1_inverse = 6.0  # mean incubation period for vaccinated
sigmaV2_inverse = 5.0  # mean latent period for vaccinated
alpha_inverse = 2.6  # mean quarantined period
# R0_star = 1.6  # basic reproduction number
initial_infected_age = 20  # some initial age to seed infections within the population
# initial_infected_age = 4  # some initial age to seed infections within the population
percent_of_initial_infected_seeds = 1e-5
timesteps = 350  # how long to run the SEAIQ model

location = 'Shanghai'
country = 'China'
level = 'subnational'

ages = get_ages(location, country, level, num_agebrackets)
susceptibility_factor_vector = get_susceptibility_factor_vector(num_agebrackets, example=2)

# param = 'a'
# param = 'b1'
# param = 'b2'
# param = 'sigmaN1_inverse'
# param = 'sigmaN2_inverse'
# param = 'sigmaV1_inverse'
# param = 'sigmaV2_inverse'
# param = 'alpha_inverse'
param = 'allocation_capacity'


allocation_capacity = 20000
# allocation_capacity = 40000
# allocation_capacity = 60000
mode = 'baseline'
# mode = 'case_based'

R0_list = []
attack_rate_list = []
compare_state = []
# for i in range(3):
#     param_value = allocation_capacity = (i+1) * 20000
contact_matrix_arr = read_contact_matrix(location, country, level, 'overall', num_agebrackets)
states, incidenceN, incidenceV, indices, C = dou_seaiq_with_age_specific_contact_martix(contact_matrix_arr, ages, a,
                                                                                        betaN,
                                                                                        betaV,
                                                                                        susceptibility_factor_vector,
                                                                                        b1, b2, sigmaN1_inverse,
                                                                                        sigmaN2_inverse,
                                                                                        sigmaV1_inverse,
                                                                                        sigmaV2_inverse,
                                                                                        alpha_inverse,
                                                                                        allocation_capacity,
                                                                                        mode,
                                                                                        initial_infected_age,
                                                                                        percent_of_initial_infected_seeds,
                                                                                        num_agebrackets, timesteps)

# 计算整个周期每天的所有年龄段人数和
total_S = np.sum(states[indices['susceptible']], axis=0)
total_exposedN = np.sum(states[indices['exposedN']], axis=0)
total_exposedV = np.sum(states[indices['exposedV']], axis=0)
total_asymptomaticN = np.sum(states[indices['asymptomaticN']], axis=0)
total_asymptomaticV = np.sum(states[indices['asymptomaticV']], axis=0)
total_infectedN = np.sum(states[indices['infectedN']], axis=0)
total_infectedV = np.sum(states[indices['infectedV']], axis=0)
total_quarantined = np.sum(states[indices['quarantined']], axis=0)
all_states = np.vstack((total_S, total_exposedN, total_exposedV, total_asymptomaticN, total_asymptomaticV,
                        total_infectedN, total_infectedV, total_quarantined))

attack_rate = total_quarantined[-1].sum() / sum(ages.values()) * 100
R0 = get_R0_with_factor_vector(betaN, betaV, a, b1, b2, sigmaN1_inverse, sigmaN2_inverse, sigmaV1_inverse,
                               sigmaV2_inverse, alpha_inverse, C, susceptibility_factor_vector, num_agebrackets,
                               contact_matrix_arr)
attack_rate_list.append(attack_rate)
# print('当 ' + param + ' = ' + str(param_value) + ' 时的发病率为：', attack_rate)
R0_list.append(R0)
# print('当 ' + param + ' = ' + str(param_value) + ' 时的R0值为：', R0)
# result_file_path = write_compare_data(country=country, location=location, state='all_states', num_agebrackets=num_agebrackets, param=param, param_value=param_value, mode=mode, allocation_capacity=allocation_capacity, value=all_states.T)
# # 将每次运行的参数保存起来
# write_param_value(a, b1, b2, betaN, betaV, sigmaN1_inverse, sigmaN2_inverse, sigmaV1_inverse, sigmaV2_inverse, alpha_inverse, C, result_file_path)

result_file_path = write_data(country=country, location=location, state='all_states', num_agebrackets=num_agebrackets, mode=mode, allocation_capacity=allocation_capacity, value=all_states.T)
# 将每次运行的参数保存起来
write_param_value(a, b1, b2, betaN, betaV, sigmaN1_inverse, sigmaN2_inverse, sigmaV1_inverse, sigmaV2_inverse, alpha_inverse, C, result_file_path)

print('发病率', attack_rate_list)
print('R0', R0_list)
# print('R0', np.array(R0_list)/8.9661787298506)
