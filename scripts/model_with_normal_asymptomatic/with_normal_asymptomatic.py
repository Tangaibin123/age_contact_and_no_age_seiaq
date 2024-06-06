import csv
import numpy as np
import copy
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# set some initial paths

# path to the directory where this script lives
thisdir = os.path.abspath('')

# path to the scripts directory of the repository
scriptsdir = os.path.split(thisdir)[0]  #script目录

# path to the main directory of the repository
maindir = os.path.split(scriptsdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(os.path.split(maindir)[0], 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')

# path to the output_source subsubdirectory
output_source_dir = os.path.join(datadir, 'output_source')


# %% 将得到的结果读入csv文件中，没有使用
def write_compare_data(country, location, state, num_agebrackets, mode, param, param_value, allocation_capacity, value,
                       overwrite=True):
    if mode == 'baseline':
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets:.0f}_{mode}_{param}={param_value:.4f}.csv"
    else:
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets:.0f}_{mode}_{allocation_capacity:.0f}_{param}={param_value:.4f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
        f.close()
    return file_path


# %% 将得到的结果读入csv文件中， 没有使用
def write_data_param(country, location, state, value, p, p_v, l, l_v, k, num_agebrackets=18, overwrite=True):
    file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets:.0f}_{p}={p_v:.1f}_{l}={l_v:.1f}_{k}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{value[v][0]:.0f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        print(f'{file_path} was written')
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f}\n")
        f.close()
    return file_path


# %% 将得到的结果读入csv文件中
def write_data(country, location, state, value, p, p_v, l, l_v, k, k_v, num_agebrackets=18, overwrite=True):
    file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets}_{p}={p_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'p', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        print(f'{file_path} was written')
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
        f.close()
    return file_path


# %% 将得到的结果读入csv文件中
def write_data_bracket_age(country, location, state, value, age, p, p_v, l, l_v, k, k_v, num_agebrackets=18,
                           overwrite=True):
    file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets}_age={age}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', "bracket", p + "_" + str(p_v),file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        print(f'{file_path} was written.')
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
        f.close()
    return file_path


# %% 保存每日新增病例数
# def write_per_day_new_cases(country, location, new_case_per_day, p, p_v, l, l_v, k, k_v, num_agebrackets=18,
def write_per_day_new_cases(country, location, new_case_per_day, p, p_v, num_agebrackets=18,
                            overwrite=True):
    # file_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    # file_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_NPI.csv"
    file_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_baseline.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source','p', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            # f.write(
            #     f"susceptible, exposedN, exposedV, asymptomaticN, asymptomaticV, infectedN, infectedV, quarantined, asymptomatic_total, infected_total, indectious_total")
            for v in range(len(new_case_per_day)):
                f.write(
                    f"{v:.2f},{new_case_per_day[v][0]:.2f},{new_case_per_day[v][1]:.2f},{new_case_per_day[v][2]:.2f},{new_case_per_day[v][3]:.2f},{new_case_per_day[v][4]:.2f},{new_case_per_day[v][5]:.2f},{new_case_per_day[v][6]:.2f},{new_case_per_day[v][7]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        print(f'{file_path} was written.')
        f = open(file_path, 'w+')
        # f.write(
        #     f"susceptible, exposedN, exposedV, asymptomaticN, asymptomaticV, infectedN, infectedV, quarantined, asymptomatic_total, infected_total, indectious_total")
        for v in range(len(new_case_per_day)):
            f.write(
                f"{v:.2f},{new_case_per_day[v][0]:.2f},{new_case_per_day[v][1]:.2f},{new_case_per_day[v][2]:.2f},{new_case_per_day[v][3]:.2f},{new_case_per_day[v][4]:.2f},{new_case_per_day[v][5]:.2f},{new_case_per_day[v][6]:.2f},{new_case_per_day[v][7]:.2f}\n")
        f.close()
    return file_path


# %% 保存每次运行的参数
def write_param_value(p, w, beta, q, l, sigma_inverse, k, gamma_inverse, result_file_path):
    date = datetime.datetime.now()
    p = str(p)
    w = str(w)
    beta = str(beta)
    q = str(q)
    l = str(l)
    sigma_inverse = str(sigma_inverse)
    k = str(k)
    gamma_inverse = str(gamma_inverse)
    file_path = os.path.join(datadir, 'param_run_log_age-structured.csv')
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists, a new line of data was written')
        file = open(file_path, mode='a+', encoding='utf-8', newline='')
        sWrite = csv.writer(file)
        sWrite.writerow([date, p, w, beta, q, l, sigma_inverse, k, gamma_inverse, result_file_path])
        file.close()
    else:
        print(f'{file_path} is created, a row of data is written')
        file = open(file_path, mode='w', encoding='utf-8', newline='')
        sWrite = csv.writer(file)
        sWrite.writerow(['date', 'p', 'w', 'beta', 'q', 'l', 'sigma_inverse', 'k', 'gamma_inverse', 'file_path'])
        sWrite.writerow([date, p, w, beta, q, l, sigma_inverse, k, gamma_inverse, result_file_path])
        file.close()
    return


# %% 模型的模拟
def dou_seaiq_with_age_specific_contact_martix(contact_matrix_dic, weights, ages, p, w, beta, q, l, k,alpha,
                                               susceptibility_factor_vector,
                                               sigma_inverse, gamma_inverse, initial_infected_age,
                                               percent_of_initial_infected_seeds,
                                               num_agebrackets, timesteps):
    """
    Args:
        contact_matrix_dic (dict)                 : a dictionary of contact matrices for different settings of contact. All setting contact matrices must be square and have the dimensions (num_agebrackets, num_agebrackets).
        weights (dict)                            : a dictionary of weights for each setting of contact
        p (float)                                 : percent of the population with unvaccinated
        w (float)                                 : percent of the population with unvaccinated
        beta (float)                              : the transmissibility for unvaccinated
        susceptibility_factor_vector (np.ndarray) : vector of age specific susceptibility, where the value 1 means fully susceptibility and 0 means unsusceptible.
        q (float)                                 : percentage of unvaccinated exposed becoming symptomatic infections
        l (float)                                 : percentage of vaccinated exposed becoming symptomatic infections
        sigma_inverse (float)                     : the incubation period for unvaccinated exposed
        k (float)                                 : percentage of vaccinated exposed becoming symptomatic infections
        gamma_inverse (float)                     : the quarantined period for asyptomatic infections
        initial_infected_age (int)                : the initial age seeded with infections
        percent_of_initial_infected_seeds (float) : percent of the population initially seeded with infections.
        num_agebrackets (int)                     : the number of age brackets of the contact matrix
        timesteps (int)                           : the number of timesteps

    Returns:
        A numpy array with the number of people in each disease state (Dou S-E-A-I-Q) by age for each timestep,
        a numpy array with the incidence by age for each timestep and the disease state indices.
    """
    sigma = 1. / sigma_inverse
    gamma = 1. / gamma_inverse

    total_population = sum(ages.values())  # 总人口数：各个年龄段的人数总和
    # 初始化起初的感染人数
    initial_infected_number = min(total_population * percent_of_initial_infected_seeds, ages[initial_infected_age])
    # print('初始感染的人数为：', initial_infected_number)
    # 总共有多少个年龄段
    # num_agebrackets = len(ages)

    # simulation output
    states = np.zeros((8, num_agebrackets, timesteps + 1))
    states_increase = np.zeros((8, num_agebrackets, timesteps + 1))
    cumulative_incidence = np.zeros((num_agebrackets, timesteps))
    incidenceN_by_setting = np.zeros((4, num_agebrackets, timesteps))
    incidenceV_by_setting = np.zeros((4, num_agebrackets, timesteps))

    # indices
    indices = {'susceptible': 0, 'exposedN': 1, 'exposedV': 2, 'asymptomaticN': 3, 'asymptomaticV': 4, 'infectedN': 5,
               'infectedV': 6, 'quarantined': 7}

    # setting indices
    setting_indices = {'household': 0, 'school': 1, 'workplace': 2, 'community': 3}

    settings = ['household', 'school', 'workplace', 'community']

    # initial conditions
    states[indices['infectedN']][initial_infected_age][0] = initial_infected_number
    for age in range(num_agebrackets):
        states[indices['susceptible']][age][0] = copy.deepcopy(ages[age]) - states[indices['infectedN']][age][0]

    age_effective_contact_matrix_dic = {}
    for setting in contact_matrix_dic:
        age_effective_contact_matrix_dic[setting] = get_age_effective_contact_matrix_with_factor_vector(
            contact_matrix_dic[setting], susceptibility_factor_vector)

    for t in range(timesteps):
        for i in range(num_agebrackets):
            for j in range(num_agebrackets):
                for kk, setting in enumerate(settings):
                    incidenceN_by_setting[kk][i][t] += p * beta * weights[setting] * \
                                                       age_effective_contact_matrix_dic[setting][i][j] * \
                                                       states[indices['susceptible']][i][t] * (
                                                               states[indices['infectedN']][j][t] +
                                                               states[indices['infectedV']][j][t] +
                                                               states[indices['asymptomaticN']][j][t] +
                                                               states[indices['asymptomaticV']][j][t]) / ages[j]

                    incidenceV_by_setting[kk][i][t] += (1 - p) * w * beta * weights[setting] * \
                                                       age_effective_contact_matrix_dic[setting][i][j] * \
                                                       states[indices['susceptible']][i][t] * (
                                                               states[indices['infectedN']][j][t] +
                                                               states[indices['infectedV']][j][t] +
                                                               states[indices['asymptomaticN']][j][t] +
                                                               states[indices['asymptomaticV']][j][t]) / ages[j]

            if (states[indices['susceptible']][i][t] - incidenceN_by_setting[:, i, t].sum() - incidenceV_by_setting[:,
                                                                                              i, t].sum() < 0):
                states[indices['susceptible']][i][t + 1] = 0
            else:
                states[indices['susceptible']][i][t + 1] = states[indices['susceptible']][i][t] - incidenceN_by_setting[
                                                                                                  :, i,
                                                                                                  t].sum() - incidenceV_by_setting[
                                                                                                             :, i,
                                                                                                             t].sum()
            # print(states[indices['susceptible']][i][t + 1])
            states[indices['exposedN']][i][t + 1] = states[indices['exposedN']][i][t] + incidenceN_by_setting[:, i,
                                                                                        t].sum() - sigma * \
                                                    states[indices['exposedN']][i][t]
            states[indices['exposedV']][i][t + 1] = states[indices['exposedV']][i][t] + incidenceV_by_setting[:, i,
                                                                                        t].sum() - sigma * \
                                                    states[indices['exposedV']][i][t]
            states[indices['infectedN']][i][t + 1] = states[indices['infectedN']][i][t] + (1-alpha*l)* sigma * \
                                                     states[indices['exposedN']][i][t] - gamma * \
                                                     states[indices['infectedN']][i][t]
            states[indices['infectedV']][i][t + 1] = states[indices['infectedV']][i][t] + (1-l) * q * sigma * \
                                                     states[indices['exposedV']][i][t] - gamma * \
                                                     states[indices['infectedV']][i][t]
            states[indices['asymptomaticN']][i][t + 1] = states[indices['asymptomaticN']][i][t] + alpha *l *sigma * \
                                                         states[indices['exposedN']][i][t] - k * gamma * \
                                                         states[indices['asymptomaticN']][i][t]
            states[indices['asymptomaticV']][i][t + 1] = states[indices['asymptomaticV']][i][t] + l * sigma * \
                                                         states[indices['exposedV']][i][t] - k * gamma * \
                                                         states[indices['asymptomaticV']][i][t]
            states[indices['quarantined']][i][t + 1] = states[indices['quarantined']][i][t] + gamma * (
                        states[indices['infectedV']][i][t] + states[indices['infectedN']][i][t]) + k * gamma * (
                                                                   states[indices['asymptomaticN']][i][t] +
                                                                   states[indices['asymptomaticV']][i][t])

            states_increase[indices['asymptomaticN']][i][t + 1] = l*alpha * sigma * states[indices['exposedN']][i][t]
            states_increase[indices['asymptomaticV']][i][t + 1] = l * sigma * states[indices['exposedV']][i][t]

            states_increase[indices['infectedN']][i][t + 1] = (1- alpha*l)*sigma * states[indices['exposedN']][i][t]
            states_increase[indices['infectedV']][i][t + 1] = l * q * k * sigma * states[indices['exposedV']][i][t]

            states_increase[indices['quarantined']][i][t + 1] = gamma * (
                    states[indices['infectedV']][i][t] + states[indices['infectedN']][i][t]) + k * gamma * (
                                                                        states[indices['asymptomaticN']][i][t] +
                                                                        states[indices['asymptomaticV']][i][t])
            cumulative_incidence[i][t] = (states_increase[indices['asymptomaticN']][i].sum() +
                                          states_increase[indices['asymptomaticV']][i].sum() +
                                          states_increase[indices['infectedN']][i].sum() +
                                          states_increase[indices['infectedV']][i].sum()) / ages[i] * 100
        # cumulative_all_group[t] = cumulative_incidence[t].sum

    return states, incidenceN_by_setting, incidenceV_by_setting, states_increase, cumulative_incidence, indices, setting_indices

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
    return np.round(np.loadtxt(file_path, delimiter=','), 6)


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

#%%保存总计各个年龄段的累计发病率(没写完）
def write_daily_all_cumulative_incidence(country, location, cumulative_incidence, p, p_v, l, l_v, k, k_v, num_agebrackets=18,
                               overwrite=True):
    file_name = f"{country}_{location}_cumulative_incidence_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', "bracket", 'INP',
                             file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
    return 0

#%% 保存每个年龄段的累计发病率
def write_cumulative_incidence(country, location, cumulative_incidence, p, p_v, l, l_v, k, k_v, num_agebrackets=18,
                               overwrite=True):
    file_name = f"{country}_{location}_cumulative_incidence_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', "bracket",file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(cumulative_incidence)):
                f.write(
                    f"{v:.2f},{cumulative_incidence[v][0]:.2f},{cumulative_incidence[v][1]:.2f},{cumulative_incidence[v][2]:.2f},{cumulative_incidence[v][3]:.2f},{cumulative_incidence[v][4]:.2f},{cumulative_incidence[v][5]:.2f},{cumulative_incidence[v][6]:.2f},{cumulative_incidence[v][7]:.2f},{cumulative_incidence[v][8]:.2f},{cumulative_incidence[v][9]:.2f},{cumulative_incidence[v][10]:.2f},{cumulative_incidence[v][11]:.2f},{cumulative_incidence[v][12]:.2f},{cumulative_incidence[v][13]:.2f},{cumulative_incidence[v][14]:.2f},{cumulative_incidence[v][15]:.2f},{cumulative_incidence[v][16]:.2f},{cumulative_incidence[v][17]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        print(f'{file_path} was written.')
        f = open(file_path, 'w+')
        for v in range(len(cumulative_incidence)):
            f.write(
                f"{v:.2f},{cumulative_incidence[v][0]:.2f},{cumulative_incidence[v][1]:.2f},{cumulative_incidence[v][2]:.2f},{cumulative_incidence[v][3]:.2f},{cumulative_incidence[v][4]:.2f},{cumulative_incidence[v][5]:.2f},{cumulative_incidence[v][6]:.2f},{cumulative_incidence[v][7]:.2f},{cumulative_incidence[v][8]:.2f},{cumulative_incidence[v][9]:.2f},{cumulative_incidence[v][10]:.2f},{cumulative_incidence[v][11]:.2f},{cumulative_incidence[v][12]:.2f},{cumulative_incidence[v][13]:.2f},{cumulative_incidence[v][14]:.2f},{cumulative_incidence[v][15]:.2f},{cumulative_incidence[v][16]:.2f},{cumulative_incidence[v][17]:.2f}\n")
        f.close()
    return file_path

# %%

num_agebrackets = 18  # number of age brackets for the contact matrices# R0_star = 1.6  # basic reproduction number
# initial_infected_age = 0
if num_agebrackets == 85:
    initial_infected_age = 20  # some initial age to seed infections within the population
elif num_agebrackets == 18:
    initial_infected_age = 4  # some initial age to seed infections within the population

percent_of_initial_infected_seeds = 1e-5
timesteps = 150  # how long to run the SEAIQ model

location = 'Shanghai'
country = 'China'
level = 'subnational'
state = "all_states"

contact_matrix = read_contact_matrix(location, country, level, 'overall', num_agebrackets)
contact_matrix_dic = get_contact_matrix_dic(location, country, level, num_agebrackets)
ages = get_ages(location, country, level, num_agebrackets)
susceptibility_factor_vector = get_susceptibility_factor_vector(num_agebrackets, example=2)

p = 'p'
w = 'w'
# beta = 'beta'
q = 'q'
l = 'l'
# param = 'sigma_inverse'
k = 'k'
q_v = 1
# param = 'gamma_inverse'
p_v = 0.5 # 0.1 0.5 0.9  # percent of unvaccinated
w_v = 0.5
beta = 0.4  #
l_v = 0.5 #接种后有症状的比例
sigma_inverse = 5.0  # mean incubation period
k_v = 0.5
gamma_inverse = 2  # mean quarantined period
alpha = 0.1 #接种人群相对比例



# weights = {'household': 3.5, 'school': 0, 'workplace': 0.5,
#            'community': 0.5}  # effective setting weights as found in our study

weights = {'household': 4.11, 'school': 11.41, 'workplace': 8.07,
           'community': 2.79}  # effective setting weights as found in our study

R0_list = []
attack = []

states, incidenceN, incidenceV, states_increase, cumulative_incidence, indices, setting_indices = dou_seaiq_with_age_specific_contact_martix(
    contact_matrix_dic, weights, ages, p_v, w_v, beta, q_v, l_v, k_v,alpha, susceptibility_factor_vector,
    sigma_inverse, gamma_inverse, initial_infected_age, percent_of_initial_infected_seeds,
    num_agebrackets, timesteps)



# # 写入每个年龄段的累计发病率
# write_cumulative_incidence(country, location, cumulative_incidence.T, p, p_v, l, l_v, k, k_v)
#
#
# #写每个状态的仿真
# all_states = np.round(np.sum(states, axis=1))
# result_file_path = write_data(country, location, state, np.round(all_states.T), p, p_v, l, l_v, k, k_v, num_agebrackets,
#                               overwrite=True)
#
# # 将每日增长的病例保存起来
# new_case_per_day = np.round(np.sum(states_increase, axis=1).T)
# write_per_day_new_cases(country, location, new_case_per_day, p, p_v, num_agebrackets, overwrite=True)



total_asymptomaticN = states_increase[indices['asymptomaticN']].sum()
total_asymptomaticV = states_increase[indices['asymptomaticV']].sum()
total_infectedN = states_increase[indices['infectedN']].sum()
total_infectedV = states_increase[indices['infectedV']].sum()

attack = total_asymptomaticN+total_asymptomaticV+total_infectedV+total_infectedN
total_population = sum(ages.values())
attack_rate = attack/total_population
print('总感染人数',attack)
print('总感染率',attack_rate)
print('各年龄段感染率',cumulative_incidence[:,-1],'\n',cumulative_incidence[:,120])
