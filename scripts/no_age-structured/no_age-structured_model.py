import csv

import numpy as np
import copy
import os
from decimal import Decimal
import string
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# set some initial paths

# path to the directory where this script lives
thisdir = os.path.abspath('')

# path to the scripts directory of the repository
scriptsdir = os.path.split(thisdir)[0]

# path to the main directory of the repository
maindir = os.path.split(scriptsdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(os.path.split(thisdir)[0], 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')

# path to the output_source subsubdirectory
output_source_dir = os.path.join(datadir, 'output_source')

# %% 将得到的结果读入csv文件中
def write_data(state, value, p,p_v,l,l_v,k,k_v, overwrite=True):

    file_name = f"{state}_numbers_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name)
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

# %% 保存每日新增病例数
def write_per_day_new_cases(new_case_per_day,p,p_v,l,l_v,k,k_v, overwrite=True):

    file_name = f"new_case_per_day_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(new_case_per_day)):
                f.write(
                    f"{v:.2f},{new_case_per_day[v][0]:.2f},{new_case_per_day[v][1]:.2f},{new_case_per_day[v][2]:.2f},{new_case_per_day[v][3]:.2f},{new_case_per_day[v][4]:.2f},{new_case_per_day[v][5]:.2f},{new_case_per_day[v][6]:.2f},{new_case_per_day[v][7]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        f = open(file_path, 'w+')
        for v in range(len(new_case_per_day)):
            f.write(
                f"{v:.2f},{new_case_per_day[v][0]:.2f},{new_case_per_day[v][1]:.2f},{new_case_per_day[v][2]:.2f},{new_case_per_day[v][3]:.2f},{new_case_per_day[v][4]:.2f},{new_case_per_day[v][5]:.2f},{new_case_per_day[v][6]:.2f},{new_case_per_day[v][7]:.2f}\n")
        f.close()
    return file_path

# %% 保存每次运行的参数
def write_param_value(p, w, beta, q, l, sigma_inverse, k, gamma_inverse, result_file_path):
    date = datetime.datetime.now()
    a = str(p)
    w = str(w)
    beta = str(beta)
    q = str(q)
    l = str(l)
    sigma_inverse = str(sigma_inverse)
    # k = str(k)
    gamma_inverse = str(gamma_inverse)
    file_path = os.path.join(datadir, 'param_run_log_no_age-structured.csv')
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists, a new line of data is written')
        file = open(file_path, mode='a+', encoding='utf-8', newline='')
        sWrite = csv.writer(file)
        sWrite.writerow([date, p, w, beta, q, l, sigma_inverse, k, gamma_inverse, result_file_path])
        file.close()
    else:
        print(f'{file_path} is created, a row of data is written')
        file = open(file_path, mode='w', encoding='utf-8', newline='')
        sWrite = csv.writer(file)
        sWrite.writerow(['date', 'p', 'w', 'beta', 'q', 'l', 'sigma_inverse', 'k', 'gamma_inverse', result_file_path])
        sWrite.writerow([date, p, w, beta, q, l, sigma_inverse, k, gamma_inverse, result_file_path])
        file.close()
    return

# %% 将得到的结果读入csv文件中
def write_data_param(state, value, p, p_value, l, l_value, k, overwrite=True):
    file_name = f"{state}_numbers_{p}={p_value:.1f}_{l}={l_value:.1f}_{k}.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', 'mutil_params', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f}\n")
        f.close()
    return file_path

# %% 将得到的结果读入csv文件中
def write_attack_ratio(value, p, p_value, l, k, overwrite=True):
    file_name = f"attack_ratio_{p}={p_value:.1f}_{l}_{k}.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', 'mutil_params', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f},{value[v][8]:.2f},{value[v][9]:.2f},{value[v][10]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f},{value[v][8]:.2f},{value[v][9]:.2f},{value[v][10]:.2f}\n")
        f.close()
    return file_path

# %% 将得到的结果读入csv文件中
def write_cumulative_incidence(value, p, p_value, l,l_value, k, overwrite=True):
    file_name = f"cumulative_incidence_{p}={p_value:.1f}_{l}_{l_value}_{k}.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f},{value[v][8]:.2f},{value[v][9]:.2f},{value[v][10]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f},{value[v][8]:.2f},{value[v][9]:.2f},{value[v][10]:.2f}\n")
        f.close()
    return file_path

# %% 模型的模拟
def simulation_with_no_age_structured_model(ages, p, w, beta, q, l, sigma_inverse, k, gamma_inverse,
                                            percent_of_initial_infected_seeds, timesteps):
    """
        Args:
            ages (dict)                               : a dictionary of the age count of people in the population
            p (float)                                 : percent of the population with unvaccinated
            w (float)                                 :
            beta (float)                              : the transmissibility
            q (float)                                 : percentage of unvaccinated exposed becoming symptomatic infections
            l (float)                                 : percentage of vaccinated exposed becoming symptomatic infections
            sigma_inverse (float)                     : the latent period for vaccinated exposed
            k (float)                                 :
            gamma_inverse (float)                     : the quarantined period for asyptomatic infections
            percent_of_initial_infected_seeds (float) : percent of the population initially seeded with infections.
            timesteps (int)                           : the number of timesteps

        Returns:
            A numpy array with the number of people in each disease state (Dou S-E-A-I-Q) by age for each timestep,
            a numpy array with the incidence by age for each timestep and the disease state indices.
        """
    sigma = 1. / sigma_inverse
    gamma = 1. / gamma_inverse

    total_population = sum(ages.values())  # 总人口数：各个年龄段的人数总和
    print('总人数为：', total_population)
    # 初始化起初的感染人数
    initial_infected_number = np.round(total_population * percent_of_initial_infected_seeds)
    print('初始感染的人数为：', initial_infected_number)

    # simulation output
    states = np.zeros((8, timesteps + 1))
    states_increase = np.zeros((8, timesteps+1))
    cumulative_incidence = np.zeros(timesteps)
    incidenceN = np.zeros(timesteps)
    incidenceV = np.zeros(timesteps)

    # indices
    indices = {'susceptible': 0, 'exposedN': 1, 'exposedV': 2, 'asymptomaticN': 3, 'asymptomaticV': 4, 'infectedN': 5,
               'infectedV': 6, 'quarantined': 7}

    # initial conditions
    states[indices['infectedN']][0] = initial_infected_number
    states[indices['susceptible']][0] = total_population - states[indices['infectedN']][0]

    for t in range(timesteps):
        incidenceN[t] = p * beta * states[indices['susceptible']][t] * (
                states[indices['infectedN']][t] + states[indices['infectedV']][t] +
                states[indices['asymptomaticN']][t] + states[indices['asymptomaticV']][t]) / total_population
        incidenceV[t] = (1 - p) * w * beta * states[indices['susceptible']][t] * (
                states[indices['infectedN']][t] + states[indices['infectedV']][t] +
                states[indices['asymptomaticN']][t] + states[indices['asymptomaticV']][t]) / total_population

        states[indices['susceptible']][t + 1] = states[indices['susceptible']][t] - incidenceN[t] - incidenceV[t]
        states[indices['exposedN']][t + 1] = states[indices['exposedN']][t] + incidenceN[t] - sigma * \
                                             states[indices['exposedN']][t]
        states[indices['exposedV']][t + 1] = states[indices['exposedV']][t] + incidenceV[t] - sigma * \
                                             states[indices['exposedV']][t]
        states[indices['infectedN']][t + 1] = states[indices['infectedN']][t] + q * sigma * states[indices['exposedN']][
            t] - gamma * states[indices['infectedN']][t]
        states[indices['infectedV']][t + 1] = states[indices['infectedV']][t] + l * q * sigma * \
                                              states[indices['exposedV']][t] - gamma * states[indices['infectedV']][t]
        states[indices['asymptomaticN']][t + 1] = states[indices['asymptomaticN']][t] + (1 - q) * sigma * \
                                                  states[indices['exposedN']][t] - k * gamma * \
                                                  states[indices['asymptomaticN']][t]
        states[indices['asymptomaticV']][t + 1] = states[indices['asymptomaticV']][t] + (1 - l * q) * sigma * \
                                                  states[indices['exposedV']][t] - k * gamma * \
                                                  states[indices['asymptomaticV']][t]
        states[indices['quarantined']][t + 1] = states[indices['quarantined']][t] + gamma * (
                    states[indices['infectedN']][t] + states[indices['infectedV']][t]) + k * gamma * (
                                                            states[indices['asymptomaticN']][t] +
                                                            states[indices['asymptomaticV']][t])

        # 每日新增的感染人数
        states_increase[indices['asymptomaticN']][t+1] = (1 - q) * sigma * states[indices['exposedN']][t]
        states_increase[indices['asymptomaticV']][t+1] = (1 - l * q) * sigma * states[indices['exposedV']][t]
        states_increase[indices['infectedN']][t+1] = q * sigma * states[indices['exposedN']][t]
        states_increase[indices['infectedV']][t+1] = l * q * sigma * states[indices['exposedV']][t]

        # 每日新增恢复或隔离人数
        states_increase[indices['quarantined']][t+1] = gamma * (
                    states[indices['infectedV']][t] + states[indices['infectedN']][t]) + k * gamma * (
                                                                 states[indices['asymptomaticN']][t] +
                                                                 states[indices['asymptomaticV']][t])
        cumulative_incidence[t] = (states_increase[indices['asymptomaticN']].sum() + states_increase[indices['asymptomaticV']].sum() + states_increase[indices['infectedN']].sum() + states_increase[indices['infectedV']].sum()) / sum(ages.values()) * 100

    return states, incidenceN, incidenceV, indices, states_increase, cumulative_incidence


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


# %% 计算R0
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
    R0 = beta * (p * ((1 - q) + k * q) + (1 - p) * w * ((1 - l * q) + k * l * q))/(k * gamma)
    # R0 = beta * (p * ((1 - q) + k * q) + (1 - p) * l * w * ((1 - q) + k * q)) / (k * gamma)

    return R0


# %%
# Dou-SEAIQ model parameters
num_agebrackets = 18  # number of age brackets for the contact matrices

p_v = 0.9  #0.1 0.5  0.9   # percent of unvaccinated
w_v = 0.5
beta = 0.6  #
q_v = 1  #
l_v = 0.8  # 0-1
sigma_inverse = 5.2  # mean latent period
k_v = 0.8  # 0-1
gamma_inverse = 5  # mean quarantined period

percent_of_initial_infected_seeds = 1e-5
timesteps = 300  # how long to run the SEAIQ model

location = 'Shanghai'
country = 'China'
level = 'subnational'
state = "all_states"
ages = get_ages(location, country, level, num_agebrackets)
# print(ages)
# print(ages.values())
# print(ages[1])

# param = ''
p = 'p'
# param2 = 'w'
# param = 'q'
l = 'l'
k = 'k'
# param = 'sigma_inverse'
# param = 'gamma_inverse'


R0_list = []
attack = []


states, incidenceN, incidenceV, indices, state_increase, cumulative_incidence = simulation_with_no_age_structured_model(ages, p_v, w_v, beta,
                                                                                                                                q_v, l_v,
                                                                                                                                sigma_inverse, k_v,
                                                                                                                                gamma_inverse,
                                                                                                                                percent_of_initial_infected_seeds,
                                                                                                                                timesteps)
S = states[indices['susceptible']]
exposedN = states[indices['exposedN']]
exposedV = states[indices['exposedV']]
asymptomaticN = states[indices['asymptomaticN']]
asymptomaticV = states[indices['asymptomaticV']]
infectedN = states[indices['infectedN']]
infectedV = states[indices['infectedV']]
quarantined = states[indices['quarantined']]
all_states = np.vstack((S, exposedN, exposedV, asymptomaticN, asymptomaticV, infectedN, infectedV, quarantined))

total_asymptomaticN = state_increase[indices['asymptomaticN']].sum()
total_asymptomaticV = state_increase[indices['asymptomaticV']].sum()
total_infectedN = state_increase[indices['infectedN']].sum()
total_infectedV = state_increase[indices['infectedV']].sum()

'''
print(states.shape)
# print(S[200:250])
# print(exposed[200:250])
day = [i for i in range(201)]
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(9, 8))
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
plt.suptitle("sqid", fontsize=20)
axes.set_xlabel('Day', labelpad=15, fontsize=15)
axes.set_ylabel('Number', labelpad=10, fontsize=15)
# plt.plot(day, S, label='susceptible')
# plt.plot(day, exposedN, label="exposedN")
# plt.plot(day, exposedV, label='exposedV')
plt.plot(day, asymptomaticV,label='asymptomatic')
plt.plot(day, infectedN, label='infectedN')
plt.plot(day, infectedV, label='infectedV')
# plt.plot(day, quarantined, label='quarantined')
plt.legend()
plt.show()
# subresult.append([total_asymptomaticN, total_asymptomaticV, total_infectedN, total_infectedV])'''


# # 将每次运行的结果存起来
# result_file_path = write_data(state, np.round(all_states.T), p, p_v, l, l_v, k, k_v, overwrite=True)
# # write_param_value(p_v, w_v, beta, q_v, l_v, sigma_inverse, k_v, gamma_inverse, result_file_path)
# # 将每日增长的病例保存起来
# new_case_per_day = np.round(state_increase.T)
# write_per_day_new_cases(new_case_per_day, p, p_v, l, l_v, k, k_v, overwrite=True)

# attack_rate = (state_increase[indices['asymptomaticN']].sum() + state_increase[indices['asymptomaticV']].sum() + state_increase[indices['infectedN']].sum() + state_increase[indices['infectedV']].sum()) / sum(ages.values()) * 100
# # R0 = get_R0(beta, p, w, q, l, k, gamma_inverse)
# # R0_list.append(float("%.5f" % R0))
# attack_rate_list=[]
# attack_rate_list.append(float("%.2f" % attack_rate))
# print(cumulative_incidence_list.shape)
# print(np.array(subresult).shape)
# # 将每次运行的结果存起来
# result_file_path = write_data_param(state='four_states', value=subresult, p="p", p_value=p, l="l", l_value=l, k="k")
# # 将每次运行的参数保存起来
# write_param_value(p, w, beta, q, l, sigma_inverse, "[0-1]", gamma_inverse, result_file_path)
# # print(param1 + "=" + str(p), param2, '发病率：', attack_rate_list)
# # print('基本再生数R0：', R0_list)
# attack.append(attack_rate_list)

p_v = 0.1  #0.1 0.5  0.9   # percent of unvaccinated


for i in range(0, 11):
    l_v = i / 10.0
    attack_rate_list = []
    subresult = []
    cumulative_incidence_list = []
    for j in range(0, 11):
        k_v = j / 10.0
        states, incidenceN, incidenceV, indices, state_increase, cumulative_incidence = simulation_with_no_age_structured_model(ages, p_v, w_v, beta,
                                                                                                          q_v, l_v,
                                                                                                          sigma_inverse, k_v,
                                                                                                          gamma_inverse,
                                                                                                          percent_of_initial_infected_seeds,
                                                                                                          timesteps)

        S = states[indices['susceptible']]
        exposedN = states[indices['exposedN']]
        exposedV = states[indices['exposedV']]
        asymptomaticN = states[indices['asymptomaticN']]
        asymptomaticV = states[indices['asymptomaticV']]
        infectedN = states[indices['infectedN']]
        infectedV = states[indices['infectedV']]
        quarantined = states[indices['quarantined']]
        all_states = np.vstack((S, exposedN, exposedV, asymptomaticN, asymptomaticV, infectedN, infectedV, quarantined))

        total_asymptomaticN = state_increase[indices['asymptomaticN']].sum()
        total_asymptomaticV = state_increase[indices['asymptomaticV']].sum()
        total_infectedN = state_increase[indices['infectedN']].sum()
        total_infectedV = state_increase[indices['infectedV']].sum()

        # subresult.append([total_asymptomaticN, total_asymptomaticV, total_infectedN, total_infectedV])

        # 将每次运行的结果存起来
        # result_file_path = write_data(state, np.round(all_states.T), p, p_v, l, l_v, k, k_v, overwrite=True)
        # write_param_value(p_v, w_v, beta, q_v, l_v, sigma_inverse, k_v, gamma_inverse, result_file_path)
        # 将每日增长的病例保存起来
        # new_case_per_day = np.round(state_increase.T)
        # write_per_day_new_cases(new_case_per_day, p, p_v, l, l_v, k, k_v, overwrite=True)

        # attack_rate = (state_increase[indices['asymptomaticN']].sum() + state_increase[indices['asymptomaticV']].sum() + state_increase[indices['infectedN']].sum() + state_increase[indices['infectedV']].sum()) / sum(ages.values()) * 100
        # R0 = get_R0(beta, p, w, q, l, k, gamma_inverse)
        # R0_list.append(float("%.5f" % R0))
        # attack_rate_list.append(float("%.2f" % attack_rate))
        cumulative_incidence_list.append(cumulative_incidence)

    cumulative_incidence_list = np.array(cumulative_incidence_list)
    # print(cumulative_incidence_list.shape)
    write_cumulative_incidence(cumulative_incidence_list.T, p, p_v, l, l_v, k, overwrite=True)
    # print(np.array(subresult).shape)
    # # 将每次运行的结果存起来
    # result_file_path = write_data_param(state='four_states', value=subresult, p="p", p_value=p, l="l", l_value=l, k="k")
    # # 将每次运行的参数保存起来
    # write_param_value(p, w, beta, q, l, sigma_inverse, "[0-1]", gamma_inverse, result_file_path)
    # # print(param1 + "=" + str(p), param2, '发病率：', attack_rate_list)
    # # print('基本再生数R0：', R0_list)
    # attack.append(attack_rate_list)



attack = np.array(attack)
write_attack_ratio(attack, p=p, p_value=p_v, l=l, k=k)
print(attack.shape)

