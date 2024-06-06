import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time

# set some initial paths

# path to the directory where this script lives
thisdir = os.path.abspath('')

# path to the scripts directory of the repository
scriptsdir = os.path.split(thisdir)[0]

# path to the main directory of the repository
maindir = os.path.split(scriptsdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(os.path.split(scriptsdir)[0], 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')

# path to the output_source subsubdirectory
output_source_dir = os.path.join(datadir, 'output_source')


def read_states_numbers(country, location, state, num_agebrackets=85, param='', param_value=0.0):
    """
    :param country:
    :param location:
    :param state_type (str):                读取包含所有状态的还是某单个状态的数据文件
    :param param (str):                改变的参数是哪个
    :param param_value (float):        参数的值
    :return: np.ndarray:               A numpy matrix of state value.
    """

    if param:
        file_name = f"{country}_{location}_{state}_vaccine_allocation.csv"
        file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'vaccine_allocation', file_name)
        M = np.loadtxt(file_path, delimiter=',')
    else:
        file_name = f"{country}_{location}_{state_type}_numbers_{num_agebrackets}.csv"
        file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)
        print(file_path + ' were read')
        M = np.loadtxt(file_path, delimiter=',')
    return M


# 读取需要对比状态的数据
def read_compare_state_numbers(country, location, state_type, single_state, num_agebrackets, param, param_value):
    # indices
    indices = {'susceptible': 0, 'exposedN': 1, 'exposedV': 2, 'asymptomaticN': 3, 'asymptomaticV': 4, 'infectedN': 5,
               'infectedV': 6, 'quarantined': 7}
    single_state_data = []
    for pv in param_value:
        file_name = f"{country}_{location}_{state_type}_numbers_{num_agebrackets}_{param}={pv:.4f}.csv"
        file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)#拼接output_source_dir/agestr---/file_name
        M = np.loadtxt(file_path, delimiter=',', usecols=indices[single_state])#读取indices[single_state】一列

        # M1 = np.loadtxt(file_path, delimiter=',', usecols=indices[single_state])
        # M2 = np.loadtxt(file_path, delimiter=',', usecols=indices[single_state]+1)
        # single_state_data.append(M1+M2)
        single_state_data.append(M)

    single_state_data = np.array(single_state_data).T

    return single_state_data


# 画基本传播图
def plot_basic_fig(country, location, all_state_data, num_agebrackets=85, param='', param_value=0.0, isSave=True):
    # 画图
    fontsizes = {'colorbar': 30, 'colorbarlabels': 22, 'title': 44, 'ylabel': 28, 'xlabel': 28, 'xticks': 24,
                 'yticks': 24}
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot(all_state_data[:, 0], all_state_data[:, 1])
    ax.plot(all_state_data[:, 0], all_state_data[:, 2])
    ax.plot(all_state_data[:, 0], all_state_data[:, 3])
    ax.plot(all_state_data[:, 0], all_state_data[:, 4])
    ax.plot(all_state_data[:, 0], all_state_data[:, 5])
    ax.plot(all_state_data[:, 0], all_state_data[:, 6])
    ax.plot(all_state_data[:, 0], all_state_data[:, 7])
    ax.plot(all_state_data[:, 0], all_state_data[:, 8])

    # ax = plt.gca()
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    # ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax.yaxis.get_offset_text().set_fontsize(16)  # 设置1e6的大小与位置


    ax.legend(['susceptible', 'exposedV', 'exposedN', 'asymptomaticV', 'asymptomaticN', 'infectedV', 'infectedN', 'quarantined'], loc='upper right')
    axins = inset_axes(ax, width=2.5, height=1.5, loc='center right')

    axins.plot(all_state_data[10:60, 0], all_state_data[[0 for i in range(10, 60)], 0])
    for i in range(2, 8):
        axins.plot(all_state_data[10:60, 0], all_state_data[10:60, i])
    if param:
        plt.text(140, 100000, param + '=' + str(param_value))
    title = 'basic with age structured'
    ax.set_title(title, fontsize=32)
    ax.set_ylabel('number of ' + state_type, fontsize=fontsizes['ylabel'])
    ax.set_xlabel('time', fontsize=fontsizes['xlabel'])

    plt.show()

    # 是否保存
    if (isSave):
        if param:
            file_name = country + '_' + location + '_' + state_type + '_' + str(
                num_agebrackets) + '_' + param + '=' + str(param_value) + '.pdf'
        else:
            file_name = country + '_' + location + '_' + state_type + '_' + str(num_agebrackets) + '.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        if param:
            file_name = country + '_' + location + '_' + state_type + '_' + str(
                num_agebrackets) + '_' + param + '=' + str(param_value) + '.eps'
        else:
            file_name = country + '_' + location + '_' + state_type + '_' + str(num_agebrackets) + '.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')


# 画状态对比图
def plot_compare_fig(country, location, single_state_data, param, param_value, single_state, num_agebrackets=85,
                     isSave=True):
    """
    :param all_states (str): 读取的文件是包含所有状态的数据文件
    :param param:
    :param single_state:
    :param isSave:
    :return:
    """
    time = np.arange(151)
    # 画图
    fontsizes = {'colorbar': 30, 'colorbarlabels': 22, 'title': 44, 'ylabel': 28, 'xlabel': 28, 'xticks': 24,
                 'yticks': 24}
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # param_value_length = len(param_value)
    for index, pv in enumerate(param_value):
        # plt.plot(time, single_state_data[:, index])
        plt.plot(time[:80], single_state_data[:80, index])

    plt.legend([param + '=' + str(pv) for pv in param_value])

    # plt.legend([(param + '=' + str(pv)) for pv in param_value])

    title = single_state
    ax.set_title(title, fontsize=fontsizes['title'])
    ax.set_ylabel('numbers', fontsize=fontsizes['ylabel'])
    ax.set_xlabel('time', fontsize=fontsizes['xlabel'])

    plt.show()

    if isSave:
        file_name = country + '_' + location + '_' + str(num_agebrackets) + '_' + single_state + '_' + param + '.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')


# 画每日病例新增图
def plot_per_day_new_cases_fig(country, location, single_state, num_agebrackets=85, param='', param_value=[],
                               isSave=True):
    # indices
    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8, 'asymptomatic': 9, 'infected': 10, 'infectious': 11}
    if param:
        file_name = f"{country}_{location}_per_day_new_increase_infectious_numbers_{num_agebrackets}_{param}={param_value:.4f}.csv"
    else:
        file_name = f"{country}_{location}_per_day_new_increase_infectious_numbers_{num_agebrackets}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)
    print(file_path + ' were read')
    per_day_new_cases = np.loadtxt(file_path, delimiter=',', skiprows=1)
    # 画图
    fontsizes = {'colorbar': 30, 'colorbarlabels': 22, 'title': 44, 'ylabel': 28, 'xlabel': 28, 'xticks': 24,
                 'yticks': 24}
    fig = plt.figure(figsize=(10, 10))
    params = ' p = 0.2 \n w = 0.8 \n q = 0.6 \n l  = 0.8 \n k = 0.5'
    if param:
        pass
    else:
        plt.text(60, 30000, params, bbox=dict(facecolor='orange', alpha=0.4))

    ax = fig.add_subplot(111)
    # plt.plot(per_day_new_cases[:70, 0], per_day_new_cases[:70, indices['asymptomatic']], label='asymptomatic')
    # plt.plot(per_day_new_cases[:70, 0], per_day_new_cases[:70, indices['infected']], label='infected')
    # plt.plot(per_day_new_cases[:70, 0], per_day_new_cases[:70, indices['infectious']], label='infectious')
    # plt.plot(per_day_new_cases[:70, 0], per_day_new_cases[:70, indices['asymptomaticN']], label='asymptomaticN')
    # plt.plot(per_day_new_cases[:70, 0], per_day_new_cases[:70, indices['asymptomaticV']], label='asymptomaticV')
    plt.plot(per_day_new_cases[:70, 0], per_day_new_cases[:70, indices['infectedN']], label='infectedN')
    # plt.plot(per_day_new_cases[:70, 0], per_day_new_cases[:70, indices['infectedV']], label='infectedV')

    plt.legend()
    title = 'per day new cases'
    ax.set_title(title, fontsize=fontsizes['title'])
    ax.set_ylabel('number of new cases', fontsize=fontsizes['ylabel'])
    ax.set_xlabel('time', fontsize=fontsizes['xlabel'])

    plt.show()

    # 是否保存
    if (isSave):
        name = 'infectious'
        if param:
            fig_name = f"{country}_{location}_per_day_new_cases_numbers_{num_agebrackets}_{name}_{param}={param_value:.4f}.pdf"
        else:
            fig_name = f"{country}_{location}_per_day_new_cases_numbers_{num_agebrackets}_{name}.pdf"
        fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
        fig.savefig(fig_path, format='pdf')


def plot_per_day_new_cases(country, location, num_agebrackets=85, isSave=True):
    file_name = f"{country}_{location}_per_day_new_increase_infectious_numbers_{num_agebrackets}.csv"
    # file_name = f"China_Shanghai_new_case_per_day_18_p=0.50_INP.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)
    print(file_path + ' was read')
    per_day_new_cases = np.loadtxt(file_path, skiprows=1, delimiter=',')
    per_day_new_cases = per_day_new_cases[:60].T
    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8, 'asymptomatic': 9, 'infected': 10, 'infectious': 11}

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    plt.suptitle('per day new cases with age-structured')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.4)

    ax1[0].set_title('asymptomatic and no-vaccine')
    ax1[0].plot(per_day_new_cases[indices['asymptomaticN']])
    max = np.max(per_day_new_cases[indices['asymptomaticN']])
    ax1[0].axhline(y=max, color='r', linestyle='--')
    ax1[0].text(0, max + 500, str(max), color='r')

    ax1[1].set_title('symptomatic and no-vaccine')
    ax1[1].plot(per_day_new_cases[indices['infectedN']])
    max = np.max(per_day_new_cases[indices['infectedN']])
    ax1[1].axhline(y=max, color='r', linestyle='--')
    ax1[1].text(0, max + 500, str(max), color='r')

    ax2[0].set_title('asymptomatic and vaccine')
    ax2[0].plot(per_day_new_cases[indices['asymptomaticV']])
    max = np.max(per_day_new_cases[indices['asymptomaticV']])
    ax2[0].axhline(y=max, color='r', linestyle='--')
    ax2[0].text(0, max + 500, str(max), color='r')

    ax2[1].set_title('symptomatic and vaccine')
    ax2[1].plot(per_day_new_cases[indices['infectedV']])
    max = np.max(per_day_new_cases[indices['infectedV']])
    ax2[1].axhline(y=max, color='r', linestyle='--')
    ax2[1].text(0, max + 500, str(max), color='r')

    ax3[0].set_title('total asymptomatic')
    ax3[0].plot(per_day_new_cases[indices['asymptomatic']])
    max = np.max(per_day_new_cases[indices['asymptomatic']])
    ax3[0].axhline(y=max, color='r', linestyle='--')
    ax3[0].text(0, max - 5000, str(max), color='r')

    ax3[1].set_title('total symptomatic')
    ax3[1].plot(per_day_new_cases[indices['infected']])
    max = np.max(per_day_new_cases[indices['infected']])
    ax3[1].axhline(y=max, color='r', linestyle='--')
    ax3[1].text(0, max - 5000, str(max), color='r')

    plt.show()

    if isSave:
        file_name = f"{country}_{location}_per_day_new_cases_of_infectious_{num_agebrackets}.pdf"
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')
        file_name = f"{country}_{location}_per_day_new_cases_of_infectious_{num_agebrackets}.eps"
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')


# 画每日病例新增图
def plot_per_day_new_cases_compare_fig(country, location, single_state, num_agebrackets=18, param='', param_value=[],
                                       isSave=True):
    # indices
    indices = {'susceptible': 1, 'exposedN': 2, 'exposedV': 3, 'asymptomaticN': 4, 'asymptomaticV': 5, 'infectedN': 6,
               'infectedV': 7, 'quarantined': 8, 'asymptomatic': 9, 'infected': 10, 'infectious': 11}
    per_day_new_cases_data = []
    for pv in param_value:
        file_name = f"{country}_{location}_per_day_new_increase_infectious_numbers_{num_agebrackets}_{param}={pv:.4f}.csv"
        file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)
        print(file_path + ' were read')
        per_day_new_cases = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=indices['asymptomatic'])
        per_day_new_cases_data.append(per_day_new_cases)

    per_day_new_cases_data = np.array(per_day_new_cases_data).T
    print(per_day_new_cases_data.shape)
    # 画图
    fontsizes = {'colorbar': 30, 'colorbarlabels': 22, 'title': 44, 'ylabel': 28, 'xlabel': 28, 'xticks': 24,
                 'yticks': 24}
    fig = plt.figure(figsize=(10, 10))
    params = ''
    if param == 'p':
        params = ' w = 0.8 \n q = 0.6 \n l  = 0.8 \n k = 0.5'
    if param == 'w':
        params = ' p = 0.2 \n q = 0.6 \n l  = 0.8 \n k = 0.5'
    if param == 'q':
        params = ' p = 0.2 \n w = 0.8 \n l  = 0.8 \n k = 0.5'
    if param == 'l':
        params = ' p = 0.2 \n w = 0.8 \n q = 0.6 \n k = 0.5'
    if param == 'k':
        params = ' p = 0.2 \n w = 0.8 \n q = 0.6 \n l  = 0.8'
    plt.text(90, 8000, params, bbox=dict(facecolor='orange', alpha=0.4))

    ax = fig.add_subplot(111)
    time = np.arange(150)
    for index, pv in enumerate(param_value):
        plt.plot(time[:100], per_day_new_cases_data[:100, index])

    plt.legend([param + '=' + str(pv) for pv in param_value])

    title = 'per day new asymptomatic cases'
    ax.set_title(title, fontsize=30)
    ax.set_ylabel('number of new cases', fontsize=fontsizes['ylabel'])
    ax.set_xlabel('time', fontsize=fontsizes['xlabel'])

    plt.show()

    # 是否保存
    if (isSave):
        name = 'asymptomatic'
        fig_name = f"{country}_{location}_per_day_new_cases_numbers_{num_agebrackets}_{name}_{param}.pdf"
        fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
        fig.savefig(fig_path, format='pdf')


location = 'Shanghai'
country = 'China'
num_agebrackets = 18

state_type = 'all_states'
# 画每个状态的对比图
param = ''
# param = 'p'
# param = 'w'
# param = 'q'
# param = 'l'
# param = 'k'
# param = 'sigma_inverse'
# param = 'gamma_inverse'
param_value = 0.0

# for i in range(10):
#     param_value = k = (i+1) / 10.0
# 得到整个传播过程的数据
all_states = read_states_numbers(country, location, state_type, num_agebrackets, param, param_value)
# # all_state_data, param='', param_value=0, isSave=True
plot_basic_fig(country, location, all_states, num_agebrackets, param=param, param_value=param_value, isSave=True)


# 画每个状态的对比图
# param = 'p'
# param = 'w'
# param = 'q'
param = 'l'
# param = 'k'
# param = 'sigma_inverse'
# param = 'gamma_inverse'

# single_state = 'susceptible'
# single_state = 'exposedN'
# single_state = 'exposedV'
# single_state = 'asymptomaticN'
# single_state = 'asymptomaticV'
# single_state = 'infectedN'
single_state = 'infectedV'
# single_state = 'quarantined'

# param_value = [pv / 10.0 for pv in range(11)]
# param_value = [(pv+1) / 10.0 for pv in range(10)]
# single_state_data = read_compare_state_numbers(country, location, state_type, single_state, num_agebrackets, param, param_value)
# plot_compare_fig(country, location, single_state_data, param, param_value, single_state, num_agebrackets)
#
#
# # 每日新增状态图
# plot_per_day_new_cases_fig(country, location, single_state, num_agebrackets, param='', param_value=0.0, isSave=True)


#画每日新增对比图
param = 'p'
param = 'w'
param = 'q'
param = 'l'
param = 'k'
param = 'sigma_inverse'
param = 'gamma_inverse'
param = ''

single_state = 'susceptible'
single_state = 'exposedN'
single_state = 'exposedV'
single_state = 'asymptomaticN'
single_state = 'asymptomaticV'
single_state = 'infectedN'
single_state = 'infectedV'
single_state = 'quarantined'
# param_value = [pv / 10.0 for pv in range(11)]
# param_value = [(pv+1) / 10.0 for pv in range(10)]

# plot_per_day_new_cases_compare_fig(country, location, single_state, num_agebrackets, param, param_value, isSave=False)
#
# plot_per_day_new_cases(country, location, num_agebrackets)
