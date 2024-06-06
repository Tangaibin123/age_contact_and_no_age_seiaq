
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



def read_states_numbers(country, location, state,p_v, num_agebrackets=18, param='', param_value=0.0):
    """
    :param country:
    :param location:
    :param state_type (str):                读取包含所有状态的还是某单个状态的数据文件
    :param param (str):                改变的参数是哪个
    :param param_value (float):        参数的值
    :return: np.ndarray:               A numpy matrix of state value.
    """


    file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets}_p={p_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'p' + "_" + str(p_v), file_name)
    print(file_path + ' were read')
    M = np.loadtxt(file_path, delimiter=',')
    return M

def plot_basic_fig(country, location, all_state_data,p_v ,num_agebrackets=18, param='', param_value=0.0, isSave=True):
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


    ax.legend(['susceptible', 'exposedN', 'exposedV', 'asymptomaticN', 'asymptomaticV', 'infectedN', 'infectedV', 'quarantined'], loc='upper right')
    axins = inset_axes(ax, width=2.5, height=1.5, loc='center right')

    axins.plot(all_state_data[10:60, 0], all_state_data[[0 for i in range(10, 60)], 0])
    for i in range(2, 8):
        axins.plot(all_state_data[10:60, 0], all_state_data[10:60, i])
    if param:
        plt.text(140, 100000, param + '=' + str(param_value))
    title = f'basic with age structured p={p_v}'
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

location = 'Shanghai'
country = 'China'
num_agebrackets = 18

state_type = 'all_states'
# 画每个状态的对比图
p_v =0.5
param = ''
param_value = 0.0
all_states = read_states_numbers(country, location, state_type,p_v,num_agebrackets, param, param_value)
plot_basic_fig(country, location, all_states,p_v ,num_agebrackets, param=param, param_value=param_value, isSave=False)