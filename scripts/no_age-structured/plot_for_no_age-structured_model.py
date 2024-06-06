import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sbn
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap

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


# 替换字符串中的空格为"_"
def replaceSpace(s: str) -> str:
    # 定义一个列表用来存储结果
    res = []

    # 遍历循环字符串s
    # 当 i 是空格的时候，res拼接“%20”
    # 当 i 非空格时，res拼接当前字符i
    for i in s:
        if i == ' ':
            res.append("_")
        else:
            res.append(i)

    # 将列表转化为字符串返回
    return "".join(res)


def swap(matrix):
    len = matrix.shape
    a = np.ones(len)

    for i in range(len[0]):
        a[i, :] = matrix[len[0] - 1 - i, :]
    return a


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


location = 'Shanghai'
country = 'China'
level = 'subnational'
num_agebrackets = 18  # number of age brackets for the contact matrices

ages = get_ages(location, country, level, num_agebrackets)
total_population = sum(ages.values())


# 读取需要对比状态的数据
def read_data_param(state_type, p, p_value, l, l_value, k, k_value):
    res_data = []
    for kv in k_value:
        data = []
        for lv in l_value:
            file_name = f"new_case_per_day_no_age_{p}={p_value:.2f}_{l}={lv:.2f}_{k}={kv:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name)

            if state_type == 'asymptomaticN':
                #  asymptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=4).sum()
            elif state_type == 'asymptomaticV':
                #  asymptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=5).sum()
            elif state_type == 'symptomaticN':
                #  symptomaticN
                M = np.loadtxt(file_path, delimiter=',', usecols=6).sum()
            elif state_type == 'symptomaticV':
                #  symptomaticV
                M = np.loadtxt(file_path, delimiter=',', usecols=7).sum()
            elif state_type == 'asymptomatic':
                #  total asymptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=(4, 5)).sum(axis=1).sum()
            elif state_type == "symptomatic":
                #  total symptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=(6, 7)).sum(axis=1).sum()
            elif state_type == "total":
                #  total infected
                M = np.loadtxt(file_path, delimiter=',', usecols=(4, 5, 6, 7)).sum(axis=1).sum()
            else:
                #  total infected
                M = np.loadtxt(file_path, delimiter=',', usecols=(4, 5, 6, 7)).sum(axis=1).sum()

            data.append(M)
        res_data.append(data)
    res = np.array(res_data)

    return res


def plot_p_l_k(state_type, data, p, p_value, isSave=True):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(9, 8))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(state_type, fontsize=20)
    x_label = [i / 10.0 for i in range(0, 11)]
    y_label = [i / 10.0 for i in range(10, -1, -1)]
    my_colormap = LinearSegmentedColormap.from_list("black", ["green", "yellow", "orange", "red"])
    sbn.heatmap(swap(data) / total_population, vmin=0, cmap=my_colormap, yticklabels=y_label, xticklabels=x_label, ax=axes)

    axes.set_xlabel('k', labelpad=15, fontsize=15)
    axes.set_ylabel('l', labelpad=10, fontsize=15)
    plt.show()

    if isSave:
        file_name = state_type + '_' + p + '=' + str(p_value) + '.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = state_type + '_' + p + '=' + str(p_value) + '.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='eps')


def plot_p_l_k_mutil(state_type, data, p, p_value, isSave=True):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12, 5))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.18, top=0.80, wspace=0.1, hspace=0.1)
    plt.suptitle(state_type + " infection ratio", fontsize=20)
    x_label = [i / 10.0 for i in range(0, 11)]
    y_label = [i / 10.0 for i in range(10, -1, -1)]

    sbn.heatmap(swap(data[0]) / total_population, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label,
                ax=axes[0])
    axes[0].set_title("p = 0.1")
    axes[0].set_xlabel('k', labelpad=15, fontsize=15)
    axes[0].set_ylabel('l', labelpad=10, fontsize=15)

    sbn.heatmap(swap(data[1]) / total_population, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label,
                ax=axes[1])
    axes[1].set_title("p = 0.5")
    axes[1].set_xlabel('k', labelpad=15, fontsize=15)

    sbn.heatmap(swap(data[2]) / total_population, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label,
                ax=axes[2])
    axes[2].set_title("p = 0.9")
    axes[2].set_xlabel('k', labelpad=15, fontsize=15)

    plt.show()

    if isSave:
        file_name = state_type + '_infection_ratio_' + p + '.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = state_type + '_infection_ratio_' + p + '.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='eps')

def read_attack_ratio(p, p_value):
    file_name = f"attack_ratio_{p}={p_value:.1f}_l_k.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', 'mutil_params', file_name)

    M = np.loadtxt(file_path, delimiter=',')

    return M

def plt_attack_ratio(attack_data, p, p_value, l, k, isSave=True):
    x = [i/10 for i in range(11)]
    Y = X = np.array(x)
    X, Y = np.meshgrid(X, Y)
    Z = attack_data

    # 绘制表面
    fig = plt.figure(figsize=(6,5))

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel(l)
    ax.set_ylabel(k)
    # 添加将值映射到颜色的颜色栏
    fig.colorbar(surf, shrink=0.7, aspect=6)
    ax.set_title("attack ratio with p = " + str(p_value), fontsize=16)
    plt.show()

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    # plt.suptitle('attack rate', fontsize=15)
    # plt.subplots_adjust(left=0.15, right=0.90, bottom=0.14, top=0.90, wspace=0.1, hspace=0.4)
    #
    # ax.set_xlabel(l)
    # ax.set_ylabel('k')


    if isSave:
        file_name = 'attack_ratio' + '_' + p + '=' + str(p_value) + '_l_k.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = 'attack_ratio' + '_' + p + '=' + str(p_value) + '_l_k.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

p = "p"
p_value = 0.1  # 0.1 0.5  0.9   # percent of unvaccinated
l = "l"
l_value = [i / 10 for i in range(0, 11)]
k = "k"
k_value = [i / 10 for i in range(0, 11)]
state_type = "total"  # asymptomaticN, asymptomaticV, symptomaticN, symptomaticV, asymptomatic, symptomatic, total

data = read_data_param(state_type, p, p_value, l, l_value, k, k_value)
# print(data)
print(data.shape)
# plot_p_l_k(state_type, data, p, p_value, isSave=False)
#
# data1 = read_data_param(state_type, p, 0.1, l, l_value, k, k_value)
# data2 = read_data_param(state_type, p, 0.5, l, l_value, k, k_value)
# data3 = read_data_param(state_type, p, 0.9, l, l_value, k, k_value)
#
# data = [data1, data2, data3]
# plot_p_l_k_mutil(state_type, data, p, p_value, isSave=True)

# 画不同接种比例下的发病率情况
data = read_attack_ratio(p, p_value)
plt_attack_ratio(data, p, p_value, l, k, isSave=True)
