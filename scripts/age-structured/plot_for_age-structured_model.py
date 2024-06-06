import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sbn
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
def read_data_param(location, country, state_type, num_agebrackets, p, p_value, l, l_value, k, k_value):
    res_data = []
    for kv in k_value:
        data = []
        for lv in l_value:
            file_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={p_value:.2f}_{l}={lv:.2f}_{k}={kv:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'age-structured_output_source', f"{p}_{p_value:.1f}", file_name)

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
                M = np.loadtxt(file_path, delimiter=',', usecols=(4, 5)).sum()
            elif state_type == "symptomatic":
                #  total symptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=(6, 7)).sum()
            elif state_type == "total":
                #  total infected
                M = np.loadtxt(file_path, delimiter=',', usecols=(4, 5, 6, 7)).sum()
            else:
                #  total infected
                M = np.loadtxt(file_path, delimiter=',', usecols=(4, 5, 6, 7)).sum()

            data.append(M)
        res_data.append(data)
    res = np.array(res_data)

    return res


# 读取需要对比状态的数据
def read_data_param_percent(location, country, state_type, num_agebrackets, p, p_value, l, l_value, k, k_value):
    res_data = []
    for kv in k_value:
        data = []
        for lv in l_value:
            file_name = f"{country}_{location}_four_states_numbers_{num_agebrackets}_{p}={p_value:.2f}_{l}={lv:.2f}_{k}={kv:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'mutil_params', file_name)

            if state_type == 'asymptomaticN':
                #  asymptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=0)
            elif state_type == 'asymptomaticV':
                #  asymptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=1)
            elif state_type == 'symptomaticN':
                #  symptomaticN
                M = np.loadtxt(file_path, delimiter=',', usecols=2)
            elif state_type == 'symptomaticV':
                #  symptomaticV
                M = np.loadtxt(file_path, delimiter=',', usecols=3)
            elif state_type == 'asymptomatic':
                #  total asymptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=(0, 1)).sum(axis=1)
            elif state_type == "symptomatic":
                #  total symptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=(2, 3)).sum(axis=1)
            elif state_type == "total":
                #  total infected
                M = np.loadtxt(file_path, delimiter=',').sum(axis=1)
            else:
                #  total infected
                M = np.loadtxt(file_path, delimiter=',').sum(axis=1)

            data.append(M)
        res_data.append(data)
    res = np.array(res_data)

    return res

#%%plot_p_l_k
def plot_p_l_k(location, country, state_type, data, p, p_value, isSave=True):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(9, 8))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(state_type, fontsize=20)
    x_label = [i / 10.0 for i in range(0, 11)]
    y_label = [i / 10.0 for i in range(10, -1, -1)]
    sbn.heatmap(swap(data) / total_population, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label, ax=axes)

    axes.set_xlabel('k', labelpad=15, fontsize=15)
    axes.set_ylabel('l', labelpad=10, fontsize=15)
    plt.show()

    if isSave:
        file_name = f"{country}_{location}_{state_type}_{num_agebrackets}_{p}={p_value:.1f}.pdf"
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f"{country}_{location}_{state_type}_{num_agebrackets}_{p}={p_value:.1f}.eps"
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='eps')
#%%%

def plot_p_l_k_mutil(location, country, state_type, data, p, num_agebrackets, p_value, isSave=True):
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
        file_name = f"{country}_{location}_{state_type}_{num_agebrackets}_infection_ratio_{p}.pdf"
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f"{country}_{location}_{state_type}_{num_agebrackets}_infection_ratio_{p}.eps"
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='eps')


def read_attack_ratio(p, p_value):
    file_name = f"attack_ratio_{p}={p_value:.1f}_l_k.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', 'mutil_params', file_name)

    M = np.loadtxt(file_path, delimiter=',')

    return M


def plt_attack_ratio(attack_data, p, p_value, l, k, isSave=True):
    x = [i / 10 for i in range(11)]
    Y = X = np.array(x)
    X, Y = np.meshgrid(X, Y)
    Z = attack_data

    # 绘制表面
    fig = plt.figure(figsize=(6, 5))

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

    if isSave:
        file_name = 'attack_ratio' + '_' + p + '=' + str(p_value) + '_l_k.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = 'attack_ratio' + '_' + p + '=' + str(p_value) + '_l_k.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')


def plot_four_infectious_bar(location, country, state, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=True):
    file_name = f"{country}_{location}_{state}_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:0.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', f"{p}_{p_v:.1f}", file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5, 6, 7)).T

    fig = plt.figure(figsize=(8, 5))
    plt.xlabel("Time")
    plt.ylabel("Number")
    plt.bar(x=M[0, :90], height=M[1, :90], color='#FF0000', label="not vaccinated asymptomatic")
    plt.bar(x=M[0, :90], bottom=M[1, :90], height=M[2, :90], color="#FFD700", label="vaccinated asymptomatic")
    plt.bar(x=M[0, :90], bottom=M[1, :90] + M[2, :90], height=M[3, :90], color="#DC143C",
            label="not vaccinated symptomatic")
    plt.bar(x=M[0, :90], bottom=M[1, :90] + M[2, :90] + M[3, :90], height=M[4, :90], color="#FF8C00",
            label="vaccinated symptomatic")
    plt.text(x=80, y=40000, s=f"{p} = {p_v}\n{l}  = {l_v}\n{k} = {k_v}",
             bbox=dict(facecolor="dimgray", alpha=0.5, boxstyle="round"))
    plt.legend()
    #
    plt.show()

    if isSave:
        file_name = f"{country}_{location}_infection_{num_agebrackets}_{p}={p_v}_{l}={l_v}_{k}={k_v}.pdf"
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f"{country}_{location}_infection_{num_agebrackets}_{p}={p_v}_{l}={l_v}_{k}={k_v}.eps"
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')


location = 'Shanghai'
country = 'China'
level = 'subnational'
state = "all_states"
p = "p"
l = "l"
k = "k"
p_v = 0.5  # 0.1 0.5 0.9
l_v = 0.8
k_v = 0.5
l_value = [i / 10 for i in range(0, 11)]
k_value = [i / 10 for i in range(0, 11)]

state_type = "total"  # asymptomaticN, asymptomaticV, symptomaticN, symptomaticV, asymptomatic, symptomatic, total

# data = read_data_param(location, country, state_type, num_agebrackets, p, p_v, l, l_value, k, k_value)
# print(data)
# print(data.shape)
# plot_p_l_k(location, country, state_type, data, p, p_v, isSave=True)

# data1 = read_data_param(location, country, state_type, num_agebrackets, p, 0.1, l, l_value, k, k_value)
# data2 = read_data_param(location, country, state_type, num_agebrackets, p, 0.5, l, l_value, k, k_value)
# data3 = read_data_param(location, country, state_type, num_agebrackets, p, 0.9, l, l_value, k, k_value)

# plot_p_l_k_mutil(location, country, state_type, data, num_agebrackets, p, p_v, isSave=True)

plot_four_infectious_bar(location, country, state, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=True)

def plot_all():
    p_v = 0.1
    l_v = 0.8
    k_v = 0.5
    file_name = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name)
    M = np.loadtxt(file_path, delimiter=',')
    M = M.T
    ms = M[1:] / total_population
    print(ms.shape)
    print(M.shape)

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 6))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.90, wspace=0.1, hspace=0.1)
    # plt.suptitle(state_type + " infection ratio", fontsize=20)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    axes.set_ylabel("比例")
    axes.set_xlabel("时间")
    axes.plot(M[0, :100], ms[0, :100], linestyle="-", label="S  易感染者")
    axes.plot(M[0, :100], ms[1, :100], linestyle="--", label="   未接种潜伏者")
    axes.plot(M[0, :100], ms[2, :100], marker="*", linestyle="none", label="   已接种潜伏者")
    axes.plot(M[0, :100], ms[3, :100], marker="+", linestyle="none", label="   未接种无症状感染者")
    axes.plot(M[0, :60], ms[4, :60], marker="x", linestyle="none", label="   已接种无症状感染者")
    axes.plot(M[0, :60], ms[5, :60], marker=".", linestyle="none", label="   未接种有症状感染者")
    axes.plot(M[0, :60], ms[6, :60], marker="1", linestyle="none", label="   已接种有症状感染者")
    axes.plot(M[0, :100], ms[7, :100], linestyle="-.", label="Q  隔离者/恢复者")
    axes.legend()
    plt.show()


# plot_all()

def plot_all2():
    p_v = 0.5
    l_v = 0.8
    k_v = 0.5
    file_name = f"all_states_numbers_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', 'no_age', file_name)
    M = np.loadtxt(file_path, delimiter=',')
    M = M.T
    ms = M[1:] / total_population
    print(ms.shape)
    print(M.shape)

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 6))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.90, wspace=0.1, hspace=0.1)
    # plt.suptitle(state_type + " infection ratio", fontsize=20)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    axes.set_ylabel("比例")
    axes.set_xlabel("时间")
    # axes.plot(M[0, 50:200], ms[0, 50:200], linestyle="-", label="S  易感染者")
    # axes.plot(M[0, 50:200], ms[1, 50:200], linestyle="--", label="   未接种潜伏者")
    # axes.plot(M[0, 50:200], ms[2, 50:200], marker="*", linestyle="none", label="   已接种潜伏者")
    # axes.plot(M[0, 50:200], ms[3, 50:200], marker="+", linestyle="none", label="   未接种无症状感染者")
    # axes.plot(M[0, 50:200], ms[4, 50:200], marker="x", linestyle="none", label="   已接种无症状感染者")
    # axes.plot(M[0, 50:200], ms[5, 50:200], marker=".", linestyle="none", label="   未接种有症状感染者")
    # axes.plot(M[0, 50:200], ms[6, 50:200], marker="1", linestyle="none", label="   已接种有症状感染者")
    # axes.plot(M[0, 50:200], ms[7, 50:200], linestyle="-.", label="Q  隔离者/恢复者")

    axes.plot(M[0, 100:160], [0 for i in range(60)], linestyle="-", label="S  易感染者", alpha=0)
    axes.plot(M[0, 100:160], ms[1, 100:160], linestyle="--", label="   未接种潜伏者", linewidth=4)
    axes.plot(M[0, 100:160], ms[2, 100:160], marker="*", linestyle="none", label="   已接种潜伏者", markersize=10)
    axes.plot(M[0, 100:160], ms[3, 100:160], marker="+", linestyle="none", label="   未接种无症状感染者", markersize=10)
    axes.plot(M[0, 100:160], ms[4, 100:160], marker="x", linestyle="none", label="   已接种无症状感染者", markersize=10)
    axes.plot(M[0, 100:160], ms[5, 100:160], marker=".", linestyle="none", label="   未接种有症状感染者", markersize=10)
    axes.plot(M[0, 100:160], ms[6, 100:160], marker="1", linestyle="none", label="   已接种有症状感染者", markersize=10)
    axes.plot(M[0, 100:160], [0 for i in range(60)], linestyle="-.", label="Q  隔离者/恢复者", alpha=0)
    # axes.legend()
    plt.show()
# plot_all2()

def plot_p_I():
    p_v = 0.1
    l_v = 0.8
    k_v = 0.5
    file_name = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5))
    ms = M.T[1:, :].sum(axis=0)/total_population

    p_v = 0.5
    file_name1 = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path1 = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name1)
    M1 = np.loadtxt(file_path1, delimiter=',', usecols=(0, 4, 5))
    ms1 = M1.T[1:, :].sum(axis=0)/total_population

    p_v = 0.9
    file_name2 = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path2 = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name2)
    M2 = np.loadtxt(file_path2, delimiter=',', usecols=(0, 4, 5))
    ms2 = M2.T[1:, :].sum(axis=0)/total_population
    # print(M.shape)
    # print(M[:, 0].shape)
    # print(ms.shape)

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.90, wspace=0.1, hspace=0.1)
    # plt.suptitle(state_type + " infection ratio", fontsize=20)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    axes.set_ylabel("无症状感染者占比", fontsize=12)
    axes.set_xlabel("时间", fontsize=12)
    axes.plot(M[:100, 0], ms[:100], marker="*", linestyle="-.", label="p = 0.1")
    axes.plot(M1[:100, 0], ms1[:100], marker=">", linestyle="-.", label="p = 0.5")
    axes.plot(M2[:100, 0], ms2[:100], marker="x", linestyle="-.", label="p = 0.9")
    # axes.plot(M[0, :100], ms[1, :100], linestyle="--", label="   未接种潜伏者")
    # axes.plot(M[0, :100], ms[2, :100], marker="*", linestyle="none", label="   已接种潜伏者")
    # axes.plot(M[0, :100], ms[3, :100], marker="+", linestyle="none", label="   未接种无症状感染者")
    # axes.plot(M[0, :60], ms[4, :60], marker="x", linestyle="none", label="   已接种无症状感染者")
    # axes.plot(M[0, :60], ms[5, :60], marker=".", linestyle="none", label="   未接种有症状感染者")
    # axes.plot(M[0, :60], ms[6, :60], marker="1", linestyle="none", label="   已接种有症状感染者")
    # axes.plot(M[0, :100], ms[7, :100], linestyle="-.", label="Q  隔离者/恢复者")
    axes.legend()
    plt.show()
# plot_p_I()


def plot_k_Q():
    p_v = 0.1
    l_v = 0.8
    k_v = 0.1
    file_name = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'mixed_output_source', 'age', p + "_" + str(p_v), file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5))
    ms = M.T[1:].sum(axis=0)/total_population

    k_v = 0.3
    file_name1 = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path1 = os.path.join(output_source_dir, 'mixed_output_source', 'age', p + "_" + str(p_v), file_name1)
    M1 = np.loadtxt(file_path1, delimiter=',', usecols=(0, 4, 5))
    ms1 = M1.T[1:].sum(axis=0)/total_population

    k_v = 0.6
    file_name2 = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path2 = os.path.join(output_source_dir, 'mixed_output_source', 'age', p + "_" + str(p_v), file_name2)
    M2 = np.loadtxt(file_path2, delimiter=',', usecols=(0, 4, 5))
    ms2 = M2.T[1:].sum(axis=0)/total_population
    # print(M.shape)
    # print(M[:, 0].shape)
    # print(ms.shape)

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.90, wspace=0.1, hspace=0.1)
    # plt.suptitle(state_type + " infection ratio", fontsize=20)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    axes.set_ylabel("感染者占比", fontsize=12)
    axes.set_xlabel("时间", fontsize=12)
    axes.plot(M[:100, 0], ms[:100], marker="*", linestyle="-.", label="k = 0.1")
    axes.plot(M1[:100, 0], ms1[:100], marker=">", linestyle="-.", label="k = 0.3")
    axes.plot(M2[:100, 0], ms2[:100], marker="x", linestyle="-.", label="k = 0.6")
    # axes.plot(M[0, :100], ms[1, :100], linestyle="--", label="   未接种潜伏者")
    # axes.plot(M[0, :100], ms[2, :100], marker="*", linestyle="none", label="   已接种潜伏者")
    # axes.plot(M[0, :100], ms[3, :100], marker="+", linestyle="none", label="   未接种无症状感染者")
    # axes.plot(M[0, :60], ms[4, :60], marker="x", linestyle="none", label="   已接种无症状感染者")
    # axes.plot(M[0, :60], ms[5, :60], marker=".", linestyle="none", label="   未接种有症状感染者")
    # axes.plot(M[0, :60], ms[6, :60], marker="1", linestyle="none", label="   已接种有症状感染者")
    # axes.plot(M[0, :100], ms[7, :100], linestyle="-.", label="Q  隔离者/恢复者")
    axes.legend()
    plt.show()

# plot_k_Q()

def plot4():
    p_v = 0.1
    l_v = 0.2
    k_v = 0.5
    file_name = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'mixed_output_source', 'age', p + "_" + str(p_v), file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5, 6, 7))
    ms = M.T[1:, :].sum(axis=0)/total_population

    l_v = 0.4
    file_name1 = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path1 = os.path.join(output_source_dir, 'mixed_output_source', 'age', p + "_" + str(p_v), file_name1)
    M1 = np.loadtxt(file_path1, delimiter=',', usecols=(0, 4, 5, 6, 7))
    ms1 = M1.T[1:, :].sum(axis=0)/total_population

    l_v = 0.6
    file_name2 = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path2 = os.path.join(output_source_dir, 'mixed_output_source', 'age', p + "_" + str(p_v), file_name2)
    M2 = np.loadtxt(file_path2, delimiter=',', usecols=(0, 4, 5, 6, 7))
    ms2 = M2.T[1:, :].sum(axis=0)/total_population
    # print(M.shape)
    # print(M[:, 0].shape)
    # print(ms.shape)

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.90, wspace=0.1, hspace=0.1)
    # plt.suptitle(state_type + " infection ratio", fontsize=20)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    axes.set_ylabel("感染者占比", fontsize=12)
    axes.set_xlabel("时间", fontsize=12)
    axes.plot(M[:100, 0], ms[:100], marker="*", linestyle="-.", label="l = 0.2")
    axes.plot(M1[:100, 0], ms1[:100], marker=">", linestyle="-.", label="l = 0.4")
    axes.plot(M2[:100, 0], ms2[:100], marker="x", linestyle="-.", label="l = 0.6")
    # axes.plot(M[0, :100], ms[1, :100], linestyle="--", label="   未接种潜伏者")
    # axes.plot(M[0, :100], ms[2, :100], marker="*", linestyle="none", label="   已接种潜伏者")
    # axes.plot(M[0, :100], ms[3, :100], marker="+", linestyle="none", label="   未接种无症状感染者")
    # axes.plot(M[0, :60], ms[4, :60], marker="x", linestyle="none", label="   已接种无症状感染者")
    # axes.plot(M[0, :60], ms[5, :60], marker=".", linestyle="none", label="   未接种有症状感染者")
    # axes.plot(M[0, :60], ms[6, :60], marker="1", linestyle="none", label="   已接种有症状感染者")
    # axes.plot(M[0, :100], ms[7, :100], linestyle="-.", label="Q  隔离者/恢复者")
    axes.legend()
    plt.show()

# plot4()

def plot_l_E():
    p_v = 0.1
    l_v = 0.2
    k_v = 0.5
    file_name = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'mixed_output_source', 'age', p + "_" + str(p_v), file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 8))
    ms = M.T[1:, :].sum(axis=0)/total_population

    l_v = 0.5
    file_name1 = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path1 = os.path.join(output_source_dir, 'mixed_output_source', 'age', p + "_" + str(p_v), file_name1)
    M1 = np.loadtxt(file_path1, delimiter=',', usecols=(0, 8))
    ms1 = M1.T[1:, :].sum(axis=0)/total_population

    l_v = 0.8
    file_name2 = f"{country}_{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path2 = os.path.join(output_source_dir, 'mixed_output_source', 'age', p + "_" + str(p_v), file_name2)
    M2 = np.loadtxt(file_path2, delimiter=',', usecols=(0, 8))
    ms2 = M2.T[1:, :].sum(axis=0)/total_population
    # print(M.shape)
    # print(M[:, 0].shape)
    # print(ms.shape)

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.90, wspace=0.1, hspace=0.1)
    # plt.suptitle(state_type + " infection ratio", fontsize=20)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    axes.set_ylabel("潜伏者占比", fontsize=12)
    axes.set_xlabel("时间", fontsize=12)
    axes.plot(M[:100, 0], ms[:100], marker="*", linestyle="-.", label="l = 0.2")
    axes.plot(M1[:100, 0], ms1[:100], marker=">", linestyle="-.", label="l = 0.5")
    axes.plot(M2[:100, 0], ms2[:100], marker="x", linestyle="-.", label="l = 0.8")
    # axes.plot(M[0, :100], ms[1, :100], linestyle="--", label="   未接种潜伏者")
    # axes.plot(M[0, :100], ms[2, :100], marker="*", linestyle="none", label="   已接种潜伏者")
    # axes.plot(M[0, :100], ms[3, :100], marker="+", linestyle="none", label="   未接种无症状感染者")
    # axes.plot(M[0, :60], ms[4, :60], marker="x", linestyle="none", label="   已接种无症状感染者")
    # axes.plot(M[0, :60], ms[5, :60], marker=".", linestyle="none", label="   未接种有症状感染者")
    # axes.plot(M[0, :60], ms[6, :60], marker="1", linestyle="none", label="   已接种有症状感染者")
    # axes.plot(M[0, :100], ms[7, :100], linestyle="-.", label="Q  隔离者/恢复者")
    axes.legend()
    plt.show()

# plot_l_E()