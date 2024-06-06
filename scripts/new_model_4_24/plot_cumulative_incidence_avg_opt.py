import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sbn
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

font_path = 'C:\Windows\Fonts\STXIHEI.TTF'
# font = FontProperties(fname=r"simsun.ttc", size=18)
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()


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

def get_label(num_agebrackets=18):
    if num_agebrackets == 85:
        age_brackets = [str(i) + '-' + str((i + 1)) for i in range(0, 84)] + ['85+']
    elif num_agebrackets == 18:
        age_brackets = [str(5 * i) + '-' + str(5 * (i + 1) - 1) for i in range(17)] + ['85+']
    return age_brackets


def read_cumulative_incidence_age(location,p,p_v,l,l_v,k,k_v,num_agebrackets):
    file_name = f"{country}_{location}_cumulative_incidence_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', "bracket",file_name)
    print(file_path+'was read')

    M = np.loadtxt(file_path, delimiter=',')
    return M

def read_cumulative_incidence_age_opt(location,p,p_v,l,l_v,k,k_v,num_agebrackets):
    file_name = f"{country}_{location}_allocation_cumulative_incidence_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', "bracket", file_name)

    M = np.loadtxt(file_path, delimiter=',')
    return M
def plot_cumulative_incidence_age(location, p, l, l_v, k, k_v,num_agebrackets=18, isSave=True):
    data = read_cumulative_incidence_age(location, p, p_v,l,l_v,k,k_v,num_agebrackets).T
    data1 = read_cumulative_incidence_age_opt(location, p, 0.5,l,l_v,k,k_v,num_agebrackets).T
    # data2 = read_cumulative_incidence_age(location, p, 0.9,l,l_v,k,k_v,num_agebrackets).T
    age_brackets = get_label()
    age_brackets = np.array(age_brackets).reshape(3, 6)
    # print(age_brackets)
    left = 0.08
    right = 0.97
    top = 0.9
    bottom = 0.15
    fig, axes = plt.subplots(3, 6, figsize=(9, 4))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.3, hspace=0.8)
    for i in range(3):
        axes[1][0].set_ylabel('感染率（%）', fontsize=14)
        for j in range(6):
            axes[2][j].set_xlabel('时间（天）', fontsize=14)
            axes[i][j].set_title(f'年龄组:{age_brackets[i][j]}', fontsize=10)
            axes[i][j].xaxis.set_major_locator(plt.MaxNLocator(3))
            axes[i][j].yaxis.set_major_locator(plt.MaxNLocator(2))
            axes[i][j].set_ylim((0, 100))
            axes[i][j].tick_params(labelsize=10)
            axes[i][j].plot(data[0][0:100], data[(i+1)*(j+1)][0:100],label='平均分配',color='coral')
            axes[i][j].plot(data1[0][0:100], data1[(i+1)*(j+1)][0:100],label='最优分配')
            axes[i][j].axhline(50, linewidth=0.5, linestyle='--', color='r')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',ncol=len(labels))
    plt.subplots_adjust(top=0.85)
    plt.show()

    if isSave:
        # file_name = f'{location}_cumulative_incidence_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.pdf'
        # fig_path = os.path.join(resultsdir, 'new_paper', file_name)
        # fig.savefig(fig_path, format='pdf')
        # print('fig is save')
        #
        # file_name = f'{location}_cumulative_incidence_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.eps'
        # fig_path = os.path.join(resultsdir, 'new_paper', file_name)
        # fig.savefig(fig_path, format='eps')

        file_name = f'opt_vs_avg_cumulative_incidence_all_age_.png'
        fig_path = os.path.join(resultsdir, 'Master_graduation', file_name)
        fig.savefig(fig_path, format='png',dpi=300)
        print('fig is save')


location = 'Shanghai'
country = 'China'
level = 'subnational'
num_agebrackets = 18
p = 'p'
w = 'w'
# beta = 'beta'
q = 'q'
l = 'l'
# param = 'sigma_inverse'
k = 'k'
q_v = 1
# param = 'gamma_inverse'
p_v = 0.5  # 0.1 0.5 0.9  # percent of unvaccinated
w_v = 0.5
beta = 0.5  #
l_v = 0.5 #接种后有症状的比例
sigma_inverse = 5.0  # mean incubation period
k_v = 0.5
gamma_inverse = 1.6  # mean quarantined period

ages = get_ages(location, country, level, num_agebrackets)
total_population = sum(ages.values())

plot_cumulative_incidence_age(location, p, l, l_v, k, k_v,num_agebrackets, isSave=True)