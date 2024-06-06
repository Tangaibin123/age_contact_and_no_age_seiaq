import matplotlib.pyplot as plt
import os
plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体
import matplotlib as mpl
# mpl.use('pdf')
# path to the directory where this script lives
thisdir = os.path.abspath('')

# path to the scripts directory of the repository
scriptsdir = os.path.split(thisdir)[0]

# path to the main directory of the repository
maindir = os.path.split(scriptsdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(maindir, 'results')


stockpile = [0.5, 0.4, 0.3, 0.2, 0.1, 0]
avg = [0.895685296, 0.917582569, 0.937012, 0.954370447, 0.969973174,0.9840739343442247]
optimal = [0.8391851591886917, 0.8614009603209863, 0.891439381952178, 0.9188546564824531, 0.9474730479546403,0.9840739343442247]

plt.plot(stockpile, avg, label='平均分配')
plt.plot(stockpile, optimal, label='最优分配',color='coral')
plt.xlabel('疫苗可以覆盖的人群比例')
plt.ylabel('总感染率')
# plt.title('Stockpile vs. Value')
plt.legend(loc='upper right',fontsize=18)
plt.show()
fig_name=f'Total infected rate.pdf'
fig_path=os.path.join(resultsdir, 'new_paper', fig_name)
# plt.savefig(fig_path, format='pdf')