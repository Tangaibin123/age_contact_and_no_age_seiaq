import matplotlib.pyplot as plt
import numpy as np
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
stockpile = [0.6,0.5, 0.4, 0.3, 0.2, 0.1]
avg = [0.8705819347043213,0.895685296, 0.917582569, 0.937012, 0.954370447, 0.969973174]
optimal = [0.8286619666319929,0.8391851591886917, 0.8614009603209863, 0.891439381952178, 0.9188546564824531, 0.9474730479546403]

x = np.arange(len(stockpile))
width = 0.35

stockpile = stockpile[::-1]
avg = avg[::-1]
optimal = optimal[::-1]

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, avg, width, label='平均分配',color='coral')
rects2 = ax.bar(x + width/2, optimal, width, label='最优分配')
ax.set_ylim([0.8, 1.0])
ax.set_xlabel('疫苗覆盖率')
ax.set_ylabel('总感染率')
ax.set_xticks(x)
ax.set_xticklabels(stockpile)
ax.legend()

fig.tight_layout()

fig_name=f'Total infected rate.png'
fig_path=os.path.join(resultsdir, 'Master_graduation', fig_name)
plt.savefig(fig_path,dpi=300)
plt.show()