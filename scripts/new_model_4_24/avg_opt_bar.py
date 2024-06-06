import matplotlib.pyplot as plt
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

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题

stockpile = [0.5, 0.4, 0.3, 0.2, 0.1, 0]
avg = [0.895685296, 0.917582569, 0.937012, 0.954370447, 0.969973174, 0.9840739343442247]
optimal = [0.8391851591886917, 0.8614009603209863, 0.891439381952178, 0.9188546564824531, 0.9474730479546403, 0.9840739343442247]

plt.figure(figsize=(10, 7))

# 绘制平均值柱状图
plt.bar(stockpile, avg, width=0.05, color='coral', align='center', label='平均分配')

# 绘制最优值柱状图
plt.bar(stockpile, optimal, width=0.05, align='edge', label='权重算法分配')

plt.ylim(0.8, 1.0)
# 添加图例
plt.legend()

# 添加标题和标签
# plt.title('Comparison of Average and Optimal Allocation')
plt.xlabel('疫苗能够覆盖的人群比例',fontsize=14)
plt.ylabel('总感染率',fontsize=14)

# 显示图形
plt.show()

fig_name=f'Total infected rate.png'
fig_path=os.path.join(resultsdir, 'Master_graduation', fig_name)
plt.savefig(fig_path,dpi=300)