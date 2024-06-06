"""
    1、总共有多少个节点
    2、每个节点包含的属性：
        1 当前节点的状态（有八个状态：S、En、Ev、An、Av、In、Iv、Q）
        2 节点的邻居节点（设置一个平均度，每个节点只需保存邻居结点的id即可，应该是一个数组）
        3 当前节点是属于哪个年龄段（年龄段分为18个/85个）
    3、需要更新每天每个节点的状态

    结构设计：
        对于整个人群，应该是一个大数组，每个元素代表一个节点。
        每个节点应该也是一个数组或字典
"""
# 整个人群结构
population = {
    1: {"state": "S", "age": 12, 'neighbor': [1, 2, 3]},
    2: {"state": "En", "age": 14, 'neighbor': [16, 2, 3]},
    3: {"state": "S", "age": 15, 'neighbor': [1, 2, 3]},
    4: {"state": "S", "age": 3, 'neighbor': [1, 32, 23]}
}

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

"""
BA无标度、小世界网络：
https://blog.csdn.net/dog250/article/details/104105711

ER随机图、规则图、BA、WS小世界
https://blog.csdn.net/un357951/article/details/103514682


https://www.cnblogs.com/forstudy/archive/2012/03/20/2407954.html
"""

"""
RG规则图
"""
# random_graphs.random_regular_graph(d, n)方法
# 可以生成一个含有n个节点，每个节点有d个邻居节点的规则图。
# regular graphy
# generate a regular graph which has 20 nodes & each node has 3 neghbour nodes.
RG = nx.random_graphs.random_regular_graph(4, 1000)
RGAdj = np.array(nx.adjacency_matrix(RG).todense())
print(RGAdj)
# the spectral layout
pos = nx.spectral_layout(RG)
# draw the regular graphy
# 画节点
nx.draw_networkx_nodes(RG, pos=pos, node_color='b', node_size=20, alpha=0.6)
# 画边
nx.draw_networkx_edges(RG, pos=pos, width=0.3, alpha=0.3)
# nx.draw(RG, pos, with_labels=False, node_size=30)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.title('RG规则图')
plt.show()

"""
ER随机图
"""
# 以概率p连接N个节点中的每一对节点。
# 用random_graphs.erdos_renyi_graph(n,p)方法
# 生成一个含有n个节点、以概率p连接
# erdos renyi graph
# generate a graph which has n=20 nodes, probablity p = 0.2.
ER = nx.random_graphs.erdos_renyi_graph(1000, 0.001)
ERAdj = np.array(nx.adjacency_matrix(ER).todense())
# the shell layout
pos = nx.kamada_kawai_layout(ER)
# 画节点
nx.draw_networkx_nodes(ER, pos=pos, node_color='b', node_size=20, alpha=0.6)
# 画边
nx.draw_networkx_edges(ER, pos=pos, width=0.05, alpha=0.3)
# nx.draw(ER, pos, with_labels=False, node_size=30)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.title('ER随机图')
plt.show()

"""
BA无标度网络
"""
# random_graphs.barabasi_albert_graph(n, m)方法
# 生成一个含有n个节点、每次加入m条边的BA无标度网络。
BA = nx.random_graphs.barabasi_albert_graph(1000, 2)
BAAdj = np.array(nx.adjacency_matrix(BA).todense())
pos = nx.spring_layout(BA)
# 画节点
nx.draw_networkx_nodes(BA, pos=pos, node_color='b', node_size=20, alpha=0.6)
# 画边
nx.draw_networkx_edges(BA, pos=pos, width=0.3, alpha=0.3)
# nx.draw(ba, ps, with_labels=False, node_size=50)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.title('BA无标度网络')
plt.show()

"""
小世界网络
"""
# 用random_graphs.watts_strogatz_graph(n, k, p)方法
# 生成一个含有n个节点、每个节点有k个邻居、以概率p随机化断边重连的WS小世界网络。
WS = nx.random_graphs.watts_strogatz_graph(1000, 4, 0.4)
WSAdj = np.array(nx.adjacency_matrix(WS).todense())
pos = nx.spring_layout(WS)
# 画节点
nx.draw_networkx_nodes(WS, pos=pos, node_color='b', node_size=20, alpha=0.6)
# 画边
nx.draw_networkx_edges(WS, pos=pos, width=0.3, alpha=0.3)
# nx.draw(WS, pos=pos, with_labels=False, node_size=50)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
# plt.rcParams['axes.unicode_minus'] = False
plt.title('WS 小世界网络')
plt.show()

"""
计算网络的度
"""


def calculateDegreeDistribution(Network, NetMap):
    avedegree = 0.0  # 计算度分布
    identify = 0.0  # 若算法正确，度概率分布总和应为0
    p_degree = np.zeros((len(NetMap)), dtype=float)
    # statistic下标为度值
    # (1)先计数该度值的量
    # (2)再除以总节点数N得到比例
    degree = np.zeros(len(NetMap), dtype=int)
    # degree用于存放各个节点的度之和
    for i in range(len(NetMap)):
        for j in range(len(NetMap)):
            degree[i] = degree[i] + NetMap[i][j]
            # 汇总各个节点的度之和
    for i in range(len(NetMap)):
        avedegree += degree[i]
        # 汇总每个节点的度之和
    print(Network + '网络模型平均度为\t' + str(avedegree / len(NetMap)))
    # 计算平均度
    for i in range(len(NetMap)):
        p_degree[degree[i]] = p_degree[degree[i]] + 1
        # 先计数该度值的量
    for i in range(len(NetMap)):  # 再除以总节点数N得到比例
        p_degree[i] = p_degree[i] / len(NetMap)
        identify = identify + p_degree[i]
        # 将所有比例相加，应为1
    identify = int(identify)

    plt.figure(figsize=(10, 4), dpi=120)
    plt.title(Network)
    # 绘制度分布图
    plt.xlabel('$Degree$', fontsize=21)
    # 横坐标标注——Degrees
    plt.ylabel('$P$', fontsize=26)
    # 纵坐标标注——P
    plt.plot(list(range(len(NetMap))), list(p_degree), '-*', markersize=15, label='度', color="#ff9c00")
    # 自变量为list(range(N)),因变量为list(p_degree)
    # 图形标注选用星星*与线条-，大小为15，标注图例为度，颜色是水果橘

    plt.xlim([0, 12])  # 给x轴设限制值，这里是我为了美观加入的，也可以不写
    plt.ylim([-0.05, 0.5])  # 给y轴设限制值，也是为了美观加入的，不在意可以跳过
    plt.xticks(fontsize=20)  # 设置x轴的字体大小为21
    plt.yticks(fontsize=20)  # 设置y轴的字体大小为21
    plt.legend(fontsize=21, numpoints=1, fancybox=True, prop={'family': 'SimHei', 'size': 15}, ncol=1)
    # plt.savefig('./SEIR_ER_files/度分布图.pdf')
    plt.show()  # 展示图片
    print('算法正常运行则概率之和应为1 当前概率之和=\t' + str(identify))
    # 用于测试算法是否正确


calculateDegreeDistribution('RG', RGAdj)
calculateDegreeDistribution('ER', ERAdj)
calculateDegreeDistribution('BA', BAAdj)
calculateDegreeDistribution('WS', WSAdj)

