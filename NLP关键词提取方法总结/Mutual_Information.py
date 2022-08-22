from sklearn import metrics
import numpy as np

"""
在概率论和信息论中，两个随机变量的互信息或转移信息（transinformation）是变量间相互依赖性的量度。
互信息算法实现
"""

# 训练集和训练标签
x_train = [[1, 2, 3, 4, 5],
           [5, 4, 3, 2, 1],
           [3, 3, 3, 3, 3],
           [1, 1, 1, 1, 1]]
y_train = [0, 1, 0, 1]
# 测试集和测试标签
x_test = [[2, 2, 2, 2, 2], [2, 1, 1, 2, 1]]

x_train = np.array(x_train)  # 转为array

# 存储每个特征与标签相关性得分
features_score_list = []
for i in range(len(x_train[0])):
    # 计算每个特征与标签的互信息
    feature_info = metrics.mutual_info_score(y_train, x_train[:, i])
    features_score_list.append(feature_info)

print(features_score_list)