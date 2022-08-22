from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

"""
基于sklearn的卡方检验实现
sklearn模块下的feature_selection这一个子模块，顾名思义，就是“特征选择”的意思，该模块负责特征选择而不是特征提取！
在该模块中给出了多种特征选择的算法，其中：
SelectKBest，顾名思义，据某中检验方法（比如chi2），选择k个最高分数的特征，属于单变量特征选择的一种，
更详细的可见官网介绍：https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

chi2方法的官方介绍：https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
用于计算每个非负特征和类之间的卡方统计数据。
此分数可用于从 X 中选择具有最高值的 n_features 个特征，用于测试卡方统计量，必须仅包含非负特征，例如布尔值或频率（例如，文档分类中的术语计数）。
"""


# 训练集和训练标签
x_train = [[1, 2, 3, 4, 5],
           [5, 4, 3, 2, 1],
           [3, 3, 3, 3, 3],
           [1, 1, 1, 1, 1]]
y_train = [0, 1, 0, 1]

# 测试集和测试标签
x_test = [[2, 2, 2, 2, 2], [2, 1, 1, 2, 1]]
y_test = [1, 1]

# 卡方检验选择特征
chi2_model = SelectKBest(chi2, k=3)  # 选择k个最佳特征
# 该函数选择训练集里的k个特征，并将训练集转化所选特征
x_train_chi2 = chi2_model.fit_transform(x_train, y_train)
# 将测试集转化为所选特征
x_test_chi2 = chi2_model.transform(x_test)

print('各个特征的得分：', chi2_model.scores_)
print('各个特征的p值：', chi2_model.pvalues_)  # p值越小，置信度越高，得分越高
print('所选特征的索引：', chi2_model.get_support(True))
print('特征提取转换后的训练集和测试集...')
print('x_train_chi2:', x_train_chi2)
print('x_test_chi2:', x_test_chi2)