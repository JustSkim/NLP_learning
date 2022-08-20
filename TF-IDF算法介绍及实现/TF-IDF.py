# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import operator

"""
模块说明：
collection模块实现了特定目标的容器，以提供Python标准内建容器 dict , list , set , 和 tuple 的替代选择，详见https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict
operator 模块提供了一套与Python的内置运算符对应的高效率函数。例如，operator.add(x, y) 与表达式 x+y 相同。 许多函数名与特殊方法名相同，只是没有双下划线。为了向后兼容性，也保留了许多包含双下划线的函数。为了表述清楚，建议使用没有双下划线的函数。
详见https://docs.python.org/zh-cn/3.6/library/operator.html#module-operator

函数说明:创建数据样本
Returns:
    dataset - 实验样本切分的词条
    classVec - 类别标签向量
"""

"""
TF-IDF（term frequency–inverse document frequency，词频-逆向文件频率）是一种用于信息检索（information retrieval）与文本挖掘（text mining）的常用加权技术。
TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。
主要思想是：如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现（也即逆向文件频率IDF大），则认为此词或者短语具有很好的类别区分能力，适合用来分类。

TF是词频(Term Frequency)，表示词条（关键字）在文本中出现的频率，通常会被归一化，以防止偏向文本长的文件
IDF是逆向文件频率(Inverse Document Frequency)。某一特定词语的IDF，可以由总文件数目除以包含该词语的文件的数目（也就是分母越小，IDF越高），再将得到的商取对数得到。
注意，分母是表示包含该词语的文件数目，但若该词语不在语料库中，会导致分母为零，运算报错，因此一般情况下，默认分母+1，以避免运算出错。

比如，单词"to"这类介词可能在所有文章中出现的频率TF都高，但其IDF低，因为包含词语"to"的文件数目太多了。但是某些领域名词可能IDF就会很高
 
TF-IDF算法实际要求的是TF-IDF值，也就是tf✖idf，二者的乘积。
这是因为综合起来，某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。
 
 
TF-IDF算法优点：非常容易理解，并且很容易实现；
        缺点：其简单结构并没有考虑词语的语义信息，无法处理一词多义与一义多词的情况。
"""

def loadDataSet():
    dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条，43个
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表好，0代表不好
    return dataset, classVec

"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     dict_feature_select:特征选择词字典
"""


def feature_select(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1

    # 计算每个词的TF值
    word_tf = {}  # 存储每个词的tf值
    sum_values = sum(doc_frequency.values())    #总共加起来43个
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum_values

    # 计算每个词的IDF值
    doc_num = len(list_words)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    '''
    defaultdict(function_factory)构建的是一个类似dictionary的对象，其中keys的值，自行确定赋值，但是values的类型，是function_factory的类实例，而且具有默认值。
    '''

    """
    这一段循环可能有点绕，之前我们使用collection.defaultdict类来存储二维矩阵中每个单词，对于二维矩阵每一行，
    将存储的单词拿出来验证是否在这一行里，是则放进去我们新建的collection.defaultdict类对象word_doc
    我们代码和所用数据，导致刚生成的word_doc和doc_frequency完全一致
    """
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / (word_doc[i] + 1))#这一步计算出每个词的IDF值

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]#tf*idf，二者相乘

    # 对字典按值由大到小排序 某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select


if __name__ == '__main__':
    data_list, label_list = loadDataSet()  # 加载数据
    features = feature_select(data_list)  # 所有词的TF-IDF值
    print(features)
    print(len(features))


"""
博客链接：https://blog.csdn.net/asialee_bird/article/details/81486700
TF-IDF算法的不足：
TF-IDF 采用文本逆频率 IDF 对 TF 值加权取权值大的作为关键词，但 IDF 的简单结构并不能有效地反映单词的重要程度和特征词的分布情况，使其无法很好地完成对权值调整的功能，所以 TF-IDF 算法的精度并不是很高，尤其是当文本集已经分类的情况下。
在本质上 IDF 是一种试图抑制噪音的加权，并且单纯地认为文本频率小的单词就越重要，文本频率大的单词就越无用。这对于大部分文本信息，并不是完全正确的。IDF 的简单结构并不能使提取的关键词， 十分有效地反映单词的重要程度和特征词的分布情 况，使其无法很好地完成对权值调整的功能。尤其是在同类语料库中，这一方法有很大弊端，往往一些同类文本的关键词被盖。
TF-IDF算法实现简单快速，但是仍有许多不足之处：
（1）没有考虑特征词的位置因素对文本的区分度，词条出现在文档的不同位置时，对区分度的贡献大小是不一样的。
（2）按照传统TF-IDF，往往一些生僻词的IDF(反文档频率)会比较高、因此这些生僻词常会被误认为是文档关键词。
（3）传统TF-IDF中的IDF部分只考虑了特征词与它出现的文本数之间的关系，而忽略了特征项在一个类别中不同的类别间的分布情况。
（4）对于文档中出现次数较少的重要人名、地名信息提取效果不佳。

针对以上问题，这篇13年发表的论文中讲到了改进后的TF-IWF算法，链接为：https://image.hanspub.org/pdf/CSA20130100000_81882762.pdf
"""