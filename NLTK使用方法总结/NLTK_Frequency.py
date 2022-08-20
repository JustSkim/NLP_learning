import nltk
"""
NLTK实现词频统计
NLTK 中的FreqDist( ) 类主要记录了每个词出现的次数，根据统计数据生成表格或绘图。其结构简单，用一个有序词典进行实现。
"""
tokens = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please',
          'maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid',
          'my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him']
# 统计词频
freq = nltk.FreqDist(tokens)
"""
查看源码可发现，FreqDist继承自Counter类，而Count类又继承自dict，所以我们可以像操作字典一样操作FreqDist对象。
关于继承可见：https://justskim.github.io/2021/10/31/python3%E4%B8%AD%E7%9A%84%E4%B8%80%E4%BA%9B%E5%B8%B8%E8%A7%81%E8%AF%AD%E6%B3%95%E7%B3%96/#python-%E7%B1%BB%E7%9A%84%E7%BB%A7%E6%89%BF
在本例中，FreqDist中的键为单词，值为单词的出现总次数
"""

# 输出词和相应的频率
for key, val in freq.items():
    print(str(key) + ':' + str(val))
    #python中，items() 方法把字典中每对 key 和 value 组成一个元组,并把这些元组放在列表中返回

# 可以把最常用的5个单词拿出来
standard_freq = freq.most_common(5)#most common函数也是Counter类中的函数，FreqDist也继承了
print(standard_freq)

# 绘图函数为这些词频绘制一个图形
freq.plot(20, cumulative=False)