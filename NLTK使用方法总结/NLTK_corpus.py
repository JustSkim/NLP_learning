from nltk.corpus import brown
"""
corpus有文集，全集；语料库，人体，本金的意思
nltk.corpus模块负责获取和处理语料库，会提供语料库和词典的标准化接口
详见 https://www.nltk.org/api/nltk.corpus.html#module-nltk.corpus
nltk.corpus下的brown，是一个百万词级的英语语料库，按文体进行分类
"""
print(brown.categories())  # 输出brown语料库的类别
print(len(brown.sents()))  # 输出brown语料库的句子数量 sentence为句子的意思
print(len(brown.words()))  # 输出brown语料库的词数量

'''
结果为：
['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 
'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 
'science_fiction']
57340
1161192
'''