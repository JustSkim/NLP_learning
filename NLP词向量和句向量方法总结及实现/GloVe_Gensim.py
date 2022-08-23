from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

"""
 Gensim加载GloVe训练的词向量
 源码地址：https://asialee.blog.csdn.net/article/details/100124565
 需要在linux环境下运行demo.sh生成vectors.txt，里面就对应每个词的向量表示。
 这里为了方便我们直接复制粘贴博客里边的
"""

"""
# 输入文件
glove_file = 'vectors.txt'
# 输出文件
w2v_file = 'w2v.txt'
# 开始转换
glove2word2vec(glove_file, w2v_file)
# 加载转化后的文件
model = KeyedVectors.load_word2vec_format(w2v_file)  # 该加载的文件格式需要转换为utf-8
print(model['时间'])  # 这个就和Word2Vec训练的模型使用方法一样了
"""

glove2word2vec("./vectors.txt", "./vectors_wd2.txt")

new_model = KeyedVectors.load_word2vec_format('vectors_wd2.txt', binary=False)
print(new_model)

vectors = new_model.wv.vectors
print(vectors)

words = new_model.wv.index2word
print(words)

vec = new_model['is']
print(vec)