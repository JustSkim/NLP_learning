from gensim.models.word2vec import Word2Vec
import pandas as pd

"""
word2vec 相关的API都在包gensim.models.word2vec中，主要参数如下：　　　
sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。后面我们会有从文件读出的例子。
size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。
window：即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为cc，window越大，则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间。
sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。
hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
negative:即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。
cbow_mean: 仅用于CBOW在做投影的时候，为0，则算法中的xwxw为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示xwxw,默认值也是1,不推荐修改默认值。
min_count:需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。
iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
alpha: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为ηη，默认是0.025。
min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。
"""


# 构建word2vec模型，词向量的训练与生成
def get_dataset_vec(dataset):
    n_dim = 300
    w2v_model = Word2Vec(dataset, sg=1, size=n_dim, min_count=10, hs=0)  # 初始化模型并训练
    # 在测试集上训练
    # w2v_model.train(x_test,total_examples=w2v_model.corpus_count,epochs=w2v_model.iter) #追加训练模型
    # 将imdb_w2v模型保存，训练集向量，测试集向量保存到文件
    # print(w2v_model['会议'])
    w2v_model.save('data/w2v/w2v_model_300.pkl')  # 保存训练结果


if __name__ == '__main__':
    # 数据集获取
    #train_data = pd.read_csv('data/clean_data_train.csv', sep=',', names=['contents', 'labels']).astype(str)
    train_data = pd.read_csv('clean_data_training.csv', sep=',', names=['contents', 'labels']).astype(str)
    test_data = pd.read_csv('data/clean_data_test.csv', sep=',', names=['contents', 'labels']).astype(str)
    cw = lambda x: str(x).split()
    train_data['words'] = train_data['contents'].apply(cw)
    test_data['words'] = train_data['contents'].apply(cw)
    dataset = pd.concat([train_data, test_data])

    # word2vec词向量训练
    get_dataset_vec(dataset['words'])

    # 词向量模型加载
    # w2v_model = Word2Vec.load('data/w2v/w2v_model_300.pkl')