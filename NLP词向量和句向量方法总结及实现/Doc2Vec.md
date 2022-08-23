# Doc2Vec

背景：Word2Vec表示的词向量不仅考虑了词之间的语义信息，还压缩了维度。但是，有时候当我们需要得到Sentence/Document的向量表示，虽然可以直接将Sentence/Document中所有词的向量取均值作为Sentence/Document的向量表示，但是这样会忽略单词之间的排列顺序对句子或文本信息的影响。

论文：https://cs.stanford.edu/~quocle/paragraph_vector.pdf

更为详细的介绍可见：https://blog.csdn.net/Walker_Hao/article/details/78995591

Doc2vec是在Word2vec的基础上做出的改进，它**不仅考虑了词和词之间的语义，也考虑了词序**。
Doc2Vec有两种模型，分别为：

- 句向量的分布记忆模型（PV-DM: Distributed Memory Model of Paragraph Vectors）
- 句向量的分布词袋（PV-DBOW: Distributed Bag of Words version of Paragraph Vector）
  

## DM模型

DM模型在给定上下文和文档向量的情况下预测单词的概率。即在训练时，首先将每个文档ID和语料库中的所有词初始化一个K维的向量，然后将文档向量和上下文词的向量输入模型，隐层将这些向量累加（或取均值、或直接拼接起来）得到中间向量，作为输出层softmax的输入。在一个文档的训练过程中，文档ID保持不变，共享着同一个文档向量，相当于在预测单词的概率时，都利用了这个句子的语义。
![DM模型](https://img-blog.csdnimg.cn/20190828221619896.png)



## DBOW模型

在给定文档向量的情况下预测文档中一组随机抽样的单词的概率。

![DBOW模型](https://img-blog.csdnimg.cn/20190828221740304.png)

Doc2vec的DM模型跟Word2vec的CBOW很像，DBOW模型跟Word2vec的Skip-gram很像。Doc2Vec为不同长度的段落训练出同一长度的向量；不同段落的词向量不共享；训练集训练出来的词向量意思一致，可以共享。