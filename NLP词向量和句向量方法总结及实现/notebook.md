# Word2vec

Word2Vec是Google在2013年开源的一款词向量计算工具，发表在[论文《Distributed Representations of Words and Phrases
and their Compositionality》](https://arxiv.org/pdf/1310.4546.pdf)上，[源码](https://github.com/tmikolov/word2vec)托管在github平台上。
特点：将所有的词向量化(该工具的命名就是“word to vector”，vector是向量的意思)，这样词与词之间就可以定量的去度量他们之间的关系，挖掘词之间的联系。
虽然源码是开源的，但是谷歌的代码库国内无法访问，因此讲解word2vec原理基本以Github上的word2vec代码为准。

## 词向量基础
Word2vec也用词向量来表示词，以下内容引自17年的博客：https://www.cnblogs.com/pinard/p/7160330.html
**One hot representation**
最早的词向量，很冗长的，使用词向量维度大小为整个词汇表的大小，对于每个具体的词汇表中的词，将对应的位置置为1。
比如有下面的5个词["King","Queen","Man","Woman","Child"]组成的词汇表，词"Queen"的序号为2， 那么它对应的词向量就是(0,1,0,0,0)。
同样地，词"Woman"的词向量就是(0,0,0,1,0)。这种词向量的编码方式一般叫做1-of-N representation或者one hot representation。

优点：表示词向量非常简单

缺点：词汇表一般都非常大，比如达到百万级别，这样每个词都用百万维的向量来表示会是内存的灾难——这样的向量其实除了一个位置是1，其余的位置全部都是0，十分浪费内存空间。因此需要把词向量的维度变小

**Distributed representation**
思路：通过训练，将**每个词都映射到一个较短的词向量上**(较短的词向量维度一般需要在训练时自己来指定)来。**所有的这些词向量就构成了向量空间**，进而可以用普通的统计学的方法来研究词与词之间的关系。

比如下图我们对词汇表里的词用"Royalty","Masculinity", "Femininity"和"Age"4个维度来表示，那么King这个词对应的词向量**可能**是(0.99,0.99,0.05,0.7)——实际情况中，我们并不能对词向量的每个维度做一个很好的解释

有了用Distributed Representation表示的较短的词向量，就可以较容易的分析词之间的关系了，比如将词的维度降维到2维，有一个有趣的研究表明，用下图的词向量表示我们的词时，我们可以发现：
$$\vec{King}-\vec{Man}+\vec{Woman}=\vec{Queen}$$
![](https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713151608181-1336632086.png)

结论：只要得到了词汇表里所有词对应的词向量，那么就可以做很多有趣的事情了

下一个问题：如何**训练得到合适的词向量**呢？一个很**常见的方法**是**使用神经网络语言模型（Neural Network Language Models，NNLM）**。



## 神经网络语言模型（Neural Network Language Models，NNLM）

根据[这篇神经网络语言模型专栏](https://zhuanlan.zhihu.com/p/109564205)，简要介绍一下：

> 语言模型是大部分NLP任务的基础，如在机器翻译（Machine Translation，MT）任务中，语言模型被用来评估翻译系统输出一个特定序列的概率，以提升其在目标语言中的流畅度。
>
> 语言模型（Language Model，LM）是自然语言处理（Natural Language Processing，NLP）系统的核心组件，能够提供词的向量表示和词序列的联合概率。神经网络语言模型（Neural Network Language Models，NNLM）克服了维度灾难，并且大大提升了传统语言模型的性能。



## CBOW与Skip-Gram用于神经网络语言模型

word2vec之前，已经有用**神经网络DNN来用训练词向量进而处理词与词之间的关系**，分为CBOW(Continuous Bag-of-Words)与Skip-Gram两种模型。采用的方法一般是一个**三层及以上的神经网络结构，分为输入层，隐藏层和输出层(softmax层)，定义数据的输入和输出非常重要**。

### CBOW(Continuous Bag-of-Words)模型

**CBOW**模型的训练**输入**是**某一个特征词的上下文相关的词对应的词向量**，而**输出就是这特定的一个词的词向量**。比如"an efficient method for learning high quality distributed vector"，我们的上下文大小取值为4，特定的这个词是"Learning"，也就是我们需要的输出词向量，上下文对应的词有8个，前后各4个，这8个词就是我们模型的输入。由于**CBOW使用词袋模型**，因此这8个词都是**平等**的——也就是**无需考虑他们和我们关注的词之间的距离大小，只要在我们上下文之内即可**。

这样在这个CBOW的例子里，我们的**输入是8个词向量，输出是所有词的softmax概率（训练的目标是期望训练样本特定词对应的softmax概率最大）**，对应的**CBOW神经网络模型输入层**有**8个神经元**，**输出层有词汇表大小个神经元**。**隐藏层的神经元个数可以自己指定**。通过**DNN的反向传播算法，求出DNN模型的参数**，同时**得到所有的词对应的词向量**。这样，当我们有新的需求，要求出某8个词对应的最可能的输出中心词时，我们可以通过一次DNN前向传播算法并通过softmax激活函数找到概率最大的词对应的神经元即可。



### Skip-Gram模型

Skip-Gram模型思路和CBOW相反——输入是特定的一个词的词向量，而输出是特定词对应的上下文词向量：

> 还是上面的例子，上下文大小取值为4， 特定的这个词"Learning"是我们的输入，而这8个上下文词是我们的输出。
>
> 这样在这个Skip-Gram的例子里，**输入是特定词， 输出是softmax概率排前8的8个词**，对应的Skip-Gram神经网络模型**输入层有1个神经元，输出层有词汇表大小个神经元**。隐藏层的神经元个数同样可以自己指定。
>
> 通过DNN的反向传播算法，可以求出DNN模型的参数，同时得到所有的词对应的词向量。这样当我们有新的需求，要求出某1个词对应的最可能的8个上下文词时，我们可以通过一次DNN前向传播算法得到概率大小排前8的softmax概率对应的神经元所对应的词即可。



## Word2vec在DNN模型基础上的优化

**word2vec也使用了CBOW与Skip-Gram来训练模型与得到词向量，但是并没有使用传统的DNN模型**，而是继续优化出新方法。传统DNN模型最主要的问题是处理过程非常耗时——词汇表一般在百万级别以上，这意味着DNN的输出层需要进行softmax计算各个词的输出概率的计算量很大，必须进行简化。

### word2vec基础之huffman树

最先优化使用的数据结构是用**霍夫曼树**来代替隐藏层和输出层的[神经元](https://so.csdn.net/so/search?q=神经元&spm=1001.2101.3001.7020)，即霍夫曼树的：
- 叶子节点：起到输出层神经元的作用，叶子节点的个数即为词汇表的小大(叶子节点的权重是词频)；
- 内部节点：起到隐藏层神经元的作用。
目的：词频越高的词，希望编码长度越短。
这么做的优点：得到霍夫曼树后，会对叶子节点进行霍夫曼编码，由于**权重高的叶子节点越靠近根节点，而权重低的叶子节点会远离根节点**，这样我们的**高权重节点编码值较短，而低权重值编码值较长**。这**保证树的带权路径最短**，也符合我们的信息论，即我们希望**越常用的词拥有更短的编码**。

编码方式：在word2vec中，约定编码方式和上面的例子相反，即约定左子树编码为1，右子树编码为0，同时约定左子树的权重不小于右子树的权重。




## Word2Vec的句向量计算
公式如下：
![Word2Vec的句向量计算](https://img-blog.csdnimg.cn/20190828182949999.png)

实现代码如下：

```python
#对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(sentence,size,w2v_model):
    sen_vec=np.zeros(size).reshape((1,size))
    count=0
    for word in sentence:
        try:
            sen_vec+=w2v_model[word].reshape((1,size))
            count+=1
        except KeyError:
            continue
    if count!=0:
        sen_vec/=count
    return sen_vec
```



## 基于加权Word2Vec的句向量

计算公式如下：

![](https://img-blog.csdnimg.cn/20190828183007343.png)

实现代码如下：

```python
 
#对每个句子的所有词向量取加权均值，来生成一个句子的vector
def build_sentence_vector_weight(sentence,size,w2v_model,key_weight):
    key_words_list=list(key_weight)
    sen_vec=np.zeros(size).reshape((1,size))
    count=0
    for word in sentence:
        try:
            if word in key_words_list:
                sen_vec+=(np.dot(w2v_model[word],math.exp(key_weight[word]))).reshape((1,size))
                count+=1
            else:
                sen_vec+=w2v_model[word].reshape((1,size))
                count+=1
        except KeyError:
            continue
    if count!=0:
        sen_vec/=count
    return sen_vec
```



## 基于Word2Vec的文本向量化实现

```python
 
# 将文本数据转换为文本向量
def doc_vec():
    train_data = pd.read_csv('data/clean_data_train.csv', sep=',',names=['contents', 'labels']).astype(str)
    test_data = pd.read_csv('data/clean_data_test.csv', sep=',', names=['contents', 'labels']).astype(str)
    w2v_model = Word2Vec.load('data/w2v/w2v_model_300.pkl')   #加载训练好的Word2Vec模型
 
    #读取词权重字典
    #with open('data/key_words_importance', 'r') as f:
       #key_words_importance = eval(f.read())
 
    cw=lambda x:int(x)
    y_train = np.array(train_data['labels'].apply(cw))
    y_test=np.array(test_data['labels'].apply(cw))
 
    #训练集转换为向量
    train_lenth=len(train_data)
    train_data_list=[]
    for i in range(train_lenth):
        train_data_list.append(str(train_data['contents'][i]).split())
    train_docvec_list=np.concatenate([build_sentence_vector(sen,300,w2v_model) for sen in train_data_list])
 
    #测试集转换为向量
    test_lenth = len(test_data)
    test_data_list = []
    for i in range(test_lenth):
        test_data_list.append(str(test_data['contents'][i]).split())
    test_docvec_list = np.concatenate([build_sentence_vector(sen, 300, w2v_model) for sen in test_data_list])
 
    return train_docvec_list,y_train,test_docvec_list,y_test
```

