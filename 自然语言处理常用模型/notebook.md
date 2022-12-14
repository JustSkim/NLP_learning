# 自然语言处理常用模型与部分概率论知识



## N-gram模型

N-gram是计算机语言学和概率论范畴内的概念，是指给定的一段文本或语音中N个项目（item）的序列。项目（item）可以是音节、字母、单词或碱基对。通常N-grams取自文本或语料库。

N=1时称为uni-gram，N=2称为bi-gram，N=3称为tri-gram，以此类推，但一般N=3已经够用了。

N-Gram是基于一个假设：第n个词出现与前n-1个词相关，而与其他任何词不相关。（这也是隐马尔可夫当中的假设。）**整个句子出现的概率就等于各个词出现的概率乘积**。而各个**词的概率可以通过语料中统计计算得到**。

（`Pycharm`中似乎不支持markdown的公式语法规则）

假设有一个由 $ n $ 个词组成的句子$S = (w_1,w_2,...,w_n)$，要衡量其概率，则可以假设每个单词$w_i$都要依赖于从第一个单词$w_1$到它前一个单词$w_{i-1}$的影响：

$$p(S) = p(w_1 w_2 w_3 ... w_n)=p(w_1)p(w_2|w_1)p(w_3|w_1 w_2)...p(w_n|w_1 w_2 ... w_{n-1})$$

这是因为根据贝叶斯公式，已知事件A发生的条件下，另一个事件B发生的概率称为条件概率，即为：$P(A|B)$

**事件A与事件B是有关系的，事件A的发生与否会影响事件B发生的概率**。

联合概率：**P(A,B)也就是P(AB)**。事件A和事件B可以相互影响的，不过也可以相互独立，不过没有讨论的意义。对应的P(A)和P(B)是边缘概率。

$P(A,B)=P(B)P(A|B)$ AB同时发生的概率 = 事件B发生的概率，乘以事件A在事件B发生的条件下发生的概率。二者相互独立时$P(A|B)=P(A)$

$$ P(A|B)=\frac{P(B/A)P(A)}{/P(B)} $$

但是$P(w_n|w_1 w_2 ... w_{n-1})$要考虑的实在太多了，因此有了**马尔科夫模型**，该模型认为，**一个词的出现仅仅依赖于它前面出现的几个词**。

因此，使用bi-gram计算的公式：

$$p(S) = p(w_1 w_2 w_3 ... w_n)=p(w_1)p(w_2|w_1)p(w_3|w_1)...p(w_n|w_{n-1})$$

实际就是认为$p(w_n|w_1 w_2 ... w_{n-1})≈p(w_n|w_{n-1})$。同样地，N=3，使用tri-gram计算：

$$p(S) = p(w_1 w_2 w_3 ... w_n)=p(w_1)p(w_2|w_1)p(w_3|w_1 w_2)...p(w_n|w_{n-1} w_{n-2})$$



接下来，要构造语言模型，就要计算其中的每一项条件概率，就要使用**极大似然估计（Maximum Likelihood Estimation，`MLE`）**，也就是数频次：$p(w_n|w_{n-1})=\frac{C(w_{n-1} w_n)}{C(w_{n-1})}$，同样地，$p(w_n|w_{n-1} w_{n-2})=\frac{C(w_{n-2} w_{n-1} w_n)}{C(w_{n-2}w_{n-1})}$

 **N-gram中N的确定**

更大的n：对下一个词出现的约束信息更多，具有更大的辨别力；
更小的n：在训练语料库中出现的次数更多，具有更可靠的统计信息，具有更高的可靠性。
**理论上，n越大越好，经验上，`tri-gram`用的最多**，尽管如此，**原则上，能用`bi-gram`解决，绝不使用`tri-gram`**：

> 为了确定$N$的取值，论文《Language Modeling with N-grams》使用了 Perplexity 这一指标，该指标越小表示一个语言模型的效果越好。针对不同的N-gram，分别计算各自的`Perplexity`进行比较（注意latex开n次方根和连乘符号的书写）：

> $PP(W)=\sqrt[n]{\frac{1}{P(w_1 w_2 ... w_n)}}=\sqrt[n]{\prod \limits_{i=0}^n{\frac{1}{p(w_i|w_{i-1}...w_1)}}}$

> 实验的结果是`tri-gram`的`Perplexity`值明显大于Unigram和Bigram。n的值越大效果越好。

**[N-gram中的数据平滑方法](https://blog.csdn.net/songbinxu/article/details/80209197#n-gram)**

N-gram的N越大，模型 Perplexity 越小，表示模型效果越好——毕竟依赖的词越多，我们获得的信息量越多，对未来的预测就越准确。然而，由于语言，以及一些语种创造性很强，当N很大时，容易出现这样的状况：某些n-gram从未出现过——也就是**稀疏问题（Sparsity）**。例如，词库中20乘1000个词，两两组合就有将近2亿个可能的词，而**其中很多组合在语料库中没有出现，根据极大似然估计得到的组合概率将会是0，从而整个句子的概率就会为0。模型只能计算零星的几个句子的概率，而大部分的句子算得的概率是0，这显然是不合理的，因此需要进行必要的数据平滑（data Smoothing）**。
所有数据平滑方法的**目的：所有的n-gram概率都不为0并且所有的N-gram概率之和为1**。
**本质：重新分配整个概率空间，使已经出现过的n-gram的概率降低，补充给未曾出现过的n-gram**。

更多N-gram的数据平滑方法可参考：https://blog.csdn.net/songbinxu/article/details/80209197#n-gram，这里只简单介绍以下几个

**拉普拉斯平滑**

分为两种方法：Add-one和Add-K

**Add-one**强制让所有的n-gram至少出现一次，只需要在分子和分母上分别做加法：
$$p(w_n| w_{n-1})=\frac{C(w_{n-1}w_n)+1}{C(w_{n-1})+\|V\|}$$

弊端：由于大部分n-gram都是没有出现过的，这样做很容易为他们分配过多的概率空间。

**Add-K**将参数改为加上一个小于1的常数K，缺点就是K需要人工确定，**不同语料库K可能不同**：

$$p(w_n|w_{n-1})=\frac{C(w_{n-1}w_n)+k}{C(w_{n-1})+k\|V\|}$$

**内插法（Interpolation）**

核心思想：既然高阶组合可能出现次数为0，那稍微低阶一点的组合总有不为0的：

$\hat{p}(w_n|w_{n-1} w_{n-2})=\lambda_3 p(w_n|w_{n-1} w_{n-2})+\lambda_2 p(w_n|w_{n-1})+\lambda_1 p(w_n)$



**N-gram对训练数据集的要求**
  关于N-gram的训练数据，**并非只要是单一语种的就可以**:

> 文献《Language Modeling with N-grams》的作者做了个实验，分别用莎士比亚文学作品，以及华尔街日报作为训练集训练两个N-gram，他认为，两个数据集都是英语，那么用他们生成的文本应该也会有所重合。然而结果是，用两个语料库生成的文本没有任何重合性，即使在语法结构上也没有。
> 这意味着，N-gram的训练集选择十分重要，若要训练一个问答系统，就需用问答的语料库来训练；要训练一个金融分析系统，就要用类似于华尔街日报这样的语料库来训练。



**N-gram的进化版：`NNLM`**，即 Neural Network based Language Model，神经网络语言模型，由Bengio在2003年提出，是一个很简单的模型，由四层组成，输入层、嵌入层、隐层和输出层。模型接收的输入是长度为n的词序列，输出的是下一个词的类别。
![Neural Network based Language Model](https://img-blog.csdn.net/20180507203905676?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NvbmdiaW54dQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**`NNLM`的进化版：`RNNLM`**

针对`NNLM`存在的问题，`Mikolov`在2010年提出了`RNNLM`。`RNNLM`的[论文](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)可见：http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf。其结构实际上是用`RNN`代替`NNLM`里的隐层，这样做的好处包括减少模型参数、提高训练速度、接受任意长度输入、利用完整的历史信息。同时，`RNN`的引入意味着可以使用`RNN`的其他变体，像`LSTM`、`BLSTM`、`GRU`等等，从而在时间序列建模上进行更多更丰富的优化。

**`Word2Vec`**

  `Word2Vec`解决的问题和上面讲到的`N-gram`、`NNLM`等不一样——`Word2Vec`学习一个从高维稀疏离散向量到低维稠密连续向量的映射。该映射的特点是，近义词向量的欧氏距离比较小，词向量之间的加减法有实际物理意义。`Word2Vec`由两部分组成：`CBoW`和Skip-Gram。其中`CBoW`的结构很简单，在`NNLM`的基础上去掉隐层，Embedding层直接连接到`Softmax`，`CBoW`的输入是某个Word的上下文（例如前两个词和后两个词），`Softmax`的输出是关于当前词的某个概率，即`CBoW`是从上下文到当前词的某种映射或者预测。`Skip-Gram`则是反过来，从当前词预测上下文
