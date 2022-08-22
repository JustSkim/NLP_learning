# PageRank 算法
谷歌的网页重要性排序算法，通过计算网页链接的数量和质量来粗略估计网页的重要性，在[论文《The PageRank Citation Ranking: Bringing Order to the Web》](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf)中被提出，被应用在谷歌的搜索引擎中，对网页进行排名。 
[核心思想](https://blog.csdn.net/asialee_bird/article/details/96894533)有以下两点：
> 1. 如果一个网页被越多的其他网页链接，说明这个网页越重要，即该网页的PR值（PageRank值）会相对较高；
> 2. 如果一个网页被一个越高权值的网页链接，也能表明这个网页越重要，即一个PR值很高的网页链接到一个其他网页，那么被链接到的网页的PR值会相应地因此而提高。

计算公式：
![PageRank算法公式](https://img-blog.csdnimg.cn/20190722210238175.png)

# TextRank 算法
一种基于图的用于关键词抽取和文档摘要的排序算法，**由PageRank算法改进而来**，利用一篇文档内部的词语间的共现信息(语义)便可以抽取关键词，它能够从一个给定的文本中抽取出该文本的关键词、关键词组，并使用抽取式的自动文摘方法抽取出该文本的关键句。
TextRank由[论文《TextRank: Bringing Order into Texts》](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)提出，链接地址：https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
基本思想：将文档看作一个词的网络，该网络中的链接表示词与词之间的语义关系，计算公式如下：
![Text计算公式](https://img-blog.csdnimg.cn/2019072221192141.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FzaWFsZWVfYmlyZA==,size_16,color_FFFFFF,t_70)
[TextRank算法主要包括三部分：关键词抽取、关键短语抽取、关键句抽取](https://blog.csdn.net/asialee_bird/article/details/96894533):
## 关键词抽取（keyword extraction）
关键词抽取：从文本中确定一些能够描述文档含义的术语的过程。
对关键词抽取而言，用于构建顶点集的文本单元可以是句子中的1个或多个字；根据这些字之间的关系（比如：在一个框中同时出现）来构建边；根据任务的需要，可以使用语法过滤器（syntactic filters）对顶点集进行优化。语法过滤器的主要作用：将某一类或者某几类词性的字过滤出来作为顶点集。

## 关键短语抽取（keyphrase extration）
关键词抽取结束后，得到N个关键词，在原始文本中相邻的关键词构成关键短语。
因此，从`get_keyphrases`函数的源码中我们可以看到，它先调用get_keywords抽取关键词，然后**分析关键词是否存在相邻的情况**，最后**确定**哪些是**关键短语**。

## 关键句抽取（sentence extraction）
句子抽取任务主要针对的是**自动摘要**这个场景——将每一个sentence作为一个顶点，根据两个句子之间的内容重复程度来计算他们之间的“相似度”，以这个相似度作为联系，由于不同句子之间相似度大小不一致，在这个场景下构建的是**以相似度大小作为edge权重的有权图**。

## TextRank与TF-IDF
1. TextRank与TFIDF**均严重依赖于分词结果**——如果某词在分词时被切分成了两个词，那么在做关键词提取时无法将两个词黏合在一起（TextRank有部分黏合效果，但需要这两个词均为关键词）。因此**是否添加标注关键词进自定义词典，将会造成准确率、召回率大相径庭**。
2. **TextRank的效果并不优于TFIDF**。
3. **TextRank**虽然考虑到了词之间的关系，但是仍然**倾向于将频繁词作为关键词**。
4. TextRank**涉及到构建词图及迭代计算，所以提取速度较慢**。

# PageRank算法和TextRank算法对比
二者区别：
+ PageRank算法根据网页之间的链接关系构造网络，TextRank算法根据词之间的共现关系构造网络；
+ PageRank算法构造的网络中的边是有向无权边，TextRank算法构造的网络中的边是无向有权边。

关于二者数学上的区别更多可见：https://blog.csdn.net/wotui1842/article/details/80351386