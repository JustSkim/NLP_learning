import os

# jieba分词库
import jieba.analyse

# gensim词向量训练库
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import word2vec

"""
中文的词向量训练和英文的差不多，输入数据的格式都一样，均需要可迭代的句子列表。但有一点需要注意的是，在英文句子里，单词之间自然地就很清楚哪个是哪个单词了，而中文句子则不然，计算机需要知道哪个部分称之为一个“词”。
所以，中文词向量的训练关键在于分词的处理。通常使用jieba分词工具库来对语料库进行处理。
————————————————
下面我们依照CSDN博主「Eureka丶」的原创文章https://blog.csdn.net/Jeremiah_/article/details/121245000的指引，进行《倚天屠龙记》小说训练
"""

#首先是对默认txt文件做处理，这一步之前我们要删掉txt中可能有的广告语句防止干扰
with open('F:\\自然语言处理数据集\\倚天屠龙记.txt', encoding='utf-8') as f1:
    document = f1.read()  # 一行一行地读取小说文本的句子

    document_cut = jieba.cut(document, cut_all=False)  # 分词

    result = ' '.join(document_cut).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('；', '').replace('…', '').replace('（', '').replace(
        '）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')  # 词与词之间用空格隔开并去掉标点符号

    # result是得到分词之后的文本语料库，我们将其写入文件倚天屠龙记_segment.txt中
    with open('F:\\自然语言处理数据集\\倚天屠龙记_segment.txt', 'w', encoding="utf-8") as f2:
        f2.write(result)

# 加载语料
sentences = word2vec.LineSentence('F:\\自然语言处理数据集\\倚天屠龙记_segment.txt')
"""
gensim官方英文文档https://radimrehurek.com/gensim/models/word2vec.html?highlight=linesentence#gensim.models.word2vec.LineSentence上
函数LineSentence的源码解释：迭代包含句子的文件：一行=一个句子。单词必须已经过预处理，并用空格分隔。
传入的单词必须已经过预处理，并用空格分隔。
参数有两个：
source：指向磁盘上文件的字符串或类似文件的对象路径，或已打开的文件对象（必须支持“seek（0）”）。
limit：int类型或None，将文件剪切到第一个“limit”行。如果“limit=None”（默认值），则不进行剪切。

Debugger过程可以看到，经过此步处理，得到的语料sentences是一个gensim.models.word2vec.LineSentence对象
"""

#训练语料，得到训练模型model_1
model_1 = word2vec.Word2Vec(sentences, sg=1, vector_size=100, negative=3, sample=0.001, hs=1, window=5, min_count=1, workers=4)

"""
word2vec.Word2Vec相关参数：
1. sentences：预处理后的训练语料库。是可迭代列表，但是对于较大的语料库，可以考虑直接从磁盘/网络传输句子的迭代。
这一个参数尤其重要，在Word2Vec函数的源码中这样写到：
句子：iterable of iterable，可选“句子”iterable可以只是一个标记列表列表，但对于更大的语料库，请考虑直接从磁盘\/网络流式传输句子的iterable。
参见：类：`~gensim.models.word2vec。BrownCorpus`，：类：`~gensim.models.word2vec。Text8Corpus`or:class:`~gensim.models.word2vec。在：mod:`~gensim.models。word2vec’模块用于此类示例。
另请参见“Python中的数据流教程”：https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/
如果不提供“语句”，则模型将保持未初始化状态——如果您计划以其他方式初始化它，请使用。

2. sg：skip-gram算法，对低频词敏感；默认sg=0为CBOW算法。
3. vector_size(int) ：是输出词向量的维数，默认值是100。这个维度的取值与我们的语料的大小相关，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间，不过见的比较多的也有300维的。
4. window(int)：是一个句子中当前单词和预测单词之间的最大距离，window越大，则和某一词较远的词也会产生上下文关系。默认值为5。window越大所需要枚举的预测此越多，计算的时间越长。
5. min_count：忽略所有频率低于此值的单词。默认值为5。
6. workers：表示训练词向量时使用的线程数，默认是当前运行机器的处理器核数。
还有关采样和学习率的，一般不常设置：
1. negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。
2. hs=1表示分层softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
"""

# 保存模型
path = get_tmpfile("word2vec_1.model")
#gensim.test.utils.get_tmpfile(suffix) 获取临时文件夹中文件后缀的完整路径。
#此函数不创建文件（仅生成唯一名称）。此外，它可以在连续呼叫中返回不同的路径。
#使用此函数，我们可以获得临时文件的路径，并使用它来存储临时模型，以备后续需要。

# 保存方式对于路径的选择
#model_1.save(path) 保存到临时文件夹中
#model_1.save("word2vec_1.model") 存储到gensim定义的路径中
model_1.save("./word2vec_1.model")#保存到我们定义的当前文件夹下，用记事本打开可以发现是乱码的

#保存到bin和txt格式文件需要使用save_word2vec_format方式
#model_1.wv.save_word2vec_format('word2vec_1.bin')
#model_1.wv.save_word2vec_format('word2vec_1.txt')

newmodel = word2vec.Word2Vec.load("./word2vec_1.model")
#使用gensim.models.Word2Vec.load加载模型，然后我们下面就使用这一个模型

# 输出和“张三丰”最相似的前10个词和相似度度量
for key in newmodel.wv.similar_by_word('张三丰', topn=10):
    print(key)
