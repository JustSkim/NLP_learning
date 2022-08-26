import os

# jieba分词库
import jieba.analyse

# gensim词向量训练库
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import word2vec

"""
THUCNews数据集训练：
THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。包含财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐14个领域。
每个领域下面都有若干这样的txt文本数据。所以现在需要做的第一步就是如何把这些分散开的语料喂入模型中。
原文链接：https://blog.csdn.net/Jeremiah_/article/details/121245000
"""


# 读取一个目录下的所有文件
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                yield line.split()


# 针对THUCNews数据集的财经领域
File = 'F:\\自然语言处理数据集\\THUCNews\\THUCNews\\财经'
files = os.listdir(File)  # 得到该文件夹下的所有文件名（包含后缀），输出为列表形式
for file in files:
    with open(File + '/' + file, encoding='utf-8') as f1:
        # 对每个文档的内容进行分词
        document = f1.read()
        document_cut = jieba.cut(document, cut_all=False)

        # 分词之后用空格隔开并去掉标点符号
        result = ' '.join(document_cut).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
            .replace('“', '').replace('”', '').replace('：', '').replace('；', '').replace('…', '').replace('（',
                                                                                                          '').replace(
            '）', '') \
            .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
            .replace('’', '')

        # 处理好的语料文档的路径
        datapath = 'F:\\自然语言处理数据集\\THUCNews\\THUCNews\\财经' + file
        with open(datapath, 'w', encoding="utf-8") as f2:
            f2.write(result)

# 利用自定义的读取一个文件夹下(这里我们选择的是THUCNews中的 财经 领域)所有文件的函数来加载语料
sentences = MySentences('F:\\自然语言处理数据集\\THUCNews\\THUCNews\\财经')


try:
    #有训练好的模型就引入
    model_2 = word2vec.Word2Vec.load("./word2vec_2.model")
    print("已有模型word2vec_2.model存在，直接使用")
except:
    #没有的话，就新训练语料模型
    model_2 = word2vec.Word2Vec(sentences, window=5, min_count=1, workers=4)
    model_2.save("./word2vec_2.model")


# 任意查看与“中国银行”最相似的前5个词语及相似度度量
for key in model_2.wv.similar_by_word('基金', topn=20):
    print(key)
"""
 模块word2vec.py中有这么一句：
 self.wv = KeyedVectors(vector_size)
 Pycharm中点击"similar_by_word"会跳转到keyedvectors.py中，可以发现，
 函数gensim.models.keyedvectors.KeyedVectors.similar_by_word自带的函数说明：
 "Compatibility alias for similar_by_key()." 
 意味着其等同于函数similar_by_key()，这从两个函数的源码定义中可以看出来，设置成两个函数只是方便使用
 函数similar_by_key(self, key, topn=10, restrict_vocab=None)作用是找到前N个最相似的，参数:
 key:要比对的词，topN: int或None，默认为None，则返回相似性得分的向量
 restrict_vocab:int，可选，用于限制搜索最相似值的向量范围。例如，restrict_vocab=10000将仅检查词汇表顺序中的前10000个关键向量。
 如果按照降序频率对词汇进行排序，则这个参数可能更有意义。

"""