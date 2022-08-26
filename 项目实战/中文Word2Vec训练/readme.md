# Introduction
跟随[博客——中文Word2Vec训练](https://blog.csdn.net/Jeremiah_/article/details/121245000)进行的项目练习。

在该项目中，会使用到两个数据集：txt格式的《倚天屠龙记》小说文本和清华大学的THUCNews数据集，在网上可以容易找到下载地址，不再赘述。

gensim的官方文档：https://radimrehurek.com/gensim/auto_examples/index.html#documentation

[在保存模型方面，gensim生成模型有三种](https://blog.csdn.net/WangYouJin321/article/details/123234002)格式，对应的函数和加载方法也不一样：
1. 默认的model文件（可以继续进行tuning)
2. bin文件(c风格）
3. txt文件（比较大），我们可以在这种格式的文件中看到一大堆独立的中文字符和数字
```python
import gensim
from gensim.models import word2vec
# 第一种
model = word2vec.Word2Vec.load(word2vec.model) 
model.save('word2vec.model')
# 第二种
model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.bin',binary=True)
model.wv.save_word2vec_format('word2vec.bin')
# 第三种
gensim.models.KeyedVectors.load_word2vec_format('word2vec.txt',binary=False)
model.wv.save_word2vec_format('word2vec.txt')
```

 可供参考使用
## 倚天屠龙记项目练习
参考文件`train_yitiantulong.py`，将训练好的模型存储到本地的word2vec_1.model中。
然后输出和“张三丰”最相似的前10个词和相似度度量，下面是第一次训练得到的结果：
```mysql based
('这人定', 0.6976600289344788)
('张真人', 0.6911172866821289)
('俞莲舟', 0.6873677968978882)
('致贺', 0.6717900633811951)
('常遇春', 0.6696386933326721)
('刁顽', 0.668433666229248)
('太师父', 0.6664665937423706)
('安好', 0.6550816297531128)
('拱手行礼', 0.6546619534492493)
('殷素素', 0.6546303033828735)
```
再一次训练，发现得到的结果不一样：
```mysql based
('张真人', 0.6830218434333801)
('法号', 0.6695971488952637)
('常遇春', 0.6572055816650391)
('致贺', 0.6557248830795288)
('太师父', 0.6471196413040161)
('俞岱岩', 0.6451671123504639)
('四弟', 0.6393076777458191)
('勿予', 0.637441873550415)
('爷', 0.6347135901451111)
('平起平坐', 0.6341426968574524)
```

## THUCNews 项目练习
THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。包含财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐14个领域。
每个领域下面都有若干这样的txt文本数据。需要做的第一步就是如何把这些分散开的语料喂入模型中。

对作者的代码进行了部分修改——有训练好的模型就直接引入，没有则重新训练一个并存储：
```python
try:
    #有训练好的模型就引入
    model_2 = word2vec.Word2Vec.load("./word2vec_2.model")
except:
    #没有的话，就新训练语料模型
    model_2 = word2vec.Word2Vec(sentences, window=5, min_count=1, workers=4)
    model_2.save("./word2vec_2.model")
```

跑了两次模型，利用函数wv.similar_by_word，输出和“张三丰”最相似的前10个词和相似度度量
```python
newmodel = word2vec.Word2Vec.load("./word2vec_1.model")
# 输出和“张三丰”最相似的前10个词和相似度度量
for key in newmodel.wv.similar_by_word('张三丰', topn=10):
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
```
但在不进行改进的情况下，训练出的结果总是如下：
```mysql based
('2', 0.6704797744750977)
('期', 0.655806839466095)
('1', 0.6329173445701599)
('★★★★★', 0.6328134536743164)
('年', 0.6306853890419006)
```

为了探究原因，首先我们更改代码，使用`try...except...`结构，在已有模型的情况下，使用保存了的模型来跑。
可以发现，即使已有现有的模型，依然要耗费很长时间，原因主要在于`model_2.wv.similar_by_word`这个函数
结果如下：
```mysql based
已有模型word2vec_2.model存在，直接使用
('2', 0.6704908013343811)
('期', 0.6558108329772949)
('1', 0.6329302787780762)
('★★★★★', 0.6328170895576477)
('年', 0.6307034492492676)
('26', 0.6221688985824585)
('0.00', 0.6174758076667786)
('深TMT联接', 0.6155266761779785)
('月', 0.6079839468002319)
('Trust', 0.6050586700439453)
('广发制造', 0.5996891856193542)
('银华永祥', 0.5981618762016296)
('广发标普', 0.5929902791976929)
('泰信中证200', 0.5927817821502686)
('鹏华产业', 0.5912985801696777)
('panamax', 0.5908035635948181)
('D', 0.5888797640800476)
('(Baltic', 0.5866135954856873)
('日', 0.5823621153831482)
('周度变化', 0.5771374702453613)
```

因此，如何改掉这些影响我们实验结果的无用字词，就是一个很大的问题