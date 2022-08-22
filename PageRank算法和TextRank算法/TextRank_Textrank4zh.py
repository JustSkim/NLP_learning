# coding=utf-8
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import jieba.analyse
from snownlp import SnowNLP
import pandas as pd
import numpy as np

"""
基于Textrank4zh的TextRank算法实现。
https://pypi.org/project/textrank4zh/#description会将我们引导到https://github.com/letiantian/TextRank4ZH，这里介绍：
TextRank算法可以用来从文本中提取关键词和摘要（重要的句子）。TextRank4ZH是针对中文文本的TextRank算法的python算法实现。

要运行本代码，我们还需要安装snownlp，作者在网站中提到https://pypi.org/project/snownlp/：
SnowNLP是一个python写的类库，可以方便的处理中文文本内容，是受到了[TextBlob](https://github.com/sloria/TextBlob)的启发而写的，由于现在大部分的自然语言处理库基本都是针对英文的，于是写了一个方便处理中文的类库，并且和TextBlob不同的是，这里没有用NLTK，所有的算法都是自己实现的，并且自带了一些训练好的字典。注意本程序都是处理的unicode编码，所以使用时请自行decode成unicode
"""


# 关键词抽取
def keywords_extraction(text):
    tr4w = TextRank4Keyword(allow_speech_tags=['n', 'nr', 'nrfg', 'ns', 'nt', 'nz'])
    # allow_speech_tags   --词性列表，用于过滤某些词性的词
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                 pagerank_config={'alpha': 0.85, })
    # text    --  文本内容，字符串
    # window  --  窗口大小，int，用来构造单词之间的边。默认值为2
    # lower   --  是否将英文文本转换为小写，默认值为False
    # vertex_source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点
    #                -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
    # edge_source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边
    #              -- 默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数

    # pagerank_config  -- pagerank算法参数配置，阻尼系数为0.85
    keywords = tr4w.get_keywords(num=6, word_min_len=2)
    # num           --  返回关键词数量
    # word_min_len  --  词的最小长度，默认值为1
    return keywords


# 关键短语抽取
def keyphrases_extraction(text):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                 pagerank_config={'alpha': 0.85, })
    keyphrases = tr4w.get_keyphrases(keywords_num=6, min_occur_num=1)
    # keywords_num    --  抽取的关键词数量
    # min_occur_num   --  关键短语在文中的最少出现次数
    return keyphrases


# 关键句抽取
def keysentences_extraction(text):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text, lower=True, source='all_filters')
    # text    -- 文本内容，字符串
    # lower   -- 是否将英文文本转换为小写，默认值为False
    # source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来生成句子之间的相似度。
    # 		  -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
    # sim_func -- 指定计算句子相似度的函数

    # 获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要
    keysentences = tr4s.get_key_sentences(num=3, sentence_min_len=6)
    return keysentences


"""
这一步是基于jieba的TextRank算法实现的关键，调用jieba.analyse模块的textrank方法，直到调用jieba的TextRank类构造方法
基于Textrank的关键词提取
函数：jieba.analyse.textrank(string, topK=20, withWeight=True, allowPOS=())
string：待处理语句
topK：关键字的个数，默认20
withWeight：是否返回权重值，默认false
allowPOS：是否仅返回指定类型，默认为空
"""
def keywords_textrank(text):
    keywords = jieba.analyse.textrank(text, topK=6)
    return keywords


if __name__ == "__main__":
    text = "来源：中国科学报本报讯（记者肖洁）又有一位中国科学家喜获小行星命名殊荣！4月19日下午，中国科学院国家天文台在京举行“周又元星”颁授仪式，" \
           "我国天文学家、中国科学院院士周又元的弟子与后辈在欢声笑语中济济一堂。国家天文台党委书记、" \
           "副台长赵刚在致辞一开始更是送上白居易的诗句：“令公桃李满天下，何须堂前更种花。”" \
           "据介绍，这颗小行星由国家天文台施密特CCD小行星项目组于1997年9月26日发现于兴隆观测站，" \
           "获得国际永久编号第120730号。2018年9月25日，经国家天文台申报，" \
           "国际天文学联合会小天体联合会小天体命名委员会批准，国际天文学联合会《小行星通报》通知国际社会，" \
           "正式将该小行星命名为“周又元星”。"

    # 基于SnowNLP的textrank算法实现
    print("下面开始基于SnowNLP的textrank算法实现\n================")
    snlp = SnowNLP(text)
    print(snlp.keywords(6))  # 关键词抽取
    #['行星', '中国', '小', '周', '天文台', '月']

    print(snlp.summary(3))  # 关键句抽取
    #['这颗小行星由国家天文台施密特CCD小行星项目组于1997年9月26日发现于兴隆观测站', '中国科学院国家天文台在京举行“周又元星”颁授仪式', '国际天文学联合会《小行星通报》通知国际社会']
    print("======================\n基于SnowNLP的textrank算法实现完毕")


    # 关键词抽取
    keywords = keywords_extraction(text)
    print(keywords)

    # 关键短语抽取
    keyphrases = keyphrases_extraction(text)
    print(keyphrases)

    # 关键句抽取
    keysentences = keysentences_extraction(text)
    print(keysentences)
