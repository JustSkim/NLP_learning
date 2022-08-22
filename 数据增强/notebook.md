# 数据增强
概念：充数据样本规模的一种有效地方法，数据的规模越大、质量越高越好，模型才能够有着更好的泛化能力。
关于EDA的学术介绍可见：https://arxiv.org/abs/1901.11196

## 1. 简单数据增强(Easy Data Augmentation, EDA)
- 同义词替换(Synonym Replacement, SR)：从句子中随机选取n个不属于停用词集的单词，并随机选择其同义词替换它们；
- 随机插入(Random Insertion, RI)：随机的找出句中某个不属于停用词集的词，并求出其随机的同义词，将该同义词插入句子的一个随机位置。重复n次；
- 随机交换(Random Swap, RS)：随机的选择句中两个单词并交换它们的位置。重复n次；
- 随机删除(Random Deletion, RD)：以 $p$ 的概率，随机的移除句中的每个单词。
使用EDA前，要先将需要处理的语料整理成特定的格式，可以使用相关EDA工具：
- 中文语料的EDA数据增强工具：https://github.com/Asia-Lee/EDA_NLP_for_Chinese
- Synonyms中文近义词工具包：https://github.com/huyingxi/Synonyms/
- 中文常用的停用词表：https://github.com/goto456/stopwords 

## 2. 回译

概念：用机器翻译把一段中文翻译成另一种语言，然后再翻译回中文。
回译的方法不仅有类似同义词替换的能力，还具有在保持原意的前提下增加或移除单词并重新组织句子的能力。
回译可使用python translate包和textblob包（少量翻译），或者使用**百度翻译或谷歌翻译的api**通过python实现。
百度开放的翻译接口——[百度翻译开放平台](http://api.fanyi.baidu.com/api/trans/product/apidoc)支持每月200万字的免费翻译，提供了各种语言的使用demo，本篇使用Python3调用百度API实现自然语言的翻译

缺点：在短文本回译过程中, 新语料与原语料可能存在很高的重复率, 并不能有效增大样本的特征空间。

## 3. 基于上下文的数据增强方法
可以参考论文[Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://arxiv.org/pdf/1805.06201.pdf)的方法实现代码，使用**双向循环神经网络**进行数据增强，链接地址：https://arxiv.org/pdf/1805.06201.pdf 。
不足之处在于：该方法目前只针对英文数据进行增强，实验工具：spacy（NLP自然语言工具包）和chainer（深度学习框架）。

