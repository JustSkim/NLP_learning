import nltk
"""
NLTK词性标注
具体词性tag含义可见 https://blog.csdn.net/john159151/article/details/50255101
其中WDT表示Wh限定词 who,which,when,what,where,how
"""
text = nltk.word_tokenize('what does the fox say')
print(text)
print(nltk.pos_tag(text))

'''
结果为：
['what', 'does', 'the', 'fox', 'say']
输出是元组列表，元组中的第一个元素是单词，第二个元素是词性标签
[('what', 'WDT'), ('does', 'VBZ'), ('the', 'DT'), ('fox', 'NNS'), ('say', 'VBP')]
'''