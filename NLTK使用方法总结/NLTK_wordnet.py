from nltk.corpus import wordnet
"""
wordnet 是NLTK中为自然语言处理构建的数据库。它包括部分词语的一个同义词组和一个简短的定义。
通过 wordnet可以得到给定词的定义和例句
"""
syn = wordnet.synsets("pain")  # 获取“pain”的同义词集
print(syn[0].definition())
print(syn[0].examples())

'''
结果为：
a symptom of some physical hurt or disorder
['the patient developed severe pain and distension']
'''

#获取同义词
synonyms = []
for syn in wordnet.synsets('Computer'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print(synonyms)
'''
结果为：
['computer', 'computing_machine', 'computing_device', 'data_processor', 'electronic_computer', 'information_processing_system', 'calculator', 'reckoner', 'figurer', 'estimator', 'computer']
'''

#获取反义词
antonyms = []
for syn in wordnet.synsets("small"):
    for l in syn.lemmas():
        if l.antonyms():  # 判断是否是正确的反义词
            antonyms.append(l.antonyms()[0].name())
print(antonyms)

'''
结果为：
['large', 'big', 'big']
'''
