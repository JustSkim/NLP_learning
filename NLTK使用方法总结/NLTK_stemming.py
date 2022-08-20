from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
"""
NLTK词干提取
3种主流的词干提取方法各有优劣：https://easyai.tech/ai-definition/stemming-lemmatisation/
Porter词干算法比较旧，主要关注点是删除单词的共同结尾，以便将它们解析为通用形式。它不是太复杂，它的开发停止了。
通常情况下，它是一个很好的起始基本词干分析器，但并不建议将它用于复杂的应用。相反，它在研究中作为一种很好的基本词干算法，可以保证重复性。与其他算法相比，它也是一种非常温和的词干算法。
Snowball算法也称为 Porter2 词干算法。普遍认为比 Porter 更好，甚至发明 Porter 的开发者也这么认为。
Snowball 在 Porter 的基础上加了很多优化。Snowball 与 Porter 相比差异约为5％。
Lancaster 的算法比较激进，有时候会处理成一些比较奇怪的单词。如果在 NLTK 中使用词干分析器，则可以非常轻松地将自己的自定义规则添加到此算法中。
"""
porter_stemmer = PorterStemmer()
print(porter_stemmer.stem('working'))
#结果为：work

lancaster_stemmer = LancasterStemmer()
print(lancaster_stemmer.stem('working'))

print(SnowballStemmer.languages)
#SnowballStemmer 类，除了英语外，还可以适用于其他 13 种语言

french_stemmer = SnowballStemmer('french')
print(french_stemmer.stem("French word"))
#使用 SnowballStemmer 类的 stem() 函数来提取非英语单词
