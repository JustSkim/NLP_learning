
from nltk.stem import WordNetLemmatizer
"""
NLTK词性还原
词形还原与词干提取类似， 但不同之处在于词干提取经常可能创造出不存在的词汇，词形还原的结果是一个真正的词汇。
lemmatize有word和pos两个参数，pos表示词性，默认为"n"，即名词
"""
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('increases'))
# 结果为：increase，取得了第三人称动词的原形


"""
词性还原结果可能是同义词或具有相同含义的不同词语。有时，如果你试图还原一个词，比如 playing,还原的结果还是 playing。这是因为默认还原的结果是名词，如果你想得到动词，可以通过以下的方式指定。
"""
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('playing', pos="v"))

"""
实际上，这是一个非常好的文本压缩水平。最终压缩到原文本的 50％ 到 60％ 左右。结果可能是动词，名词，形容词或副词
"""
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('playing', pos="v"))
print(lemmatizer.lemmatize('playing', pos="n"))
print(lemmatizer.lemmatize('playing', pos="a"))
print(lemmatizer.lemmatize('playing', pos="r"))
