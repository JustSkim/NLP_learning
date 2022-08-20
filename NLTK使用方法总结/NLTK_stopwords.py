from nltk.corpus import stopwords
"""
NLTK去除停用词，corpus模块会根据相应的api接口去下载stopwords的压缩包。
本人在路径"G:\anaconda3\envs\py37\nltk_data\packages"下已经准备好了相应的压缩包，因此无需联网也可使用
点击readme文件，可以发现关于stopwords数据包的描述：
“stopwords库包含多种语言的停止词列表。这些是在文本检索应用中通常被忽略的高频语法词。”
"""
tokens = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please',
          'maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid',
          'my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him']

clean_tokens = tokens[:]
stwords = stopwords.words('english')
for token in tokens:
    if token in stwords:
        clean_tokens.remove(token)

print(clean_tokens)