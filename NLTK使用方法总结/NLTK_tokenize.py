from nltk.tokenize import sent_tokenize,word_tokenize #看引进的两个函数名称就知道，前一个分句，后一个负责分词功能
"""
nltk分句
"""
mytext = "Hello Adam, how are you? I hope everything is going well. Today is a good day, see you dude."
print(sent_tokenize(mytext))

"""
nltk分词，注意标点符号无论中英，也会被划分为单独的词，但多个连续空格会被忽略
"""
mytext = "Hello Mr.   Adam,。 how are you? I hope everything is going well. Today is a good day, see you dude."
print(word_tokenize(mytext))

"""
nltk还可以标记非英语语言的文本，比如这里把法语标记出来
这得益于sent_tokenize只有两个参数，第一个是要处理的文本，第二个是语言的选择。
该函数会去找到"nltk_data\tokenizers\punkt"路径下的各语言对应的pickle文件
"""
mytext = "Bonjour M. Adam, comment allez-vous? J'espère que tout va bien. Aujourd'hui est un bon jour."
print(sent_tokenize(mytext,"french"))