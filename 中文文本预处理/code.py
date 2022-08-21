import langconv
import re
#from langconv import *

"""
中文文本预处理总结
"""
print(langconv.Node)

# 读取文件列表数据,返回文本数据的内容列表和标签列表
def filelist_contents_labels(filelist):
    contents = []
    labels = []
    for file in filelist:
        with open(file, "r", encoding="utf-8") as f:
            for row in f.read().splitlines():
                sentence = row.split('\t')
                contents.append(sentence[-1])
                if sentence[0] == 'other':
                    labels.append(0)
                else:
                    labels.append(1)
    return contents, labels


# 全角转半角
def full_to_half(sentence):  # 输入为一个句子
    change_sentence = ""
    for word in sentence:
        inside_code = ord(word)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        change_sentence += chr(inside_code)
    return change_sentence

#半角转全角
def hulf_to_full(sentence):      #输入为一个句子
    change_sentence=""
    for word in sentence:
        inside_code=ord(word)
        if inside_code==32:    #半角空格直接转换
            inside_code=12288
        elif inside_code>=32 and inside_code<=126:  #半角字符（除空格）根据关系转化
            inside_code+=65248
        change_sentence+=chr(inside_code)
    return change_sentence

#大写数字转换为小写数字
def big2small_num(sentence):
    numlist = {"一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","零":"0"}
    for item in numlist:
        sentence = sentence.replace(item, numlist[item])
    return sentence

#大写字母转为小写字母
def upper2lower(sentence):
    new_sentence=sentence.lower()
    return new_sentence

#去除文本中的表情字符（只保留中英文和数字）
def clear_character(sentence):
    pattern1= '\[.*?\]'
    pattern2 = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line1=re.sub(pattern1,'',sentence)
    line2=re.sub(pattern2,'',line1)
    new_sentence=''.join(line2.split()) #去除空白
    return new_sentence

#去除字母数字表情和其它字符
def clear_character(sentence):
    pattern1='[a-zA-Z0-9]'
    pattern2 = '\[.*?\]'
    pattern3 = re.compile(u'[^\s1234567890:：' + '\u4e00-\u9fa5]+')
    pattern4='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    line1=re.sub(pattern1,'',sentence)   #去除英文字母和数字
    line2=re.sub(pattern2,'',line1)   #去除表情
    line3=re.sub(pattern3,'',line2)   #去除其它字符
    line4=re.sub(pattern4, '', line3) #去掉残留的冒号及其它符号
    new_sentence=''.join(line4.split()) #去除空白
    return new_sentence


# 转为简体
def Traditional2Simplified(sentence):
    sentence = Converter('zh-hans').convert(sentence)
    return sentence
# 转为繁体
def Simplified2Traditional(sentence):
    sentence = Converter('zh-hant').convert(sentence)
    return sentence

#去除停用词，返回去除停用词后的文本列表
def clean_stopwords(contents):
    contents_list=[]
    stopwords = {}.fromkeys([line.rstrip() for line in open('data/stopwords.txt', encoding="utf-8")]) #读取停用词表
    stopwords_list = set(stopwords)
    for row in contents:      #循环去除停用词
        words_list = jieba.lcut(row)
        words = [w for w in words_list if w not in stopwords_list]
        sentence=''.join(words)   #去除停用词后组成新的句子
        contents_list.append(sentence)
    return contents_list

# 将清洗后的文本和标签写入.csv文件中
def after_clean2csv(contents, labels): #输入为文本列表和标签列表
    columns = ['contents', 'labels']
    save_file = pd.DataFrame(columns=columns, data=list(zip(contents, labels)))
    save_file.to_csv('data/clean_data.csv', index=False, encoding="utf-8")