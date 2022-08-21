# langconv
langconv这一个模块的功能：对中文字符串进行繁体和简体中文的转换。
在参考[博客——中文自然语言处理](https://asialee.blog.csdn.net/article/details/94208759)编写代码的时候，发现使用传统的
`pip install langconv`无法安装所需模块，在pypi.org中也找不到模块langconv，因此
按照[教程](https://www.cnblogs.com/qytang/p/5588205.html)需要先下载zh_wiki.py及langconv，这里给出二者下载链接：
- langconv文件的源码地址：https://github.com/skydark/nstools/blob/master/zhtools/langconv.py
- zh_wiki文件下载：https://github.com/skydark/nstools/blob/master/zhtools/zh_wiki.py
python导入其他文件中的py格式文件的方式：
```python
#直接导入同一目录下的py文件
import zh_wiki

"""
从同级的文件夹中导入子文件，这种方式要求文件夹zhtool下有一为空的__init__.py文件
"""
from zhtool.zh_wiki import zh_wiki
```
在Pycharm中，如果以上做法无法生效，可以考虑采用以下两个办法：
1. 在文件开头加入`import sys`和`sys.path.append( ' ' )`
2. "File --- Setting --- Project: xxx --- Project Structure --- Add Content Root"将文件夹设置为资源路径，也可以右击文件夹完成这一步操作