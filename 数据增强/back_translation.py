# 假设取两条已经存在的正样本和两条负样本
# 将基于这四条样本产生新的同标签的四条样本
p_sample1 = "酒店设施非常不错"
p_sample2 = "这家价格很便宜"
n_sample1 = "拖鞋都发霉了, 太差了"
n_sample2 = "电视不好用, 没有看到足球"

# 导入google翻译接口工具
from google_trans_new import google_translator
"""
由于google翻译对接口进行了更新，之前用的googletrans已经不能用了。
因此我们要使用网上大神已经开发出的新方法：https://github.com/lushan88a/google_trans_new
使用该库需要先安装request模块，在运行代码中有报错，可以参考以下链接寻求解决办法：
https://blog.csdn.net/a1397852386/article/details/120742059
https://github.com/lushan88a/google_trans_new/issues/36
"""

# 实例化翻译对象
translator = google_translator()

print(translator.translate('안녕하세요.'))


# 进行第一次批量翻译, 翻译目标是韩语
translations = translator.translate([p_sample1, p_sample2, n_sample1, n_sample2], lang_tgt='ko', lang_src='zh')
"""
lang_src是原来文本语言，lang_tgt是目标文本语言，默认为英语。需要填写符合googletrans.LANGUAGES的，参数未被识别出则默认为英语
"""
print("打印中间结果：")
print(translations)



# 最后在翻译回中文, 完成回译全部流程
cn_res = translator.translate(translations, lang_tgt='zh',lang_src='ko')
print("回译得到的增强数据:")
print(cn_res)