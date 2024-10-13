import re

# 定义匹配中文标点符号的正则表达式
# 中文标点符号集合：^[。！？：；，、（）《》“”‘’——……]
CHINESE_PUNCTUATION_PATTERN = re.compile(r'^[。！？：；，、]')


# 判断字符串是否以中文标点符号开头
def is_starts_with_chinese_punctuation(s):
    return bool(CHINESE_PUNCTUATION_PATTERN.match(s))