import re
import os

# 定义输入的 srt 文件名
input_filename = 'video/B_HeiTianEr.srt'

# 定义时间戳的正则表达式模式（匹配形如 00:00:00,000 --> 00:00:04,460 的行）
timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}')

# 读取 srt 文件所有行
with open(input_filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 存放清洗后的文本
cleaned_lines = []

# 遍历每一行进行判断
for line in lines:
    line = line.strip()  # 去除首尾空白字符
    if not line:
        continue  # 跳过空行
    # 跳过仅包含数字的行（序号行）
    if line.isdigit():
        continue
    # 跳过时间戳行
    if timestamp_pattern.match(line):
        continue
    # 跳过包含 "< No Speech >" 的行
    if '< No Speech >' in line:
        continue
    # 剩下的就是需要保留的内容
    cleaned_lines.append(line)

# 生成输出文件名：在原文件名的基础上加上 _clean.txt
base, ext = os.path.splitext(input_filename)
output_filename = f"{base}_clean.txt"

# 将清洗后的结果写入到输出文件
with open(output_filename, 'w', encoding='utf-8') as out_file:
    for text in cleaned_lines:
        out_file.write(text + "\n")

print(f"清洗后的文件已保存为: {output_filename}")
