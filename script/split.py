import pandas as pd
from docx import Document

# 替换为你的Word文件路径
word_file_path = "..\\data\\src.docx"
# 替换为你想要的Excel文件路径
excel_file_path = "..\\data\\src.xlsx"
# 定义分割符号，这里以换行符为例
delimiter = '：->'

# 加载Word文档
document = Document(word_file_path)

# 初始化一个空列表来存储分割后的数据
data_list = []

# 遍历Word文档中的每个段落
for para in document.paragraphs:
    # 去除段落中的多余空格，并按照特定符号分割
    split_data = para.text.strip().split(delimiter)
    # 确保分割后的数据至少有两个部分
    if len(split_data) >= 2:
        # 将分割后的两部分数据作为一行添加到列表中
        data_list.append(split_data[:2])  # 只取前两个部分，放入两列

# 将数据转换为pandas DataFrame
# 假设每个分割后的部分都应该作为单独的一列
df = pd.DataFrame(data_list, columns=['Column1', 'Column2'])

# 将DataFrame保存为Excel文件
df.to_excel(excel_file_path, index=False) 

print(f'数据已成功导入到 {excel_file_path}')