import openpyxl
import json
from datetime import time

# 输入文件名
input_file_name = '..\\data\\src.xlsx'
# 输出文件名
output_file_name = '..\\data\\output.jsonl'

# 加载 Excel 文件
wb = openpyxl.load_workbook(input_file_name)
sheet = wb.active

# 初始化一个空列表来保存对话
conversations = []

# 遍历 Excel 工作表，提取数据并构建 JSONL 格式的列表
for row in sheet.iter_rows(min_row=2, values_only=True):  # 假设第一行是标题行，从第二行开始读取数据
    input_text = str(row[0]) if row[0] is not None else ''  # 将值转换为字符串，以处理 time 类型
    output_text = str(row[1]) if row[1] is not None else ''  # 同上

    # 将提取到的数据添加到 conversations 列表中
    conversation = {
        "conversation": [
            {
                "system": "你是一位专业、经验丰富的公考常识题问答助手。你始终根据用户的问题提供准确、全面和详细的答案",
                "input": input_text,
                "output": output_text
            }
        ]
    }

    conversations.append(conversation)

# 将构建的 conversations 列表转换为 JSONL 格式的字符串列表
jsonl_data = [json.dumps(conversation, ensure_ascii=False) for conversation in conversations]

# 将 JSONL 数据写入到输出文件中
with open(output_file_name, 'w', encoding='utf-8') as f:
    for line in jsonl_data:
        f.write(line + '\n')

print(f'数据已成功写入 {output_file_name}')