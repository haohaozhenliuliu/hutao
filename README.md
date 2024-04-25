

# 公考知识问答助手

![R-C](.\\image\R-C.jpg)

## 一. 介绍


## 二. OpenXlab 模型

&emsp;&emsp;公考知识问答助手使用的是InternLM的7B模型，模型参数量为7B，模型已上传，可以直接下载推理。

| 基座模型| 微调数据量 | 训练次数 | 下载地址 |
|:------:|:------:|:-------:|:---------|
|InternLM-chat-7b|46933 conversations|12 epochs|[xiaomile/ChineseMedicalAssistant_internlm](https://openxlab.org.cn/models/detail/xiaomile/ChineseMedicalAssistant_internlm)|

## 三. 数据处理

1. 原始数据样例

   ![处理后数据样例](.\\image\处理后数据样例.png)

2. 处理后数据样例

   ![处理前数据样例](.\\image\处理前数据样例.png)

3. 数据处理脚本

   ```python
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
   ```

   ```python
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
   ```

   

4. 公考知识问答助手 数据集采用中的从互联网中收集到理念公考常识题的数据，共计 40000 余条，数据集样例：

```text
input:世界上最大的宫殿是
output:故宫
input:世界上最靠北的首都是
output:雷克亚未克
input:蒙古首都乌兰巴托的意思是
output:红色英雄
input:世界上第一辆摩托车其主要材料是
output:木头
```



5. > 数据收集和整理过程

> 通过split.py脚本将准备好的src.docx文件中全部题目的题干和答案分离并写入新的src.xlsx文件中，其中题干写入第一列中，答案写入第二列中，经过分离后还需经过人工修正才能得到[最终的效果,并准备创建数据集。
>
> 使用transform.py脚本将xlsx文件转化成微调用的jsonl格式数据集`output.jsonl`。

## 四. 模型微调

&emsp;&emsp;使用 XTuner 训练， XTuner 有各个模型的一键训练脚本，很方便。且对 InternLM2 的支持度最高。通过微调帮助模型对公考一些常识题目有更清晰的认识.

- 微调流程

  ![Snipaste_2024-04-25_20-19-30](.\image\Snipaste_2024-04-25_20-19-30.png)

### XTuner

&emsp;&emsp;使用 XTuner 进行微调，具体脚本可参考`configs`文件夹下的脚本，脚本内有较为详细的注释。

|基座模型|配置文件|
|:---:|:---:|
|internlm-chat-7b|[internlm_chat_7b_qlora_oasst1_e3_copy.py](configs/internlm_chat_7b_qlora_oasst1_e3_copy.py)|

配置文件修改

![xtunerConfig](.\\image\xtunerConfig.png)

微调方法如下：


1. 找到config文件夹中配置文件internlm_chat_7b_qlora_oasst1_e3_copy，将模型地址`pretrained_model_name_or_path`和数据集地址`data_path`修改成自己的相应文件的地址，其他参数根据自己的需求修改，然后就可以开始微调。

   ```bash
   xtuner train /root/InternLM-GK/config/internlm_chat_7b_qlora_oasst1_e3_copy.py
   ```

2. 将保存的 `.pth` 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 HuggingFace Adapter 模型，即：生成 Adapter 文件夹：

   ```bash
   # 创建用于存放Hugging Face格式参数的hf文件夹
   mkdir /root/InternLM-GK/config/work_dirs/hf
   
   export MKL_SERVICE_FORCE_INTEL=1
   
   # 配置文件存放的位置
   export CONFIG_NAME_OR_PATH=/root/InternLM-GK/config/internlm_chat_7b_qlora_oasst1_e3_copy.py
   
   # 模型训练后得到的pth格式参数存放的位置
   export PTH=/root/InternLM-GK/work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_3.pt
   
   # pth文件转换为Hugging Face格式后参数存放的位置
   export SAVE_PATH=/root/InternLM-GK/work_dirs/hf
   
   # 执行参数转换
   xtuner convert pth_to_hf $CONFIG_NAME_OR_PATH $PTH $SAVE_PATH
   ```

3. 将 HuggingFace Adapter 模型合并入 HuggingFace 模型：

    ```bash
    export MKL_SERVICE_FORCE_INTEL=1
    export MKL_THREADING_LAYER='GNU'
    
    # 原始模型参数存放的位置
    export NAME_OR_PATH_TO_LLM=/root/InternLM-GK/model/Shanghai_AI_Laboratory/internlm-chat-7b
    
    # Hugging Face格式参数存放的位置
    export NAME_OR_PATH_TO_ADAPTER=/root/InternLM-GK/work_dirs/hf
    
    # 最终Merge后的参数存放的位置
    mkdir /root/InternLM-GK/work_dirs/hf_merge
    export SAVE_PATH=/root/InternLM-GK/work_dirs/hf_merge
    
    # 执行参数Merge
    xtuner convert merge \
        $NAME_OR_PATH_TO_LLM \
        $NAME_OR_PATH_TO_ADAPTER \
        $SAVE_PATH \
        --max-shard-size 2GB
    ```

### Chat

微调结束后可以使用xtuner查看对话效果

```shell
xtuner chat .work_dirs/hf_merge --prompt-template internlm_chat
```



### 本地网页部署

微调结束后可以使用xtuner查看对话效果

```shell
streamlit run /root/InternLM-GK/code/InternLM/web_demo.py --server.address 127.0.0.1 --server.port 6006
```

![web_chat](.\\image\web_chat.png)

## 五. LmDeploy部署

![LMDeploy](.\\image\LMDeploy.png)

- 首先安装LmDeploy

```shell
pip install -U 'lmdeploy[all]==v0.1.0'
```

- 然后转换模型为`turbomind`格式。使用 TurboMind 推理模型需要先将模型转化为 TurboMind 的格式，，目前支持在线转换和离线转换两种形式。TurboMind 是一款关于 LLM 推理的高效推理引擎，基于英伟达的 FasterTransformer 研发而成。它的主要功能包括：LLaMa 结构模型的支持，persistent batch 推理模式和可扩展的 KV 缓存管理器。
本项目采用离线转换，需要在启动服务之前，将模型转为 lmdeploy TurboMind 的格式，如下所示。

```shell
lmdeploy convert internlm-chat-7b  /root/InternLM-GK/work_dirs/hf_merge/
```
执行完成后将会在当前目录生成一个 workspace 的文件夹。

- LmDeploy Chat对话。模型转换完成后，我们就具备了使用模型推理的条件，接下来就可以进行真正的模型推理环节。
  本地对话（Bash Local Chat）模式，它是跳过API Server直接调用TurboMind。简单来说，就是命令行代码直接执行 TurboMind。
```shell
lmdeploy chat turbomind ./workspace
```

​		网页Demo演示。本项目采用将TurboMind推理作为后端，将Gradio作为前端Demo演示。

```shell
lmdeploy serve gradio ./workspace #转换后的turbomind模型地址
```
​		就可以直接启动 Gradio，此时没有API Server，TurboMind直接与Gradio通信。

## 六. OpenCompass 评测

- 安装 OpenCompass

```shell
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

- 支持数据集

  ![support-data](.\image\support-data.png)
  
- 在opencompass/configs目录下新建自定义数据集测评配置文件 `eval_internlm_7b_custom.py` 和 `eval_internlm_chat_turbomind_api_custom.py`

  /root/personal_assistant/config/work_dirs/hf_merge

```python
from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

with read_base():
    from .summarizers.medium import summarizer

datasets = [
    {"path": "/root/data/testdata/hutao/output.jsonl", "data_type": "qa", "infer_method": "gen"}, # your custom dataset
]

internlm_chat_7b = dict(
       type=HuggingFaceCausalLM,
       # `HuggingFaceCausalLM` 的初始化参数
       path='/root/personal_assistant/config/work_dirs/hf_merge', # your model path
       tokenizer_path='/root/personal_assistant/config/work_dirs/hf_merge', # your model path
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto',trust_remote_code=True),
       # 下面是所有模型的共同参数，不特定于 HuggingFaceCausalLM
       abbr='internlm_chat_7b',               # 结果显示的模型缩写
       max_seq_len=2048,             # 整个序列的最大长度
       max_out_len=100,              # 生成的最大 token 数
       batch_size=64,                # 批量大小
       run_cfg=dict(num_gpus=1),     # 该模型所需的 GPU 数量
    )

models=[internlm_chat_7b]
```

```python
from mmengine.config import read_base
from opencompass.models.turbomind_api import TurboMindAPIModel

with read_base():
    from .summarizers.medium import summarizer

datasets = [
    {"path": "/root/data/testdata/hutao/output.jsonl", "data_type": "qa", "infer_method": "gen"}, # your custom dataset
]


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
    eos_token_id=103028)

models = [
    dict(
        type=TurboMindAPIModel,
        abbr='internlm-chat-7b-turbomind',
        path="./model/workspace_4bit",
        api_addr='http://0.0.0.0:23333',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
```

- 评测启动！

```shell
python run.py configs/eval_internlm_7b_custom.py
```

- 量化评测，先启动turbomind作为服务端

```shell
lmdeploy serve api_server ./workspace_4bit --server_name 0.0.0.0 --server_port 23333 --instance_num 64 --tp 1
```

```shell
python run.py eval_internlm_chat_turbomind_api_custom.py
```

评测结果

![openCompass_result](.\\image\openCompass_result.png)

## 致谢

<div align="center">

***感谢上海人工智能实验室组织的 书生·浦语实战营 学习活动~***

***感谢 OpenXLab 对项目部署的算力支持~***

***感谢 浦语小助手 对项目的支持~***
</div>



![internstudio_logo](.\image\internstudio_logo.svg)
