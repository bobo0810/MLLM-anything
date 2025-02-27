# Transformers基础



## Transformers库

### Tokenizer用法


```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/xxx/xxx") # 加载tokenizer

# 对文本进行编码,生成token_id
text = "这是一个示例文本。"
encoded_text = tokenizer.encode(text)

# 将编码后的序列id解码为原始文本
decoded_text = tokenizer.decode(encoded_text)

print("原始文本：", text)
print("解码后的文本：", decoded_text)
```







## 名词解释

### 全局标记BOS  EOS 

- BOS 是 “Beginning of Sentence” 的缩写，告诉模型何时开始生成，表示句子的开始。
- EOS 是 “End of Sentence” 的缩写，告诉模型何时停止生成，表示句子的结束。

### 局部标记<|im_start|>   <|im_end|>

- <|im_start|>  表示消息开始，缩写“input message start”。
- <|im_end|>   表示消息结束，缩写“input message end”。

### 两者区别

|                                | 用途                                       |
| ------------------------------ | ------------------------------------------ |
| BOS  EOS                       | 用于**序列生成**，标记整个文本的开始和结束 |
| <\|im_start\|>    <\|im_end\|> | 用于**对话生成**，区分对话中的不同角色     |

```
<|im_start|>user
Hi, how are you? <|im_end|>
<|im_start|>assistant
I'm good, thank you! How can I help you today? <|im_end|>
<|im_start|>user
Can you help me with a math problem? <|im_end|>
```

```
<BOS> <|im_start|>user
Hi, how are you? <|im_end|> <|im_start|>assistant
I'm good, thank you! How can I help you today? <|im_end|> <|im_start|>user
Can you help me with a math problem? <|im_end|> <EOS>
```

### Special Token

| **Special Token** | **含义**             | **用途**                                                     |
| ----------------- | -------------------- | ------------------------------------------------------------ |
| `[UNK]`           | Unknown Token        | 未知标记，用于表示输入序列中不在词汇表中的单词。             |
| `[CLS]`           | Classification Token | 输入序列的开始。                                             |
| `[SEP]`           | Separator Token      | 输入序列的结束。                                             |
| `[PAD]`           | Padding Token        | 用于填充短序列，使得所有序列长度一致。                       |
| `[MASK]`          | Mask Token           | 在掩码语言建模任务中，用于掩盖部分词汇，模型需要预测这些词。 |

### Base/Instruction/Chat区别

| 类型            | 经历阶段                     | 特点                                                         |
| --------------- | ---------------------------- | ------------------------------------------------------------ |
| Base模型        | Pretrain                     | 经过大规模数据的无监督学习，掌握基本的语言生成能力和背景知识，具备语言理解能力。 |
| Instruction模型 | Pretrain + SFT + RLHF        | 经过监督微调和强化学习，理解和执行复杂的自然语言指令。       |
| Chat模型        | Pretrain + SFT + RLHF + 微调 | 经过进一步微调，专注于对话的上下文理解能力和多轮交流连贯性。 |

### 输入参数input_ids、attention_mask

| **参数**             | **含义**                 | **作用**                                                    |
| -------------------- | ------------------------ | ----------------------------------------------------------- |
| **`input_ids`**      | 输入文本的 token ID 序列 | 表示文本的实际内容，用数字标识每个词汇或子词。              |
| **`attention_mask`** | 注意力掩码               | 哪些token需要关注，哪些是填充token 应该被忽略。1有效、0忽略 |

### chat_template教程

https://huggingface.co/docs/transformers/zh/chat_templating

### QwenModel 和 QwenForCausalLM两者区别

| 模型            | 适用场景 | 作用                                                         |
| --------------- | -------- | ------------------------------------------------------------ |
| QwenModel       | 特征提取 | 基础类，仅通过输入的 token 序列生成隐藏层的输出，不包含特定任务的头部head。 |
| QwenForCausalLM | 文本生成 | 模型类，用于因果语言建模（Causal Language Modeling）。 除了包含QwenModel，额外添加用于生成和语言建模任务的解码头部，接受输入并预测后续的token序列。 |

### LLaVA模型架构图

![image-20250226103138835](./assets/image-20250226103138835.png)

