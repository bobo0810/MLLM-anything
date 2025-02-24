# LLaMA Factory踩坑指南

## 1. DPO训练多模态相关参数

（1）训练参数  [link](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/trainers.html#dpo)

train_dpo.yaml的内容

```bash
# 训练配置
stage: dpo
pref_beta: 0.1
pref_loss: sigmoid  # choices: [sigmoid (dpo), orpo, simpo]
# 数据集配置
dataset: rlhf_v # dataset_info.json包含的名称
```

(2) 数据集配置和格式

dataset_info.json的部分内容，DPO训练时必须开启ranking参数

```json
  "rlhf_v": {
    "file_name": "/xxx/rlhf_v.json",
    "ranking": true, 
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected",
      "images": "images"
    }
  },
```

rlhf_v.json内容格式如下    [示例数据集](https://huggingface.co/datasets/llamafactory/RLHF-V/viewer?row=0)

```json
[
    {
        "images": ["/xxx/image_0.jpeg"],
        "conversations": [
            {
                "from": "human",
                "value": "<image>What are the key features you observe in the image?"
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": "A young man standing on stage wearing a white shirt and black pants."
        },
        "rejected": {
            "from": "gpt",
            "value": "A young man standing on stage wearing white pants and shoes."
        }
    },
    xxxxx
]
```

## 2. 冻结指定模块训练

train.yaml的内容

```yaml
finetuning_type: full  # 全参放开 支持3种类型full/freeze/lora
freeze_vision_tower: true # true表示冻结指定模块
```

src/llamafactory/model/model_utils/visual.py  控制冻结模块

```python
_register_composite_model(
    model_type="minicpmo", # 模型权重的config.json中的"model_type"值
    vision_model_keys=["vision_tower", "proj"], # 指定冻结的模块
    language_model_keys=["llm"],
)
```
