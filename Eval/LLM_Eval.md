# OpenCompass-LLM评测库

## 支持新模型

（1）模型定义和推理

`opencompass/models`目录下新建`MyModel.py`，内容如下。

```python
from opencompass.models.base import BaseModel
class MyModel(BaseModel):
    def __init__(self,path):
        self.template_parser=LMTemplateParser(meta_template)
        # 加载模型
        self.model=xxx
        self.tokenizer=xx
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        xxx
        return out # list类型  
```

然后在`opencompass/models/__init__.py`导入MyModel类

（2）模型配置

`opencompass/configs/models/`目录下新建`MyModel.py`，内容如下。

```python
from opencompass.models import QwenOmni
models = [
    dict(
        # 以下参数为初始化类需要的参数
        type=MyModel, # 模型类名
        path='/xxx/xxx', # 模型权重路径
        max_seq_len=2048,
        # 以下参数为各类模型都必须设定的参数，非HuggingFaceCausalLM的初始化参数
        abbr='xxx', # 模型简称，用于结果展示
        max_out_len=1024,    # 最长生成token数
        batch_size=1,        # 批次大小
        run_cfg=dict(num_gpus=1), # 运行模型所需的GPU数
    )
]
```

（3）评测配置

`opencompass/configs/`目录下新建`MyEval.py`，内容如下。

```python
from mmengine.config import read_base
with read_base():
    from .datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    from .datasets.demo.demo_math_chat_gen import math_datasets
    from .models.MyModel import MyModel as my_model

datasets = gsm8k_datasets + math_datasets # 选定要评测的评测集
models = my_model  # 选定要评测的模型
```

（4）开启评测

根目录下执行  `python run.py  opencompass/configs/MyEval.py  --max-num-worker 6`。表示并行开启6个任务，加速评测。评测日志默认保存在`outputs`目录下

参考：[官方文档—支持新模型](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/new_model.html)
