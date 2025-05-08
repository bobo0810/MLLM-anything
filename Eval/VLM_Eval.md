# VLMEvalKit-多模评测库

## OpenCompass榜单

基于VLMEvalKit多模评测库进行评测，涵盖8个评测集，

- 规则评分：MMMU   MMStar   AI2D  OCRBench
- GPT评分：MMVet  MathVista_MINI   HallusionBench   （默认GPT-4-Turbo-1106）
- 提交官网评分： MMBench

## 裁判模型适配GPT接口

背景：评测库支持OpenAI  API判分，但目前只有内部封装后的API，无法直接使用，需要适配。

方案：`vlmeval/api/gpt.py`修改`generate_inner`方法

```python
def generate_inner(self, inputs, **kwargs) -> str:
    xxx
    #----------------------自定义GPT API-------------------------------
    prompt= input_msgs[0]["content"][0]["text"] # 无任何格式的正常文本 
    
    # 请求参数
    payload = {
        "model": 'gpt-4o-xxx',  # 模型名称
        "prompt": prompt, 
        "params": {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }
    headers = {
        "Content-Type": "application/json",
        "token": "xxxxxxx"  # 使用提供的 API 密钥
    }
    url = "https://xxxx.com/v1/api/chat"  

    # 请求重试机制
    for i in range(self.retry):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
            response_data = response.json()
            if response_data["code"] == 0:
                # 返回成功结果
                answer= response_data["xxx"].strip()
                break
            else:
                time.sleep(1)
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Request failed: {e}")
            time.sleep(2)

    ret_code = response_data["code"]
    print("answer---------->",answer)
    return ret_code, answer, response
```

## 支持新模型

（1）模型定义和推理

`vlmeval/vlm`目录下新建`MyModel.py`，内容如下。随后在`opencompass/models/__init__.py`导入MyModel类

```python
from .base import BaseModel
from ..smp import *

class MyModel(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False # 是否支持图文交错
    VIDEO_LLM = False # 是否支持视频评测
    def __init__(self,model_path, **kwargs):
        # 加载模型
        self.model=xxx
        self.tokenizer=xx
    def generate_inner(self, message, dataset=None):
        user_question = '\n'.join([msg['value'] for msg in message if msg['type'] == 'text'])
        images = [Image.open(msg['value']).convert('RGB') for msg in message if msg['type'] == 'image']
        xxx
        return response
```

（2）模型配置

`vlmeval/vlm/__init__.py`导入该模型

```python
from .MyModel import MyModel
```

`vlmeval/config.py`配置该模型

```python
my_series = {
    'MyModel': partial(MyModel, model_path='xxx'),
}
model_groups = [xxx, my_series ,xxx]
```

（3）开启评测

```bash
torchrun   --nproc-per-node=1 run.py  \
    --model MyModel --verbose \
    --data  MMMU_DEV_VAL
```

参考 [官方文档](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/Development.md)



