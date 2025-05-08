

# UltraEval-Audio语音评测库

## 支持新模型

参考https://github.com/OpenBMB/UltraEval-Audio/blob/main/docs%2Fhow%20eval%20your%20model.md

```html
├── audio_evals/
│   ├── models/
│   │   └── mymodel.py # 模型定义和infer的具体实现 
├── registry/
│   └── model/
│       └── mymodel.yaml # 确定模型权重和生成配置
│   └── prompt/
│       └── mymodel.yaml # prompt细节
```

（1）`audio_evals/models`/目录下新建mymodel.py   

```python
from audio_evals.models.model import Model
from audio_evals.base import PromptStruct

class MyModel(Model):
    def __init__(self, path: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params) # sample_params参数传给infer方法的kwargs
        logger.debug("start load model from {}".format(path))
        self.model = xxx
        self.processor=xxx


    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        # 解析registry/prompt/mymodel.yaml格式的数据，构造格式进行推理
        return response
```

（2）`registry/model/`目录下新建mymodel.yaml  

```yaml
MyModel: # 模型名，bash用
  class: audio_evals.models.mymodel.MyModel  # 模型定义层级
  args:
    path: /xxx/xx # 模型权重路径
    sample_params:  # 以下是生成配置,会传给MyModel的__init__(sample_params)
      num_beams: 3
      top_k: 20
      top_p: 0.5
      temperature: 0.7
      repetition_penalty: 1.0
      do_sample: false
      max_new_tokens: 256
      min_new_tokens: 1
```

（3）registry/prompt/目录下新建mymodel.yaml   

```yaml
mymodel-asr-en: #prompt名称
  class: audio_evals.prompt.base.Prompt
  args:
    template:
    - role: user
      contents:
      - type: text
        value: 'Transcribe the audio into plain text' # prompt内容
      - type: audio
        value: '{{WavPath}}'
```

（4）开启评估

```bash
prompt="mymodel-asr-en"  #yaml文件的prompt名称
model="MyModel"  #yaml文件的模型名称
dataset="librispeech-test-clean" # 指定评测集
python audio_evals/main.py --dataset $dataset --prompt $prompt --model $model
```



## 支持新数据集

参考https://github.com/OpenBMB/UltraEval-Audio/blob/main/docs%2Fhow%20add%20a%20dataset.md

（1）准备数据data.jsonl

```json
{"WavPath": "path/to/audio2.wav", "gt": "this is the first audio"}
```

（1）registry/dataset/目录下新建mydataset.yaml

```bash
mydataset: # 评测集名称
  class: audio_evals.dataset.dataset.RelativePath
  args:
    default_task: asr-zh # 所属评测任务类别
    f_name: /xx/data.jsonl # 评测集路径
    file_path_prefix: /aaa/bbb  # jsonl中WavPath前缀
    ref_col: gt # jsonl中真值标签对应的key名
```

（2）开启评测

```bash
dataset="mydataset" # yaml文件的评测集名称
python audio_evals/main.py --dataset $dataset --prompt xxx --model xxx
```

