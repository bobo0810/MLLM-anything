# VLMEvalKit多模评测库

## OpenCompass榜单

基于VLMEvalKit多模评测库进行评测，涵盖8个评测集，

- 规则评分：MMMU   MMStar   AI2D  OCRBench
- GPT评分：MMVet  MathVista_MINI   HallusionBench  
- 提交官网评分： MMBench

## 适配GPT接口

背景：评测库支持OpenAI  API判分，但目前只有内部封装后的API，无法直接使用，需要适配。

方案：`vlmeval/api/gpt.py`修改`generate_inner`方法

```python
def generate_inner(self, inputs, **kwargs) -> str:
    xxx
    #----------------------自定义GPT API-------------------------------
    prompt= input_msgs[0]["content"][0]["text"] # 纯文本 
    
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

