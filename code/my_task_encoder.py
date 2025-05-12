from dataclasses import dataclass
from typing import Callable, Tuple, Union, List, Dict, Any, Optional
import torch
from megatron.energon import CaptioningSample, DefaultTaskEncoder, batch_list, batch_stack
from megatron.energon import WorkerConfig
worker_config = WorkerConfig.default_worker_config() # 获取当前Rank状态

# Type for intermediate batch, after batching operation  多个样本打包后的batch格式
@dataclass
class CaptioningRawBatch:
    # (n,)
    __key__: List[str]
    __restore_key__: Tuple[Union[str, int], ...]
    __subflavor__: List[str]
    __subflavors__: List[Dict[str, Any]]
    # (n, c, h, w)
    image: torch.Tensor
    # (n,)
    caption: List[str]


# Typing for the resulting batch data  模型输入的batch格式
@dataclass
class CaptioningBatch:
    __keys__: List[str]
    # (n, c, h, w)
    images: torch.Tensor
    # (n, c)
    text_tokens: torch.Tensor
    # (n, c, c)
    text_attn_mask: torch.Tensor


# All the typing is optional
class CaptioningTaskEncoder(
    DefaultTaskEncoder[CaptioningSample, CaptioningSample, CaptioningRawBatch, CaptioningBatch]
):
    """A simple task encoder for captioning."""
    # TaskEncoder需要传入4个参数[T_sample原始样本格式, T_encoded_sample预处理后的样本格式, T_raw_batch多个样本打包后的batch格式, T_batch模型输入的batch格式]

    def __init__(
        self,
        tokenizer,
        image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_length: int = 128,
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by overwriting the `batch`
        # method)
        super().__init__(batch_type=CaptioningRawBatch)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def encode_sample(self, sample: CaptioningSample) -> CaptioningSample:
        '''对原始样本预处理'''
        sample.image = self.image_transform(sample.image) 
        return sample
    
    def batch(self, samples: List[CaptioningSample]) -> CaptioningRawBatch:
        '''将encode_sample方法预处理后的单个样本转为batch'''
        # Batch the samples 
        # The actions dict specifies how to batch each field of the sample. In addition to these, you may use 
        # `batch_pad_stack` as well.
        # By default, `batch_pad_stack` is used for all tensor fields, and `batch_list` is used for all non-tensor 
        # fields. This example matches the default implementation (not overwriting the `batch` method).
        # filtered_samples = [
        #     CaptioningSample(
        #         __key__=s.__key__,
        #         image=s.image,
        #         caption=s.caption,
        #     ) for s in samples
        # ]
        return self._batch(samples, result_type=CaptioningRawBatch, actions={"image": batch_stack, "caption": batch_list})

    def encode_batch(self, batch_data: CaptioningRawBatch) -> CaptioningBatch:
        '''打包后的batch转为模型输入的batch'''
        # Run the encoder on the batch of captions.
        tokenized = self.tokenizer(batch_data.caption)
        # Return the final batch, going into the network
        return CaptioningBatch(
            __keys__=batch_data.__key__,
            images=batch_data.image,
            text_tokens=tokenized["input_ids"],
            text_attn_mask=tokenized["attention_mask"],
        )



# ------------------------    
from torchvision import transforms
from transformers import AutoTokenizer
from megatron.energon import get_loader, get_train_dataset

# 自定义图像预处理
train_img_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
)

train_loader = get_loader(get_train_dataset(
    'xxx/demo_tar',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config = worker_config,
    task_encoder=CaptioningTaskEncoder(
        tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'),
        image_transform=train_img_transform,
    ),
    # Change this to set how images are decoded.
    # E.g. "pil" is another commonly used valid option.
    # See `webdataset.imagehandler` for more options.
    image_decode="torchrgb",
))

for data in train_loader:
    # data is a CaptioningBatch 已处理好的模型输入batch
    print(data)