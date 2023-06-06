# utc_pytorch
Paddle通用文本分类UTC（Universal Text Classification）的pytorch实现.
### 模型加载
```
from transformers import BertTokenizer
from model import UTC

tokenizer = BertTokenizer.from_pretrained("rrsong/utc-base")

model = UTC.from_pretrained("rrsong/utc-base")
```

