from mmengine.config import read_base

with read_base():
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .models.llama_alpaca_7b import models


datasets = [*siqa_datasets]
