from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        meta_template = dict(
            round=[
                dict(role='HUMAN', begin='### Instruction:\n', end='\n\n'),
                dict(role='BOT', begin='### Response:\n', end='\n\n', generate=True),
                ],
            begin="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
            ),
        abbr='llama-alpaca_7b',
        path="/home/zhen/models/llama-alpaca-lora-7b",
        tokenizer_path='/home/zhen/models/llama-alpaca-lora-7b',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
