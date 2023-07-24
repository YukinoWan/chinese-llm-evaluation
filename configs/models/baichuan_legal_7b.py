from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        meta_template = dict(
            round=[
                dict(role='HUMAN', begin='Human: ', end='\n'),
                dict(role='BOT', begin='Assistant: ', end='\n', generate=True),
                ],
            begin="A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.",
            ),
        abbr='baichuan-legal_7b',
        path="baichuan-inc/baichuan-7B",
        tokenizer_path='baichuan-inc/baichuan-7B',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
