To reproduce the results, you can run the following command:

```bash
python main.py --model /data/ke.yi/model/meta-llama/Meta-LLaMA-3-8B --a_bits 8 --v_bits 16 --k_bits 16 --w_bits 4 --wandb_id Meta-Llama-3-8B-4-8-16-gptq --wandb_project a8w4 --quant_func fpe2m2 --w_rtn
```

To get quantized model, you can run the following command:

```bash
python main.py --model /data/ke.yi/model/meta-llama/Meta-LLaMA-3-8B --a_bits 8 --v_bits 16 --k_bits 16 --w_bits 4 --wandb_id Meta-Llama-3-8B-4-8-16-gptq --wandb_project a8w4 --quant_func fpe2m2 --w_rtn --save_qmodel_path {path_to_save_quantized_model}
```

To resume from a checkpoint, you can run the following command:

```python
def get_model(model_name, hf_token=None):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                                use_auth_token=hf_token,
                                                                low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model

model = get_model({path_to_origin_model}, None)
model.load_state_dict(torch.load({path_to_quantized_model}))
```
