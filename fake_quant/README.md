To reproduce the results, you can run the following command:

```bash
python main.py --model /root/data/model/meta-llama/Meta-Llama-3-8b --a_bits 8 --v_bits 16 --k_bits 16 --w_bits 4 --wandb_id Meta-Llama-3-8B-4-8-16-gptq --wandb_project a8w4 --quant_func fpe2m2 --w_rtn
```