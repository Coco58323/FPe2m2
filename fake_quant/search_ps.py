import gc
import torch
import torch.nn as nn

import tqdm
from typing import List

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from quant_utils import sym_quant_fpe2m2_ps


def auto_config_search(module):
    def _auto_get_pow_scale(weight):
        pow_scale = []
        recon_loss = []
        # for j in tqdm.tqdm(range(40,101), desc="Searching for power scale"):
        for j in range(50,121):
        # for i in tqdm.tqdm(range(weight.shape[0]), desc="Searching for power scale"):
            
            scale = j/100
            w = sym_quant_fpe2m2_ps(weight,power_scale=scale)
            loss = (weight-w).float().pow(2).mean(dim=-1,keepdim=True)
            recon_loss.append(loss)
        recon_loss = torch.cat(recon_loss, dim=-1)
        _, best_idx = torch.min(recon_loss, dim=-1,keepdim=True)
        pow_scale = (best_idx+50)/100
        # print(f"Best power scale: {pow_scale}")
        return pow_scale

    scales = {}
        # attention input
    scales["q_proj"] = _auto_get_pow_scale(module.self_attn.q_proj.weight)
    scales["k_proj"] = _auto_get_pow_scale(module.self_attn.k_proj.weight)
    scales["v_proj"] = _auto_get_pow_scale(module.self_attn.v_proj.weight)
    gc.collect()
    torch.cuda.empty_cache()
    # attn out
    scales["o_proj"] = _auto_get_pow_scale(module.self_attn.o_proj.weight)
    gc.collect()
    torch.cuda.empty_cache()
    # fc1
    scales["gate_proj"] = _auto_get_pow_scale(module.mlp.gate_proj.weight)
    scales["up_proj"] = _auto_get_pow_scale(module.mlp.up_proj.weight)
    gc.collect()
    torch.cuda.empty_cache()
    # fc2
    scales["down_proj"] = _auto_get_pow_scale(module.mlp.down_proj.weight)
    gc.collect()
    torch.cuda.empty_cache()

    return scales



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    layers = model.model.layers
    return layers


@torch.no_grad()
def run_config_search(
    model,
):
    layers = get_blocks(model)
    quant_func = {}
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running Searching..."):
        layer = layers[i]
        layer = layer.cuda()

        func_dict = auto_config_search(
            layer,
        )
        quant_func[i] = func_dict


        torch.cuda.empty_cache()

        layer = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    return quant_func

if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    model_path = "/root/data/model/meta-llama/Llama-2-13b-hf"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
    config.use_cache = False

    enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, **kwargs
    )

    model.eval()
    quant_func = run_config_search(model)
    print(quant_func)