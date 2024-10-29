import gc
import torch
import torch.nn as nn

import tqdm
import functools
from collections import defaultdict
from typing import List

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

@torch.no_grad()
def auto_config_search(module, module_kwargs,input_feat):
    from quant_utils import sym_quant_fpe2m2_ps
    
    # find the best scale ratio
    def _search_module_scale(block, linears2quant: list, x, kwargs={}):
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]
        # candidate_func = [sym_quant_fpe2m2, sym_quant_fpe2m2_autoscale]
        history = []
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        best_func_name = []
        for fc in linears2quant:
            best_error = float("inf")
            for idx in range(1,20,1):
                power_scale = idx/100
                fc.weight.data = sym_quant_fpe2m2_ps(fc.weight.data,power_scale=power_scale)

                out = block(x, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]

                loss = (
                    (org_out - out).float().pow(2).mean().item()
                )
                print(loss)
                history.append(loss)
                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    temp_power_scale = power_scale
                gc.collect()
                torch.cuda.empty_cache()
            block.load_state_dict(org_sd)
            best_func_name.append(temp_power_scale)
            print(f"Best quantization function: {temp_power_scale}")
        # del best_func_name
        gc.collect()
        torch.cuda.empty_cache()
        # return [None]*len(linears2quant)
        return best_func_name
    def _auto_get_quant_func(layers, inp, module2inspect=None, kwargs={}):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]
        
        quant_func = _search_module_scale(module2inspect, layers, inp, kwargs)
        return quant_func

    func_name = {}
    if isinstance(module, LlamaDecoderLayer):
        # attention input
        
        temp = _auto_get_quant_func(
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        func_name["q_proj"] = temp[0]
        func_name["k_proj"] = temp[1]
        func_name["v_proj"] = temp[2]
        gc.collect()
        torch.cuda.empty_cache()
        # attn out
        temp =  _auto_get_quant_func(
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
        func_name["o_proj"] = temp[0]
        gc.collect()
        torch.cuda.empty_cache()
        # fc1
        temp=    _auto_get_quant_func(
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        func_name["gate_proj"] = temp[0]
        func_name["up_proj"] = temp[1]
        gc.collect()
        torch.cuda.empty_cache()
        # fc2
        func_name["down_proj"] = _auto_get_quant_func(
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )[0]
        gc.collect()
        torch.cuda.empty_cache()

    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return func_name



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        from datasets import load_dataset
        dataset = load_dataset("/root/data/dataset/mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


@torch.no_grad()
def run_config_search(
    model,
    enc,
    n_samples=512,
    seqlen=512,
    # some configs for ablation study
    calib_data="pileval",
):

    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()
    quant_func = {}
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        torch.cuda.empty_cache()

        func_dict = auto_config_search(
            layer,
            layer_kwargs,
            input_feat=input_feat,
        )
        quant_func[i] = func_dict


        # Clear GPU memory
        torch.cuda.empty_cache()

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
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
    quant_func = run_config_search(model, enc, n_samples=512, seqlen=512, calib_data="pileval")
    print(quant_func)