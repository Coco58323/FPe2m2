import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import gptq_utils
import eval_utils
import search_ps

def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_id)
        wandb.config.update(args)

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    if model.lm_head.weight.device == torch.device('meta'):
        import torch.nn as nn
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.clone())


    if args.power_scale:
        power_scales = search_ps.run_config_search(model)
    else:
        power_scales = None
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict)
        elif not args.w_rtn: # GPTQ Weight Quantization
            # assert "llama" in args.model, "Only llama is supported for GPTQ!"
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            _ = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args,power_scales=power_scales)
        else: # RTN Weight Quantization
            _ = gptq_utils.rtn_fwrd(model, utils.DEV, args,power_scales=power_scales)
            
        if args.save_qmodel_path:
            torch.save(model.state_dict(), args.save_qmodel_path)

    quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)
                qlayers[name].out_runtime_smooth = args.a_runtime_smooth
            
            if 'k_proj' in name and args.k_bits < 16: #Set the k_proj precision
                qlayers[name].out_quantizer.configure(bits=args.k_bits,
                                              groupsize=args.k_groupsize,
                                              sym=not(args.k_asym),
                                              clip_ratio=args.k_clip_ratio)
                qlayers[name].out_runtime_smooth = args.a_runtime_smooth
            
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize
            qlayers[name].runtime_smooth = args.a_runtime_smooth
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip)

    # Evaluating on dataset
    testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )
    

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    if args.wandb:
            wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})

    if not args.lm_eval:
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM

        
    
    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token,trust_remote_code=True)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto', apply_chat_template=args.apply_chat_template,trust_remote_code=True)

    if len(args.tasks) == 1:
        args.tasks = args.tasks[0].split(',')  
    task_names = args.tasks
    results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=args.lm_eval_batch_size)['results']

    # metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    # metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    # print(metric_vals)
    metric_vals = results

    # task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    # tasks = []
    # from typing import List
    # if isinstance(args.tasks, List):
    #     for task in args.tasks:
    #         tasks.extend(task.split(','))
    # results = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size='auto')['results']
    # metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}

    # metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    # print(metric_vals)
    if args.wandb:
        wandb.log(metric_vals)

if __name__ == '__main__':
    main()

