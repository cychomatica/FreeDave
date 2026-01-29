from utils.determinism_utils import deterministic

def main(config):
    # All imports inside main() so they happen after deterministic context is entered
    import torch
    from modeling.sdar import SDARForCausalLM
    import modeling.sdar.modeling_sdar as sdar_module
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from generate.trado_generate import block_diffusion_generate, block_diffusion_generate_FreeDave
    from utils.monitor_utils import ForwardHookCounter

    import time, json, os
    from functools import partial
    from utils.eval_utils import data_prepare, output_process, get_token_lengths, reward, execute
    from tqdm import tqdm
    from termcolor import cprint

    # Verify determinism settings
    cprint(f"flash_attn_available: {sdar_module.flash_attn_available}", 'yellow')
    cprint(f"liger_kernel_is_available: {sdar_module.liger_kernel_is_available}", 'yellow')
    cprint(f"use_eager_attn: {sdar_module.USE_EAGER_ATTN}", 'yellow')

    dataset = config.dataset.eval_dataset
    data_path = 'data/' + dataset + '.json'
    data = data_prepare(config.model_base, config.dataset.data_type, data_path, config.rollout.start_with_think)

    sampling_mode = 'normal' if config.rollout.draft_steps == 1 else 'fast-draft={}'.format(config.rollout.draft_steps)
    remasking_strategy = 'static' if config.rollout.remasking_strategy == "low_confidence_static" else 'dynamic'
    # wandb.init(project='freedave', name='{}-{}-{}-max_gen_length={}-block_size={}-block_denoising_steps={}-{}'.format(os.path.basename(config.model), sampling_mode, remasking_strategy, config.rollout.max_token, config.rollout.block_size, config.rollout.denoising_steps_per_block, dataset))

    model_path = config.model
    # Use local SDARForCausalLM instead of AutoModelForCausalLM to control determinism
    model = SDARForCausalLM.from_pretrained(
        model_path, torch_dtype='float16', device_map='cuda'
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path, trust_remote_code=True, torch_dtype='float16', device_map='cuda'
    # )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    nfe_counter = ForwardHookCounter(model)
    
    if config.rollout.draft_steps > 1 and config.rollout.fast_sampling_version != 'NA':
        generate_func = partial(block_diffusion_generate_FreeDave, 
                                model=model,
                                mask_id=151669,
                                gen_length=config.rollout.max_token,
                                block_length=config.rollout.block_size,
                                denoising_steps=config.rollout.denoising_steps_per_block,
                                draft_steps=config.rollout.draft_steps,
                                temperature=config.rollout.temperature,
                                top_k=config.rollout.top_k,
                                top_p=config.rollout.top_p,
                                remasking_strategy=config.rollout.remasking_strategy,
                                confidence_threshold=config.rollout.dynamic_threshold,
                                eager_acceptance_mode=config.rollout.eager_acceptance_mode,
                                )
        cprint('Evaluating {} on {}.\nUsing FreeDave ({}, eager acceptance {}) with draft steps={}'.format(
            os.path.basename(model_path), 
            dataset, 
            config.rollout.remasking_strategy, 
            'enabled' if config.rollout.eager_acceptance_mode else 'disabled', 
            config.rollout.draft_steps), 
            color='green')
    else:
        generate_func = partial(block_diffusion_generate, 
                                model=model,
                                mask_id=151669,
                                gen_length=config.rollout.max_token,
                                block_length=config.rollout.block_size,
                                denoising_steps=config.rollout.denoising_steps_per_block,
                                temperature=config.rollout.temperature,
                                top_k=config.rollout.top_k,
                                top_p=config.rollout.top_p,
                                remasking_strategy=config.rollout.remasking_strategy,
                                confidence_threshold=config.rollout.dynamic_threshold,
                                )
        cprint('Evaluating {} on {}.\nUsing normal sampling ({})'.format(os.path.basename(model_path), dataset, config.rollout.remasking_strategy), color='green')

    total_sampling_time = 0
    total_generated_tokens = 0
    total_nfe = 0
    total_accepted_steps = 0
    total_draft_steps = 0

    for i in tqdm(range(len(data)), dynamic_ncols=True):        
        prompt = data[i]['prompt']
        tokens = tokenizer.batch_encode_plus([prompt], return_tensors='pt', padding=True, truncation=False)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        prompt_len = tokens['input_ids'].shape[1]
        
        nfe_counter.reset_count()
        start_time = time.time()
        with nfe_counter.count_context():
            output_ids, accepted_steps, draft_steps = generate_func(prompt=tokens)
        end_time = time.time()
        sampling_time = end_time - start_time
        nfe = nfe_counter.counter.count

        output_ids = output_ids.cpu()
        torch.cuda.empty_cache()

        output_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)
        cleaned_text = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
        num_generated_tokens = len(output_ids[0][prompt_len:])
        
        data[i]['full_output'].append(output_text)
        data[i]['cleaned_output'].append(cleaned_text)
        data[i]['generated_tokens'].append(num_generated_tokens)
        data[i]['response_tokens'] = get_token_lengths(data[i]['cleaned_output'], tokenizer)
        data[i]['response_time'].append(sampling_time)
        data[i]['response_nfe'].append(nfe)
        
        total_sampling_time += sampling_time
        total_generated_tokens += sum(data[i]['generated_tokens'])
        total_nfe += nfe
        total_accepted_steps += accepted_steps
        total_draft_steps += draft_steps

    cprint('Generation done!', color='green')
    cprint('Avg throughput (tokens/s): {}'.format(total_generated_tokens / total_sampling_time), color='green')
    cprint('Avg throughput (tokens/nfe): {}'.format(total_generated_tokens / total_nfe), color='green')
    if total_draft_steps > 0:
        cprint('Avg acceptance rate: {}'.format(total_accepted_steps / total_draft_steps), color='green')
    data = output_process(config.dataset.data_type, data)

    save_dir = os.path.join('exp_results', config.experiment.project, 'temp_data')
    os.makedirs(save_dir, exist_ok=True)
    save_filename = '{}-{}-{}-max_gen_length={}-block_size={}-block_denoising_steps={}-{}.json'.format(os.path.basename(model_path), sampling_mode, remasking_strategy, config.rollout.max_token, config.rollout.block_size, config.rollout.denoising_steps_per_block, dataset)
    
    with open(os.path.join(save_dir, save_filename), 'w') as f:
        json.dump(data, f, indent=4)
    
    if config.dataset.data_type == 'code':
        cprint(f"\ncode execution started", color = "yellow")
        execute(config, save_dir, save_filename)
        cprint(f"code execution completed\n", color = "yellow")

    reward(config, save_dir, save_filename)

if __name__ == '__main__':
    from termcolor import cprint
    from utils.eval_utils import get_config

    config = get_config()
    cprint('Experiment config:\n{}'.format(config), color='green')
    with deterministic(enabled=config.experiment.deterministic, seed=config.experiment.seed):
        main(config)
