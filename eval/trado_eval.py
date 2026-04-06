from utils.determinism_utils import deterministic

def main(config):
    # All imports inside main() so they happen after deterministic context is entered
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from generation.generation_core import DLMGeneration
    from utils.monitor_utils import ForwardMonitor

    import time, json, os
    from functools import partial
    from utils.eval_utils import data_prepare, output_process, get_token_lengths, reward, execute
    from tqdm import tqdm
    from termcolor import cprint

    dataset = config.dataset.eval_dataset
    data_path = 'data/' + dataset + '.json'
    data = data_prepare(config.model_base, config.dataset.data_type, data_path, config.rollout.start_with_think)

    model_path = config.model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype='float16', device_map='cuda'
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inference_monitor = ForwardMonitor(model)
    DLM_Gen = DLMGeneration()
    
    if config.rollout.draft_steps > 1 and config.rollout.draft_mode is not None:
        generate_func = partial(DLM_Gen.block_decode_with_block_attention_FreeDave,
                                model=model,
                                temperature=config.rollout.temperature,
                                top_k=config.rollout.top_k,
                                top_p=config.rollout.top_p,
                                alg_temp=config.rollout.alg_temp,
                                block_length=config.rollout.block_size,
                                max_gen_length=config.rollout.max_token,
                                decoding_steps=config.rollout.max_token,
                                mask_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['mask_token']],
                                eos_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['eos_token']],
                                pad_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['pad_token']],
                                eager_acceptance_mode=config.rollout.eager_acceptance_mode,
                                draft_steps=config.rollout.draft_steps,
                                draft_mode=config.rollout.draft_mode,
                                confidence_threshold=None,
                                early_exit=config.rollout.early_exit,
                                use_cache=True,
            )
        decoding_strategy = 'FreeDave_eager_acceptance_{}_{}_d={}'.format(
            'enabled' if config.rollout.eager_acceptance_mode else 'disabled',
            config.rollout.draft_mode,
            config.rollout.draft_steps,
        )
        cprint('Evaluating {} on {}.\nUsing FreeDave (eager acceptance {}, draft mode={}, d={})'.format(
            os.path.basename(model_path), 
            dataset, 
            'enabled' if config.rollout.eager_acceptance_mode else 'disabled', 
            config.rollout.draft_mode,
            config.rollout.draft_steps), 
            color='green')
    else:
        generate_func = partial(DLM_Gen.block_decode_with_block_attention,
                                model=model,
                                temperature=config.rollout.temperature,
                                top_k=config.rollout.top_k,
                                top_p=config.rollout.top_p,
                                alg_temp=config.rollout.alg_temp,
                                block_length=config.rollout.block_size,
                                max_gen_length=config.rollout.max_token,
                                decoding_steps=config.rollout.max_token,
                                mask_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['mask_token']],
                                eos_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['eos_token']],
                                pad_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['pad_token']],
                                confidence_threshold=config.rollout.confidence_threshold,
                                early_exit=config.rollout.early_exit,
                                use_cache=True,
                            )
        if config.rollout.confidence_threshold is not None:
            decoding_strategy = 'Parallel_confidence_threshold={}'.format(config.rollout.confidence_threshold)
            cprint('Evaluating {} on {}.\nUsing parallel decoding (confidence threshold: {})'.format(os.path.basename(model_path), dataset, config.rollout.confidence_threshold), color='green')
        else:
            decoding_strategy = 'Static'
            cprint('Evaluating {} on {}.\nUsing static decoding'.format(os.path.basename(model_path), dataset), color='green')

    inference_stats = {
        'total_generated_tokens': 0,
        'total_sampling_time': 0,
        'total_nfe': 0,
        'Avg TPS': None,
        'Avg TPF': None,
        'gpu_peak_memory (MB)': None,
        # 'total_accepted_steps': 0,
        # 'total_draft_steps': 0,
    }

    for i in tqdm(range(len(data)), dynamic_ncols=True):        
        prompt = data[i]['prompt']
        tokens = tokenizer.batch_encode_plus([prompt], return_tensors='pt', padding=True, truncation=False)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        prompt_len = tokens['input_ids'].shape[1]
        
        inference_monitor.reset()
        with inference_monitor.count():
            # output_ids, accepted_steps, draft_steps = generate_func(prompt=tokens)
            output = generate_func(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        sampling_time = inference_monitor.get_elapsed_time()
        nfe = inference_monitor.get_nfe()

        output_ids = output.sequences.cpu()
        trajectory_step_map = output.trajectory_step_map.cpu()
        torch.cuda.empty_cache()

        output_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)
        cleaned_text = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
        num_generated_tokens = len(output_ids[0][prompt_len:])
        
        data[i]['full_output'].append(output_text)
        data[i]['cleaned_output'].append(cleaned_text)
        data[i]['generated_tokens'].append((trajectory_step_map >= 0).sum().item())
        data[i]['response_tokens'] = get_token_lengths(data[i]['cleaned_output'], tokenizer)
        data[i]['response_time'].append(sampling_time)
        data[i]['response_nfe'].append(nfe)
        
        inference_stats['total_sampling_time'] += sampling_time
        inference_stats['total_generated_tokens'] += sum(data[i]['generated_tokens'])
        inference_stats['total_nfe'] += nfe
        # inference_stats['total_accepted_steps'] += accepted_steps
        # inference_stats['total_draft_steps'] += draft_steps

    cprint('Generation done!', color='green')
    inference_stats['Avg TPS'] = inference_stats['total_generated_tokens'] / inference_stats['total_sampling_time']
    inference_stats['Avg TPF'] = inference_stats['total_generated_tokens'] / inference_stats['total_nfe']
    peak_memory = torch.cuda.max_memory_reserved()
    peak_memory_mb = peak_memory / (1024 ** 2)
    inference_stats['gpu_peak_memory (MB)'] = peak_memory_mb

    # if inference_stats['total_draft_steps'] > 0:
    #     cprint('Avg acceptance rate: {}'.format(inference_stats['total_accepted_steps'] / inference_stats['total_draft_steps']), color='green')
    data = output_process(config.dataset.data_type, data)

    save_dir = os.path.join('exp_results', config.experiment.project, 'temp_data')
    os.makedirs(save_dir, exist_ok=True)
    save_filename = '{}-{}-max_gen_length={}-block_size={}-{}.json'.format(os.path.basename(model_path), decoding_strategy, config.rollout.max_token, config.rollout.block_size, dataset)
    
    with open(os.path.join(save_dir, save_filename), 'w') as f:
        json.dump(data, f, indent=4)
    
    if config.dataset.data_type == 'code':
        cprint(f"\ncode execution started", color = "yellow")
        execute(config, save_dir, save_filename)
        cprint(f"code execution completed\n", color = "yellow")

    reward(config, save_dir, save_filename, inference_stats)

if __name__ == '__main__':
    from termcolor import cprint
    from utils.eval_utils import get_config

    config = get_config()
    cprint('Experiment config:\n{}'.format(config), color='green')
    with deterministic(enabled=config.experiment.deterministic, seed=config.experiment.seed):
        main(config)
