import torch
from generate.llada_generate import generate_with_prefix_cache, generate_with_prefix_cache_FreeDave
from modeling.llada.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer
from utils.monitor_utils import ForwardHookCounter

import time, json, os
from functools import partial
from utils.eval_utils import data_prepare, output_process, reward, get_token_lengths, get_config
from tqdm import tqdm
from termcolor import cprint

if __name__ == "__main__":
    
    config = get_config()
    cprint('Experiment config:\n{}'.format(config), color='green')

    dataset = config.dataset.eval_dataset
    data_path = 'data/' + dataset + '.json'
    data = data_prepare(config.model_base, config.dataset.data_type, data_path)

    # Load model and tokenizer
    model_path = config.model
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    nfe_counter = ForwardHookCounter(model)

    mask_id = tokenizer.encode('<|mdm_mask|>')[0]
    if config.rollout.use_cache == False:
        config.rollout.further_horizon = None
    if config.rollout.remasking_strategy == "low_confidence_static":
        unmask_threshold = None
    else:
        unmask_threshold = config.rollout.dynamic_threshold

    if config.rollout.draft_steps > 1:
        generate_func = partial(generate_with_prefix_cache_FreeDave, 
                                model=model,
                                steps=config.rollout.steps,
                                draft_steps=config.rollout.draft_steps,
                                gen_length=config.rollout.max_gen_length,
                                block_length=config.rollout.block_size, 
                                temperature=config.rollout.temperature,
                                target=config.rollout.target, 
                                mask_id=mask_id, 
                                further_horizon=config.rollout.further_horizon,
                                use_cache=config.rollout.use_cache, 
                                unmask_threshold = unmask_threshold
                                )
        cprint('Evaluating {} on {}.\nUsing FreeDave ({}) with draft steps={}'.format(os.path.basename(model_path), dataset, config.rollout.remasking_strategy, config.rollout.draft_steps), color='green')
    else:
        generate_func = partial(generate_with_prefix_cache, 
                                model=model,
                                steps=config.rollout.steps,
                                gen_length=config.rollout.max_gen_length,
                                block_length=config.rollout.block_size, 
                                temperature=config.rollout.temperature,
                                target=config.rollout.target, 
                                mask_id=mask_id, 
                                further_horizon=config.rollout.further_horizon,
                                use_cache=config.rollout.use_cache, 
                                unmask_threshold = unmask_threshold
                                )
        cprint('Evaluating {} on {}.\nUsing normal sampling ({})'.format(os.path.basename(model_path), dataset, config.rollout.remasking_strategy), color='green')

    total_sampling_time = 0
    total_response_tokens = 0
    total_nfe = 0
    for i in tqdm(range(len(data))):
        prompt = data[i]['prompt']
        messages = [{'role': 'user', 'content': prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        prompt_ids = prompt.input_ids.to(device="cuda")

        nfe_counter.reset_count()
        start_time = time.time()
        with nfe_counter.count_context():
            output = generate_func(
                input_ids=prompt_ids,
            )
        end_time = time.time()
        sampling_time = end_time - start_time
        nfe = nfe_counter.counter.count

        output.sequences = output.sequences.cpu()
        torch.cuda.empty_cache()

        output_text = tokenizer.decode(output.sequences[0][len(prompt_ids[0]):].tolist())
        cleaned_text = output_text.split(tokenizer.eos_token)[0].strip()

        data[i]['full_output'].append(output_text)
        data[i]['cleaned_output'].append(cleaned_text)
        data[i]['response_tokens'] = get_token_lengths(data[i]['cleaned_output'], tokenizer)
        data[i]['response_time'].append(sampling_time)
        data[i]['response_nfe'].append(nfe)

        total_sampling_time += sampling_time
        total_response_tokens += sum(data[i]['response_tokens'])
        total_nfe += nfe

    cprint('Generation done!', color='green')
    cprint('Avg throughput (tokens/s): {}'.format(total_response_tokens / total_sampling_time), color='green')
    cprint('Avg throughput (tokens/nfe): {}'.format(total_response_tokens / total_nfe), color='green')

    data = output_process(config.dataset.data_type, data)

    save_dir = os.path.join('exp_results', config.experiment.project, 'temp_data')
    os.makedirs(save_dir, exist_ok=True)

    sampling_mode = 'normal' if config.rollout.draft_steps == 1 else 'fast-draft_steps={}'.format(config.rollout.draft_steps)
    remasking_strategy = 'static' if config.rollout.remasking_strategy == "low_confidence_static" else 'dynamic'
    save_filename = '{}-{}-{}-max_gen_length={}-block_size={}-steps={}-{}.json'.format(os.path.basename(model_path), sampling_mode, remasking_strategy, config.rollout.max_gen_length, config.rollout.block_size, config.rollout.steps, dataset)
    
    with open(os.path.join(save_dir, save_filename), 'w') as f:
        json.dump(data, f, indent=4)

    reward(config, save_dir, save_filename)