import torch
from dream_generate import block_diffusion_generate, block_diffusion_generate_FreeDave
from modeling.dream.generation_utils_block import DreamGenerationConfig
from modeling.dream.tokenization_dream import DreamTokenizer
from modeling.dream.modeling_dream import DreamModel
from monitor_utils import ForwardHookCounter

import time
import json, os
from functools import partial
from omegaconf import OmegaConf
from eval_utils import data_prepare, output_process, reward, get_token_lengths
from tqdm import tqdm
from termcolor import cprint

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def generation_tokens_hook_func(step, x, logits):
    print(f"############ Step {step} ############")
    # print(tokenizer.decode(h[0].tolist()))
    print(tokenizer.decode(x[0].tolist()).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, " "), end="\r")
    time.sleep(0.01)
    return x

if __name__ == "__main__":
    
    config = get_config()
    cprint('Experiment config:\n{}'.format(config), color='green')

    dataset = config.dataset.eval_dataset
    data_path = 'data/' + dataset + '.json'
    data = data_prepare(config.model_base, config.dataset.data_type, data_path)

    # Load model and tokenizer
    model_path = config.model
    model = DreamModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = DreamTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    nfe_counter = ForwardHookCounter(model)

    generation_config = DreamGenerationConfig(
            output_history=True,            
            return_dict_in_generate=True,   
            max_gen_length=config.rollout.max_gen_length,     
            steps=config.rollout.steps,        
            draft_steps=config.rollout.draft_steps,
            temperature=config.rollout.temperature,  
            top_p=config.rollout.top_p,               
            top_k=config.rollout.top_k,            
            tar=config.rollout.target,               
            alg_temp=config.rollout.alg_temp,        
        )
    if config.rollout.remasking_strategy == "low_confidence_static":
        unmask_threshold = None
    else:
        unmask_threshold = config.rollout.dynamic_threshold

    if config.rollout.draft_steps > 1:
        generate_func = partial(block_diffusion_generate_FreeDave, 
                                model=model,
                                generation_config=generation_config,
                                block_length=config.rollout.block_size,
                                use_cache=config.rollout.use_cache,
                                further_horizon=config.rollout.further_horizon,
                                mask_token_id = model.config.mask_token_id,
                                eos_token_id = model.config.eos_token_id,
                                pad_token_id = model.config.pad_token_id,
                                pad_target_penalty = config.rollout.pad_target_penalty,
                                unmask_threshold = unmask_threshold
                                )
        cprint(f'Evaluating {os.path.basename(model_path)} on {dataset}.\nUsing FreeDave with draft steps={config.rollout.draft_steps}', color='green')
    else:
        generate_func = partial(block_diffusion_generate, 
                                model=model,
                                generation_config=generation_config,
                                block_length=config.rollout.block_size,
                                use_cache=config.rollout.use_cache,
                                further_horizon=config.rollout.further_horizon,
                                mask_token_id = model.config.mask_token_id,
                                eos_token_id = model.config.eos_token_id,
                                pad_token_id = model.config.pad_token_id,
                                pad_target_penalty = config.rollout.pad_target_penalty,
                                unmask_threshold = unmask_threshold
                                )
        cprint(f'Evaluating {os.path.basename(model_path)} on {dataset}.\nUsing normal sampling', color='green')

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
        attention_mask = prompt.attention_mask.to(device="cuda")

        nfe_counter.reset_count()
        start_time = time.time()
        with nfe_counter.count_context():
            output = generate_func(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
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

    cprint(f'Generation done!', color='green')
    cprint(f'Avg throughput (tokens/s): {total_response_tokens / total_sampling_time}', color='green')
    cprint(f'Avg throughput (tokens/nfe): {total_response_tokens / total_nfe}', color='green')

    data = output_process(config.dataset.data_type, data)

    save_dir = os.path.join(config.experiment.project, 'temp_data')
    os.makedirs(save_dir, exist_ok=True)

    sampling_mode = 'normal' if config.rollout.draft_steps == 1 else f'fast-draft_steps={config.rollout.draft_steps}'
    remasking_strategy = 'static' if config.rollout.remasking_strategy == "low_confidence_static" else 'dynamic'
    save_filename = f'{os.path.basename(model_path)}-{sampling_mode}-{remasking_strategy}-max_gen_length={config.rollout.max_gen_length}-block_size={config.rollout.block_size}-steps={config.rollout.steps}-{dataset}.json'
    
    with open(os.path.join(save_dir, save_filename), 'w') as f:
        json.dump(data, f, indent=4)

    reward(config, save_dir, save_filename)