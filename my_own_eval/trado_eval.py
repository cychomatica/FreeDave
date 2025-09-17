from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import block_diffusion_generate, block_diffusion_generate_fast
from monitor_utils import ForwardHookCounter
import time
import json, os
from functools import partial
from omegaconf import OmegaConf
from utils import data_prepare, output_process
from tqdm import tqdm
from termcolor import cprint

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == '__main__':
    config = get_config()

    dataset = config.dataset.eval_dataset
    data_path = 'data/' + dataset + '.json'
    data = data_prepare(config.dataset.data_type, data_path, config.rollout.start_with_think)

    model_path = config.model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype='float16', device_map='cuda'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    nfe_counter = ForwardHookCounter(model)
    
    if config.rollout.draft_steps > 1:
        generate_func = partial(block_diffusion_generate_fast, 
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
                                )
        cprint(f'Using fast sampling with draft steps={config.rollout.draft_steps}', color='green')
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
        cprint(f'Using normal sampling', color='green')

    total_sampling_time = 0
    total_response_tokens = 0
    total_nfe = 0
    for i in tqdm(range(len(data))):
        prompt = data[i]['prompt']
        messages = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.batch_encode_plus([text], return_tensors='pt', padding=True, truncation=True, max_length=200)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        nfe_counter.reset_count()
        start_time = time.time()
        with nfe_counter.count_context():
            output_ids = generate_func(prompt=tokens)
        end_time = time.time()
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        cleaned_text = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '')
        sampling_time = end_time - start_time
        nfe = nfe_counter.counter.count

        data[i]['full_output'].append(output_text)
        data[i]['cleaned_output'].append(cleaned_text)
        data[i]['response_tokens'].append(len(output_ids[0]))
        data[i]['response_time'].append(sampling_time)
        data[i]['response_nfe'].append(nfe)
        
        total_sampling_time += sampling_time
        total_response_tokens += len(output_ids[0])
        total_nfe += nfe

    cprint(f'Avg throughput (tokens/s): {total_response_tokens / total_sampling_time}', color='green')
    cprint(f'Avg throughput (tokens/nfe): {total_response_tokens / total_nfe}', color='green')

    data = output_process(config.dataset.data_type, data)

    save_dir = os.path.join(config.experiment.project, 'temp_data')
    os.makedirs(save_dir, exist_ok=True)

    sampling_mode = 'normal' if config.rollout.draft_steps == 1 else f'fast-draft={config.rollout.draft_steps}'
    save_filename = f'{os.path.basename(model_path)}-{sampling_mode}-block_size={config.rollout.block_size}-block_denoising_steps={config.rollout.denoising_steps_per_block}-{dataset}.json'
    
    with open(os.path.join(save_dir, save_filename), 'w') as f:
        json.dump(data, f, indent=4)

        
