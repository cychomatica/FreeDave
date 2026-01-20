# built on https://github.com/Gen-Verse/dLLM-RL/blob/main/reward/reward.py

import json
from jinja2 import Template
from termcolor import cprint
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import asyncio
from utils import math_utils
from omegaconf import OmegaConf
import os
import re
import subprocess

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def get_prompt(data_i, system_prompts):
    return Template(system_prompts).render(problem = data_i['question'])

def get_system_prompts(model_base='trado', data_type='math', start_with_think=False):
    if data_type == 'code':
        code_eval = True
        system_prompts_function = '''<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
        system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
        if start_with_think:
            system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant<think>\n'''
        system_prompts = {
            'function': system_prompts_function,
            'stdio': system_prompts_stdio
        }
    elif data_type == 'option':
        code_eval = False
        system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>\n<|im_start|>assistant\n'''
        if start_with_think:
            system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>\n<|im_start|>assistant<think>\n'''
    else:
        code_eval = False
        if model_base in ['trado', 'sdar']:
            system_prompts = '''<|im_start|>user\n{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''
            if start_with_think:
                system_prompts = '''<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant<think>\n'''
        elif model_base == 'dream':
            system_prompts = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
        else:
            system_prompts = '''<|startoftext|><|start_header_id|>user<|end_header_id|>You need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n'''

    return system_prompts, code_eval

def get_token_lengths(strings, tokenizer):
    lengths = []
    for s in strings:
        token_ids = tokenizer.encode(s, add_special_tokens=False)
        token_ids = [t for t in token_ids if t != tokenizer.pad_token_id]
        lengths.append(len(token_ids))
    return lengths

def data_prepare(model_base, data_type, data_path, start_with_think=False):

    with open(data_path, 'r') as f:
        data = json.load(f)

    system_prompts, code_eval = get_system_prompts(model_base, data_type, start_with_think)

    for i in range(len(data)):

        if data_type in ['math', 'option']:
            data[i]['prompt'] = get_prompt(data[i], system_prompts)
            data[i]['full_output'] = []
            data[i]['cleaned_output'] = []
            data[i]['extracted_output'] = []
            data[i]['step_map'] = []
            data[i]['generated_tokens'] = []
            data[i]['response_tokens'] = []
            data[i]['response_time'] = []
            data[i]['response_nfe'] = []
        elif data_type == 'code':
            if data[i]["test_method"] == "stdio":
                data[i]['prompt'] = get_prompt(data[i], system_prompts['stdio'])
            else:
                data[i]['prompt'] = get_prompt(data[i], system_prompts['function'])
            data[i]['full_output'] = []
            data[i]['cleaned_output'] = []
            data[i]['extracted_output'] = []
            data[i]['step_map'] = []
            data[i]['generated_tokens'] = []
            data[i]['response_tokens'] = []
            data[i]['response_time'] = []
            data[i]['response_nfe'] = []
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    
    return data

def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return 'Can not extract the answer!'

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else 'Can not extract the answer!'

def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output

def output_process(data_type, data):
    for i in range(len(data)):
        for each_output in data[i]['full_output']:
            if data_type == 'code':
                data[i]['extracted_output'].append(extract_code(each_output))
            else:
                data[i]['extracted_output'].append(extract_final_boxed_answer(each_output))
    return data

def reward(config, outputs_dir, outputs_filename):

    with open(os.path.join(outputs_dir, outputs_filename), 'r') as f:
        data = json.load(f)

    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    generated_length_list = []
    response_length_list = []
    response_time_list = []
    response_nfe_list = []
    for i in range(len(data)):
        response_time_list = response_time_list + data[i]['response_time']
        response_nfe_list = response_nfe_list + data[i]['response_nfe']
        generated_length_list = generated_length_list + data[i]['generated_tokens']
        response_length_list = response_length_list + data[i]['response_tokens']
        index_list = index_list + [i] * len(data[i]['extracted_output'])
        extracted_output_list = extracted_output_list + data[i]['extracted_output']
        if config.dataset.data_type == 'math':
            data[i]['correctness'] = []
            ground_truth_list = ground_truth_list + [data[i]['ground_truth_answer']] * len(data[i]['extracted_output'])
        elif config.dataset.data_type == 'option':
            data[i]['correctness'] = []
            ground_truth_list = ground_truth_list + [data[i]['answer']] * len(data[i]['extracted_output'])

    if config.dataset.data_type in ['math', 'option']:

        nest_asyncio.apply()

        async def get_correctness():
            executor = ThreadPoolExecutor(max_workers=64)
            tasks = []
            for i in range(len(index_list)):
                tasks.append(math_utils.is_equal(extracted_output_list[i], ground_truth_list[i], executor))
            results = await asyncio.gather(*tasks)
            return results
    
        correctness_list = asyncio.run(get_correctness())
        for i in range(len(index_list)):
            index_i = index_list[i]
            data[index_i]['correctness'].append(correctness_list[i])

    def z_score_normalize(lst):
        mean = sum(lst) / len(lst)
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        if std == 0:
            return [0 for x in lst]
        return [(x - mean) / std for x in lst]

    def set_last_t(lst: list, t: int) -> None:
        new_lst = lst.copy()
        new_val = max(lst) + 1
        new_lst[-t:] = [new_val] * t
        return new_lst

    if config.dataset.data_type in ['math', 'option']:
        acc = sum(correctness_list)/len(correctness_list)
    else:
        num_task   = 0
        num_correct_task = 0
        for x in data:
            for y in x['correctness']:
                num_correct_task += all(y)
                num_task += 1
        acc = num_correct_task / num_task if num_task else 0

    if config.rollout.output_unmasking_history == False:
        for i in range(len(data)):
            data[i]['step_map'] = []
        
    with open(os.path.join(outputs_dir, outputs_filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    results_dir = os.path.join(os.path.dirname(outputs_dir), 'results')
    results_filename = outputs_filename.replace('.json', '.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, results_filename), 'a') as f:
        # Save + print
        def save_and_print(text):
            cprint('*'*10 + 'eval results' + '*'*10 + '\n' + text, color='green')
            f.write(text + '\n')

        avg_generated_len = sum(generated_length_list)/len(generated_length_list)
        avg_valid_len = sum(response_length_list)/len(response_length_list)
        avg_time = sum(response_time_list)/len(response_time_list)
        avg_nfe = sum(response_nfe_list)/len(response_nfe_list)

        save_and_print(f'acc: {acc}\navg generated tokens: {avg_generated_len}\navg valid tokens: {avg_valid_len}\navg time: {avg_time}\navg nfe: {avg_nfe}')

def execute(config, outputs_dir, outputs_filename):
    subprocess.run(
        f'python utils/code_execution.py '
        f'--outputs_dir {outputs_dir} '
        f'--outputs_filename {outputs_filename} '
        f'--num_chunk {config.execute.num_chunk} ',
        shell=True,
        cwd='./',
        check=True,
    )

if __name__ == '__main__':
    config = get_config()
    reward(config, config.outputs_dir, config.outputs_filename)