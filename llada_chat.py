import torch
from llada_generate import generate_with_prefix_cache, generate_with_prefix_cache_FreeDave
import time
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from modeling.llada.modeling_llada import LLaDAModelLM
from monitor_utils import ForwardHookCounter
from termcolor import cprint

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def generation_tokens_hook_func(step, x, logits):
            print(f'############ Step {step} ############')
            # print(tokenizer.decode(h[0].tolist()))
            print(tokenizer.decode(x[0].tolist()).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, ' '), end='\r')
            time.sleep(0.01)
            return x

if __name__ == '__main__':
    
    config = get_config()
    # Load model and tokenizer
    model_path = config.model
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    forward_counter = ForwardHookCounter(model)

    # Initialize conversation history
    messages = []

    print('Multi-turn conversation with Dream-v0-Instruct-7B')
    print('Type ''exit'' to end the conversation')
    print('-'*66)

    while True:
        # Get user input
        user_input = input('You: ')
        print('-'*66)

        # Check if user wants to exit
        if user_input.lower() == 'exit':
            print('Conversation ended.')
            break

        # Add user message to conversation history
        messages.append({'role': 'user', 'content': user_input})

        # Format input with chat template
        prompt = tokenizer.apply_chat_template(
            messages, return_tensors='pt', return_dict=True, add_generation_prompt=True
        )
        prompt_ids = prompt.input_ids.to(device='cuda')

        mask_id = tokenizer.encode('<|mdm_mask|>')[0]
        if config.rollout.use_cache == False:
            config.rollout.further_horizon = None
        if config.rollout.remasking_strategy == "low_confidence_static":
            unmask_threshold = None
        else:
            unmask_threshold = config.rollout.dynamic_threshold

        # Generate response
        forward_counter.reset_count()
        start_time = time.time()
        with forward_counter.count_context():
            output = generate_with_prefix_cache(
                model,
                prompt_ids,
                steps=config.rollout.steps,
                gen_length=config.rollout.max_gen_length,
                block_length=config.rollout.block_size, 
                temperature=config.rollout.temperature,
                strategy=config.rollout.target, 
                mask_id=mask_id, 
                further_horizon=config.rollout.further_horizon,
                use_cache=config.rollout.use_cache, 
                unmask_threshold = unmask_threshold
            )
        end_time = time.time()
        output.sequences = output.sequences.cpu()
        torch.cuda.empty_cache()

        # Process response
        generation = tokenizer.decode(output.sequences[0][len(prompt_ids[0]):].tolist())
        generation = generation.split(tokenizer.eos_token)[0].split('<|eot_id|>')[0].strip()

        # Print response
        print('Model:', generation)
        cprint(f'Normal generation: (time: {end_time - start_time} seconds; num of forward passes: {forward_counter.counter.count}; avg step forward time: {(end_time - start_time) / forward_counter.counter.count} seconds)', 'cyan')
        print('-'*66)

        # Generate response
        forward_counter.reset_count()
        start_time = time.time()
        with forward_counter.count_context():
            output = generate_with_prefix_cache_FreeDave(
                model,
                prompt_ids,
                steps=config.rollout.steps, 
                draft_steps=config.rollout.draft_steps,
                gen_length=config.rollout.max_gen_length,
                block_length=config.rollout.block_size, 
                temperature=config.rollout.temperature,
                strategy=config.rollout.target, 
                mask_id=mask_id, 
                further_horizon=config.rollout.further_horizon,
                use_cache=config.rollout.use_cache, 
                unmask_threshold = unmask_threshold
            )
        end_time = time.time()
        output.sequences = output.sequences.cpu()
        torch.cuda.empty_cache()

        # Process response
        generation = tokenizer.decode(output.sequences[0][len(prompt_ids[0]):].tolist())
        generation = generation.split(tokenizer.eos_token)[0].split('<|eot_id|>')[0].strip()
        # Print response
        print('Model:', generation)
        cprint(f'Fast generation: (time: {end_time - start_time} seconds; num of forward passes: {forward_counter.counter.count}; avg step forward time: {(end_time - start_time) / forward_counter.counter.count} seconds)', 'cyan')
        print('-'*66)

        # Add model response to conversation history
        messages.append({'role': 'assistant', 'content': generation})


'''An example conversation (maybe different due to randomness)
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|im_end|>
<|im_start|>assistant
Janet sells 16 - 3 - 4 = 9 eggs per day.
She makes 9 * $2 = $18 per day.<|im_end|>
<|im_start|>user
what if her duck lay three more eggs<|im_end|>
<|im_start|>assistant
If Janet's ducks lay three more eggs per day, she would have 16 + 3 = 19 eggs per day.<|im_end|>
<|im_start|>user
yes, so how many dollars she make<|im_end|>
<|im_start|>assistant
Janet sells 19 - 3 - 4 = 12 eggs per day.
She makes 12 * $2 = $24 per day.
'''