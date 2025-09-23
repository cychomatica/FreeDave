import torch
from generate.dream_generate import block_diffusion_generate, block_diffusion_generate_FreeDave, block_diffusion_generate_batch
import time
from omegaconf import OmegaConf
from modeling.dream.generation_utils_block import DreamGenerationConfig
from modeling.dream.tokenization_dream import DreamTokenizer
from modeling.dream.modeling_dream import DreamModel
from utils.monitor_utils import ForwardHookCounter
from termcolor import cprint

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == '__main__':
    
    config = get_config()
    
    # Load model and tokenizer
    model_path = config.model
    model = DreamModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda')
    tokenizer = DreamTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
        attention_mask = prompt.attention_mask.to(device='cuda')

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
        if config.rollout.remasking_strategy == 'low_confidence_static':
            unmask_threshold = None
        else:
            unmask_threshold = config.rollout.dynamic_threshold

        # Generate response
        forward_counter.reset_count()
        start_time = time.time()
        with forward_counter.count_context():
            output = block_diffusion_generate(
                model,
                prompt_ids,
                attention_mask=attention_mask,
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
        end_time = time.time()
        output.sequences = output.sequences.cpu()
        torch.cuda.empty_cache()

        # Process response
        generation = tokenizer.decode(output.sequences[0][len(prompt_ids[0]):].tolist())
        generation = generation.split(tokenizer.eos_token)[0].strip()

        # Print response
        print('Model:', generation)
        cprint('Normal generation: (time: {} seconds; num of forward passes: {}; avg step forward time: {} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count), 'cyan')
        print('-'*66)

        # # Generate response
        # forward_counter.reset_count()
        # start_time = time.time()
        # with forward_counter.count_context():
        #     output = block_diffusion_generate_batch(
        #         model,
        #         prompt_ids,
        #         attention_mask=attention_mask,
        #         generation_config=generation_config,
        #         block_length=config.rollout.block_size,
        #         use_cache=config.rollout.use_cache,
        #         further_horizon=config.rollout.further_horizon,
        #         mask_token_id = model.config.mask_token_id,
        #         eos_token_id = model.config.eos_token_id,
        #         pad_token_id = model.config.pad_token_id,
        #         pad_target_penalty = config.rollout.pad_target_penalty,
        #         unmask_threshold = unmask_threshold
        #     )
        # end_time = time.time()
        # output.sequences = output.sequences.cpu()
        # torch.cuda.empty_cache()

        # # Process response
        # generation = tokenizer.decode(output.sequences[0][len(prompt_ids[0]):].tolist())
        # generation = generation.split(tokenizer.eos_token)[0].strip()

        # # Print response
        # print('Model:', generation)
        # cprint('Normal batch generation: (time: {} seconds; num of forward passes: {}; avg step forward time: {} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count), 'cyan')
        # print('-'*66)

        # Generate response
        forward_counter.reset_count()
        start_time = time.time()
        with forward_counter.count_context():
            output = block_diffusion_generate_FreeDave(
                model,
                prompt_ids,
                attention_mask=attention_mask,
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
        end_time = time.time()
        output.sequences = output.sequences.cpu()
        torch.cuda.empty_cache()

        # Process response
        generation = tokenizer.decode(output.sequences[0][len(prompt_ids[0]):].tolist())
        generation = generation.split(tokenizer.eos_token)[0].strip()

        # Print response
        print('Model:', generation)
        cprint('Fast generation: (time: {} seconds; num of forward passes: {}; avg step forward time: {} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count), 'cyan')
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