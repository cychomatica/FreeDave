import torch
from generate.llada_generate import block_diffusion_generate, block_diffusion_generate_FreeDave
import time, os
from transformers import AutoTokenizer
from modeling.llada.modeling_llada import LLaDAModelLM
from utils.monitor_utils import ForwardHookCounter
from termcolor import cprint

if __name__ == '__main__':
    
    # Load model and tokenizer
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    forward_counter = ForwardHookCounter(model)

    # Initialize conversation history
    remasking_strategy = "low_confidence_static"
    messages = []

    print('Multi-turn conversation with {}'.format(os.path.basename(model_path)))
    print('Type ''exit'' to end the conversation')
    print('-'*100)

    while True:
        # Get user input
        user_input = input('You: ')
        print('-'*100)

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
        if remasking_strategy == "low_confidence_static":
            unmask_threshold = None
        else:
            unmask_threshold = 0.95

        # Generate response
        forward_counter.reset_count()
        start_time = time.time()
        with forward_counter.count_context():
            output = block_diffusion_generate(
                model,
                prompt_ids,
                steps=256,
                gen_length=256,
                block_length=32, 
                temperature=0.1,
                target="confidence", 
                mask_id=mask_id, 
                further_horizon=128,
                use_cache=True, 
                unmask_threshold = unmask_threshold
            )
        end_time = time.time()
        output.sequences = output.sequences.cpu()
        torch.cuda.empty_cache()

        # Process response
        generation = tokenizer.decode(output.sequences[0][len(prompt_ids[0]):].tolist())
        cleaned_generation = generation.split(tokenizer.eos_token)[0].split('<|eot_id|>')[0].strip()

        # Print response
        cprint('Normal generation: (time: {} seconds; nfe: {}; avg forward time: {} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count), 'yellow')
        print('Model\'s Response:', cleaned_generation)
        print('-'*100)

        # Generate response
        forward_counter.reset_count()
        start_time = time.time()
        with forward_counter.count_context():
            output_fast = block_diffusion_generate_FreeDave(
                model,
                prompt_ids,
                steps=256, 
                draft_steps=4,
                gen_length=256,
                block_length=32, 
                temperature=0.1,
                target="confidence", 
                mask_id=mask_id, 
                further_horizon=128,
                use_cache=True, 
                unmask_threshold = unmask_threshold
            )
        end_time = time.time()
        output_fast.sequences = output_fast.sequences.cpu()
        torch.cuda.empty_cache()

        # Process response
        generation_fast = tokenizer.decode(output_fast.sequences[0][len(prompt_ids[0]):].tolist())
        cleaned_generation_fast = generation_fast.split(tokenizer.eos_token)[0].split('<|eot_id|>')[0].strip()
        # Print response
        cprint('Fast generation: (time: {} seconds; nfe: {}; avg forward time: {} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count), 'cyan')
        print('Model\'s Response:', cleaned_generation_fast)
        print('-'*100)

        # Add the response from normal generation to the conversation history by default
        messages.append({'role': 'assistant', 'content': cleaned_generation})