from utils.determinism_utils import deterministic

def main(chat_history=False):
    # All imports inside main() so they happen after deterministic context is entered
    import torch
    from modeling.sdar import SDARForCausalLM
    import modeling.sdar.modeling_sdar as sdar_module
    from transformers import AutoTokenizer
    from generate.trado_generate import block_diffusion_generate, block_diffusion_generate_FreeDave
    import time, os
    from utils.monitor_utils import ForwardHookCounter
    from termcolor import cprint

    
    cprint(f'flash_attn_available: {sdar_module.flash_attn_available}', 'yellow')
    cprint(f'liger_kernel_is_available: {sdar_module.liger_kernel_is_available}', 'yellow')
    cprint(f'use_eager_attn: {sdar_module.USE_EAGER_ATTN}', 'yellow')

    model_name = 'Gen-Verse/TraDo-4B-Instruct' # use local SDARForCausalLM instead of AutoModelForCausalLM to control the determinism
    model = SDARForCausalLM.from_pretrained(
        model_name, 
        torch_dtype='float16', 
        device_map='cuda'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    forward_counter = ForwardHookCounter(model)

    # Initialize conversation history
    messages = []

    print('Multi-turn conversation with {}'.format(os.path.basename(model_name)))
    print('Type ''exit'' to end the conversation')
    print('-'*100)
    
    while True:

        if not chat_history:
            messages = []

        prompt = input('Enter your question: \n')
        print('-'*100)

        # Check if user wants to exit
        if prompt.lower() == 'exit':
            print('Conversation ended.')
            break

        # Add user message to conversation history
        messages.append({'role': 'user', 'content': prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.batch_encode_plus(
            [text], return_tensors='pt', padding=True, truncation=True, max_length=200
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        # Generate response
        forward_counter.reset_count()
        start_time = time.time()
        with forward_counter.count_context():
            output_ids, _, _ = block_diffusion_generate(
                model,
                prompt=tokens,
                mask_id=151669,
                gen_length=256,
                block_length=4, 
                denoising_steps=4,
                temperature=1.0, 
                top_k=1, 
                top_p=1.0,
                remasking_strategy='low_confidence_static',
                confidence_threshold=0.9
            )
        end_time = time.time()
        output_ids = output_ids.cpu()
        torch.cuda.empty_cache()

        # Process response
        output_text = tokenizer.decode(output_ids[0][len(tokens['input_ids'][0]):], skip_special_tokens=False)
        cleaned_text = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()

        # Print response
        cprint('Normal generation: (time: {:.4f} seconds; nfe: {}; avg forward time: {:.4f} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count), 'yellow')
        print('Model\'s Response:', cleaned_text)
        print('-'*100)

        # Generate response
        forward_counter.reset_count()
        start_time = time.time()
        with forward_counter.count_context():
            output_ids_fast, _, _ = block_diffusion_generate_FreeDave(
                model,
                prompt=tokens,
                mask_id=151669,
                gen_length=256,
                block_length=4, 
                denoising_steps=4, 
                draft_steps=8,
                temperature=1.0, 
                top_k=1, 
                top_p=1.0,
                remasking_strategy='low_confidence_static',
                confidence_threshold=0.9,
                eager_acceptance_mode=True
            )
        end_time = time.time()
        output_ids_fast = output_ids_fast.cpu()
        torch.cuda.empty_cache()

        # Process response
        output_text_fast = tokenizer.decode(output_ids_fast[0][len(tokens['input_ids'][0]):], skip_special_tokens=False)
        cleaned_text_fast = output_text_fast.replace('<|MASK|>', '').replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()

        # Print response
        cprint('FreeDave generation: (time: {:.4f} seconds; nfe: {}; avg forward time: {:.4f} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count), 'cyan')
        print('Model\'s Response:', cleaned_text_fast)
        print('-'*100)

        # Add the response from normal generation to the conversation history by default
        messages.append({'role': 'assistant', 'content': cleaned_text_fast})

if __name__ == '__main__':
    with deterministic(enabled=True, seed=42):
        main(chat_history=True)