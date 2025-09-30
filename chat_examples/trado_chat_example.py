from transformers import AutoModelForCausalLM, AutoTokenizer
from generate.trado_generate import block_diffusion_generate, block_diffusion_generate_FreeDave
from utils.monitor_utils import ForwardHookCounter
import time
import os
from termcolor import cprint

model_name = "Gen-Verse/TraDo-4B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype="float16", device_map="cuda"
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

    prompt = input("Enter your question: \n")
    print('-'*100)

    # Check if user wants to exit
    if prompt.lower() == 'exit':
        print('Conversation ended.')
        break

    # messages = [{"role": "user", "content": prompt}]
    messages.append({'role': 'user', 'content': prompt})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer.batch_encode_plus(
        [text], return_tensors='pt', padding=True, truncation=True, max_length=200
    )
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    forward_counter.reset_count()
    start_time = time.time()
    with forward_counter.count_context():
        output_ids = block_diffusion_generate(
            model,
            prompt=tokens,
            mask_id=151669,
            gen_length=256,
            block_length=4, 
            denoising_steps=4,
            temperature=1.0, 
            top_k=1, 
            top_p=1.0,
            remasking_strategy="low_confidence_static",
            confidence_threshold=0.9
        )
    end_time = time.time()
    output_text = tokenizer.decode(output_ids[0][len(tokens['input_ids'][0]):], skip_special_tokens=False)
    cleaned_text = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
    cprint('Normal generation: (time: {:.4f} seconds; nfe: {}; avg forward time: {:.4f} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count), 'yellow')
    print('Model\'s Response:', cleaned_text)
    print('-'*100)

    forward_counter.reset_count()
    start_time = time.time()
    with forward_counter.count_context():
        output_ids_fast = block_diffusion_generate_FreeDave(
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
            remasking_strategy="low_confidence_static",
            confidence_threshold=0.9,
            eager_acceptance_mode=True
        )
    end_time = time.time()
    output_text_fast = tokenizer.decode(output_ids_fast[0][len(tokens['input_ids'][0]):], skip_special_tokens=False)
    cleaned_text_fast = output_text_fast.replace('<|MASK|>', '').replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
    cprint('Fast generation: (time: {:.4f} seconds; nfe: {}; avg forward time: {:.4f} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count), 'cyan')
    print('Model\'s Response:', cleaned_text_fast)
    print('-'*100)

    # Add the response from normal generation to the conversation history by default
    messages.append({'role': 'assistant', 'content': cleaned_text_fast})