from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import block_diffusion_generate, block_diffusion_generate_fast
from monitor_utils import ForwardHookCounter
import time

model_name = "Gen-Verse/TraDo-4B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype="float16", device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
forward_counter = ForwardHookCounter(model)

# prompt = "What's the solution of x^2 - 2x + 1 = 0\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
prompt = 'Who is Terrence Tao? Why is he famous?'
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

tokens = tokenizer.batch_encode_plus([text], return_tensors='pt', padding=True, truncation=True, max_length=200)
tokens = {k: v.to(model.device) for k, v in tokens.items()}

forward_counter.reset_count()
start_time = time.time()
with forward_counter.count_context():
    output_ids = block_diffusion_generate(
        model,
        prompt=tokens,
        mask_id=151669,
        gen_length=256,
        block_length=16, denoising_steps=16,
        temperature=1.0, top_k=0, top_p=1.0,
        remasking_strategy="low_confidence_static",
        confidence_threshold=0.9
    )
end_time = time.time()
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
cleaned_text = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '')
print('-'*100)
print('Normal generation: (time: {:.4f} seconds; num of forward passes: {}; avg step forward time: {:.4f} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count))
print(cleaned_text)

forward_counter.reset_count()
start_time = time.time()
with forward_counter.count_context():
    output_ids = block_diffusion_generate_fast(
        model,
        prompt=tokens,
        mask_id=151669,
        gen_length=256,
        block_length=16, denoising_steps=16, draft_steps=8,
        temperature=1.0, top_k=0, top_p=1.0,
        remasking_strategy="low_confidence_static",
        confidence_threshold=0.9
    )
end_time = time.time()
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
cleaned_text = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '')
print('-'*100)
print('Fast generation: (time: {:.4f} seconds; num of forward passes: {}; avg step forward time: {:.4f} seconds)'.format(end_time - start_time, forward_counter.counter.count, (end_time - start_time) / forward_counter.counter.count))
print(cleaned_text)
