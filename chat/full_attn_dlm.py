import torch
from modeling import get_model
from transformers import AutoTokenizer
from termcolor import cprint
import os
from generation.monitor_utils import ForwardMonitor
from generation import DLMGeneration
import argparse

def main(args):
    
    model_name = args.model_name
    model = get_model(
        model_name=model_name, 
        torch_dtype='auto', 
        device_map='cuda',
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    DLM = DLMGeneration(sdpa_additive_attention_mask=args.sdpa_additive_attention_mask)
    inference_monitor = ForwardMonitor(model)

    # Initialize conversation history
    messages = []

    print('Multi-turn conversation with {}'.format(os.path.basename(model_name)))
    print('Type ''exit'' to end the conversation')
    print('-'*100)
    
    while True:

        if not args.chat_history:
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

        with inference_monitor.count():
            output = DLM.block_decode_with_full_attention(
                model=model,
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                temperature=0.0,
                top_p=None,
                top_k=None,
                alg_temp=None,
                block_length=args.block_length,
                max_gen_length=args.max_gen_length,
                decoding_steps=args.max_gen_length,
                use_cache=True,
                dual_cache=args.dual_cache,
                mask_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['mask_token']],
                eos_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['eos_token']],
                pad_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['pad_token']],
            )
        output_ids = output.sequences.cpu()
        torch.cuda.empty_cache()

        output_text = tokenizer.decode(output_ids[0][len(tokens['input_ids'][0]):], skip_special_tokens=False)
        cleaned_text = output_text.replace(tokenizer.special_tokens_map['mask_token'], '').replace(tokenizer.special_tokens_map['eos_token'], '').replace('<|im_end|>', '').strip()

        cprint(
            'Normal generation ({}): ({})'.format(
                'dual cache' if args.dual_cache else 'prefix cache',
                inference_monitor,
            ),
            'yellow',
        )
        print('Model\'s Response:', cleaned_text)
        print('-'*100)

        with inference_monitor.count():
            output = DLM.block_decode_with_full_attention_FreeDave(
                model=model,
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                temperature=0.0,
                top_p=None,
                top_k=None,
                alg_temp=None,
                block_length=args.block_length,
                max_gen_length=args.max_gen_length,
                decoding_steps=args.max_gen_length,
                use_cache=True,
                dual_cache=args.dual_cache,
                eager_acceptance_mode=args.eager_acceptance_mode,
                draft_steps=args.draft_steps,
                draft_mode=args.draft_mode,
                mask_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['mask_token']],
                eos_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['eos_token']],
                pad_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['pad_token']],
            )
        output_ids = output.sequences.cpu()
        torch.cuda.empty_cache()

        output_text = tokenizer.decode(output_ids[0][len(tokens['input_ids'][0]):], skip_special_tokens=False)
        cleaned_text = output_text.replace(tokenizer.special_tokens_map['mask_token'], '').replace(tokenizer.special_tokens_map['eos_token'], '').replace('<|im_end|>', '').strip()

        cprint(
            'FreeDave generation ({}, {}): ({})'.format(
                'dual cache' if args.dual_cache else 'prefix cache',
                args.draft_mode,
                inference_monitor,
            ),
            'green',
        )
        print('Model\'s Response:', cleaned_text)
        # print('Trajectory:', trajectory)
        print('-'*100)

        # Add the response from normal generation to the conversation history by default
        messages.append({'role': 'assistant', 'content': cleaned_text})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat_history', type=bool, default=True)
    parser.add_argument('--model_name', type=str, default='Dream-org/Dream-v0-Instruct-7B')
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--max_gen_length', type=int, default=256)
    parser.add_argument('--dual_cache', action='store_true', default=False)
    parser.add_argument('--eager_acceptance_mode', action='store_true', default=False)
    parser.add_argument('--draft_steps', type=int, default=4)
    parser.add_argument('--draft_mode', type=str, default='tree_attention')
    parser.add_argument('--sdpa_additive_attention_mask', action='store_true', default=False, help='Set to True for Dream')
    args = parser.parse_args()
    main(args)