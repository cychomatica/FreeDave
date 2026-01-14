import json
import torch
from transformers import AutoTokenizer

def calculate_token_match_rate(file1, file2, model_path="Gen-Verse/TraDo-4B-Instruct"):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    num_items = min(len(data1), len(data2))
    total_tokens = 0
    total_matches = 0
    sequence_matches = 0
    
    mismatches = []

    for i in range(num_items):
        # We assume the first response in the list
        s1 = data1[i].get('full_output', [""])[0]
        s2 = data2[i].get('full_output', [""])[0]
        
        # Tokenize the full strings
        tokens1 = tokenizer.encode(s1, add_special_tokens=False)
        tokens2 = tokenizer.encode(s2, add_special_tokens=False)
        
        # Identify where the assistant response starts to only count generated tokens
        # The prompt is common to both.
        # Let's find the first index where tokens differ or the length of the shorter one.
        
        # Actually, let's find the prompt length by tokenizing the prompt field if available
        prompt_str = data1[i].get('prompt', "")
        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        
        # Generated tokens are those after prompt_len
        gen_tokens1 = tokens1[prompt_len:]
        gen_tokens2 = tokens2[prompt_len:]
        
        max_gen_len = max(len(gen_tokens1), len(gen_tokens2))
        match_count = 0
        for t1, t2 in zip(gen_tokens1, gen_tokens2):
            if t1 == t2:
                match_count += 1
            else:
                break # Once it diverges, we stop counting matches for this sequence
        
        total_matches += match_count
        total_tokens += max_gen_len # Use the max length as denominator for the rate per sample
        
        if match_count == max_gen_len:
            sequence_matches += 1
        else:
            mismatches.append({
                'index': i,
                'prompt': prompt_str,
                'output1': s1,
                'output2': s2,
                'matched_tokens': match_count,
                'total_gen_tokens1': len(gen_tokens1),
                'total_gen_tokens2': len(gen_tokens2),
                'first_mismatch_token1': tokenizer.decode([gen_tokens1[match_count]]) if match_count < len(gen_tokens1) else "EOS",
                'first_mismatch_token2': tokenizer.decode([gen_tokens2[match_count]]) if match_count < len(gen_tokens2) else "EOS",
            })

    token_match_rate = (total_matches / total_tokens) * 100 if total_tokens > 0 else 0
    sequence_match_rate = (sequence_matches / num_items) * 100 if num_items > 0 else 0
    
    print(f"Comparison across {num_items} items:")
    print(f"Total Generated Tokens (max of both): {total_tokens}")
    print(f"Total Matching Prefix Tokens: {total_matches}")
    print(f"Token-level Match Rate: {token_match_rate:.4f}%")
    print(f"Sequence-level Match Rate: {sequence_match_rate:.2f}% ({sequence_matches}/{num_items})")
    
    if mismatches:
        with open('token_mismatches.json', 'w', encoding='utf-8') as f:
            json.dump(mismatches, f, indent=2, ensure_ascii=False)
        print(f"Token mismatch details saved to token_mismatches.json")

if __name__ == "__main__":
    # f1 = 'exp_results/trado_eval/temp_data/TraDo-4B-Instruct-normal-static-max_gen_length=2048-block_size=4-block_denoising_steps=4-MATH500_subset.json'
    f1 = 'exp_results/trado_eval/temp_data/TraDo-4B-Instruct-fast-draft=8-static-max_gen_length=2048-block_size=4-block_denoising_steps=4-MATH500.json'
    f2 = 'exp_results/trado_eval/temp_data/TraDo-4B-Instruct-normal-static-max_gen_length=2048-block_size=4-block_denoising_steps=4-MATH500.json'
    calculate_token_match_rate(f1, f2)
