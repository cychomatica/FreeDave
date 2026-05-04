## Evaluation
The evaluation pipeline is built on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). 

First, install lm-evaluation-harness:
```
cd <dir_to_install_lm_eval>
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
``` 
Then, you can go back to this directory and run the eval scripts:
- LLaDA
    ```
    bash scripts/eval_llada.sh
    ``` 
- Dream
    ```
    bash scripts/eval_dream.sh
    ``` 
- TraDo & SDAR Models
    ```
    bash scripts/eval_trado.sh
    ``` 

The original `gsm8k` task from lm-evaluation-harness uses `exact_match`, which might not correctly extract all the answers from responses. Here we use our modified `gsm8k_math_vefify` task instead, where the only difference from the original lm_eval `gsm8k` is the additional `math_verify` metric.