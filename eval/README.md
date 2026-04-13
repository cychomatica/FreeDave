## Evaluation
The evaluation is built on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). 

First, install it with:
```
cd <dir_to_install_lm_eval>
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
``` 
Then, you can go back to this directory and run the eval scripts:
```
bash scripts/eval_llada.sh
``` 
```
bash scripts/eval_dream.sh
``` 
```
bash scripts/eval_trado.sh
``` 