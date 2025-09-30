<div align="center">
<br>
<h3>Free Draft-and-Verification: Towards Lossless Lossless Parallel Decoding for Diffusion Large Language Models</h3>
<p align="center">
  <a href="#">
    <img
      src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&logoColor=red"
      alt="CURE Paper on arXiv"
    />
  </a>
</p>
</div>

## Updates
- **(09/30/2025) Add Supported Model**: [LLaDA](https://github.com/ML-GSAI/LLaDA).

- **(09/23/2025) Current Supported Models**: [TraDo](https://github.com/Gen-Verse/dLLM-RL), [SDAR](https://github.com/JetAstra/SDAR), [Dream](https://github.com/DreamLM/Dream)

## Overview

We propose **FreeDave** (**Free** **D**raft-**a**nd-**Ve**rification), a fast sampling algorithm for diffusion language models, which achieves lossless parallel decoding.

FreeDave utilizes the property of diffusion language models

<p align="center">
  <img src="assets/FreeDave_pipeline.png" width="100%"/>
</p>

## Quick Start
### Environment Setup
```bash
conda create --name freedave python=3.10
source activate freedave
pip install torch==2.6.0
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
```
### Chat Examples
We provide some examples of multi-turn chat for a quick start. 
```bash
python -m chat_examples.trado_chat_example
# python -m chat_examples.dream_chat_example
# python -m chat_examples.llada_chat_example
```

## Dataset Preparation

You can navigate to `./data` to download datasets for evaluation and training, for example as follows. In that directory, you will also find detailed instructions on how to modify your own dataset.

```bash
cd data
python download_data.py --dataset MATH500
python download_data.py --dataset GSM8K
python download_data.py --dataset AIME2024
cd ..
```

After downloading the data, you are almost ready to evaluate or train diffusion language models. The only remaining step is to select (or create) a config file in `./configs` that corresponds to your project, and then use the following commands. Details on how to select and modify (or create) a config file are provided in `./configs`.


## Inference & Evaluations

After downloading the data, take the [TraDo](https://github.com/Gen-Verse/dLLM-RL) models as an example. You can set the configurations in `configs/trado_eval.yaml` (see instructions and details in `./configs`) and run the following commands to perform inference with different sampling strategies.

```bash
python -m eval.trado_eval config=configs/trado_eval.yaml
# see details in ./configs
```

Use `configs/trado_eval.yaml` for TraDo models' inference, `configs/sdar_eval.yaml` for SDAR, `configs/dream_eval.yaml` for Dream, and `configs/llada_eval.yaml` for LLaDA. A example script `run_exp.sh` is also provided for reference. 

<!-- Instructions on how to set the configurations are provided in the corresponding configuration files.   -->

<!-- We support both general tasks and coding tasks (including automated execution of code) in evaluation.   -->

Now only math tasks are supported. Support on coding tasks is in progress.

There are two main sampling methods you can choose:

- **Static Sampling:** unmask fixed number of tokens each time
- **Dynamic Sampling:** unmask tokens based on a chosen threshold, faster than static


## Citation

```
@article{wang2025trado,
  title={Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models},
  author={Wang, Yinjie and Yang, Ling and Li, Bowen and Tian, Ye and Shen, Ke and Wang, Mengdi},
  journal={arXiv preprint arXiv:2509.06949},
  year={2025}
}
```

## Acknowledgement

This repository is heavily built on [TraDo](https://github.com/Gen-Verse/dLLM-RL), and references the following open-source projects:
- [SDAR](https://github.com/JetAstra/SDAR)
- [Dream](https://github.com/DreamLM/Dream)
- [LLaDA](https://github.com/ML-GSAI/LLaDA)

and theoretical foundations:
- [MDLM](https://arxiv.org/pdf/2406.07524)
- [DiffuLLaMA](https://arxiv.org/abs/2410.17891)
- [Block Diffusion](https://arxiv.org/abs/2503.09573)