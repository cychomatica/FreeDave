<div align="center">
<br>
<h3>Free Draft-and-Verification: Towards Lossless Sampling Acceleration for Diffusion Large Language Models</h3>
<p align="center">
  <a href="#">
    <img
      src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&logoColor=red"
      alt="CURE Paper on arXiv"
    />
  </a>
</p>
</div>

## Features 

- **Model Support**: [TraDo](https://github.com/Gen-Verse/dLLM-RL), [SDAR](https://github.com/JetAstra/SDAR), [Dream](https://github.com/DreamLM/Dream), [LLaDA](https://github.com/ML-GSAI/LLaDA), [MMaDA](https://github.com/Gen-Verse/MMaDA), [Diffu-Coder](https://github.com/apple/ml-diffucoder) We support models with diverse structures, including full attention models, adapted models, and block attention models.
- **Inference Acceleration**: improved [KV-cache](https://github.com/NVlabs/Fast-dLLM/tree/main), [jetengine](https://github.com/Labman42/JetEngine/tree/0ddc55ad3fb712b6374515b78d656f420e1a7243) (based on nano-vllm), different sampling strategies, support multi-nodes, easy to build your own accelerated inference methods


## Overview

We propose **FreeDave** (**Free** **D**raft-**a**nd-**Ve**rification), a fast sampling algorithm for diffusion language models, which achieves multi-token without the sacrifice of generation quanlity. 

FreeDave utilizes the property of diffusion language models 

<p align="center">
  <img src="assets/FreeDave Pipeline.png" width="100%"/>
</p>


## Quick Start


```bash
conda create --name freedave python=3.10
source activate freedave
pip install torch==2.6.0
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
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

After downloading the data, take [TraDo](https://github.com/Gen-Verse/dLLM-RL) models as an example. You can set the configurations in `configs/trado_eval.yaml` (see instructions and details in `./configs`) and run the following commands to perform inference with different sampling strategies.
```bash
python -m eval.trado_eval config=configs/trado_eval.yaml
# see details in ./configs
```
Use `trado_eval.yaml` for TraDo models' inference, `sdar_eval.yaml` for SDAR, `dream_eval.yaml` for Dream and Diffu-Coder, and `llada_eval.yaml` for LLaDA and MMaDA. 
<!-- Instructions on how to set the configurations are provided in the corresponding configuration files.   -->
<!-- We support both general tasks and coding tasks (including automated execution of code) in evaluation.   -->
Now only math tasks are supported. Support on coding tasks is in progress.

There are two main sampling methods you can choose:

- **Static Sampling:** unmask fixed number of tokens each time

- **Dynamic Sampling:** unmask tokens based on a chosen threshold, faster than static

To have a look how diffusion language models sample, open `./sample/trace.viewer.html` in your browser, or generate trajectory by your self with `./sample/get_trace_viewer.py`.


You can also perform inference across multiple nodes using `multinode_eval.py` with the same configuration files, with only minor modifications as instructed in the configuration files.
In multi-node setup, the first node controls the others. You can run  
`python multinode_eval.py config=configs/dream_multinode_eval.yaml` on the first node to eval, or submit the following as the entry command for a job:


## 📖 Citation

```
@article{wang2025trado,
  title={Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models},
  author={Wang, Yinjie and Yang, Ling and Li, Bowen and Tian, Ye and Shen, Ke and Wang, Mengdi},
  journal={arXiv preprint arXiv:2509.06949},
  year={2025}
}
```


## 🤝 Acknowledgement

This repository is heavily built on the following open-source projects:

- [TraDO](https://github.com/Gen-Verse/dLLM-RL)
- [SDAR](https://github.com/JetAstra/SDAR)
- [Dream](https://github.com/DreamLM/Dream)
- [LLaDA](https://github.com/ML-GSAI/LLaDA)
<!-- [MMaDA](https://github.com/Gen-Verse/MMaDA/tree/main). -->

and theoretical foundations:

- [MDLM](https://arxiv.org/pdf/2406.07524)
- [DiffuLLaMA](https://arxiv.org/abs/2410.17891)
- [Block Diffusion](https://arxiv.org/abs/2503.09573)


## 💬 Discussion

Please do not hesitate to report any issues or difficulties you encounter.







