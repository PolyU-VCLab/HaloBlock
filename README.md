<div align="center">

# HaloBlock: Faster and Better Decoding for Diffusion Large Language Models

<div align="center">
  <a href="https://opensource.org/license/mit-0">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
  <a href="https://arxiv.org/pdf/2601.xxxxx">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-red.svg">
  </a>
</div>

</div>

Official implementation of **HaloBlock**, a training-free difficulty-aware adaptive halo mechanism for block-wise decoding in diffusion large language models (dLLMs). It mitigates the left-dominance effect and boundary contextual discontinuity in block-wise decoding, achieving better generation quality and faster inference convergence.

## Installation

```bash
conda create -n haloblock python==3.10
conda activate haloblock
pip install -r requirements.txt
```

## Evaluation

For LLaDA evaluation,
```
cd llada

# GSM8K (5-shot)
sh eval_gsm8k.sh

# MATH (4-shot)
sh eval_math.sh

# HumanEval (0-shot)
sh eval_humaneval.sh

# MBPP (3-shot)
sh eval_mbpp.sh
```

For Dream evaluation,
```
cd dream

# GSM8K (5-shot)
sh eval_gsm8k.sh

# MATH (4-shot)
sh eval_math.sh

# HumanEval (0-shot)
sh eval_humaneval.sh

# MBPP (3-shot)
sh eval_mbpp.sh
```

## Citation

If you find this work useful, please cite our paper:

```
@Article{zheng2026haloblock,
  title={HaloBlock: Faster and Better Decoding for Diffusion Large Language Models},
  author={Zheng, Zhaohui and He, Chenhang and Zhang, Jinrui and Cheng, Ming-Ming and Zhang, Lei},
  journal={arXiv preprint arXiv:2601.xxxxxx},
  year={2026}
}
```

## Acknowledgements

This codebase is inherited from [Fast dLLM](https://github.com/NVlabs/Fast-dLLM) and [dParallel](https://github.com/czg1225/dParallel). We would like to thank the authors of [LLaDA](https://github.com/ML-GSAI/LLaDA) and [Dream](https://github.com/HKUNLP/Dream) for open-sourcing the models, and the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) team for open-sourcing the evaluation framework.
