# UrbanMind (KDD 2025)

This repository releases the research code for our KDD 2025 paper: *UrbanMind: Urban Dynamics Prediction with Multifaceted Spatial-Temporal Large Language Models*.

arXiv: https://arxiv.org/abs/2505.11654

## Notes on Code Status

- This repository is provided as a research code draft for transparency and reproducibility.
- Running the code in a new environment may require small changes, including:
  - updating dataset / checkpoint paths in data loaders and configuration sections;
  - adjusting imports and module paths if local refactoring introduced name differences.
- The code is intended for re-implementation, inspection, and extension of the methods described in the paper.

## Environment

Example tested environment (server):

- PyTorch: 2.1.2+cu121  
- CUDA (PyTorch runtime): 12.1  

## Project Structure

- models/: Core model implementations and training scripts.
  - masking/: Muffin-MAE style masking pretraining for learning multifaceted and target dynamics representations.
    - multifaced_mask/: multifaceted masking variant.
    - target_mask/: target masking variant.
  - fine_tune/: Prompt-based LLM fine-tuning and forecasting head for downstream prediction.
  - adaptation/: Test-time adaptation utilities (reconstructor-based adaptation workflow).
  - util.py: Shared utilities for model components and training helpers.
- data/: Data samples (subset / examples) illustrating expected formats and pipeline usage.

## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{liu2025urbanmind,
  title={UrbanMind: Urban Dynamics Prediction with Multifaceted Spatial-Temporal Large Language Models},
  author={Liu, Yuhang and Zhang, Yingxue and Zhang, Xin and Tian, Ling and Li, Yanhua and Luo, Jun},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={1951--1962},
  year={2025}
}

```

## License Notice

This project uses **LLaMA 3.2**, which is licensed under the LLaMA 3.2 Community License(https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE),  
Copyright © Meta Platforms, Inc. All Rights Reserved.
