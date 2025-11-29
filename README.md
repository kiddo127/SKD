# Stratified Knowledge-Density Super-Network for Scalable Vision Transformers (AAAI 2026)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This repository contains the official implementation of Stratified Knowledge-Density Super-Network for Scalable Vision Transformers (AAAI 2026), a novel approach for building scalable Vision Transformers (ViTs) that enable efficient sub-network extraction for diverse deployment scenarios.

## 📖 Overview
Traditional approaches require training and maintaining multiple ViT variants for different resource constraints, which is computationally expensive and inefficient. Our method transforms a pre-trained ViT into a Stratified Knowledge-Density Super-Network, where knowledge is hierarchically organized across weights, allowing flexible extraction of sub-networks that retain maximal knowledge for varying model sizes.

<img width="4487" height="3355" alt="overview" src="https://github.com/user-attachments/assets/ab707633-55cd-4e4b-8e04-6524fe707abb" />

## 🚀 Key Features
- 🔄 **One-shot Transformation**: Convert pre-trained ViTs into scalable super-networks
- 📊 **Knowledge Stratification**: Hierarchical organization of knowledge across parameter dimensions
- ⚡ **Zero-cost Extraction**: Extract sub-networks of arbitrary sizes at $\mathcal{O}(1)$ cost
- 🎯 **State-of-the-Art Performance**: Outperforms existing model compression and expansion methods
- 🔧 **Easy Deployment**: Support for diverse resource constraints from edge devices to servers

## 🏗️ Method
### Weighted PCA for Attention Contraction (WPAC)
WPAC concentrates knowledge into a compact set of critical weights through function-preserving transformations:

<img width="7544" height="3318" alt="WPAC" src="https://github.com/user-attachments/assets/f705e47b-6584-4af0-a0b7-8b2e98cdc2b2" />

- **Token-wise Weighted PCA**: Applies PCA to intermediate features with Taylor-based importance weighting
- **Function Preservation**: Mathematical equivalence maintained through transformation matrix injection
- **Information Concentration**: Knowledge condensed into top principal components

### Progressive Importance-Aware Dropout (PIAD)
PIAD enhances knowledge stratification through adaptive dropout:

<img width="5454" height="2475" alt="PIAD" src="https://github.com/user-attachments/assets/8055b200-764b-4c35-93b6-b256e2b94630" />

- **Progressive Evaluation**: Dynamically assesses importance of weight groups
- **Importance-Aware Sampling**: Lower dropout probabilities for important parameters
- **Hierarchical Training**: Promotes knowledge stratification across different model sizes


## 📊 Performance
### Model Compression Results
|Backbone|Method|MACs|Params|Top-1 Acc|
|-|-|-|-|-|
_|DeiT-Small|Original|4.26 G|22.05 M|79.83|_
|DeiT-Small|SKD (Ours)|3.07 G|16.03 M|79.42|
|DeiT-Small|SKD (Ours)|2.43 G|12.78 M|78.71|
### Sub-network Extract Results

## 💻 Usage


## 📝 Citation
If you use this work in your research, please cite our paper:

```
@inproceedings{li2026stratified,
  title={Stratified Knowledge-Density Super-Network for Scalable Vision Transformers},
  author={Li, Longhua and Qi, Lei and Geng, Xin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## 🙏 Acknowledgments
*   **Codebase:** Our work is based on the [DeiT (Data-efficient Image Transformers)](https://github.com/facebookresearch/deit) repository. We sincerely thank the authors for open-sourcing their excellent work.
*   **Funding:** This research was supported by the Jiangsu Science Foundation (BG2024036, BK20243012), the National Science Foundation of China (62125602, U24A20324, 92464301),  CAAI-Lenovo Blue Sky Research Fund, and the Fundamental Research Funds for the Central Universities (2242025K30024).

