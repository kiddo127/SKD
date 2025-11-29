# Stratified Knowledge-Density Super-Network for Scalable Vision Transformers (AAAI 2026)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This repository contains the official implementation of [Stratified Knowledge-Density Super-Network for Scalable Vision Transformers](https://arxiv.org/abs/2511.11683) (AAAI 2026), a novel approach for building scalable Vision Transformers (ViTs) that enable efficient sub-network extraction for diverse deployment scenarios.

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
### Sub-network Extraction Results
Sub-networks are extracted from the trained SKD Super-Network and evaluated directly on ImageNet-1k **_without fine-tuning_**.
|Backbone|Method|MACs|Top-1 Acc|
|-|-|-|-|
|***DeiT-Base***|***Original***|***16.88 G***|***81.8***|
|DeiT-Base|SKD (Ours)|14.07 G|81.5|
|DeiT-Base|SKD (Ours)|11.25 G|80.9|
|DeiT-Base|SKD (Ours)|8.44 G|80.4|
|DeiT-Base|SKD (Ours)|5.63 G|77.0|
|***DeiT-Small***|***Original***|***4.26 G***|***79.8***|
|DeiT-Small|SKD (Ours)|3.55 G|79.0|
|DeiT-Small|SKD (Ours)|2.84 G|78.2|
|DeiT-Small|SKD (Ours)|2.13 G|76.2|
|DeiT-Small|SKD (Ours)|1.42 G|70.6|
|***DeiT-Tiny***|***Original***|***1.08 G***|***72.1***|
|DeiT-Tiny|SKD (Ours)|0.90 G|70.0|
|DeiT-Tiny|SKD (Ours)|0.72 G|68.6|
|DeiT-Tiny|SKD (Ours)|0.54 G|65.8|
|DeiT-Tiny|SKD (Ours)|0.36 G|61.4|

### Model Compression Results
Results after fine-tuning for 30 epochs on ImageNet-1k.
|Backbone|Method|MACs|Params|Top-1 Acc|
|-|-|-|-|-|
|***DeiT-Base***|***Original***|***16.88 G***|***86.57 M***|***81.80***|
|DeiT-Base|SKD (Ours)|10.57 G|54.55 M|81.45|
|DeiT-Base|SKD (Ours)|8.47 G|43.92 M|81.24|
|***DeiT-Small***|***Original***|***4.26 G***|***22.05 M***|***79.83***|
|DeiT-Small|SKD (Ours)|3.07 G|16.03 M|79.42|
|DeiT-Small|SKD (Ours)|2.43 G|12.78 M|78.71|
|***DeiT-Tiny***|***Original***|***1.08 G***|***5.72 M***|***72.14***|
|DeiT-Tiny|SKD (Ours)|0.89 G|4.77 M|71.40|


## 💻 Usage
1. **Convert Pre-trained Model to SKD Super-Network**

Execute `run_reparam.sh` to apply Weighted PCA for Attention Contraction (WPAC) for reparameterizing the pre-trained model. Subsequently, run `run_train.sh` to train the reparameterized model into an SKD Super-Network using Progressive Importance-Aware Dropout (PIAD).

**Note**: Replace the `--data-path` parameter with the path to your ImageNet2012 dataset (e.g., ImageNet2012/Data/CLS-LOC).

**Example Commands**:
```
# Step 1: Reparameterize the model with WPAC
sh run_reparam.sh

# Step 2: Train the SKD Super-Network with PIAD
sh run_train.sh
```

2. **Extract Sub-networks**

Execute `run_test.sh` to comprehensively evaluate the performance of sub-networks extracted from the pre-trained SKD Super-Network. This script facilitates rapid validation across multiple model sizes and configurations without requiring fine-tuning.

**Note**: Replace the `--data-path` parameter with the path to your ImageNet2012 dataset (e.g., ImageNet2012/Data/CLS-LOC).

**Example Commands**:
```
# Evaluate sub-network performance directly from SKD Super-Network
sh run_test.sh
```

3. **Fine-tuning (Optional)**

Execute `run_train_peeled.sh` to perform fine-tuning on sub-networks extracted from the SKD Super-Network, optimizing their performance for specific tasks or datasets. This process enhances the capabilities of the pre-structured sub-networks while maintaining their efficient architecture.

**Note**: Replace the `--data-path` parameter with the path to your ImageNet2012 dataset (e.g., ImageNet2012/Data/CLS-LOC).

**Example Commands**:
```
# Fine-Tune extracted sub-networks for enhanced performance
sh run_train_peeled.sh
```


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

