# Stratified Knowledge-Density Super-Network for Scalable Vision Transformers (AAAI 2026)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This repository contains the official implementation of Stratified Knowledge-Density Super-Network for Scalable Vision Transformers (AAAI 2026), a novel approach for building scalable Vision Transformers (ViTs) that enable efficient sub-network extraction for diverse deployment scenarios.

## 📖 Overview
Traditional approaches require training and maintaining multiple ViT variants for different resource constraints, which is computationally expensive and inefficient. Our method transforms a pre-trained ViT into a Stratified Knowledge-Density Super-Network, where knowledge is hierarchically organized across weights, allowing flexible extraction of sub-networks that retain maximal knowledge for varying model sizes.

<img width="4487" height="3355" alt="overview" src="https://github.com/user-attachments/assets/ab707633-55cd-4e4b-8e04-6524fe707abb" />

## 🚀 Key Features
🔄 One-shot Transformation: Convert pre-trained ViTs into scalable super-networks
📊 Knowledge Stratification: Hierarchical organization of knowledge across parameter dimensions
⚡ Zero-cost Extraction: Extract sub-networks of arbitrary sizes at O(1)cost
🎯 State-of-the-Art Performance: Outperforms existing model compression and expansion methods
🔧 Easy Deployment: Support for diverse resource constraints from edge devices to servers
