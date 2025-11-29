# Stratified Knowledge-Density Super-Network for Scalable Vision Transformers (AAAI 2026)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This repository contains the official implementation of Stratified Knowledge-Density Super-Network for Scalable Vision Transformers (AAAI 2026), a novel approach for building scalable Vision Transformers (ViTs) that enable efficient sub-network extraction for diverse deployment scenarios.

## 📖 Overview
Traditional approaches require training and maintaining multiple ViT variants for different resource constraints, which is computationally expensive and inefficient. Our method transforms a pre-trained ViT into a Stratified Knowledge-Density Super-Network, where knowledge is hierarchically organized across weights, allowing flexible extraction of sub-networks that retain maximal knowledge for varying model sizes.
