# BEVCA

## Overview

BEVCA is a framework for generating and optimizing adversarial camouflage textures targeting multi-view 3D perception models, such as BEVFormer, to evaluate and improve robustness against adversarial attacks in autonomous driving systems.

## Installation and Setup

### 1. Install PyTorch3D

You need to install PyTorch3D to support 3D operations in BEVCA. Follow the official installation guide here:  
[PyTorch3D Installation Guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

### 2. Prepare Environment from BEVFormer Repository

BEVCA depends on the environment and dependencies from the [BEVFormer repository](https://github.com/fundamentalvision/BEVFormer). Please clone and set up the BEVFormer repo as instructed there to ensure all necessary packages and configurations are ready.

### 3. Download and Set Up the Dataset

Download the dataset required for camouflage generation and testing (will put in online drive shortly). Follow the dataset preparation instructions provided in the BEVFormer repo or associated documentation to organize data correctly for BEVCA.

### 4. Run Adversarial Texture Optimization

Run the BEVCA_generate_camouflage.py to generate and optimize adversarial camouflage textures



