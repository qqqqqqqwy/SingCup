# SingCup: Solute-level Sugar Concentration Detection via Variable Acoustic Resonance Channel Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation for the paper: **"SingCup: Solute-level Sugar Concentration Detection via Variable Acoustic Resonance Channel Modeling"**.

SingCup is a low-cost, compact system designed to detect sugar concentration in liquids using active acoustic sensing. [cite_start]Unlike traditional static methods, SingCup leverages **Variable Resonance Spectrograms (VRS)** generated during the liquid pouring process to capture rich, dynamic channel responses[cite: 7, 47].

## 🏗️ System Architecture

The software pipeline consists of three main components:

1.  [cite_start]**VRS Generation:** Captures dynamic resonance patterns via STFT as liquid is poured[cite: 7].
2.  [cite_start]**Denoising Module (U-Net):** A customized U-Net model designed to suppress structural vibration noise and environmental interference[cite: 8, 207].
3.  [cite_start]**CoLA-Net (Convolutional LSTM with Attention):** The core regression model that extracts resonance textures (1D-CNN), fuses mass features, and models temporal dependencies (LSTM + Attention) for precise concentration estimation[cite: 9, 321].

## 📂 Project Structure

```text
.
├── dataloader.py    # Custom Dataset classes for Liquid and Denoise tasks
├── main.py          # Entry point for training and evaluation
├── model.py         # Model definitions (CoLA-Net, UNet2D, ResNet18, TCN, etc.)
├── run.sh           # Shell script to reproduce all experiments
├── trainer.py       # Trainer for the main regression task (CoLA-Net)
├── trainer_cla.py   # Trainer for classification tasks (Solute/Concentration)
├── trainer_noise.py # Trainer for the U-Net denoising module
├── utils.py         # Utility functions (seeding, metrics, preprocessing)
└── scripts/         # Sub-scripts called by run.sh
