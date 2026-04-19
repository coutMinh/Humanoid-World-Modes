# Project: Humanoid World Models (HWM) 🤖
**Lightweight Open World Foundation Models for Humanoid Robotics**

**Course:** Introduction to Machine Learning  
**Instructor:** 
**Students:** 
- Tran Ta Quang Minh
- Le Nguyen Khang
- Nguyen Tan Hung

---

## 📖 Introduction
This project is based on the research paper *"Humanoid World Models: Open World Foundation Models for Humanoid Robotics" (Qasim Ali et al., 2025)*.

The goal of this project is to explore and build **World Models** specifically designed for humanoid robots. In our setting, this system acts as an action-conditioned video generator: it **predicts future video frames** based on past visual observations and intended robotic control actions.

By simulating the outcomes of actions before trying them in the real world, these models enable long-horizon planning and safe policy learning in complex, open-world environments. A core advantage of this project is its focus on **lightweight** architectures that can be trained and deployed on modest academic hardware (e.g., 1–2 GPUs).

## 🚀 Methodology
This project investigates and implements two distinct generative paradigms for video generation:

1. **Masked Humanoid World Model (Masked-HWM):** 
   - Operates in a discrete latent space (via VQ-VAE).
   - Trained using the Masked Video Modelling (MVM) paradigm with a non-autoregressive Transformer architecture.
2. **Flow Humanoid World Model (Flow-HWM):** 
   - Operates in a continuous latent space.
   - Trained using the Flow Matching framework to learn a time-dependent velocity field that transforms Gaussian noise into video samples.

**Architectural Optimizations (Efficiency):**
To make the models lightweight, we explore different Transformer block designs:
- **Attention Mechanisms:** Comparing *Joint Attention* (processing all tokens together) vs. *Split Attention* (self-attention followed by cross-attention).
- **Parameter Sharing:** Testing *Modality Sharing* (sharing weights within video/action streams) and *Full Sharing* (sharing across all streams). These techniques reduce model size by **33%–53%** with minimal impact on visual fidelity.

## 📊 Dataset
- **1xGPT Dataset:** Contains 100 hours of egocentric (first-person) video captured from the Humanoid EVE robot.
- **Inputs:** 256x256 resolution RGB frames paired with a 25-dimensional action vector (representing joint pitch-yaw-roll angles, movement velocities, and hand closure states).

## 🏆 Key Findings
Based on the experiments (from the original paper):
- The **Masked-HWM** consistently outperforms the Flow-HWM variant in both overall visual quality and sampling speed under restricted compute conditions.
- The **Full Sharing** parameter strategy proves to be highly effective. It halves the parameter count and achieves the fastest inference speed while maintaining competitive image quality metrics (FID, PSNR).

## 📂 Repository Structure
```text
├── data/                  # Scripts for downloading and preprocessing the 1xGPT dataset
├── models/                # Source code defining model architectures (Masked-HWM, Flow-HWM)
│   ├── modules/           # Building blocks (Transformer layers, Attention, Parameter Sharing)
│   └── vae/               # Tokenizers (VQ-VAE / Continuous VAE)
├── train.py               # Script for training the world models
├── evaluate.py            # Script for evaluation (calculating FID, PSNR)
├── inference.py           # Script to generate future video predictions from a checkpoint
├── requirements.txt       # Python dependencies
└── README.md              # This documentation file
