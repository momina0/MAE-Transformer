# Masked Autoencoder (MAE) from Scratch

Self-supervised image representation learning with a **Masked Autoencoder** built from base PyTorch layers, with no `timm` or pretrained weights.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![ViT](https://img.shields.io/badge/ViT--B%2F16-encoder-blue) ![Self-Supervised](https://img.shields.io/badge/self--supervised-learning-8A2BE2)

## Overview
The model hides **75% of image patches** and learns to reconstruct them, which forces the encoder to learn semantic visual features without labels.

- **Encoder:** ViT-Base (B/16) that only sees the visible patches
- **Decoder:** lightweight Transformer that reconstructs the masked pixels
- **Dataset:** TinyImageNet
- **Evaluation:** reconstruction quality with **PSNR** and **SSIM**, plus visualisations of masked, reconstructed and original images

## Pipeline
1. Patchify images into 16x16 tokens and apply random masking (ratio 0.75)
2. Encode visible tokens, then append mask tokens and positional embeddings
3. Decode and compute MSE loss on the masked patches only
4. Visualise reconstructions and report PSNR/SSIM

## Run
Open `genas2.ipynb` in Jupyter, Colab or Kaggle (GPU recommended) and run the cells in order.
