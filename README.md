# 3D-QSSM

**3D-QSSM: A Quasi-Separable Vision State-Space Model for Scalable Foundation Models in Volumetric Medical Imaging**


<p align="center">
  <img src="3d_qssm_architecture.png" width="900">
</p>

<p align="center">
  <b>Overview of the 3D-QSSM framework.</b>
</p>

This repository provides the implementation of **3D-QSSM**, a fully three-dimensional quasi-separable state-space model designed for scalable self-supervised learning in volumetric medical imaging.

The framework combines a bidirectional QSSM backbone with 3D masked autoencoder pretraining.

## Model Configuration

The default configuration used in the paper is:

* Input volume: `160 × 160 × 160`
* Patch size: `16 × 16 × 16`
* Number of tokens: `1000`
* Encoder depth: `12`
* Encoder embedding dimension: `384`
* Decoder depth: `12`
* Decoder embedding dimension: `192`
* Masking ratio: `75%`
  
## Hydra dependency

3D-QSSM uses the official Hydra implementation of bidirectional
quasiseparable state-space mixing:

https://github.com/goombalab/hydra

Hydra is included as an external dependency and should be installed
before running the 3D-QSSM models.

## Downstream Tasks

3D-QSSM is evaluated on:

* Brain age prediction
* Alzheimer's disease classification
* Brain tumor segmentation

## Repository Status

The public implementation is currently being organized and documented.

Training scripts, pretrained weights, downstream fine-tuning code, and detailed installation instructions will be added progressively.

## Key Results

3D-QSSM was evaluated across regression, classification, and segmentation tasks using 5-fold cross-validation and few-shot settings.

* **Brain age prediction:** up to **25% lower MAE** compared with transformer-based MAE baselines.
* **Alzheimer's disease classification:** approximately **3–5% improvement** in classification performance.
* **Brain tumor segmentation:** approximately **6–25% improvement in Dice score** across tumor regions and data regimes.
* **Few-shot learning:** consistently stronger performance when only a fraction of labelled data is available.
* **Memory efficiency:** mean GPU memory usage of **39.17 GB**, compared with **49.58 GB** for 3D Swin-MAE and **67.95 GB** for 3D ViT-MAE.

These results indicate that 3D-QSSM combines improved downstream performance with substantially lower and more stable GPU memory consumption for volumetric medical imaging.

