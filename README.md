# 3D-QSSM

**3D-QSSM: A Quasi-Separable Vision State-Space Model for Scalable Foundation Models in Volumetric Medical Imaging**


<p align="center">
  <img src="Picture6.png" width="900">
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

## Downstream Tasks

3D-QSSM is evaluated on:

* Brain age prediction
* Alzheimer's disease classification
* Brain tumor segmentation

## Repository Status

The public implementation is currently being organized and documented.

Training scripts, pretrained weights, downstream fine-tuning code, and detailed installation instructions will be added progressively.
