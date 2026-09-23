# anonymous-ai-image-study
# Human vs. AI Image Detection

This repository contains the code, data metadata, and results associated with our research study comparing human and AI-based detection of AI-generated images.

## Research Study

The study investigates how effectively humans and an AI image detector distinguish between AI-generated and real images.

The stimulus set consists of 60 images across 10 categories:
- 30 AI-generated images
- 30 real images
- 3 AI-generated and 3 real images per category

Human participants classified the images as AI-generated or real, reported their confidence, and identified visual cues influencing their decisions.

The automated evaluation uses a pretrained image-detection model on the stimulus images, with its predictions compared against the ground-truth labels.

Purpose

The repository provides the computational materials required to reproduce the automated image-detection component of the study and supports comparison between human and machine performance.

Model

The automated evaluation uses the pretrained Steganograph-IA detector:
delpot/steganograph-ia-detector

Model: https://huggingface.co/delpot/steganograph-ia-detector
