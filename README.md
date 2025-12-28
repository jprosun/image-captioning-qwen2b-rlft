# Qwen2-VL-2B Captioning — SFT + RLFT (LoRA)

## Overview
This project fine-tunes `Qwen2-VL-2B-Instruct` for image captioning using a two-stage pipeline:

- **Stage 0 — Supervised Fine-Tuning (SFT):** Trains the model to produce factual, coherent captions.
- **Stage 1 — Reinforcement Learning (RLFT):** Further optimizes the SFT model with a custom reward model to:
  - encourage richer descriptions
  - penalize unnecessary verbosity
  - reduce visual hallucination

All fine-tuning uses **LoRA**, enabling training and inference on Kaggle / Colab GPUs.

## Model Pipeline
```
Qwen2-VL-2B (Base)
        ↓
Supervised Fine-Tuning (Stage 0, LoRA)
        ↓
Reinforcement Learning (Stage 1, LoRA + Reward Model)
```

At inference time, the project supports side-by-side comparison between the **Base**, **SFT**, and **RLFT** models.

## Key Features
- **Base model:** `Qwen2-VL-2B-Instruct`
- **Parameter-efficient fine-tuning:** LoRA
- **Custom reward model:** CLIP-based
- **Length-aware reward shaping**
- **4-bit inference:** via `bitsandbytes` to avoid OOM
- **Kaggle / Colab compatible**
- **Decoding controls:** deterministic or stochastic (temperature / top-p / top-k)

## Reward Model (Stage 1)
The RL stage uses a CLIP-based reward that compares:
- image embeddings
- text embeddings
- element-wise interactions between them

Reward is computed from:
- image–text alignment
- caption descriptiveness
- length regularization (handled outside the RM)

This encourages captions that are:
- more expressive than SFT
- less prone to hallucination than naive long captions

## Data Source
The project uses **13,000 images** from the first training images of the **COCO 2017 dataset**.

### Download & Extract COCO Images
To extract the 13k images from COCO train set, run:

```bash
python scripts/extract_coco_13k.py --output_dir FINAL/data/data13k
```

This script:
- Downloads COCO train2017 annotations
- Filters and extracts the first 13,000 images
- Saves them to the specified output directory

See `scripts/extract_coco_13k.py` for details and customization options.

## Training Notebooks (Kaggle)
Interactive training and experimentation notebooks are available on Kaggle:

- **[RM Optimize](https://www.kaggle.com/code/sonjpro/rm-optimize)** — Reward model optimization and tuning
- **[FT Basic RL using RM](https://www.kaggle.com/code/sonjpro/ft-basic-rl-using-rm)** — Reinforcement learning fine-tuning workflow
- **[RL LLM V2](https://www.kaggle.com/code/tunhng223/rl-llm-v2)** — Advanced RL training pipeline

These notebooks demonstrate the full training pipeline and can be run directly in Kaggle environments.
