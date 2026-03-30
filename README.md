# Vesuvius Surface Detection

3D surface segmentation pipeline for the **Kaggle Vesuvius Challenge - Surface Detection**, built from iterative experimentation across custom 2.5D U-Net, PyTorch SMP baselines, and nnUNetv2 / ResEnc L.

## Overview

This project focuses on reconstructing the surface of carbonized Herculaneum scrolls from volumetric CT scans. Unlike a standard binary segmentation task, the competition score combines:

- **TopoScore** (30%)
- **SurfaceDice** (35%)
- **VOI_score** (35%)

That means performance depends not only on overlap quality, but also on preserving topology and avoiding split/merge errors.

## Results

### Key milestones
- Custom **Keras 2.5D U-Net** baseline: validation Dice **0.2828**
- **PyTorch SMP + pretrained encoder** baseline: validation Dice **0.4342**
- **nnUNetv2 3D full-resolution**: validation Dice up to **0.527**
- **nnUNet ResEnc L** with merged dataset and continued training: validation Dice **0.5710+**
- Best public leaderboard score recorded in project notes: **0.524**

## What this repository shows

This repository is a **cleaned public version** of a larger private experimentation workspace. It preserves the core technical ideas:

- custom nnUNet trainer monitoring and checkpoint management
- continued training across Kaggle and Colab environments
- threshold and postprocessing validation workflows
- topology-aware / VOI-aware postprocessing experiments
- legacy baselines showing model evolution

Large datasets, private checkpoints, and generated artifacts are intentionally excluded.

## Repository structure

```text
src/
  trainers/
    nnUNetTrainerWithMonitor.py      # auto-save and monitoring trainer
  training/
    continue_2540_resenc.py          # continued training on merged dataset
  inference/
    submit_resenc_l_v2.py            # leaderboard submission pipeline
  analysis/
    validate_resenc_threshold.py     # threshold/postprocessing validation
    postprocess_voi_topo.py          # topology / VOI-aware postprocessing
  legacy/
    vesuvius_smp_train.py            # SMP baseline
    vesuvius_colab_complete.py       # early Keras/JAX baseline

docs/
  research_notes.md                  # method research and leaderboard context
  experiment_timeline.md             # iteration log and results timeline
  project_context.md                 # project-specific implementation notes
```

## Technical highlights

### 1. Multi-stage model evolution
The project evolved from a custom 2.5D approach to pretrained 2D/2.5D segmentation and finally to 3D nnUNet-based pipelines, reflecting a shift from convenience-driven experimentation to architecture-task alignment.

### 2. Cross-platform training workflow
The training scripts support multi-hour experiments across:
- **Kaggle dual-T4 DDP** setups
- **Colab L4** continuation runs
- resumed checkpoint workflows with custom trainer hooks

### 3. Evaluation-aware postprocessing
The project explicitly studies how postprocessing affects not just Dice but the full leaderboard objective, including:
- dust removal
- hole filling
- hysteresis-style thresholding
- topology-aware postprocessing
- VOI-driven trade-offs between merge/split behavior

### 4. Practical competition debugging
The logs and notes capture several real failure modes that were solved during experimentation, including:
- ignore-label handling (`label=2`)
- volume axis / coordinate alignment between validation and submission
- checkpoint continuation strategy
- Kaggle filesystem constraints and disk cleanup

## Reproducibility notes

This repository is primarily intended as a **project showcase and technical reference**, not a one-command reproducible benchmark repo. Some scripts still contain cloud-platform-specific paths from the original competition workflow.

If I fully productionize this repository later, the next steps would be:
- parameterize paths and runtime configuration
- package helpers into reusable modules
- add lightweight unit tests for postprocessing utilities
- add a small synthetic example dataset for smoke tests

## Why this project matters

This project reflects the type of work I enjoy most:
- machine learning under real evaluation constraints
- long-cycle experimentation
- debugging subtle data / inference issues
- bridging research iteration with usable engineering workflows

## Related skills demonstrated

- Python
- PyTorch
- nnUNetv2
- 3D medical-image-style segmentation workflows
- experiment tracking
- postprocessing and evaluation analysis
- Kaggle / Colab / GPU pipeline adaptation

## Notes

This repository intentionally excludes the original competition data and large generated artifacts.
