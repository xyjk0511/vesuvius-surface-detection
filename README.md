# Vesuvius Surface Detection / 维苏威古卷表面分割

![Python](https://img.shields.io/badge/Python-3D%20Segmentation-blue)
![Competition](https://img.shields.io/badge/Kaggle-Vesuvius%20Challenge-20BEFF)
![Val%20Dice](https://img.shields.io/badge/Val%20Dice-0.571%2B-success)
![Leaderboard](https://img.shields.io/badge/LB-0.524-orange)

## At a glance / 项目速览

| Item | Summary |
|------|---------|
| Task | 3D surface segmentation from CT volumes / 基于 CT 体数据的 3D 表面分割 |
| Competition metric | TopoScore + SurfaceDice + VOI_score |
| Best validation result | **0.5710+** |
| Best public LB | **0.524** |
| Project value | 3D segmentation, nnUNet, evaluation-aware postprocessing |

3D surface segmentation pipeline for the **Kaggle Vesuvius Challenge - Surface Detection**, built through iterative experimentation from custom 2.5D baselines to nnUNetv2 / ResEnc L.

一个面向 **Kaggle Vesuvius Challenge - Surface Detection** 的 3D 表面分割项目，记录了从自定义 2.5D 基线到 nnUNetv2 / ResEnc L 的完整迭代过程。

---

## Overview / 项目概述

This project focuses on reconstructing the surface of carbonized Herculaneum scrolls from volumetric CT scans.

这个项目的目标，是从 3D CT 扫描中恢复被碳化卷轴的表面结构，用于后续虚拟展开与文本识别。

Unlike a standard binary segmentation task, the competition score combines:

- **TopoScore** (30%)
- **SurfaceDice** (35%)
- **VOI_score** (35%)

So performance depends not only on overlap quality, but also on preserving topology and avoiding split/merge errors.

因此这不是普通的 Dice 分割任务；模型不仅要分得准，还要尽量保持拓扑连续性，并减少分裂 / 合并错误。

---

## Results / 核心结果

### Key milestones / 关键结果
- Custom **Keras 2.5D U-Net** baseline: validation Dice **0.2828**
- **PyTorch SMP + pretrained encoder** baseline: validation Dice **0.4342**
- **nnUNetv2 3D full-resolution**: validation Dice up to **0.527**
- **nnUNet ResEnc L** with merged dataset and continued training: validation Dice **0.5710+**
- Best public leaderboard score recorded in project notes: **0.524**

### What improved over time / 提升是怎么来的
This project improved through:
- moving from 2.5D to 3D segmentation
- scaling up training data
- continued training on stronger configurations
- threshold and postprocessing analysis aligned with the competition metric

这个项目的提升主要来自：
- 从 2.5D 过渡到 3D segmentation
- 扩大训练数据规模
- 在更强配置上继续训练
- 针对竞赛指标做阈值与后处理优化

---

## What this repository shows / 这个仓库主要展示什么

This repository is a **cleaned public version** of a larger private experimentation workspace. It preserves the most representative technical ideas:

- custom nnUNet trainer monitoring and checkpoint management
- continued training across Kaggle and Colab environments
- threshold and postprocessing validation workflows
- topology-aware / VOI-aware postprocessing experiments
- legacy baselines showing model evolution

这是从一个更大的私人实验工作区中整理出来的公开版本，保留了最能体现项目价值的部分：
- 自定义 nnUNet trainer 与 checkpoint 管理
- Kaggle / Colab 跨平台继续训练
- 阈值与后处理验证流程
- 面向 TopoScore / VOI 的后处理实验
- 早期基线模型，展示技术演进路径

Large datasets, private checkpoints, and generated artifacts are intentionally excluded.

原始数据、checkpoint 和大体积生成产物没有包含在仓库中。

---

## Repository structure / 仓库结构

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

---

## Technical highlights / 技术亮点

### 1. Multi-stage model evolution / 多阶段模型演进
The project evolved from a custom 2.5D approach to pretrained segmentation baselines and finally to 3D nnUNet-based pipelines.

这个项目不是从一开始就选对路线，而是经历了从自定义 2.5D → 预训练 2D/2.5D → 3D nnUNet 的逐步演进。

### 2. Cross-platform training workflow / 跨平台训练工作流
The training scripts support long-running experiments across:
- **Kaggle dual-T4 DDP**
- **Colab L4** continuation runs
- resumed checkpoint workflows with custom trainer hooks

训练部分支持多平台、多阶段实验推进：
- Kaggle 双 T4 DDP
- Colab L4 续训
- 基于自定义 trainer hook 的 checkpoint 恢复

### 3. Evaluation-aware postprocessing / 面向竞赛指标的后处理
The repository explicitly studies how postprocessing affects not just Dice, but the full leaderboard objective:
- dust removal
- hole filling
- hysteresis-style thresholding
- topology-aware postprocessing
- VOI-oriented merge/split trade-offs

这里最有价值的一点是：后处理不是围绕 Dice 单独优化，而是围绕完整竞赛分数来分析。

### 4. Practical debugging under competition constraints / 竞赛环境下的实际调试能力
The project documents several real failure modes that had to be solved during experimentation:
- ignore-label handling (`label=2`)
- volume axis / coordinate alignment between validation and submission
- checkpoint continuation strategy
- Kaggle filesystem and disk-cleanup constraints

这些问题都非常真实，也很能体现工程与调试能力。

---

## Reproducibility / 复现说明

This repository is primarily intended as a **project showcase and technical reference**, not a one-command benchmark repo.

本仓库主要是一个**项目展示 + 技术参考**仓库，而不是一键完整复现的 benchmark 仓库。

Some scripts still contain competition-environment assumptions inherited from the original workflow.

部分脚本仍然保留了原竞赛环境中的路径和平台假设。

If I continue productionizing it, the next steps would be:
- parameterize paths and runtime configuration
- package helpers into reusable modules
- add small synthetic examples or smoke-test inputs
- add unit tests for postprocessing utilities

如果继续整理成更标准的开源项目，我下一步会：
- 参数化路径和运行配置
- 把 helper 逻辑拆成可复用模块
- 增加小型示例或 smoke tests
- 为后处理工具补测试

---

## Why this project matters / 为什么这个项目重要

This is one of the strongest examples of how I work on ML problems under real constraints:
- long-cycle experimentation
- model iteration guided by evidence
- debugging subtle validation/submission mismatches
- bridging research intuition with practical engineering

这是我最能代表自己建模方式的项目之一，体现了：
- 长周期实验推进
- 基于证据的迭代
- 处理验证 / 提交不一致这类细节问题
- 把研究直觉和工程实现结合起来
