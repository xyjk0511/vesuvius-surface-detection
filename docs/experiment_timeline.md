# Vesuvius Challenge - 项目进度记录

## 项目概述

- **竞赛**: Vesuvius Challenge - Surface Detection
- **目标**: 从 3D 体数据（TIF 格式）中预测表面掩码
- **评估指标**: Dice Score

---

## 方法演进

### 阶段 1: Keras 自定义 UNet (失败)

**文件**: vesuvius_colab_complete.py

| 项目 | 值 |
|------|-----|
| 框架 | Keras 3 + JAX |
| 模型 | 自定义 2.5D UNet + Attention |
| 参数量 | 8.1M |
| 最佳 Val Dice | **0.2828** |

---

### 阶段 2: PyTorch + 预训练编码器 (成功)

**文件**: vesuvius_smp_train.py

| 项目 | 值 |
|------|-----|
| 框架 | PyTorch |
| 模型 | smp.Unet + ResNet34 预训练 |
| 参数量 | 24.4M |
| 最佳 Val Dice | **0.4342** (Epoch 12) |

---

### 阶段 3: nnUNet 3D (进行中)

**文件**:
- nnunet_scripts/nnunet_local_preprocess.py (本地预处理)
- nnunet_scripts/kaggle_nnunet_train_only.py (Kaggle训练)
- nnunet_scripts/colab_nnunet_continue.py (Colab继续训练)

| 项目 | 值 |
|------|-----|
| 框架 | nnUNetv2 |
| 模型 | 3d_fullres (6层UNet) |
| 训练样本 | **806** (100%) |
| Patch Size | 128x128x128 |
| Batch Size | 2 |

**5-Fold 训练进度** (2026-01-27 04:30):

| Fold | Epochs | EMA Best Dice | LB Score | 状态 |
|------|--------|---------------|----------|------|
| 0 | 300 | - | **0.492** | 完成 |
| 1 | 69 | 0.4338 | - | 完成 (最低) |
| 2 | 188 | **0.527** | 0.485 | 完成 (Val最高) |
| 3 | 206 | 0.4985 | - | 完成 |
| 4 | 68 | 0.4844 | - | 完成 |

**5-Fold 全部完成！** 准备集成推理提交

**集成推理脚本** (已归档到 archive/):
- `submit_5fold_all.py` - 5-fold全部 (0,1,2,3,4)
- `submit_4fold_no1.py` - 4-fold排除Fold1 (0,2,3,4)
- `submit_3fold_best.py` - 3-fold最佳 (0,2,3)

**关键配置**:
- `use_mirroring=False` - 关闭TTA（加速8倍，避免超时）
- 后处理：移除小连通分量 + 填充小孔洞

**Checkpoint 路径 (Kaggle Dataset)**:
- Fold 0: `/kaggle/input/nnunet-ckpt-ep100`
- Fold 2: `/kaggle/input/nnunetfold2/fold2`

**关键修复**:
- Ignore label: 保留lbl=2，dataset.json添加"ignore":2
- P100不支持torch.compile: 设置nnUNet_compile=False
- Kaggle只读文件系统: 复制数据到/kaggle/working/
- 继续训练: 使用--c参数从checkpoint恢复
- **验证脚本转置** (2026-02-04): validation-samples数据集的label坐标系与npz概率图不一致，需要转置 `np.transpose(fg_prob, (2,1,0))`，Dice从0.21提升到0.51

---

## 当前状态

- **最佳方案**: nnUNet ResEnc L (2540样本)
- **最佳 LB Score**: **0.524** (v2: step=0.5, TTA开, dust_500)
- **最佳 Val Dice**: **0.5710** (ResEnc L, Epoch 219 EMA)
- **前100名门槛**: 0.548 (差距0.024)
- **v3 掉分**: 0.517 (关TTA+fill_holes导致-0.007)
- **剩余时间**: 3天 (截止 2026-02-13)

---

## ResEnc L 训练进度 (2026-02-05)

**模型**: nnUNetTrainerResEncL + nnUNetResEncUNetLPlans
**数据集**: 2540样本 (Dataset101_Vesuvius_merged)
**配置**: Patch 192x192x192, Batch 2, 1200 epochs

### 训练记录

| 日期 | Epoch | Val Dice | 平台 | 备注 |
|------|-------|----------|------|------|
| 02-05 08:00 | 87 | 0.5237 | Kaggle 2xT4 | 首次突破 |
| 02-05 19:32 | 158 | 0.5563 | Kaggle 2xT4 | 新最佳 |
| 02-05 20:22 | 159 | 0.5546 | Colab L4 | 测试速度 |
| 02-05 20:34 | 160 | 0.5549 | Colab L4 | |
| 02-05 21:12 | 159 | 0.5560 | Kaggle 2xT4 | |
| 02-06 06:18 | 219 | **0.5761** | Kaggle 2xT4 | **新最佳 EMA 0.5710** |
| 02-06 08:08 | 231 | 0.5720 | Kaggle 2xT4 | 训练中 |
| 02-06 18:24 | 273 | 0.5711 | Kaggle 2xT4 | NEW BEST |
| 02-06 18:34 | 274 | 0.5785 | Kaggle 2xT4 | checkpoint 已丢失 |
| 02-06 21:17 | 292 | 0.5674 | Kaggle 2xT4 | 训练中 |

### 平台速度对比

| 平台 | GPU | 速度 |
|------|-----|------|
| Kaggle | 2x T4 (DDP) | ~530s/epoch |
| Colab | L4 | ~644s/epoch |

**结论**: Kaggle 2xT4 更快，继续使用 Kaggle 训练

### Checkpoint 路径

- `2540-1`: Epoch 87 (Dice 0.5237)
- `2540-2`: Epoch 158 (Dice 0.5563) - Colab 测试用
- `2540-3`: Epoch 159 (Dice 0.5560) - 当前最佳，用于提交
- `2540-4`: Epoch 219 (best, Dice 0.5710) - 含旧fold0_前缀文件，需跳过

### 脚本

- `active/continue_2540_resenc.py` - Kaggle 继续训练
- `active/colab_continue_2540_resenc.py` - Colab 继续训练
- `active/submit_resenc_l_v2.py` - 提交脚本 (阈值0.25, dust_500, TTA开启)

### 验证结果对比 (2026-02-05)

| 模型 | 最佳阈值 | Val Dice | 差距 |
|------|----------|----------|------|
| **ResEnc L 2540** (Ep159) | 0.25 | **0.5680** | - |
| 806 Baseline (Ep315) | 0.35 | 0.5266 | -0.0414 |

**验证脚本**: `validate_resenc_threshold.py`, `validate_806_baseline.py`
**验证样本**: 5个 (validation-samples 数据集)
**TTA**: 开启

**坐标系说明**:
- 验证脚本: 概率图需要转置 `np.transpose(fg_prob, (2,1,0))` 匹配 label 坐标系
- 提交脚本: 概率图不需要转置，直接输出 TIF (根据 CLAUDE.md 血泪教训)

---

## nnUNet 推理方法 (LB 0.492)

### 推理参数配置

```python
predictor = nnUNetPredictor(
    tile_step_size=0.5,      # 滑窗步长比例 (0.5 = 50% overlap)
    use_gaussian=True,       # 高斯权重混合
    use_mirroring=True,      # TTA 镜像增强
    device=torch.device('cuda'),
    verbose=True
)

predictor.initialize_from_trained_model_folder(
    MODEL_DIR,
    use_folds=(0, 2),        # 使用的 fold
    checkpoint_name='checkpoint_best.pth'
)
```

### 数据预处理

1. **TIF 读取**: tifffile -> cv2 -> PIL (三级兜底)
2. **归一化**: 百分位数归一化 (1%, 99%)
   ```python
   p1, p99 = np.percentile(vol, [1, 99])
   vol = np.clip(vol, p1, p99)
   vol = (vol - p1) / (p99 - p1 + 1e-8)
   ```
3. **轴转换**: (D,H,W) -> (W,H,D) 用于 NIfTI 格式

### 后处理 (关键！)

```python
def postprocess_prediction(pred, min_size=100):
    # 1. 移除小的前景连通分量
    labeled, num_features = ndimage.label(pred)
    if num_features > 0:
        sizes = ndimage.sum(pred, labeled, range(1, num_features + 1))
        for i, size in enumerate(sizes):
            if size < min_size:
                pred[labeled == (i + 1)] = 0

    # 2. 填充小孔洞 (关键！0.492 vs 0.488 的差异)
    bg = 1 - pred
    labeled_bg, num_bg = ndimage.label(bg)
    if num_bg > 1:
        sizes_bg = ndimage.sum(bg, labeled_bg, range(1, num_bg + 1))
        max_bg_label = np.argmax(sizes_bg) + 1
        for i in range(1, num_bg + 1):
            if i != max_bg_label and sizes_bg[i-1] < min_size:
                pred[labeled_bg == i] = 1
    return pred
```

### 提交脚本

| 脚本 | 用途 | Folds |
|------|------|-------|
| submit_fold0_fold2.py | Fold 0+2 集成 | (0, 2) |
| submit_fold2_only.py | Fold 2 单独测试 | (2,) |

### 分数对比

| 配置 | LB Score | 前景像素数 | 备注 |
|------|----------|-----------|------|
| Fold 0 + 完整后处理 | **0.492** | ? | 最佳（有填充孔洞）|
| Fold 0+2 + 无孔洞填充 | **0.490** | 1,631,693 | 集成推理（无填充）|
| Fold 0+2 + 完整后处理 | 0.488 | 1,631,697 | 填充降低0.002分 |
| Fold 0 + 无孔洞填充 | 0.488 | ? | 缺少填充孔洞 |
| Fold 2 + 无孔洞填充 | 0.485 | 1,877,706 | 单fold（无填充）|
| Fold 2 + 完整后处理 | 0.482 | 1,877,725 | 填充反而降分 |
| Fold 0 + 无后处理 | 0.462 | ? | 无任何后处理 |
| tile_step_size=0.4 | 0.490 | ? | 更小步长无提升 |

---

## 问题排查

### 2026-01-21: 提交分数异常 (0.291)

**根因**: thin_surface_from_prob(thickness=3) 丢弃91%前景

**修复**: 改为直接阈值 (prob_vol > THR).astype(uint8)

---

## 待办

1. [x] 重新提交 Kaggle (LB: 0.447)
2. [x] 升级编码器 EfficientNet-B4
3. [x] 添加 TTA + stride优化
4. [x] Focal-Dice Loss
5. [ ] 训练 EfficientNet-B4 模型
6. [ ] 多模型集成提交

---

## 更新日志

### 2026-02-10 (submit_resenc_l_v3.py - LB 0.517, 掉分)

**背景**: Config A (skip_normalize) 失败 (LB 0.222)，参数扫描已排除。转向推理和后处理优化。

**公开方案研究发现**:
- 官方 baseline: 0.543 raw, 0.562 加后处理 (我们 0.524 低于 baseline)
- Top 方案关键: geodesic Voronoi sheet 分离(VOI), 填充孔洞(Topo), boundary loss
- 评估指标 65% 权重在 VOI+Topo，需要保持拓扑连续性

**v3 脚本**: `active/submit_resenc_l_v3.py`，基于 v2 三处改动:

| 改动 | v2 (0.524) | v3 (0.517) | 目的 |
|------|-----------|-----------|------|
| tile_step_size | 0.5 | 0.25 | 75%重叠，边界更精确(SurfaceDice) |
| use_mirroring | True | False | 关TTA避免超时 |
| 后处理 | dust_removal | dust_removal + fill_holes | 填充孔洞(TopoScore) |

**LB 结果**: 0.517 (比 v2 的 0.524 降了 0.007)

**掉分分析**:
- 主因: 关闭TTA (use_mirroring=False)，历史数据显示TTA贡献+0.015~0.02
- 次因: fill_holes 在806模型上曾导致-0.002~0.003
- tile_step_size 0.25 vs 0.5 影响可忽略(本地仅+0.0009)
- 关TTA是为了避免step=0.25+TTA超时(10.6h>9h限制)，但代价大于收益

**结论**: v2 配置 (step=0.5, TTA开, 仅dust) 仍然是最优，不要同时改多个变量

---

### 2026-02-09 (参数扫描提交脚本 - Config A 双重归一化测试)

**背景**: LB 0.524，训练已停止（收益太低）。发现提交脚本存在潜在的双重归一化问题（手动 percentile normalize + nnUNet 内部归一化）。

**创建脚本**: `active/submit_resenc_l_sweep.py`
- 基于 `submit_resenc_l_v2.py`，支持按 `skip_normalize` 分组批量测试
- 同组配置共享推理结果，只重跑后处理，节省推理时间

**测试矩阵**:

| 编号 | SKIP_NORMALIZE | THRESHOLD | DUST | 说明 | 状态 |
|------|---------------|-----------|------|------|------|
| A | True | 0.25 | 500 | 去掉双重归一化 | **LB 0.222** (失败) |
| B | False | 0.20 | 500 | 低阈值 | 本地验证已覆盖 |
| C | False | 0.30 | 500 | 高阈值 | 本地验证已覆盖 |
| D | False | 0.25 | 200 | 小dust | 本地验证已覆盖 |
| E | False | 0.25 | 1000 | 大dust | 本地验证已覆盖 |

**结论**: B-E 已通过 `validate_resenc_threshold.py` 本地验证，当前配置 (thr=0.25, dust=500) 已是最优。只有 Config A (skip_normalize) 无法本地验证（验证数据已是归一化后的 NIfTI），需要提交测试。

**验证脚本更新**: `validate_resenc_threshold.py` focused 模式新增 dust_200、dust_500、dust_1000 三档。

---

### 2026-02-08 (submit_resenc_l_v2.py 后处理回滚)

**问题**: topo_opt 后处理 (thr=0.35, dust+fill_holes+keep_largest) LB 得分 0.363，远低于简单后处理的 0.516
**操作**: 回滚到简单后处理配置

| 项目 | 回滚前 | 回滚后 |
|------|--------|--------|
| 阈值 | 0.35 | **0.25** |
| DUST_MIN_SIZE | 100 | **500** |
| 后处理 | dust + fill_holes + keep_largest | **仅 dust_removal** |

**结论**: 本地验证复合分数最高的 topo_opt 方法在 LB 上严重退化 (0.363 vs 0.516)，说明本地验证集与测试集分布差异大，简单后处理更稳健。

---

### 2026-02-07 (active 文件夹清理)

**保留 11 个文件**:
- 提交: submit_resenc_l_v2.py, submit_resenc_l_threshold_test.py
- 训练: continue_2540_resenc.py, colab_continue_2540_resenc.py
- 验证: validate_resenc_threshold.py, validate_806_baseline.py, validate_compare_models.py
- 工具: merge_datasets.py, make_val_samples.py, postprocess_voi_topo.py
- 配置: CLAUDE.md

**移动 24 个旧文件到 archive/**:
- 旧提交脚本: submit_nnunet.py, submit_STANDARD.py, submit_transunet_v1/v2.py 等
- 旧训练脚本: train_806_fold0.py, train_1754_ddp_patch192.py 等
- 旧验证脚本: validate_local.py, validate_postprocess.py 等
- 旧工具: explore_paths.py, check_dataset_overlap.py 等

**删除**: __pycache__ 目录

### 2026-02-07 (后处理方法对比验证 - 竞赛复合评分)

**验证脚本**: `validate_resenc_threshold.py` (focused 模式)
**模型**: ResEnc L Epoch 219 (checkpoint_best, Dice 0.5710)
**验证样本**: 20个 (validate-sample-new, seed=42)
**评分公式**: Total = 0.30 * Topo + 0.35 * SurfD + 0.35 * VOI

**Top 10 后处理方法 (按竞赛复合分数排序)**:

| 排名 | 方法 | 阈值 | Total | Dice | VOI | SurfD | Topo |
|------|------|------|-------|------|-----|-------|------|
| 1 | **topo_opt** | **0.35** | **0.5002** | 0.0606 | 0.8948 | 0.1146 | 0.4896 |
| 2 | topo_opt | 0.40 | 0.4998 | 0.0528 | 0.9075 | 0.1004 | 0.4901 |
| 3 | largest_1 | 0.37 | 0.4964 | 0.0559 | 0.9010 | 0.1049 | 0.4811 |
| 4 | topo_opt | 0.25 | 0.4952 | 0.1045 | 0.8136 | 0.1976 | 0.4709 |
| 5 | largest_1 | 0.40 | 0.4946 | 0.0528 | 0.9075 | 0.1004 | 0.4728 |
| 6 | largest_1 | 0.35 | 0.4936 | 0.0606 | 0.8948 | 0.1146 | 0.4678 |
| 7 | topo_opt | 0.30 | 0.4869 | 0.0801 | 0.8454 | 0.1515 | 0.4598 |
| 8 | largest_1 | 0.25 | 0.4826 | 0.1045 | 0.8136 | 0.1976 | 0.4288 |
| 9 | largest_1 | 0.28 | 0.4805 | 0.0832 | 0.8422 | 0.1567 | 0.4364 |
| 10 | largest_1 | 0.30 | 0.4787 | 0.0801 | 0.8454 | 0.1515 | 0.4325 |

**topo_opt 方法逻辑** (`postprocess_topo_optimized`):
1. 二值化 (threshold)
2. dust_removal: 移除小连通分量 (<100 voxels)
3. binary_fill_holes: 填充所有内部孔洞
4. keep_largest_1: 只保留最大连通分量

**关键发现**:
- topo_opt 和 largest_1 占据 Top 10 全部位置
- 高阈值 (0.35-0.40) 牺牲 Dice/SurfD 换取 VOI+Topo，总分更高
- Dice 极低 (0.05-0.10) 但 VOI 极高 (0.89-0.91)，说明预测稀疏但拓扑干净
- topo_opt 比 largest_1 多了 fill_holes 步骤，Topo 分数更高
- combined/closing_dust/gauss/voi_opt 方法均不如 topo_opt

**提交建议配置**: topo_opt thr=0.35
```python
binary = (prob > 0.35).astype(np.uint8)
binary = dust_removal(binary, min_size=100)
filled = ndimage.binary_fill_holes(binary)
labeled, num = ndimage.label(filled)
if num > 1:
    sizes = ndimage.sum(filled, labeled, range(1, num+1))
    max_label = np.argmax(sizes) + 1
    filled = (labeled == max_label)
pred = filled.astype(np.uint8)
```

---

### 2026-02-06 (Trainer 保存逻辑修复)

**问题**: checkpoint 文件 epoch 混乱，续训回退到旧 epoch
**根因**: 多个保存逻辑 bug

**修复内容** (continue_2540_resenc.py):
1. CELL 5: 跳过 fold0_ 前缀旧文件，避免覆盖新 checkpoint
2. CELL 6: checkpoint 不存在时 raise 报错
3. save_every=5 (基类默认50)，和基类保存时机同步
4. _save_one_checkpoint 替代 _save_checkpoint_copy，避免重复复制
5. on_epoch_end 在 super() 之前取 epoch，修复 +1 偏移
6. on_train_start 从日志恢复 best_dice，避免续训误报 NEW BEST
7. on_train_end 只复制 final（基类会删除 latest）
8. 内存清理和 latest 保存合并，每5个 epoch 一次

**验证**: Epoch 219 正确恢复，best_dice=0.5710 从日志恢复成功

---

### 2026-02-05 (ResEnc L 提交脚本 v2 磁盘优化)

**问题**: submit_resenc_l_v2.py 在隐藏测试集(120个样本)上 out of disk
**根因**: 批量处理模式下，120个NIfTI + 120个npz + 120个tif 同时存在磁盘

**修复** (三个关键改动):
1. **逐样本处理**: 每个样本独立走完 转换->推理->后处理->删除 的完整流程
2. **只删pth权重，不删整个nnUNet_results**: predictor推理时仍需要plans.json等配置文件，只删~820MB的pth文件
3. **直接写入zip**: 每个样本后处理完直接追加到zip，不在磁盘上累积tif文件

**磁盘峰值对比**:
- 修改前: 120个NIfTI + 120个npz + 120个tif + zip (几十GB)
- 修改后: 1个NIfTI + 1个npz + 1个临时tif + 逐渐增长的zip (常数级)

**公开测试集(1样本)验证通过**, 等待隐藏测试集结果

---

### 2026-02-04 (ResEnc L 2540样本提交脚本)

**模型**: nnUNet ResEnc L (nnUNetResEncUNetLPlans)
**数据集**: 2540样本 (2540-1 checkpoint + 2540-part1 config)
**Trainer**: nnUNetTrainerResEncL

**提交脚本**: `active/submit_resenc_l.py` (已归档, 替换为 submit_resenc_l_v2.py)

**关键配置**:
- 阈值: 0.3 (验证测试最佳)
- 后处理: dust_500 (移除<500体素连通分量)
- TTA: use_mirroring=True

**验证结果** (validate_postprocess_methods.py):
| 方法 | Dice |
|------|------|
| t0.3_dust_500 | **0.5410** |
| t0.3_none | 0.5389 |
| t0.3_dust_200 | 0.5401 |

**脚本修复**:
1. 添加 CONFIG_SRC 备用配置来源 (2540-part1)
2. 修复 npz 文件名 _0 后缀问题
3. 添加 sanity check 验证输出shape

**Sanity Check 通过**:
```
Original TIF shape: (320, 320, 320)
Prediction shape:   (320, 320, 320)
OK: Shapes match!
```

**继续训练脚本**:
- `active/continue_2540_resenc.py` - Kaggle T4x2 DDP
- `active/colab_continue_2540_resenc.py` - Colab L4/A100

**状态**: 待提交验证 LB 分数

---

### 2026-02-03 (1754样本训练失败)

**数据集**: nnunet-1754-preprocessed (1754样本, .b2nd格式)
**配置**: Patch 192, Batch 2/GPU, T4x2
**结果**: 效果差，已停止训练
**结论**: 新数据集质量可能不如原806样本，或需要不同的训练策略

---

### 2026-02-02 (1754样本DDP训练启动)

**数据集**: nnunet-1754-preprocessed (1754样本, .b2nd格式)
**配置**: Patch 192, Batch 2/GPU, T4x2, 1200 epochs
**脚本**: active/train_1754_ddp_patch192.py

**训练进度**:
- Epoch 1: Dice 0.1588, train_loss 0.1105
- 时间: ~10分钟/epoch
- 显存: 9.7/15 GB
- 自动保存: fold0_checkpoint_best.pth (249.7MB)

**状态**: 已停止 (效果差)

---

### 2026-02-03 (active 脚本归档)

**归档目录**: `archive/old_active/`

**移动内容**:
- 训练实验与启动器：`continue_cldice_fold0.py`, `train_cldice_kaggle.py`, `train_fg50_only.py`, `train_improved_v1_fg_oversample.py`, `train_improved_v2_zthick.py`, `train_improved_v3_full.py`, `train_improved_v3_lite.py`, `train_v3_dual_gpu.py`
- 提交/推理：`submit_2fold_best.py`, `submit_cldice.py`, `submit_threshold_008_streaming.py`
- 验证/对比：`validate_postprocess.py`, `validate_surface_postprocess.py`, `validate_tta.py`, `validate_transunet_params.py`, `validate_transunet_v2.py`, `compare_models_kaggle.py`, `verify_loss_quick.py`
- 分析/探索：`explore_dataset_deep.py`, `explore_local_dataset.py`, `explore_dataset_structure.py`, `explore_simple.py`, `explore_detail.py`, `analyze_deep.py`, `analyze_image_features.py`, `analyze_prediction_errors.py`, `analyze_z_detail.py`

### 2026-02-02 (新数据集下载和转换)

**新数据集**: seg-derived-recto-surfaces
- 来源: https://dl.ash2txt.org/datasets/seg-derived-recto-surfaces/
- 样本数: **1754** (比原806样本多2倍)
- 格式: 300x300x300 TIF
- 标签: 二值(0=背景, 1=表面), 无ignore label
- 前景比例: 15-19% (原数据集约5%)
- 来源卷轴: Scroll 1, 4, 5

**数据位置**:
- 转换后: `F:/nnUNet_data/Dataset001_Vesuvius/` (1754样本)
- 原始下载: `D:/local kaggle/new_data/` (18GB)

**脚本**:
- `nnunet_scripts/download_new_dataset.py` - 下载脚本
- `nnunet_scripts/convert_new_dataset.py` - 转换脚本

**下一步**: 上传到Kaggle Dataset用于训练

---

### 2026-02-01 (转置Bug修复 + LB 0.522)

**重大修复**: 提交脚本转置Bug
- 之前提交分数: 0.292, 0.327 (错误的转置)
- 修复后分数: **0.522** (不转置)
- 原因: npz概率图已经和TIF对齐，不需要转置

**提交配置**:
- 模型: 806样本 Fold 0, Epoch 315
- 阈值: 0.37
- TTA: use_mirroring=True
- 后处理: 移除小连通分量 + 填充小孔洞

**当前差距**: 前100名 0.548，差 0.026

---

### 2026-01-31 (TransUNet方案测试)

**来源**: Kaggle公开方案 (LB 0.539)
**模型**: TransUNet + SEResNeXt50 (Innat公开权重)
**框架**: Keras 3 + JAX

**脚本**:
- `active/submit_transunet_v1.py` - 提交脚本
- `active/validate_transunet_params.py` - 参数验证脚本

**模型配置**:
- 权重: `/kaggle/input/vsd-model/keras/transunetseresnext/2/transunet.seresnext50.weights.h5`
- input_shape: (128, 128, 128, 1)
- num_classes: 2
- 参数量: 70.02M

**验证结果** (5个样本):
| 方法 | Mean Dice | 配置 |
|------|-----------|------|
| **baseline_0.75** | **0.5047** | 阈值0.75, 无Hysteresis |
| hyst_0.6_0.95 | 0.5024 | T_low=0.6, T_high=0.95 |
| baseline_0.5 | 0.4958 | 阈值0.5 |
| hyst_0.5_0.9 | 0.4957 | T_low=0.5, T_high=0.9 |
| hyst_0.4_0.85 | 0.4862 | T_low=0.4, T_high=0.85 |

**Frangi测试**: 效果极差(Dice=0.0006)，原因是skimage.frangi实现与原方案不同

**最终配置**:
- TTA: 4x旋转 (0, 90, 180, 270度)
- 阈值: 0.75
- Hysteresis: 关闭
- dust_min_size: 100

**推理时间**:
- 单样本: 2分08秒 (含4x TTA)
- 120样本预计: 约4.3小时

**状态**: 提交中，等待LB分数

---

### 2026-01-31 (提交脚本转置Bug - 严重教训)

**问题**: 提交分数异常低 (0.292, 0.327)，远低于验证的 0.5+

**根因**: 验证脚本修复了转置问题，但**没有同步修复提交脚本**！

**错误代码**:
```python
pred = np.transpose(pred, (2, 1, 0))  # 这行是错误的！
```

**影响的脚本** (已全部修复):
- submit_nnunet.py
- submit_cldice.py
- submit_2fold_best.py
- submit_STANDARD.py
- test_cldice_local.py

**严重教训**:
1. 修复验证脚本时，必须同时检查所有相关的提交脚本
2. 不要假设代码是对的，要用 grep 搜索所有相关文件
3. 浪费了提交次数和时间

---

### 2026-01-31 (806样本提交脚本确认)

**提交脚本**: `active/submit_nnunet.py`
**Checkpoint**: `/kaggle/input/fold0-806-epoch315`
**配置文件**: `/kaggle/input/nnunet-ckpt-ep100`

**关键配置**:
- TTA: `use_mirroring=True`
- 阈值: 0.37
- 后处理: 移除小连通分量 + 填充小孔洞

**清理机制**:
- 逐样本清理: 每个样本处理完立即删除临时文件
- 最终清理: 删除所有工作目录

**预计时间**: 约3.4小时 (120样本)

---

### 2026-01-31 (三模型对比验证 + TTA)

**验证脚本**: `test_compare_models.py`
**测试样本**: 5个随机样本 (seed=42)
**阈值**: 0.37

**无TTA结果**:
| 模型 | 训练配置 | 平均 Dice |
|------|----------|-----------|
| 806samples | 806样本, Epoch 315 | 0.5011 |
| 80samples | 80样本, Epoch 100 | 0.4827 |
| clDice | 80样本, clDice Loss, Epoch 300 | 0.4722 |

**有TTA结果** (use_mirroring=True):
| 模型 | 平均 Dice | vs无TTA | 每样本时间 | 120样本预计 |
|------|-----------|---------|------------|-------------|
| **806samples** | **0.5174** | +0.0163 | 100.8s | 3.36h |
| 80samples | 0.5022 | +0.0195 | 102.1s | 3.40h |
| clDice | 0.4876 | +0.0154 | 102.6s | 3.42h |

**逐样本结果 (有TTA)**:
| Sample | clDice | 806samples | 80samples |
|--------|--------|------------|-----------|
| 4244019916 | 0.5833 | 0.5890 | 0.5889 |
| 1577382633 | 0.3351 | 0.3792 | 0.3550 |
| 1156808983 | 0.5755 | 0.5930 | 0.5750 |
| 850710964 | 0.3788 | 0.4146 | 0.4030 |
| 2373617013 | 0.5654 | 0.6111 | 0.5889 |

**结论**:
1. TTA 有效，提升约 +0.015~0.02
2. 806样本训练效果最好 (0.5174)
3. clDice Loss 反而降低性能
4. 120样本约3.4小时，不会超过9小时限制

---

### 2026-01-31 (验证脚本轴转置Bug修复 - 重要教训)

**问题**: `test_compare_models.py` 验证脚本 Dice 分数异常低 (~0.12-0.33)

**根因**:
1. 错误地对 nnUNet 输出的概率图做了转置
2. dice_score 函数没有忽略 label=2 (ignore region)
3. 使用了 `deprecated_train` 数据而不是 `train` 数据

**调试输出**:
```
prob_raw.shape: (320, 320, 320), label.shape: (320, 320, 320)
Dice (no transpose): 0.5889
Dice (transpose 2,1,0): 0.1321
>>> Using: no transpose
```

**修复**:
1. **不要盲目转置** - nnUNet npz 概率图和 TIF 标签已经对齐，不需要转置
2. **测试两种方式** - 不确定时测试 no_transpose 和 transpose(2,1,0)，选择 Dice 更高的
3. **忽略 label=2** - dice_score 必须排除 ignore 区域:
   ```python
   def dice_score(pred, label, ignore_label=2):
       mask = label != ignore_label
       pred_masked = pred[mask].astype(np.float32)
       label_masked = (label[mask] == 1).astype(np.float32)
       intersection = np.sum(pred_masked * label_masked)
       union = np.sum(pred_masked) + np.sum(label_masked)
       if union == 0:
           return 1.0
       return 2 * intersection / union
   ```
4. **用正确的数据** - 使用 `train_images`/`train_labels` 而不是 `deprecated_train`

**教训总结**:
- 写验证脚本时，先用已知正确的脚本 (validate_postprocess.py) 作为参考
- 数据对齐问题要通过测试两种方式来验证，不要假设
- dice_score 计算必须和训练时一致（忽略 label=2）

---

### 2026-01-30 (806样本 Fold 0 提交测试)

**使用脚本**: `active/submit_nnunet.py`
**Checkpoint**: `fold0-806-epoch315` (806样本训练, Epoch 315, Val Dice 0.4889)

**技术修复**:
- 配置文件从 `nnunet-ckpt-ep100` 复制 (plans.json, dataset.json)
- Checkpoint 从 `fold0-806-epoch315` 复制
- Monkey patch 注册 `nnUNetTrainerWithMonitor` 类

**提交版本**:
| 阈值 | 预测sum | LB Score | 备注 |
|------|---------|----------|------|
| 0.50 | - | 待更新 | |
| 0.37 | 2,812,496 | 待更新 | 验证集最佳阈值 |

**2-Fold集成 (submit_2fold_best.py)**:
- Fold 0 (80样本) + Fold 2, TTA开启
- 预测sum: 1,631,701
- LB Score: **0.488**
- 结论: 集成仍不如单Fold 0 (0.492)

---

### 2026-01-30 02:15 (clDice Loss训练开始)

**新方向**: 使用clDice Loss保持拓扑连续性

**TTA验证实验结果**:
| 配置 | 平均Dice | 时间/样本 | 相对baseline |
|------|----------|-----------|--------------|
| baseline_no_tta | 0.5017 | 21.8s | - |
| tta_mirror | 0.5175 | 73s | +0.0158 |
| step_0.25_no_tta | 0.5026 | 52.5s | +0.0009 |
| step_0.25_tta | 0.5169 | 346s | +0.0152 |

**结论**: TTA镜像有效(+0.016)，更小步长无效

**clDice训练配置**:
- 脚本: `active/train_cldice_kaggle.py`
- Loss: DC + CE + clDice (权重0.3)
- 样本: 80个
- GPU: Kaggle T4
- 每epoch: 约4.8分钟

**训练进度**:
| Epoch | Dice | EMA Best | Train Loss |
|-------|------|----------|------------|
| 0 | 0.154 | - | 0.834 |
| 3 | 0.181 | - | 0.665 |
| 49 | 0.4655 | 0.4123 | 0.205 |

**预计**: 6小时跑约75个epoch
**状态**: 训练中，Dice稳步提升

---

### 2026-01-29 21:00 (后处理验证实验 - 重要发现)

**806样本训练结果**:
- Fold 0: Epoch 315, Best Dice **0.4889**
- 58个epoch无改进后停止
- Checkpoint: `/kaggle/input/fold0-806-epoch315`
- 结论: 10倍数据量没有带来明显提升（80样本LB 0.492 vs 806样本Dice 0.49）

**后处理验证实验**:

创建验证脚本 `validate_postprocess.py`，测试不同后处理方法：
- baseline 0.5 (阈值0.5 + 移除小区域)
- threshold 0.75
- topology_v1 (闭运算 + 填充孔洞 + 保留最大连通分量)
- topology_v2 (更强闭运算26-连通)
- CT Frangi AND

**关键Bug发现: 轴顺序错误！**

验证脚本中对概率图做了错误的转置，导致Dice只有0.17：
```python
# 错误写法
prob = np.transpose(prob_raw, (2, 1, 0))  # Dice = 0.17

# 正确写法
prob = prob_raw  # 不需要转置, Dice = 0.50
```

**调试输出**:
```
Sample: 1004283650
  prob_raw.shape: (320, 320, 320)
  label.shape: (320, 320, 320)
  Dice (no transpose): 0.6603
  Dice (transpose 2,1,0): 0.2115
  >>> 使用: no transpose
```

**修正后验证结果**:

| 方法 | 平均Dice | 相对baseline |
|------|----------|--------------|
| **baseline 0.5** | **0.5050** | - |
| topology_v2 | 0.4324 | -0.0726 |
| threshold 0.75 | 0.3896 | -0.1153 |
| topology_v1 | 0.3321 | -0.1728 |
| CT Frangi AND | 0.0475 | -0.4574 |

**结论**:
1. **轴顺序**: npz概率图不需要转置，nii.gz分割结果需要转置
2. **后处理无效**: 所有拓扑优化方法都降低了分数
3. **baseline最优**: 阈值0.5 + 移除小区域 是最佳后处理
4. **提交脚本正确**: submit_nnunet.py的转置逻辑是对的（LB 0.492）

**验证脚本修复**:
- 添加自定义Trainer注册 (monkey patch)
- 修复checkpoint前缀处理
- 添加CONFIG_SRC配置文件来源
- 添加轴顺序调试输出

**下一步方向**:
- 后处理走不通
- 多fold集成效果不好
- 需要从模型/训练策略入手
- 可尝试: TTA、更小tile_step_size、clDice损失函数

---

### 2026-01-29 (集成推理实验)

**4-Fold集成结果 (0,2,3,4)**:
- LB Score: **0.489** (比单fold 0.492还低!)
- 配置: use_mirroring=False, Trainer=nnUNetTrainer

### 2026-01-30 (阈值扫描)

- 验证脚本: active/validate_surface_postprocess.py
- 设置: 随机抽样 50 样本 (seed=42), baseline后处理
- 结果: best_thr=0.37, mean Dice=0.5234
- 结论: Fold 3/4质量较差，拉低了集成效果

**2-Fold集成尝试 (0,2)**:
- 只用LB验证过的两个最佳fold
- Fold 0: LB 0.492
- Fold 2: LB 0.485
- 配置: use_mirroring=True (开启TTA增加多样性)
- 脚本: `submit_2fold_best.py`

**时间估算**:
- 每样本: ~2分钟 (2fold + TTA)
- 120样本: ~4小时
- Kaggle限制9小时，来得及

**分数对比更新**:

| 配置 | LB Score | 备注 |
|------|----------|------|
| Fold 0 单独 | **0.492** | 当前最佳 |
| Fold 0+2 (无TTA) | 0.490 | 集成略降 |
| 4-Fold (0,2,3,4) 无TTA | 0.489 | 集成反而更低 |
| Fold 2 单独 | 0.485 | |

**待验证**: 2-Fold (0,2) + TTA 能否突破0.492

---

### 2026-01-29 09:30 (806样本训练继续 - Fold 0)

**训练配置**:
- 样本数: 806 (从80增加10倍)
- GPU: Kaggle Tesla T4
- Trainer: nnUNetTrainerWithMonitor
- 训练/验证: 644 / 162 样本

**继续训练**:
- 从 Epoch 150 恢复 (checkpoint_latest.pth)
- 当前 Epoch 152, Dice **0.4498**
- 每个 epoch 约 168 秒 (2.8分钟)

**预计进度**:
- 12小时 = 720分钟 / 2.8 = 约 257 个 epoch
- 预计跑到 Epoch 407

**脚本**: `active/train_806_fold0.py`
**Checkpoint**: `/kaggle/input/fold0-806/`

---

### 2026-01-29 08:30 (阈值0.75提交脚本准备完成)

**关键发现**：官方baseline推荐阈值 **0.75**（不是0.5）
- nnUNet默认阈值0.5: LB 0.543
- 阈值0.75 + Frangi: LB **0.562**

**验证集调参问题**：
- 测试了阈值0.06-0.60，Dice都很低（0.12-0.22）
- 验证集和测试集分布差异大，不适合用来选阈值
- 结论：直接用官方推荐阈值0.75

**提交脚本更新** (`submit_threshold_008_streaming.py`):
- 阈值: 0.08 → **0.75**
- 后处理: 使用0.492验证过的完整版本（删除小分量+填充孔洞）
- TTA: use_mirroring=True
- 流式zip写入，避免爆硬盘

**时间估算**：
- 单样本: 3分45秒（含TTA）
- 120样本: ~7.5小时（Kaggle限制9小时）

---

### 2026-01-29 01:30 (后处理函数Bug修复 - 重要教训)

**问题**: 验证脚本阈值搜索结果异常，所有阈值Dice都是0.2621

**根因**: `remove_small_cc_fast` 函数有bug
```python
# 错误写法 (导致背景变成前景!)
counts[0] = min_size + 1  # BUG: 背景被保留为1
keep = counts >= min_size
result = keep[lab].astype(np.uint8)
```

**正确写法** (来自0.492脚本):
```python
def postprocess_prediction(pred, min_size=100):
    # 用 ndimage.sum 计算分量大小，循环遍历删除小分量
    labeled, num_features = ndimage.label(pred)
    if num_features > 0:
        sizes = ndimage.sum(pred, labeled, range(1, num_features + 1))
        for i, size in enumerate(sizes):
            if size < min_size:
                pred[labeled == (i + 1)] = 0
    return pred
```

**教训**:
1. 不要随意"优化"已验证的代码
2. 向量化版本容易引入边界条件bug
3. 直接复用已工作的代码，不要重新实现

**已修复脚本**:
- `validate_fold0_val_v3.py`
- `submit_threshold_008_streaming.py`

---

### 2026-01-28 07:30 (Frangi后处理脚本 - 待解决torch兼容性)
- **创建脚本**:
  - `submit_fold0_frangi.py` - Fold 0 + 阈值0.75 + Frangi filter
  - `submit_3fold_frangi.py` - 3-fold (0,2,3) + 阈值0.75
- **问题**: Utility Script 里的 torch 版本和 Python 3.12 不兼容
  - 错误: `ValueError: module functions cannot set METH_CLASS or METH_STATIC`
  - 尝试: 先导入系统 torch 再设置 nnunetv2 路径
- **待测试**: 新脚本需要在 Kaggle 上验证
- **隐藏测试集**: 约120个样本，9小时限制足够

### 2026-01-28 06:30 (比赛方法研究)
- **详细研究笔记**: `docs/vesuvius_research_2026-01-28.md`
- **关键发现**:
  - 榜首分数: 0.577 (1Savy)
  - 官方 baseline: nnUNet 0.543, +后处理 0.562
  - 我们当前: 0.492 (差距 0.05-0.07)
- **提分方向**:
  1. Frangi filter 后处理 (阈值0.75)
  2. 训练更多 epochs (1200+)
  3. nnUNet ResEnc L (需24GB显存)
  4. MedialSurfaceRecall 自定义 loss

### 2026-01-28 05:15 (400样本训练开始)
- **目标**: 增加训练样本从 80 -> 400 (5倍)
- **数据集**: `nnunet-400` (Kaggle Dataset)
- **数据格式**: `.b2nd` (新版 nnUNet v2)
- **训练配置**:
  - GPU: Kaggle T4
  - Fold: 0
  - Batch size: 2
  - Patch size: 128x128x128
  - 训练/验证: 320 / 80 样本
- **进度**:
  - Epoch 1: Dice = 0.1279 (NEW BEST)
  - Checkpoint 已保存: fold0_checkpoint_best.pth (249.7 MB)
- **预计时间**: ~85小时 (需多次继续训练)
- **预期提升**: LB 0.52-0.55 (从 0.492)

### 2026-01-27 04:30 (集成推理脚本准备完成)
- 准备了3个集成推理脚本
- 关闭TTA镜像避免超时（加速8倍）
- 预计4-fold无TTA推理时间：约3小时
- **提交策略**：
  1. 第1次：4-fold (0,2,3,4) 无TTA
  2. 第2次：根据结果决定

### 2026-01-27 02:00 (Fold 1/4 继续训练成功！)
- **Fold 4**: Dice 0.4518 -> **0.4792** (+0.0274)
  - Epoch 30 达到新最佳
  - 超越之前最佳 +0.0274
- **Fold 1**: Dice 0.4062 -> **0.4338** (+0.0276)
  - Epoch 41 达到新最佳
  - 超越之前最佳 +0.0276
- **方法验证**: 预训练权重 + 低学习率(1e-3) 方法有效

### 2026-01-27 02:30 (Fold 1/4 继续训练 - 使用预训练权重)
- **方法**: 使用 `--pretrained_weights` + 低学习率 (1e-3) 继续训练
- **Fold 1 进展**:
  - 从 Epoch 177 (Dice 0.4062) 的权重继续
  - 初始 Dice: 0.3905 (比从头训练高 3-4倍)
  - Epoch 7: Dice 0.4054，接近之前最佳
  - 训练速度: ~206秒/epoch
- **关键技术决策**:
  - 学习率从 0.01 降到 0.001 (10倍降低)
  - 自定义 Trainer: `nnUNetTrainerLowLR`
  - 保存完整训练状态（所有文件，不仅 .pth）
- **脚本**: `continue_fold1_v2.py`, `continue_fold4_v2.py`
- **状态**: Fold 1 训练中，Fold 4 脚本已准备

### 2026-01-27 00:15 (5-Fold训练全部完成！)
- Fold 4: Epoch 179, Dice **0.4518** (从0.4435提升+0.0083)
- **里程碑**: 5个fold全部训练完成
- **平均验证Dice**: 0.460 (Fold 1/3/4平均)
- **最佳fold**: Fold 3 (0.4985)
- **下一步**: 准备5-fold集成推理

### 2026-01-27 00:00 (Fold 3训练完成)
- Fold 3: Epoch 206, Dice **0.4985** (从0.4884提升+0.0101)
- 停滞17个epoch后手动中断，符合early stopping策略
- 状态：可用于集成推理
- **4个fold已完成**：Fold 0/1/2/3

### 2026-01-26 23:15 (Fold 1训练完成)
- Fold 1: Epoch 177, Dice **0.4062** (从0.3936提升+0.0126)
- 训练被手动中断，checkpoint已保存
- 状态：可用于集成推理

### 2026-01-26 晚 (fold0+2填充最终结果)
- fold0+2+填充孔洞：LB Score = **0.488**
- 比fold0+2无填充(0.490)低0.002分
- **结论确认**: 集成推理不应使用填充孔洞
- **最佳策略**: 单fold用填充，集成不用填充
- **当前最佳**: Fold 0单独+填充 = 0.492

### 2026-01-26 20:20 (5-Fold 训练显著提升)
- Fold 1: Epoch 122, Dice 0.3936 (从0.3815提升+0.0121)
- Fold 3: Epoch 161, Dice **0.4884** (从0.4765提升+0.0119，表现最佳)
- Fold 4: Epoch 118, Dice 0.4435 (从0.4154提升+0.0281，提升最大)
- **观察**: 所有fold都有显著提升，Fold 3表现最好
- **状态**: Fold 1持续提升中，Fold 3刚达新高，Fold 4停滞2轮

### 2026-01-26 晚 (后处理策略对比实验)
- 完成4个提交版本的对比实验
- **关键发现**: 填充孔洞对不同fold效果不同
  - Fold 0: 有填充(0.492) > 无填充(0.488) = +0.004
  - Fold 2: 无填充(0.485) > 有填充(0.482) = -0.003
  - Fold 0+2集成: 填充影响极小（仅+4像素）
- **最佳配置**: Fold 0单独 + 完整后处理 = 0.492
- **集成效果**: Fold 0+2无填充 = 0.490，提升有限
- **前景像素数**: 集成后减少13%（1,877k -> 1,631k），说明预测更保守

### 2026-01-26 06:30 (5-Fold 训练进展)
- Fold 1: Epoch 87, Best Dice 0.3815
- Fold 3: Epoch 122, Best Dice **0.4765** (表现最佳)
- Fold 4: Epoch 82, Best Dice 0.4154 (停滞7轮)
- 平均 Dice: (0.38 + 0.48 + 0.42) / 3 = 0.43

### 2026-01-26 (5-Fold 重新训练)
- Fold 1/3/4 因中断需重新训练
- Fold 3: Kaggle T4 训练中，Epoch 6，Dice 0.0684
- Fold 1/4: Colab T4 训练中，Epoch 1
- 优化 Trainer: 内置自动保存功能，NEW BEST 时自动复制 checkpoint
- Colab 启用 torch.compile 加速 (10min -> 3min/epoch)

### 2026-01-25 (nnUNet 5-Fold)
- Fold 0 完成 300 epoch，LB Score: 0.492
- Fold 1 训练失败
- Fold 2 训练中，Epoch 182，EMA Dice 0.5213
- 修复自定义 Trainer 签名错误 (移除 unpack_dataset 参数)

### 2026-01-23 (nnUNet训练)
- 本地预处理80个样本，生成7GB NIfTI数据
- 修复ignore label处理：保留lbl=2而非转为0
- Kaggle T4x2训练：Epoch 0-29，EMA Dice 0.29
- 迁移到Colab L4继续训练：Epoch 30+
- 速度对比：T4x2(2.5min/epoch) > L4(6.7min/epoch)

### 2026-01-21 (nnUNet)
- 新增 kaggle_nnunet_full.py，尝试 nnUNet + SkeletonRecall
- 修复标签验证错误: 将 lbl==2 (unlabeled) 从255改为0

### 2026-01-21 (优化)
- 推理优化: TTA(4倍) + stride=192
- 编码器升级: ResNet34 -> EfficientNet-B4
- 损失函数: BCE+Dice -> Focal-Dice Loss
- 添加多模型集成支持

### 2026-01-21
- 重新提交成功，LB Score: 0.447
- 修复推理bug后分数大幅提升

### 2026-01-21 (早)
- 首次提交 0.291，发现推理bug
- 修复 vesuvius_submit.py

### 2026-01-20
- 创建 vesuvius_smp_train.py
- Val Dice: 0.28 -> 0.4342 (+53%)
