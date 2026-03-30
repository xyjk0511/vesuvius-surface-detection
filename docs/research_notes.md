# Vesuvius Surface Detection 研究笔记

**日期**: 2026-01-28
**目的**: 搜索比赛最佳方法和提分策略

---

## 1. 比赛概述

### 基本信息
- **比赛名称**: Vesuvius Challenge - Surface Detection
- **目标**: 从 3D CT 扫描中检测和分割古代卷轴的纸莎草表面
- **截止日期**: 2026-02-13
- **奖金**: $100,000 ($60,000 第一名)

### 评分指标
评分是三个指标的加权平均：
```
Score = 0.30 * TopoScore + 0.35 * SurfaceDice + 0.35 * VOI_score
```

| 指标 | 说明 |
|------|------|
| **TopoScore** | 拓扑正确性，使用 Betti 数检测桥接/孔洞错误 |
| **SurfaceDice** | 表面感知的 Dice 系数，允许小位移 |
| **VOI_score** | 变异信息，检测过分割/欠分割 |

### 当前排行榜 (2026-01-28)
| 排名 | 团队 | 分数 |
|------|------|------|
| 1 | 1Savy | **0.577** |
| - | 官方 baseline + 后处理 | 0.562 |
| - | 官方 baseline | 0.543 |
| - | 我们当前 | 0.492 |

---

## 2. 官方 Baseline 方法

### 模型配置
- **框架**: nnUNetv2
- **训练 epochs**: ~1200
- **学习率**: 0.01
- **样本数**: 806 (全部)
- **自定义 Trainer**: MedialSurfaceRecall

### MedialSurfaceRecall Trainer
- 修改自 SkeletonRecall trainer
- 通过三轴 2D 骨架化聚合来近似标签的中轴面
- 目的：防止预测中出现孔洞，使输出更像"片状"
- 包含自定义数据增强

### 后处理 (关键！+0.019分)
1. **阈值**: softmax 输出阈值设为 **0.75** (不是0.5)
2. **修改版 Frangi filter**:
   - 原版检测"血管性"(vesselness)
   - 修改版检测"表面性"(surfaceness)
   - 通过不同的特征值处理实现

---

## 3. nnUNet 配置对比

### 标准配置 vs ResEnc 预设
nnUNet 新增了 ResEnc (Residual Encoder) 预设，性能更好：

| 配置 | 显存需求 | 训练时间(A100) | 性能 |
|------|----------|----------------|------|
| 3d_fullres (当前) | 9-11GB | ~12h | 基准 |
| **ResEnc M** | 9-11GB | ~12h | 略好 |
| **ResEnc L (推荐)** | **24GB** | ~35h | **显著更好** |
| ResEnc XL | 40GB | 更长 | 最好 |

### ResEnc L 优势
- 更多残差块
- 更深的编码器
- 在大数据集上显著提升 (KiTS2023, AMOS2022)
- **推荐作为新的默认配置**

### 使用方法
```bash
# 规划和预处理
nnUNetv2_plan_and_preprocess -d DATASET -pl nnUNetPlannerResEncL

# 如果已有预处理数据，只需重新规划
nnUNetv2_plan_experiment -d DATASET -pl nnUNetPlannerResEncL
```

**注意**: ResEnc 预设使用与标准配置相同的预处理数据

---

## 4. Frangi Filter 后处理

### 原理
Frangi filter 基于 Hessian 矩阵的特征值分析：
- 计算每个体素的 Hessian 矩阵
- 分析三个特征值 (lambda1, lambda2, lambda3)
- 根据特征值比例判断结构类型

### 结构类型判断
| 特征值关系 | 结构类型 |
|------------|----------|
| lambda1 ~ lambda2 ~ lambda3 ~ 0 | 噪声/背景 |
| lambda1 ~ lambda2 ~ 0, lambda3 大 | 片状/表面 |
| lambda1 ~ 0, lambda2 ~ lambda3 大 | 管状/血管 |
| lambda1 ~ lambda2 ~ lambda3 大 | 球状/斑点 |

### 修改版 Frangi (表面检测)
官方 baseline 修改了 Frangi filter：
- 原版：增强管状结构 (vesselness)
- 修改版：增强片状结构 (surfaceness)
- 通过调整特征值公式实现

### Python 实现
```python
# scikit-image 有内置 frangi 函数
from skimage.filters import frangi

# 或使用 ellisdg/frangi3d 库
# pip install frangi3d
```

---

## 5. 其他高分方法

### 5.1 ThaumatoAnakalyptor (2023大奖方案)
**GitHub**: `schillij95/ThaumatoAnakalyptor`

**流程**:
1. **3D 表面检测**: 用 3D Sobel 核卷积体积图像
2. **梯度过滤**: 基于梯度幅度和方向过滤体素
3. **实例分割**: 使用 Mask3D 深度网络分割点云
4. **网格生成**: 将点云分组为片状并创建网格

**特点**:
- 不是纯深度学习方法
- 结合传统图像处理和深度学习
- 需要复杂的流水线

### 5.2 Geodesic Voronoi (hengck23)
**用途**: 分离不同的纸莎草层

**流程**:
1. 在 3D 连通分量中识别不同的片
2. 使用射线测试选择每片的种子像素
3. 运行测地 Voronoi (2-pass chamfer/raster-scan)
4. 基于测地距离分配标签

**优势**:
- 处理"平滑 C 曲线"问题（相邻片接触）
- 可在 CPU 和 GPU 上运行

### 5.3 SwinUNETR (Transformer)
**特点**:
- 结合 Vision Transformer 和卷积网络
- 3D 医学图像分割 SOTA
- SwinUNETR-V2 进一步改进

**应用**:
- 在 Ink Detection 比赛中被使用
- 适合 Surface Detection 任务

---

## 6. 数据增强策略

### 标准增强
- 旋转、缩放、裁剪
- 每个 TIFF 切片独立应用

### 高级技术
- **平移不变性**: 显著提升性能
- **TTA (测试时增强)**: 多个增强版本预测后聚合

### nnUNet 内置增强
- 弹性变形
- 旋转
- 缩放
- 镜像
- 伽马变换
- 高斯噪声

---

## 7. 可用数据集

### Kaggle 数据集
| 名称 | 说明 |
|------|------|
| vesuvius-challenge-surface-detection | 原始比赛数据 |
| Vesuvius Surface: nnUNet preprocessed | 预处理好的 nnUNet 格式 |

### nnUNet 预处理数据集特点
- 806 训练样本
- 3D TIFF 格式
- 三个标签: 0=背景, 1=表面, 2=忽略
- 包含 3d_fullres 预处理输出
- 可节省 1-2 小时预处理时间

---

## 8. 提分策略总结

### 短期 (不需要重新训练)
1. **Frangi filter 后处理** (+0.02 预期)
2. **阈值调整**: 0.5 -> 0.75
3. **更好的连通分量后处理**

### 中期 (需要重新训练)
1. **增加训练 epochs**: 300 -> 1000+
2. **使用全部 806 样本**
3. **实现 MedialSurfaceRecall loss**

### 长期 (需要更好硬件)
1. **nnUNet ResEnc L** (需要 24GB 显存)
2. **尝试 SwinUNETR**
3. **ThaumatoAnakalyptor 流水线**

### 预期分数
| 策略 | 预期分数 |
|------|----------|
| 当前 | 0.492 |
| + 后处理优化 | 0.51-0.52 |
| + 更多训练 | 0.52-0.55 |
| + ResEnc L | 0.56-0.58 |
| 榜首水平 | 0.577+ |

---

## 9. 参考资源

### 官方资源
- [Kaggle 比赛页面](https://kaggle.com/competitions/vesuvius-challenge-surface-detection)
- [Vesuvius Challenge 官网](https://scrollprize.org)
- [nnUNet GitHub](https://github.com/MIC-DKFZ/nnUNet)

### 代码仓库
- [ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor)
- [Mask3D](https://github.com/JonasSchult/Mask3D)
- [frangi3d](https://github.com/ellisdg/frangi3d)

### 论文
- nnUNet: Nature Methods 2021
- SwinUNETR: CVPR 2022
- Mask3D: ICRA 2023

---

## 10. 待办事项

- [ ] 实现 Frangi filter 后处理
- [ ] 测试阈值 0.75
- [ ] 完成 806 样本训练
- [ ] 尝试 ResEnc L (需要 A100)
- [ ] 研究 MedialSurfaceRecall 实现
