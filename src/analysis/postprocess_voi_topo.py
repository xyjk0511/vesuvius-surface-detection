# ==================== VOI/TopoScore 后处理优化脚本 ====================
# 目的: 针对VOI(35%)和TopoScore(30%)优化后处理
# 评估公式: Score = 0.30*TopoScore + 0.35*SurfaceDice + 0.35*VOI_score
# 运行环境: 本地 (有标签数据) 或 Kaggle (P100/T4x2)

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize_3d, remove_small_objects, remove_small_holes
from skimage.measure import label as sk_label
import warnings
warnings.filterwarnings('ignore')

# ==================== Part 1: 评估指标实现 ====================

def dice_score(pred, label, ignore_label=2):
    """标准Dice Score"""
    mask = label != ignore_label
    pred_masked = pred[mask].astype(np.float32)
    label_masked = (label[mask] == 1).astype(np.float32)

    intersection = np.sum(pred_masked * label_masked)
    union = np.sum(pred_masked) + np.sum(label_masked)

    if union == 0:
        return 1.0
    return 2 * intersection / union


def compute_voi(pred, label, ignore_label=2):
    """
    计算 Variation of Information (VOI)
    VOI = VOI_split + VOI_merge
    VOI_score = 1 / (1 + 0.3 * VOI)

    VOI_split: 过分割程度 (一个GT被分成多个pred)
    VOI_merge: 欠分割程度 (多个GT被合并成一个pred)
    """
    mask = label != ignore_label
    pred_masked = pred[mask].ravel()
    label_masked = (label[mask] == 1).astype(np.uint8).ravel()

    # 计算联合分布
    n = len(pred_masked)
    if n == 0:
        return 0.0, 0.0, 1.0

    # 简化: 二值分割的VOI
    # P(pred=1, label=1), P(pred=1, label=0), P(pred=0, label=1), P(pred=0, label=0)
    p11 = np.sum((pred_masked == 1) & (label_masked == 1)) / n
    p10 = np.sum((pred_masked == 1) & (label_masked == 0)) / n
    p01 = np.sum((pred_masked == 0) & (label_masked == 1)) / n
    p00 = np.sum((pred_masked == 0) & (label_masked == 0)) / n

    # 边缘分布
    p_pred1 = p11 + p10
    p_pred0 = p01 + p00
    p_label1 = p11 + p01
    p_label0 = p10 + p00

    def entropy(p):
        if p <= 0 or p >= 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)

    def cond_entropy(pxy, px):
        if pxy <= 0 or px <= 0:
            return 0
        return -pxy * np.log2(pxy / px)

    # H(Label|Pred) = VOI_split
    H_label_given_pred = (cond_entropy(p11, p_pred1) + cond_entropy(p01, p_pred1) +
                          cond_entropy(p10, p_pred0) + cond_entropy(p00, p_pred0))

    # H(Pred|Label) = VOI_merge
    H_pred_given_label = (cond_entropy(p11, p_label1) + cond_entropy(p10, p_label1) +
                          cond_entropy(p01, p_label0) + cond_entropy(p00, p_label0))

    voi_split = H_label_given_pred
    voi_merge = H_pred_given_label
    voi_total = voi_split + voi_merge

    # 竞赛公式: VOI_score = 1 / (1 + alpha * VOI), alpha=0.3
    voi_score = 1.0 / (1.0 + 0.3 * voi_total)

    return voi_split, voi_merge, voi_score


def compute_surface_dice(pred, label, tolerance=2, ignore_label=2):
    """
    计算 Surface Dice @ tolerance
    只关注表面像素，容差范围内算匹配
    """
    mask = label != ignore_label

    # 提取表面 (边界像素)
    pred_binary = (pred > 0).astype(np.uint8)
    label_binary = (label == 1).astype(np.uint8)

    # 腐蚀后取差得到边界
    struct = ndimage.generate_binary_structure(3, 1)
    pred_eroded = ndimage.binary_erosion(pred_binary, struct)
    label_eroded = ndimage.binary_erosion(label_binary, struct)

    pred_surface = pred_binary & ~pred_eroded
    label_surface = label_binary & ~label_eroded

    # 应用mask
    pred_surface = pred_surface & mask
    label_surface = label_surface & mask

    if np.sum(pred_surface) == 0 and np.sum(label_surface) == 0:
        return 1.0

    # 距离变换
    if np.sum(label_surface) > 0:
        label_dist = distance_transform_edt(~label_surface)
    else:
        label_dist = np.ones_like(pred_surface, dtype=np.float32) * 1000

    if np.sum(pred_surface) > 0:
        pred_dist = distance_transform_edt(~pred_surface)
    else:
        pred_dist = np.ones_like(label_surface, dtype=np.float32) * 1000

    # 在容差范围内的匹配
    pred_matched = np.sum(pred_surface & (label_dist <= tolerance))
    label_matched = np.sum(label_surface & (pred_dist <= tolerance))

    total_surface = np.sum(pred_surface) + np.sum(label_surface)
    if total_surface == 0:
        return 1.0

    surface_dice = (pred_matched + label_matched) / total_surface
    return surface_dice


def compute_betti_numbers(binary_mask):
    """
    计算3D二值图像的Betti数
    Betti 0: 连通分量数
    Betti 1: 孔洞数 (隧道)
    Betti 2: 空腔数

    简化实现: 只计算Betti 0和近似Betti 1
    """
    # Betti 0: 前景连通分量数
    labeled, betti0 = ndimage.label(binary_mask)

    # Betti 0 (背景): 背景连通分量数
    bg_labeled, bg_components = ndimage.label(~binary_mask)

    # 近似 Betti 1: 背景连通分量数 - 1 (排除外部背景)
    # 这是一个简化，真正的Betti 1需要persistent homology
    betti1_approx = max(0, bg_components - 1)

    return betti0, betti1_approx


def compute_topo_score(pred, label, ignore_label=2):
    """
    计算 TopoScore (基于Betti数匹配)
    TopoScore惩罚拓扑错误: 合并、分裂、孔洞
    """
    mask = label != ignore_label

    pred_binary = (pred > 0).astype(np.uint8)
    label_binary = (label == 1).astype(np.uint8)

    # 只在有效区域计算
    pred_masked = pred_binary.copy()
    label_masked = label_binary.copy()
    pred_masked[~mask] = 0
    label_masked[~mask] = 0

    # 计算Betti数
    pred_b0, pred_b1 = compute_betti_numbers(pred_masked)
    label_b0, label_b1 = compute_betti_numbers(label_masked)

    # Betti数差异
    diff_b0 = abs(pred_b0 - label_b0)
    diff_b1 = abs(pred_b1 - label_b1)

    # TopoScore: 惩罚Betti数差异
    # 简化公式: score = 1 / (1 + diff_b0 + diff_b1)
    topo_score = 1.0 / (1.0 + 0.5 * diff_b0 + 0.5 * diff_b1)

    return topo_score, pred_b0, pred_b1, label_b0, label_b1


# ==================== Part 2: 后处理策略 ====================

def postprocess_baseline(prob, threshold=0.37, min_size=100):
    """基线后处理 (当前最佳 LB 0.522)"""
    binary = (prob > threshold).astype(np.uint8)

    # 移除小连通分量
    labeled, num = ndimage.label(binary)
    if num > 0:
        sizes = ndimage.sum(binary, labeled, range(1, num + 1))
        for i, size in enumerate(sizes):
            if size < min_size:
                binary[labeled == (i + 1)] = 0

    # 填充小孔洞
    bg = 1 - binary
    labeled_bg, num_bg = ndimage.label(bg)
    if num_bg > 1:
        sizes_bg = ndimage.sum(bg, labeled_bg, range(1, num_bg + 1))
        max_bg_label = np.argmax(sizes_bg) + 1
        for i in range(1, num_bg + 1):
            if i != max_bg_label and sizes_bg[i-1] < min_size:
                binary[labeled_bg == i] = 1

    return binary


def postprocess_hysteresis(prob, t_high=0.75, t_low=0.5, min_size=100):
    """
    Hysteresis阈值后处理 (Host Baseline方法)
    - 强阈值区域作为种子
    - 弱阈值区域只有连接到强阈值才保留
    - 有助于保持拓扑连续性
    """
    strong = prob >= t_high
    weak = prob >= t_low

    if not strong.any():
        return np.zeros_like(prob, dtype=np.uint8)

    # 从强阈值区域传播到弱阈值区域
    struct = ndimage.generate_binary_structure(3, 3)  # 26-连通
    binary = ndimage.binary_propagation(strong, mask=weak, structure=struct)

    # 移除小连通分量
    binary = remove_small_objects(binary, min_size=min_size)

    return binary.astype(np.uint8)


def postprocess_voi_optimized(prob, threshold=0.37, min_size=100,
                               merge_threshold=50, close_radius=2):
    """
    针对VOI优化的后处理
    - 减少VOI_split: 合并相近的小分量
    - 减少VOI_merge: 使用形态学操作分离粘连
    """
    binary = (prob > threshold).astype(np.uint8)

    # Step 1: 闭运算 - 连接断裂的表面 (减少split)
    struct = ndimage.generate_binary_structure(3, 1)
    for _ in range(close_radius):
        binary = ndimage.binary_dilation(binary, struct)
    for _ in range(close_radius):
        binary = ndimage.binary_erosion(binary, struct)

    # Step 2: 移除小分量 (减少split噪声)
    binary = remove_small_objects(binary.astype(bool), min_size=min_size)

    # Step 3: 填充小孔洞 (减少split)
    binary = remove_small_holes(binary, area_threshold=merge_threshold)

    return binary.astype(np.uint8)


def postprocess_topo_optimized(prob, threshold=0.37, min_size=100):
    """
    针对TopoScore优化的后处理
    - 保持连通性 (Betti 0)
    - 避免产生孔洞 (Betti 1)
    """
    binary = (prob > threshold).astype(np.uint8)

    # Step 1: 移除小分量
    binary = remove_small_objects(binary.astype(bool), min_size=min_size)

    # Step 2: 填充所有内部孔洞 (减少Betti 1)
    filled = ndimage.binary_fill_holes(binary)

    # Step 3: 保持最大连通分量 (确保Betti 0 = 1)
    labeled, num = ndimage.label(filled)
    if num > 1:
        sizes = ndimage.sum(filled, labeled, range(1, num + 1))
        max_label = np.argmax(sizes) + 1
        filled = (labeled == max_label)

    return filled.astype(np.uint8)


def postprocess_combined(prob, threshold=0.37, t_high=0.6, t_low=0.3,
                         min_size=100, fill_holes=True):
    """
    综合优化策略: 结合Hysteresis和形态学操作
    平衡VOI、TopoScore和SurfaceDice
    """
    # Step 1: Hysteresis阈值 (保持连续性)
    strong = prob >= t_high
    weak = prob >= t_low

    if strong.any():
        struct = ndimage.generate_binary_structure(3, 3)
        binary = ndimage.binary_propagation(strong, mask=weak, structure=struct)
    else:
        binary = prob > threshold

    # Step 2: 移除小分量
    binary = remove_small_objects(binary.astype(bool), min_size=min_size)

    # Step 3: 可选填充孔洞
    if fill_holes:
        binary = remove_small_holes(binary, area_threshold=min_size)

    return binary.astype(np.uint8)


# ==================== Part 3: 评估和对比 ====================

def evaluate_all_metrics(pred, label, ignore_label=2):
    """计算所有评估指标"""
    dice = dice_score(pred, label, ignore_label)
    voi_split, voi_merge, voi_score = compute_voi(pred, label, ignore_label)
    surf_dice = compute_surface_dice(pred, label, tolerance=2, ignore_label=ignore_label)
    topo, pb0, pb1, lb0, lb1 = compute_topo_score(pred, label, ignore_label)

    # 竞赛总分
    total_score = 0.30 * topo + 0.35 * surf_dice + 0.35 * voi_score

    return {
        'dice': dice,
        'voi_split': voi_split,
        'voi_merge': voi_merge,
        'voi_score': voi_score,
        'surface_dice': surf_dice,
        'topo_score': topo,
        'pred_betti0': pb0,
        'pred_betti1': pb1,
        'label_betti0': lb0,
        'label_betti1': lb1,
        'total_score': total_score
    }


def compare_postprocess_methods(prob, label, ignore_label=2):
    """对比不同后处理方法的效果"""
    methods = {
        'baseline': lambda p: postprocess_baseline(p, threshold=0.37),
        'hysteresis_0.75_0.5': lambda p: postprocess_hysteresis(p, t_high=0.75, t_low=0.5),
        'hysteresis_0.6_0.3': lambda p: postprocess_hysteresis(p, t_high=0.6, t_low=0.3),
        'voi_optimized': lambda p: postprocess_voi_optimized(p, threshold=0.37),
        'topo_optimized': lambda p: postprocess_topo_optimized(p, threshold=0.37),
        'combined': lambda p: postprocess_combined(p, threshold=0.37),
    }

    results = {}
    for name, method in methods.items():
        pred = method(prob)
        metrics = evaluate_all_metrics(pred, label, ignore_label)
        results[name] = metrics

    return results


def print_comparison_table(results):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("后处理方法对比")
    print("=" * 80)
    print(f"{'方法':<25} {'Dice':>8} {'VOI':>8} {'SurfDice':>8} {'Topo':>8} {'Total':>8}")
    print("-" * 80)

    for name, m in results.items():
        print(f"{name:<25} {m['dice']:>8.4f} {m['voi_score']:>8.4f} "
              f"{m['surface_dice']:>8.4f} {m['topo_score']:>8.4f} {m['total_score']:>8.4f}")

    print("=" * 80)


# ==================== Part 4: 主测试入口 ====================

if __name__ == "__main__":
    import os
    import nibabel as nib

    # 数据路径 (本地)
    DATA_DIR = r"D:\local kaggle\nnUNet_raw\Dataset001_Vesuvius"
    PROB_DIR = r"D:\local kaggle\nnUNet_results\Dataset001_Vesuvius\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\validation"

    print("VOI/TopoScore 后处理优化测试")
    print("=" * 60)

    # 查找验证集样本
    label_dir = os.path.join(DATA_DIR, "labelsTr")
    if not os.path.exists(label_dir):
        print(f"标签目录不存在: {label_dir}")
        print("请修改 DATA_DIR 路径")
        exit(1)

    # 获取样本列表
    samples = [f.replace(".nii.gz", "") for f in os.listdir(label_dir) if f.endswith(".nii.gz")][:5]
    print(f"测试样本数: {len(samples)}")

    all_results = {}
    for sample_id in samples:
        print(f"\n处理样本: {sample_id}")

        # 加载标签
        label_path = os.path.join(label_dir, f"{sample_id}.nii.gz")
        label = nib.load(label_path).get_fdata().astype(np.uint8)

        # 加载概率图 (如果存在)
        prob_path = os.path.join(PROB_DIR, f"{sample_id}.npz")
        if os.path.exists(prob_path):
            prob = np.load(prob_path)['probabilities'][1]
        else:
            print(f"  概率图不存在: {prob_path}")
            continue

        # 对比不同后处理方法
        results = compare_postprocess_methods(prob, label)
        print_comparison_table(results)
        all_results[sample_id] = results

    # 汇总平均结果
    if all_results:
        print("\n" + "=" * 80)
        print("平均结果汇总")
        print("=" * 80)
        methods = list(list(all_results.values())[0].keys())
        for method in methods:
            avg_total = np.mean([r[method]['total_score'] for r in all_results.values()])
            avg_voi = np.mean([r[method]['voi_score'] for r in all_results.values()])
            avg_topo = np.mean([r[method]['topo_score'] for r in all_results.values()])
            print(f"{method:<25} VOI={avg_voi:.4f} Topo={avg_topo:.4f} Total={avg_total:.4f}")
