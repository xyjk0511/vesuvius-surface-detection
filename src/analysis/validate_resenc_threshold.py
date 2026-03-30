# ==================== ResEnc L 阈值+后处理验证脚本 ====================
# 用 validation-samples 数据集离线测试阈值+后处理组合
# 推理一次保留npz，测试多种组合算Dice

# ===== CELL 1: 环境配置 =====
import sys
sys.path.insert(0, '/kaggle/usr/lib/nnunet_install/packages')

import os
os.environ["nnUNet_raw"] = "/kaggle/working/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/kaggle/working/nnUNet_results"

for p in [os.environ["nnUNet_raw"], os.environ["nnUNet_preprocessed"], os.environ["nnUNet_results"]]:
    os.makedirs(p, exist_ok=True)

import nnunetv2
import torch
print(f"nnunetv2: {nnunetv2.__file__}")
print(f"CUDA: {torch.cuda.is_available()}")

# ===== CELL 2: 注册 Trainer =====
from pathlib import Path
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerResEncL(nnUNetTrainer):
    pass

import nnunetv2.utilities.find_class_by_name as find_module
_original_recursive_find = find_module.recursive_find_python_class

def patched_recursive_find(folder, class_name, current_module):
    if class_name == 'nnUNetTrainerResEncL':
        return nnUNetTrainerResEncL
    return _original_recursive_find(folder, class_name, current_module)

find_module.recursive_find_python_class = patched_recursive_find
print("Trainer patched: nnUNetTrainerResEncL")

# ===== CELL 3: 设置 Checkpoint =====
import shutil

CKPT_SRC = "/kaggle/input/2540-4"
CONFIG_SRC = "/kaggle/input/2540-part1/kaggle_upload_part1"

# 检查 checkpoint epoch
ckpt_best = Path(CKPT_SRC) / "checkpoint_best.pth"
if ckpt_best.exists():
    ckpt = torch.load(str(ckpt_best), map_location="cpu", weights_only=False)
    ep = ckpt.get("current_epoch", "unknown")
    print(f"checkpoint_best.pth: epoch={ep}")
    del ckpt
else:
    raise FileNotFoundError(f"checkpoint_best.pth not found in {CKPT_SRC}")

MODEL_DIR = "/kaggle/working/nnUNet_results/Dataset101_Vesuvius/nnUNetTrainerResEncL__nnUNetResEncUNetLPlans__3d_fullres"
FOLD_DIR = f"{MODEL_DIR}/fold_0"
os.makedirs(FOLD_DIR, exist_ok=True)

for f in Path(CKPT_SRC).glob("*.pth"):
    if f.name.startswith("fold0_"):
        print(f"Skipped old: {f.name}")
        continue
    shutil.copy(f, f"{FOLD_DIR}/{f.name}")
    print(f"Copied: {f.name}")

for f in Path(CKPT_SRC).glob("*.json"):
    shutil.copy(f, MODEL_DIR)
    print(f"Copied: {f.name}")

if Path(CONFIG_SRC).exists():
    for f in Path(CONFIG_SRC).glob("*.json"):
        dst = Path(MODEL_DIR) / f.name
        if not dst.exists():
            shutil.copy(f, dst)
            print(f"Copied from CONFIG_SRC: {f.name}")

plans_src = Path(MODEL_DIR) / "nnUNetResEncUNetLPlans.json"
plans_dst = Path(MODEL_DIR) / "plans.json"
if plans_src.exists() and not plans_dst.exists():
    shutil.copy(plans_src, plans_dst)
    print("Renamed: nnUNetResEncUNetLPlans.json -> plans.json")

for f in Path(CKPT_SRC).glob("*.pkl"):
    shutil.copy(f, MODEL_DIR)

print("Checkpoint setup done!")

# ===== CELL 4: 准备验证数据 =====
import numpy as np
import nibabel as nib
import gc
import time
import inspect
import inspect
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

RAW_IMG_DIR = "/kaggle/input/validate-sample-new/imagesTr"
RAW_LBL_DIR = "/kaggle/input/validate-sample-new/labelsTr"

INPUT_DIR = "/kaggle/working/nnUNet_raw/Dataset101_Vesuvius/imagesTs"
OUTPUT_DIR = "/kaggle/working/nnUNet_output"
for d in [INPUT_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# 找到所有样本
all_files = sorted(list(Path(RAW_IMG_DIR).glob("*.nii")) + list(Path(RAW_IMG_DIR).glob("*.nii.gz")))
val_samples = []
for f in all_files:
    stem = f.stem.replace(".nii", "")
    if stem.endswith("_0000"):
        stem = stem[:-5]
    val_samples.append(stem)
val_samples = sorted(set(val_samples))
import random
random.seed(42)
if len(val_samples) > 20:
    val_samples = sorted(random.sample(val_samples, 20))
print(f"Validation samples ({len(val_samples)}): {val_samples}")

# 复制原始 nii 到输入目录
for sample_id in val_samples:
    for ext in [".nii", ".nii.gz"]:
        src = Path(RAW_IMG_DIR) / f"{sample_id}_0000{ext}"
        if src.exists():
            shutil.copy(src, Path(INPUT_DIR) / f"{sample_id}_0000{ext}")
            print(f"Copied: {src.name}")
            break

# 读取标签
labels = {}
for sample_id in val_samples:
    for ext in [".nii", ".nii.gz"]:
        lbl_path = Path(RAW_LBL_DIR) / f"{sample_id}{ext}"
        if lbl_path.exists():
            lbl_data = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)
            labels[sample_id] = lbl_data
            print(f"Label {sample_id}: shape={lbl_data.shape}, unique={np.unique(lbl_data)}")
            break

print(f"Loaded {len(labels)} labels")

# ===== CELL 5: 推理 (TTA) =====
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False,
    device=torch.device('cuda'),
    verbose=True,
    verbose_preprocessing=True
)

predictor.initialize_from_trained_model_folder(
    MODEL_DIR,
    use_folds=(0,),
    checkpoint_name='checkpoint_best.pth'
)

input_files = []
for sample_id in val_samples:
    for ext in [".nii", ".nii.gz"]:
        p = Path(INPUT_DIR) / f"{sample_id}_0000{ext}"
        if p.exists():
            input_files.append(p)
            break
print(f"Inference files: {len(input_files)}")
predictor.predict_from_files(
    [[str(f)] for f in input_files],
    OUTPUT_DIR,
    save_probabilities=True,
    overwrite=True,
    num_processes_preprocessing=2,
    num_processes_segmentation_export=2
)

print("Inference done!")

# ===== CELL 6: 后处理函数 + Dice =====

def dice_score(pred, label, ignore_label=2):
    if pred.shape != label.shape:
        return 0.0
    mask = label != ignore_label
    pred_m = pred[mask]
    label_m = (label[mask] == 1).astype(np.uint8)
    intersection = np.sum(pred_m * label_m)
    union = np.sum(pred_m) + np.sum(label_m)
    if union == 0:
        return 1.0
    return 2.0 * intersection / union

def dust_removal(pred, min_size):
    labeled, num = ndimage.label(pred)
    if num == 0:
        return pred
    sizes = np.bincount(labeled.ravel())
    if sizes.size <= 1:
        return pred
    remove = sizes < min_size
    remove[0] = False
    pred[remove[labeled]] = 0
    return pred

def hole_filling(pred, min_size=100):
    bg = (pred == 0)
    labeled_bg, num_bg = ndimage.label(bg)
    if num_bg <= 1:
        return pred
    sizes_bg = np.bincount(labeled_bg.ravel())
    if sizes_bg.size <= 1:
        return pred
    max_bg_label = np.argmax(sizes_bg[1:]) + 1
    fill = sizes_bg < min_size
    fill[0] = False
    fill[max_bg_label] = False
    pred[fill[labeled_bg]] = 1
    return pred

def morpho_closing(pred, iterations=1):
    from scipy.ndimage import binary_closing, generate_binary_structure
    struct = generate_binary_structure(3, 2)
    return binary_closing(pred, structure=struct, iterations=iterations).astype(np.uint8)

def gaussian_smooth_threshold(prob_fg, sigma, thr):
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(prob_fg, sigma=sigma)
    return (smoothed >= thr).astype(np.uint8)

def keep_largest_n(pred, n=1):
    labeled, num = ndimage.label(pred)
    if num <= n:
        return pred
    sizes = ndimage.sum(pred, labeled, range(1, num + 1))
    top_labels = np.argsort(sizes)[::-1][:n] + 1
    result = np.zeros_like(pred)
    for lbl in top_labels:
        result[labeled == lbl] = 1
    return result

# --- 竞赛评估指标 ---

def compute_voi(pred, label, ignore_label=2):
    mask = label != ignore_label
    pred_masked = pred[mask].ravel()
    label_masked = (label[mask] == 1).astype(np.uint8).ravel()
    n = len(pred_masked)
    if n == 0:
        return 0.0, 0.0, 1.0
    p11 = np.sum((pred_masked == 1) & (label_masked == 1)) / n
    p10 = np.sum((pred_masked == 1) & (label_masked == 0)) / n
    p01 = np.sum((pred_masked == 0) & (label_masked == 1)) / n
    p00 = np.sum((pred_masked == 0) & (label_masked == 0)) / n
    p_pred1 = p11 + p10
    p_pred0 = p01 + p00
    p_label1 = p11 + p01
    p_label0 = p10 + p00
    def cond_entropy(pxy, px):
        if pxy <= 0 or px <= 0:
            return 0
        return -pxy * np.log2(pxy / px)
    H_label_given_pred = (cond_entropy(p11, p_pred1) + cond_entropy(p01, p_pred1) +
                          cond_entropy(p10, p_pred0) + cond_entropy(p00, p_pred0))
    H_pred_given_label = (cond_entropy(p11, p_label1) + cond_entropy(p10, p_label1) +
                          cond_entropy(p01, p_label0) + cond_entropy(p00, p_label0))
    voi_total = H_label_given_pred + H_pred_given_label
    voi_score = 1.0 / (1.0 + 0.3 * voi_total)
    return H_label_given_pred, H_pred_given_label, voi_score

def compute_surface_dice(pred, label, tolerance=2, ignore_label=2):
    mask = label != ignore_label
    pred_binary = (pred > 0).astype(np.uint8)
    label_binary = (label == 1).astype(np.uint8)
    struct = ndimage.generate_binary_structure(3, 1)
    pred_eroded = ndimage.binary_erosion(pred_binary, struct)
    label_eroded = ndimage.binary_erosion(label_binary, struct)
    pred_surface = pred_binary & ~pred_eroded & mask
    label_surface = label_binary & ~label_eroded & mask
    if np.sum(pred_surface) == 0 and np.sum(label_surface) == 0:
        return 1.0
    if np.sum(label_surface) > 0:
        label_dist = distance_transform_edt(~label_surface)
    else:
        label_dist = np.ones_like(pred_surface, dtype=np.float32) * 1000
    if np.sum(pred_surface) > 0:
        pred_dist = distance_transform_edt(~pred_surface)
    else:
        pred_dist = np.ones_like(label_surface, dtype=np.float32) * 1000
    pred_matched = np.sum(pred_surface & (label_dist <= tolerance))
    label_matched = np.sum(label_surface & (pred_dist <= tolerance))
    total_surface = np.sum(pred_surface) + np.sum(label_surface)
    if total_surface == 0:
        return 1.0
    return (pred_matched + label_matched) / total_surface

def compute_betti_numbers(binary_mask):
    mask_bool = binary_mask.astype(bool)
    labeled, betti0 = ndimage.label(mask_bool)
    bg_labeled, bg_components = ndimage.label(~mask_bool)
    betti1_approx = max(0, bg_components - 1)
    return betti0, betti1_approx

def compute_topo_score(pred, label, ignore_label=2):
    mask = label != ignore_label
    pred_masked = (pred > 0).astype(np.uint8).copy()
    label_masked = (label == 1).astype(np.uint8).copy()
    pred_masked[~mask] = 0
    label_masked[~mask] = 0
    pred_b0, pred_b1 = compute_betti_numbers(pred_masked)
    label_b0, label_b1 = compute_betti_numbers(label_masked)
    diff_b0 = abs(pred_b0 - label_b0)
    diff_b1 = abs(pred_b1 - label_b1)
    topo_score = 1.0 / (1.0 + 0.5 * diff_b0 + 0.5 * diff_b1)
    return topo_score, pred_b0, pred_b1, label_b0, label_b1

def _downsample_3d(arr, factor=2):
    """最近邻 2x 降采样 3D 数组"""
    return arr[::factor, ::factor, ::factor]

def build_metric_context(label, ignore_label=2, tolerance=2, ds_factor=2):
    mask = label != ignore_label
    label_binary = (label == 1).astype(np.uint8)
    label_masked_flat = label_binary[mask].ravel()

    # 降采样版本 (用于 SurfaceDice 和 Betti)
    mask_ds = _downsample_3d(mask, ds_factor)
    label_binary_ds = _downsample_3d(label_binary, ds_factor)

    struct = ndimage.generate_binary_structure(3, 1)

    # SurfaceDice: 降采样计算
    label_eroded_ds = ndimage.binary_erosion(label_binary_ds, struct)
    label_surface_ds = (label_binary_ds & ~label_eroded_ds & mask_ds).astype(bool)
    tol_ds = max(1, tolerance // ds_factor)

    if np.sum(label_surface_ds) > 0:
        label_dist_ds = distance_transform_edt(~label_surface_ds).astype(np.float32)
    else:
        label_dist_ds = np.ones_like(label_surface_ds, dtype=np.float32) * 1000

    # Betti: 全分辨率计算 (ndimage.label 在 320^3 上很快)
    label_masked_full = label_binary.copy()
    label_masked_full[~mask] = 0
    label_b0, label_b1 = compute_betti_numbers(label_masked_full)

    return {
        "mask": mask,
        "label_binary": label_binary,
        "label_masked_flat": label_masked_flat,
        "label_b0": label_b0,
        "label_b1": label_b1,
        "surface_struct": struct,
        "tolerance": tolerance,
        # 降采样字段
        "ds_factor": ds_factor,
        "mask_ds": mask_ds,
        "label_surface_ds": label_surface_ds,
        "label_dist_ds": label_dist_ds,
        "tol_ds": tol_ds,
    }


def evaluate_all_metrics(pred, label, ignore_label=2, metric_ctx=None):
    if metric_ctx is None:
        d = dice_score(pred, label, ignore_label)
        _, _, voi_sc = compute_voi(pred, label, ignore_label)
        surf_d = compute_surface_dice(pred, label, tolerance=2, ignore_label=ignore_label)
        topo, pb0, pb1, lb0, lb1 = compute_topo_score(pred, label, ignore_label)
    else:
        mask = metric_ctx["mask"]
        label_binary = metric_ctx["label_binary"]
        label_masked_flat = metric_ctx["label_masked_flat"]

        pred_binary = (pred > 0).astype(np.uint8)
        pred_masked = pred_binary[mask].ravel()

        # Dice
        intersection = np.sum(pred_masked * label_masked_flat)
        union = np.sum(pred_masked) + np.sum(label_masked_flat)
        d = 1.0 if union == 0 else (2.0 * intersection / union)

        # VOI
        n = len(pred_masked)
        if n == 0:
            voi_sc = 1.0
        else:
            p11 = np.sum((pred_masked == 1) & (label_masked_flat == 1)) / n
            p10 = np.sum((pred_masked == 1) & (label_masked_flat == 0)) / n
            p01 = np.sum((pred_masked == 0) & (label_masked_flat == 1)) / n
            p00 = np.sum((pred_masked == 0) & (label_masked_flat == 0)) / n
            p_pred1 = p11 + p10
            p_pred0 = p01 + p00
            p_label1 = p11 + p01
            p_label0 = p10 + p00

            def cond_entropy(pxy, px):
                if pxy <= 0 or px <= 0:
                    return 0.0
                return -pxy * np.log2(pxy / px)

            h_label_given_pred = (cond_entropy(p11, p_pred1) + cond_entropy(p01, p_pred1) +
                                  cond_entropy(p10, p_pred0) + cond_entropy(p00, p_pred0))
            h_pred_given_label = (cond_entropy(p11, p_label1) + cond_entropy(p10, p_label1) +
                                  cond_entropy(p01, p_label0) + cond_entropy(p00, p_label0))
            voi_total = h_label_given_pred + h_pred_given_label
            voi_sc = 1.0 / (1.0 + 0.3 * voi_total)

        # SurfaceDice (降采样加速)
        ds = metric_ctx["ds_factor"]
        pred_ds = _downsample_3d(pred_binary, ds)
        mask_ds = metric_ctx["mask_ds"]
        struct = metric_ctx["surface_struct"]
        pred_eroded_ds = ndimage.binary_erosion(pred_ds, struct)
        pred_surface_ds = (pred_ds & ~pred_eroded_ds & mask_ds).astype(bool)
        label_surface_ds = metric_ctx["label_surface_ds"]
        tol_ds = metric_ctx["tol_ds"]

        if np.sum(pred_surface_ds) == 0 and np.sum(label_surface_ds) == 0:
            surf_d = 1.0
        else:
            label_dist_ds = metric_ctx["label_dist_ds"]
            if np.sum(pred_surface_ds) > 0:
                pred_dist_ds = distance_transform_edt(~pred_surface_ds).astype(np.float32)
            else:
                pred_dist_ds = np.ones_like(label_surface_ds, dtype=np.float32) * 1000
            pred_matched = np.sum(pred_surface_ds & (label_dist_ds <= tol_ds))
            label_matched = np.sum(label_surface_ds & (pred_dist_ds <= tol_ds))
            total_surface = np.sum(pred_surface_ds) + np.sum(label_surface_ds)
            surf_d = 1.0 if total_surface == 0 else ((pred_matched + label_matched) / total_surface)

        # TopoScore (全分辨率计算 Betti)
        pred_topo = pred_binary.copy()
        pred_topo[~mask] = 0
        pb0, pb1 = compute_betti_numbers(pred_topo)
        lb0 = metric_ctx["label_b0"]
        lb1 = metric_ctx["label_b1"]
        diff_b0 = abs(pb0 - lb0)
        diff_b1 = abs(pb1 - lb1)
        import math
        topo = 1.0 / (1.0 + 0.5 * math.log(1 + diff_b0) + 0.5 * math.log(1 + diff_b1))

    total = 0.30 * topo + 0.35 * surf_d + 0.35 * voi_sc
    return {
        'dice': d, 'voi_score': voi_sc, 'surface_dice': surf_d,
        'topo_score': topo, 'total_score': total,
        'pred_b0': pb0, 'pred_b1': pb1, 'label_b0': lb0, 'label_b1': lb1
    }

# --- 新增后处理方法 ---

def postprocess_hysteresis(prob, t_high=0.75, t_low=0.5, min_size=100):
    strong = prob >= t_high
    weak = prob >= t_low
    if not strong.any():
        return np.zeros_like(prob, dtype=np.uint8)
    struct = ndimage.generate_binary_structure(3, 3)
    binary = ndimage.binary_propagation(strong, mask=weak, structure=struct).astype(np.uint8)
    binary = dust_removal(binary, min_size)
    return binary

def postprocess_voi_optimized(prob, threshold=0.37, min_size=100, close_radius=2):
    binary = (prob > threshold).astype(np.uint8)
    struct = ndimage.generate_binary_structure(3, 1)
    for _ in range(close_radius):
        binary = ndimage.binary_dilation(binary, struct)
    for _ in range(close_radius):
        binary = ndimage.binary_erosion(binary, struct)
    binary = dust_removal(binary.astype(np.uint8), min_size)
    binary = hole_filling(binary.astype(np.uint8), min_size=min_size)
    return binary.astype(np.uint8)

def postprocess_topo_optimized(prob, threshold=0.37, min_size=100):
    binary = (prob > threshold).astype(np.uint8)
    binary = dust_removal(binary.astype(np.uint8), min_size)
    filled = ndimage.binary_fill_holes(binary)
    labeled, num = ndimage.label(filled)
    if num > 1:
        sizes = ndimage.sum(filled, labeled, range(1, num + 1))
        max_label = np.argmax(sizes) + 1
        filled = (labeled == max_label)
    return filled.astype(np.uint8)

def postprocess_combined(prob, threshold=0.37, t_high=0.6, t_low=0.3,
                         min_size=100, fill_holes=True):
    strong = prob >= t_high
    weak = prob >= t_low
    if strong.any():
        struct = ndimage.generate_binary_structure(3, 3)
        binary = ndimage.binary_propagation(strong, mask=weak, structure=struct)
    else:
        binary = prob > threshold
    binary = dust_removal(binary.astype(np.uint8), min_size)
    if fill_holes:
        binary = hole_filling(binary.astype(np.uint8), min_size=min_size)
    return binary.astype(np.uint8)

# ===== CELL 7: 阈值 + 后处理组合测试 (竞赛复合分数) =====

# 调试: 列出实际的 npz 文件
import time
import inspect

all_npz = sorted(Path(OUTPUT_DIR).glob("*.npz"))
print(f"NPZ files in OUTPUT_DIR ({len(all_npz)}):")
for f in all_npz:
    print(f"  {f.name}")
print(f"val_samples: {val_samples}")

RUN_MODE = "focused"  # "fast", "focused", or "full"
thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.37, 0.40, 0.50]

# 需要概率图的方法单独处理
postprocess_binary = {
    "none":           lambda p: p.copy(),
    "dust_100":       lambda p: dust_removal(p.copy(), 100),
    "dust_500":       lambda p: dust_removal(p.copy(), 500),
    "dust_500_fill":  lambda p: hole_filling(dust_removal(p.copy(), 500), 100),
    "closing_1":      lambda p: morpho_closing(p.copy(), iterations=1),
    "closing_dust":   lambda p: dust_removal(morpho_closing(p.copy(), 1), 500),
    "largest_1":      lambda p: keep_largest_n(p.copy(), n=1),
    "largest_3":      lambda p: keep_largest_n(p.copy(), n=3),
}

if RUN_MODE == "focused":
    # 更细的阈值，只保留得分高的方法
    thresholds = [0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.37, 0.40]
    postprocess_binary = {
        "dust_200":     lambda p: dust_removal(p.copy(), 200),
        "dust_500":     lambda p: dust_removal(p.copy(), 500),
        "dust_1000":    lambda p: dust_removal(p.copy(), 1000),
        "closing_dust": lambda p: dust_removal(morpho_closing(p.copy(), 1), 500),
        "largest_1":    lambda p: keep_largest_n(p.copy(), n=1),
    }
    gaussian_sigmas = [1.0]
    gaussian_thresholds = [0.25, 0.28, 0.30, 0.32, 0.35]
    hyst_configs = []
    voi_topo_thresholds = [0.25, 0.30, 0.35, 0.40]
    combined_configs = [
        ("combined_0.6_0.3", 0.37, 0.6, 0.3),
        ("combined_0.6_0.25", 0.37, 0.6, 0.25),
        ("combined_0.5_0.25", 0.37, 0.5, 0.25),
    ]
    metric_ds_factor = 2
elif RUN_MODE == "fast":
    thresholds = [0.30, 0.35, 0.37, 0.40]
    postprocess_binary = {
        "none":         lambda p: p.copy(),
        "dust_500":     lambda p: dust_removal(p.copy(), 500),
        "closing_dust": lambda p: dust_removal(morpho_closing(p.copy(), 1), 500),
        "largest_1":    lambda p: keep_largest_n(p.copy(), n=1),
    }
    gaussian_sigmas = [0.5, 1.0]
    gaussian_thresholds = [0.30, 0.35, 0.40]
    hyst_configs = [("hyst_0.6_0.3", 0.6, 0.3)]
    voi_topo_thresholds = [0.35]
    combined_configs = [("combined_0.6_0.3", 0.37, 0.6, 0.3)]
    metric_ds_factor = 4
else:
    gaussian_sigmas = [0.5, 1.0]
    gaussian_thresholds = [0.25, 0.30, 0.35, 0.40]
    hyst_configs = [
        ("hyst_0.6_0.3", 0.6, 0.3),
        ("hyst_0.5_0.25", 0.5, 0.25),
        ("hyst_0.4_0.2", 0.4, 0.2),
    ]
    voi_topo_thresholds = [0.25, 0.30, 0.35]
    combined_configs = [
        ("combined_0.6_0.3", 0.37, 0.6, 0.3),
        ("combined_0.5_0.25", 0.37, 0.5, 0.25),
        ("combined_0.4_0.2", 0.37, 0.4, 0.2),
    ]
    metric_ds_factor = 2

print(
    f"RUN_MODE={RUN_MODE} | thresholds={len(thresholds)} | binary_methods={len(postprocess_binary)} | "
    f"gauss={len(gaussian_sigmas) * len(gaussian_thresholds)} | hyst={len(hyst_configs)} | "
    f"voi_topo={len(voi_topo_thresholds) * 2} | combined={len(combined_configs)} | ds_factor={metric_ds_factor}"
)
build_ctx_has_ds_factor = "ds_factor" in inspect.signature(build_metric_context).parameters
if not build_ctx_has_ds_factor:
    print("WARNING: build_metric_context has no ds_factor in current kernel, fallback to default call")

# 收集结果
results = []

# 建立 sample_id -> npz 路径的映射 (支持前缀匹配)
npz_map = {}
for npz_f in all_npz:
    npz_map[npz_f.stem] = npz_f

def find_npz(sample_id):
    base = sample_id.replace("_0000", "")
    # 精确匹配
    for c in [sample_id, base, base + "_0", sample_id + "_0"]:
        if c in npz_map:
            return npz_map[c]
    # 前缀匹配: npz名可能是sample_id的前缀(nnUNet截断)
    for stem, path in npz_map.items():
        if base.startswith(stem) or stem.startswith(base):
            return path
    return None

total_samples = len(val_samples)
combos_binary = len(thresholds) * len(postprocess_binary)
combos_gauss = len(gaussian_sigmas) * len(gaussian_thresholds)
combos_hyst = len(hyst_configs)
combos_voi_topo = len(voi_topo_thresholds) * 2
combos_combined = len(combined_configs)
combos_per_sample = combos_binary + combos_gauss + combos_hyst + combos_voi_topo + combos_combined
overall_start = time.time()

for sample_idx, sample_id in enumerate(val_samples, start=1):
    sample_start = time.time()
    print(f"\n=== Sample [{sample_idx}/{total_samples}] {sample_id} | target combos={combos_per_sample} ===", flush=True)
    npz_path = find_npz(sample_id)
    if npz_path is None:
        print(f"WARNING: npz not found for {sample_id}, skip")
        continue

    probs = np.load(str(npz_path))['probabilities']
    prob_fg = probs[1]
    del probs

    if sample_id not in labels:
        print(f"WARNING: label not found for {sample_id}, skip")
        del prob_fg
        continue

    label = labels[sample_id]
    if build_ctx_has_ds_factor:
        metric_ctx = build_metric_context(label, ignore_label=2, tolerance=2, ds_factor=metric_ds_factor)
    else:
        metric_ctx = build_metric_context(label, ignore_label=2, tolerance=2)

    # 新数据集(val_samples_d101)已与Dataset101预处理一致
    # NIfTI输入已转置，nnUNet输出npz直接和label对齐，不需要再转
    fg_prob = prob_fg

    print(f"\n--- {sample_id} ---")
    print(f"fg_prob: {fg_prob.shape}, label: {label.shape}")
    print(f"prob range: [{fg_prob.min():.4f}, {fg_prob.max():.4f}]")

    # 二值后处理方法
    for thr in thresholds:
        binary = (fg_prob >= thr).astype(np.uint8)
        for pp_name, pp_fn in postprocess_binary.items():
            pred = pp_fn(binary)
            m = evaluate_all_metrics(pred, label, metric_ctx=metric_ctx)
            results.append({
                "sample": sample_id, "thr": thr,
                "method": pp_name, **m
            })
            del pred
    print(f"  binary methods done: {combos_binary} combos, elapsed={time.time() - sample_start:.1f}s", flush=True)

    # Gaussian smooth 方法 (直接用概率图)
    for sigma in gaussian_sigmas:
        for thr in gaussian_thresholds:
            pred = gaussian_smooth_threshold(fg_prob, sigma, thr)
            m = evaluate_all_metrics(pred, label, metric_ctx=metric_ctx)
            results.append({
                "sample": sample_id, "thr": thr,
                "method": f"gauss_s{sigma}", **m
            })
            del pred
    print(f"  gaussian done: +{combos_gauss} combos, elapsed={time.time() - sample_start:.1f}s", flush=True)

    # Hysteresis 方法 (需要概率图)
    for name, t_high, t_low in hyst_configs:
        pred = postprocess_hysteresis(fg_prob, t_high=t_high, t_low=t_low, min_size=100)
        m = evaluate_all_metrics(pred, label, metric_ctx=metric_ctx)
        results.append({
            "sample": sample_id, "thr": t_low,
            "method": name, **m
        })
        del pred
    print(f"  hysteresis done: +{combos_hyst} combos, elapsed={time.time() - sample_start:.1f}s", flush=True)

    # VOI/Topo 优化方法
    for thr in voi_topo_thresholds:
        pred = postprocess_voi_optimized(fg_prob, threshold=thr, min_size=100, close_radius=2)
        m = evaluate_all_metrics(pred, label, metric_ctx=metric_ctx)
        results.append({
            "sample": sample_id, "thr": thr,
            "method": "voi_opt", **m
        })
        del pred

        pred = postprocess_topo_optimized(fg_prob, threshold=thr, min_size=100)
        m = evaluate_all_metrics(pred, label, metric_ctx=metric_ctx)
        results.append({
            "sample": sample_id, "thr": thr,
            "method": "topo_opt", **m
        })
        del pred
    print(f"  voi/topo done: +{combos_voi_topo} combos, elapsed={time.time() - sample_start:.1f}s", flush=True)

    # Combined 方法
    for name, thr, t_high, t_low in combined_configs:
        pred = postprocess_combined(fg_prob, threshold=thr, t_high=t_high, t_low=t_low, min_size=100)
        m = evaluate_all_metrics(pred, label, metric_ctx=metric_ctx)
        results.append({
            "sample": sample_id, "thr": t_low,
            "method": name, **m
        })
        del pred
    print(f"  combined done: +{combos_combined} combos, elapsed={time.time() - sample_start:.1f}s", flush=True)

    del prob_fg, fg_prob
    gc.collect()
    sample_elapsed = time.time() - sample_start
    total_elapsed = time.time() - overall_start
    avg_per_sample = total_elapsed / sample_idx
    eta = avg_per_sample * (total_samples - sample_idx)
    print(
        f"=== Sample done in {sample_elapsed/60:.2f} min | total elapsed {total_elapsed/60:.2f} min | ETA {eta/60:.2f} min ===",
        flush=True
    )

print("\n\nScoring done!")

# ===== CELL 8: 汇总结果 =====

from collections import defaultdict

# 按 method+thr 聚合所有指标
agg = defaultdict(lambda: defaultdict(list))
for r in results:
    key = (r["method"], r["thr"])
    for metric in ["dice", "voi_score", "surface_dice", "topo_score", "total_score"]:
        agg[key][metric].append(r[metric])

actual_samples = len(set(r["sample"] for r in results))

print(f"\n{'='*90}")
print(f"Mean scores across {actual_samples} samples (sorted by Total)")
print(f"{'='*90}")
print(f"{'Method':<20} {'Thr':<6} {'Total':>8} {'Dice':>8} {'VOI':>8} {'SurfD':>8} {'Topo':>8}")
print("-" * 70)

sorted_results = []
for (method, thr), metrics in agg.items():
    row = {m: np.mean(v) for m, v in metrics.items()}
    sorted_results.append((method, thr, row))

sorted_results.sort(key=lambda x: -x[2]["total_score"])

for method, thr, row in sorted_results:
    print(f"{method:<20} {thr:<6.2f} {row['total_score']:>8.4f} {row['dice']:>8.4f} "
          f"{row['voi_score']:>8.4f} {row['surface_dice']:>8.4f} {row['topo_score']:>8.4f}")

print(f"\nTop 10:")
for i, (method, thr, row) in enumerate(sorted_results[:10]):
    print(f"  {i+1}. {method} thr={thr:.2f} -> Total={row['total_score']:.4f} "
          f"(Dice={row['dice']:.4f} VOI={row['voi_score']:.4f} "
          f"SurfD={row['surface_dice']:.4f} Topo={row['topo_score']:.4f})")

# 逐样本最佳
print(f"\nPer-sample best (by total_score):")
for sid in val_samples:
    sample_results = [r for r in results if r["sample"] == sid]
    if sample_results:
        best = max(sample_results, key=lambda x: x["total_score"])
        print(f"  {sid}: {best['method']} thr={best['thr']:.2f} "
              f"Total={best['total_score']:.4f} Dice={best['dice']:.4f}")

print("\nAll done!")
