# ==================== ResEnc L 提交脚本 v2 ====================
# 模型: ResEnc L (2540样本, Epoch 219, Dice 0.5710)
# 后处理: dust_removal(500) (thr=0.25)
# 基于 STANDARD 模式: 批量处理，推理完成后再清理

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
    # 跳过带fold0_前缀的旧文件
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

# ===== CELL 4: 初始化推理器 =====
import numpy as np
import nibabel as nib
import tifffile
import pandas as pd
from tqdm import tqdm
from scipy import ndimage

test_csv = "/kaggle/input/vesuvius-challenge-surface-detection/test.csv"
test_df = pd.read_csv(test_csv)
print(f"Test samples: {len(test_df)}")

INPUT_DIR = "/kaggle/working/nnUNet_raw/Dataset101_Vesuvius/imagesTs"
OUTPUT_DIR = "/kaggle/working/nnUNet_output"
PRED_DIR = "/kaggle/working/preds_tif"
for d in [INPUT_DIR, OUTPUT_DIR, PRED_DIR]:
    os.makedirs(d, exist_ok=True)

THRESHOLD = 0.25
DUST_MIN_SIZE = 500

def read_tif_volume(tif_path):
    try:
        vol = tifffile.imread(tif_path)
    except:
        import cv2
        ret, frames = cv2.imreadmulti(str(tif_path), flags=cv2.IMREAD_UNCHANGED)
        vol = np.array(frames)
    return vol

def normalize_volume(vol):
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-8)
    return vol.astype(np.float32)

def postprocess_prediction(pred, min_size=500):
    labeled, num_features = ndimage.label(pred)
    if num_features > 0:
        sizes = ndimage.sum(pred, labeled, range(1, num_features + 1))
        for i, size in enumerate(sizes):
            if size < min_size:
                pred[labeled == (i + 1)] = 0
    return pred

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    device=torch.device('cuda'),
    verbose=True,
    verbose_preprocessing=True
)

predictor.initialize_from_trained_model_folder(
    MODEL_DIR,
    use_folds=(0,),
    checkpoint_name='checkpoint_best.pth'
)

# 删除pth权重文件释放磁盘（模型已加载到GPU显存）
# 注意：不能删整个nnUNet_results，predictor推理时仍需要plans.json等配置
for pth_file in Path(FOLD_DIR).glob("*.pth"):
    os.remove(pth_file)
    print(f"Removed weight file: {pth_file.name}")
print("Predictor initialized, weight files cleaned from disk")

# ===== CELL 5: 逐样本推理 + 直接写入zip (最小磁盘占用) =====
import gc
import zipfile
import io

TMP_TIF = "/kaggle/working/tmp_pred.tif"
count = 0

print(f"Processing {len(test_df)} samples, writing directly to zip...")
with zipfile.ZipFile("/kaggle/working/submission.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        sample_id = row['id']

        # 1) 转换TIF -> NIfTI
        tif_path = Path(f"/kaggle/input/vesuvius-challenge-surface-detection/test_images/{sample_id}.tif")
        if not tif_path.exists():
            tif_path = Path(f"/kaggle/input/vesuvius-challenge-surface-detection/deprecated_test_images/{sample_id}.tif")

        vol = read_tif_volume(tif_path)
        vol = normalize_volume(vol)
        vol = np.transpose(vol, (2, 1, 0))
        nii_path = f"{INPUT_DIR}/{sample_id}_0000.nii.gz"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), nii_path)
        del vol
        gc.collect()

        # 2) 推理单个样本
        predictor.predict_from_files(
            [[nii_path]],
            OUTPUT_DIR,
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1
        )

        # 3) 删除输入NIfTI
        os.remove(nii_path)

        # 4) 后处理: npz -> tif -> 直接写入zip
        npz_path = Path(OUTPUT_DIR) / f"{sample_id}_0000.npz"
        if not npz_path.exists():
            npz_path = Path(OUTPUT_DIR) / f"{sample_id}.npz"

        probs = np.load(str(npz_path))['probabilities']
        prob_fg = probs[1]
        del probs
        pred = (prob_fg >= THRESHOLD).astype(np.uint8)
        del prob_fg
        pred = postprocess_prediction(pred, min_size=DUST_MIN_SIZE)

        # 写临时tif再追加到zip，然后立即删除
        tifffile.imwrite(TMP_TIF, pred, compression='lzw')
        del pred
        zf.write(TMP_TIF, f"{sample_id}.tif")
        os.remove(TMP_TIF)

        # 5) 删除npz和nii.gz输出
        for f in Path(OUTPUT_DIR).glob(f"{sample_id}*"):
            os.remove(f)

        gc.collect()
        count += 1
        print(f"  [{count}/{len(test_df)}] {sample_id} done")

print(f"submission.zip created with {count} predictions!")

# 最终清理
shutil.rmtree(PRED_DIR, ignore_errors=True)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
shutil.rmtree("/kaggle/working/nnUNet_raw", ignore_errors=True)
shutil.rmtree("/kaggle/working/nnUNet_preprocessed", ignore_errors=True)
shutil.rmtree("/kaggle/working/nnUNet_results", ignore_errors=True)
print("All cleanup done!")
