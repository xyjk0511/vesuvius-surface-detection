# ==================== 继续训练脚本 ====================
# 从 checkpoint 继续训练 2540样本 ResEnc L Fold 0
# 配置: Patch 192, T4x2 DDP, 1200 epochs

# ===== CELL 1: 安装 (单独运行) =====
# !pip install -q nnunetv2 nibabel

# ===== CELL 2: 环境配置 =====
import os
import json
import shutil
from pathlib import Path

os.environ["nnUNet_raw"] = "/kaggle/working/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/kaggle/working/nnUNet_results"
os.environ["nnUNet_n_proc_DA"] = "2"

for p in [os.environ["nnUNet_raw"], os.environ["nnUNet_results"]]:
    os.makedirs(p, exist_ok=True)

print("Environment setup done!")

# ===== CELL 3: 自定义 Trainer =====
TRAINER_CODE = '''
import os
import gc
import shutil
import torch
from pathlib import Path
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerResEncL(nnUNetTrainer):
    """ResEnc L 1200 epochs + auto save + memory cleanup"""
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1200
        self.best_dice = -1.0
        self.best_epoch = 0
        self.save_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else None
        self.save_every = 5  # 基类默认50，改成5让latest每5轮更新

    def on_train_start(self):
        super().on_train_start()
        # 从checkpoint恢复best_dice，避免续训时误报NEW BEST
        log = self.logger.my_fantastic_logging
        if "ema_fg_dice" in log and len(log["ema_fg_dice"]) > 0:
            self.best_dice = max(log["ema_fg_dice"])
            self.best_epoch = log["ema_fg_dice"].index(self.best_dice)
            if self._is_main_process():
                print(f">>> Restored best_dice={self.best_dice:.4f} from epoch {self.best_epoch}")

    def _is_main_process(self):
        if not self.is_ddp:
            return True
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
        return self.local_rank == 0

    def _save_one_checkpoint(self, src_name):
        if self.save_dir is None or not self._is_main_process():
            return
        src = Path(self.output_folder) / src_name
        if not src.exists():
            return
        dst = f"{self.save_dir}/fold{self.fold}_{src_name}"
        shutil.copy(src, dst)
        print(f">>> SAVED: {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")

    def on_epoch_end(self):
        ep = self.current_epoch  # 在super()之前取，避免+1偏移
        super().on_epoch_end()

        # 基类在 (ep+1) % save_every == 0 时保存latest，即ep=4,9,14...
        if (ep + 1) % self.save_every == 0 and ep != self.num_epochs - 1:
            torch.cuda.empty_cache()
            gc.collect()
            self._save_one_checkpoint("checkpoint_latest.pth")
            if self._is_main_process():
                print(f">>> Epoch {ep}: memory cleaned, latest saved")

        log = self.logger.my_fantastic_logging
        if "ema_fg_dice" not in log or len(log["ema_fg_dice"]) == 0:
            return
        dice = log["ema_fg_dice"][-1]

        if dice > self.best_dice:
            self.best_dice = dice
            self.best_epoch = ep
            if self._is_main_process():
                print(f"\\n{'='*60}\\nNEW BEST! Epoch {ep}: Dice = {dice:.4f}\\n{'='*60}\\n")
            # 只在new best时复制best
            self._save_one_checkpoint("checkpoint_best.pth")

        if self._is_main_process():
            print(f">>> Best={self.best_dice:.4f}@Ep{self.best_epoch}, Current Ep{ep}")

    def on_train_end(self):
        super().on_train_end()
        # final只在训练结束时生成，latest和best已在on_epoch_end处理
        self._save_one_checkpoint("checkpoint_final.pth")
        if self._is_main_process():
            print(f">>> Training finished. Final checkpoint saved to {self.save_dir}")
'''

import nnunetv2
trainer_dir = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer"
trainer_file = trainer_dir / "nnUNetTrainerResEncL.py"
with open(trainer_file, 'w') as f:
    f.write(TRAINER_CODE)
print(f"Trainer installed: {trainer_file}")

# ===== CELL 4: 合并数据集 =====
PART1 = Path("/kaggle/input/2540-part1/kaggle_upload_part1")
PART2 = Path("/kaggle/input/2540-part2/kaggle_upload_part2")
DST = Path("/kaggle/working/nnUNet_preprocessed/Dataset101_Vesuvius_merged")

DST.mkdir(parents=True, exist_ok=True)
dst_data = DST / "nnUNetPlans_3d_fullres"
dst_data.mkdir(exist_ok=True)

for cfg in ["dataset.json", "dataset_fingerprint.json", "nnUNetResEncUNetLPlans.json"]:
    src = PART1 / cfg
    if src.exists() and not (DST / cfg).exists():
        shutil.copy2(src, DST / cfg)
        print(f"Copied {cfg}")

link_count = 0
for part in [PART1, PART2]:
    src_dir = part / "nnUNetPlans_3d_fullres"
    if src_dir.exists():
        for f in src_dir.iterdir():
            dst_f = dst_data / f.name
            if not dst_f.exists():
                try:
                    os.symlink(str(f), str(dst_f))
                    link_count += 1
                except OSError:
                    pass

count = len(list(dst_data.glob("*.b2nd")))
print(f"Merged data files: {count}")

# ===== CELL 5: 复制 Checkpoint =====
# !!! 修改这里 !!!
CKPT_DATASET = "2540-4"  # checkpoint 数据集
FOLD = 0

CKPT_SRC = Path(f"/kaggle/input/{CKPT_DATASET}")
CKPT_DST = Path(f"/kaggle/working/nnUNet_results/Dataset101_Vesuvius_merged/nnUNetTrainerResEncL__nnUNetResEncUNetLPlans__3d_fullres/fold_{FOLD}")

CKPT_DST.mkdir(parents=True, exist_ok=True)

if CKPT_SRC.exists():
    for f in CKPT_SRC.glob("*"):
        name = f.name
        # 跳过带fold前缀的旧文件，避免覆盖同名新文件
        if name.startswith(f"fold{FOLD}_"):
            print(f"Skipped old: {name}")
            continue
        dst = CKPT_DST / name
        shutil.copy2(f, dst)
        print(f"Copied: {name}")
else:
    print(f"WARNING: Checkpoint not found: {CKPT_SRC}")
    print("Available datasets:")
    for d in Path("/kaggle/input").iterdir():
        print(f"  {d.name}")

# ===== CELL 6: 选择从哪个checkpoint继续 =====
# "latest" = 训练到的最远位置, "best" = 最佳Dice的位置
USE_CHECKPOINT = "best"  # "latest" 或 "best"

src_ckpt = CKPT_DST / f"checkpoint_{USE_CHECKPOINT}.pth"
final_ckpt = CKPT_DST / "checkpoint_final.pth"
if not src_ckpt.exists():
    raise FileNotFoundError(f"checkpoint_{USE_CHECKPOINT}.pth not found in {CKPT_DST}")
if src_ckpt != final_ckpt:
    if final_ckpt.exists():
        shutil.move(final_ckpt, CKPT_DST / "checkpoint_final_backup.pth")
    shutil.copy(src_ckpt, final_ckpt)
    print(f"Using checkpoint_{USE_CHECKPOINT}.pth as checkpoint_final.pth")

# ===== CELL 7: 检查状态 =====
import torch
print(f"\nGPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

ckpt_files = list(CKPT_DST.glob("*.pth"))
print(f"\nCheckpoint files: {len(ckpt_files)}")
for f in ckpt_files:
    print(f"  {f.name}")

print("\nReady to continue training!")

# ===== CELL 8: 继续训练 (单独运行) =====
# 双GPU DDP
# !nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainerResEncL -p nnUNetResEncUNetLPlans -num_gpus 2 --c

# 单GPU (备用)
# !nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainerResEncL -p nnUNetResEncUNetLPlans --c
