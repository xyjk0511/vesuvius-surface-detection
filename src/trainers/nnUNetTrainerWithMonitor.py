# ==================== 带早停监控 + 自动保存的Trainer ====================
# 显示距离最高分多少epoch，方便判断是否该停止
# NEW BEST 时自动保存到 /kaggle/working/ 或 /content/drive/
# 保存为: nnunetv2/training/nnUNetTrainer/nnUNetTrainerWithMonitor.py

import os
import shutil
from pathlib import Path
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerWithMonitor(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.best_dice = -1.0
        self.best_epoch = 0
        self.epochs_since_best = 0

        # 自动检测保存目录
        if os.path.exists("/kaggle/working"):
            self.save_dir = "/kaggle/working"
        elif os.path.exists("/content/drive/MyDrive"):
            self.save_dir = f"/content/drive/MyDrive/vesuvius_nnunet/fold{fold}"
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

    def _save_checkpoint_copy(self):
        """复制 checkpoint 到保存目录"""
        if self.save_dir is None:
            return
        fold = self.fold
        ckpt_dir = Path(self.output_folder)
        for f in ckpt_dir.glob("*.pth"):
            dst = f"{self.save_dir}/fold{fold}_{f.name}"
            shutil.copy(f, dst)
            size_mb = os.path.getsize(dst) / 1e6
            print(f">>> SAVED: fold{fold}_{f.name} ({size_mb:.1f} MB)")

    def on_epoch_end(self):
        super().on_epoch_end()

        current_dice = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
        current_epoch = self.current_epoch

        if current_dice > self.best_dice:
            self.best_dice = current_dice
            self.best_epoch = current_epoch
            self.epochs_since_best = 0
            print(f"\n{'='*60}")
            print(f"NEW BEST! Epoch {current_epoch}: Dice = {current_dice:.4f}")
            print(f"{'='*60}\n")
            # 自动保存
            self._save_checkpoint_copy()
        else:
            self.epochs_since_best = current_epoch - self.best_epoch

        print(f"\n>>> MONITOR: Best Dice = {self.best_dice:.4f} @ Epoch {self.best_epoch}")
        print(f">>> MONITOR: Current Epoch {current_epoch}, {self.epochs_since_best} epochs since best")

        if self.epochs_since_best >= 50:
            print(f">>> WARNING: No improvement for {self.epochs_since_best} epochs!")
        print()

# 使用方法:
# nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerWithMonitor
