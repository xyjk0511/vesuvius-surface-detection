# Vesuvius Challenge - SMP 预训练编码器版本
# 使用 segmentation_models_pytorch 的 ResNet34 预训练编码器

import os
import warnings
warnings.filterwarnings('ignore')

import subprocess
subprocess.run(["pip", "install", "-q", "segmentation-models-pytorch", "imagecodecs"], check=False)

import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# ==================== 配置 ====================
class CFG:
    # 路径 (Colab)
    BASE_PATH = "/content/drive/MyDrive/vesuvius-challenge-surface-detection"
    OUTPUT_PATH = "/content/drive/MyDrive/vesuvius_output"

    # 模型 - EfficientNet-B4 (更强的编码器)
    ENCODER = "efficientnet-b4"
    ENCODER_WEIGHTS = "imagenet"
    K_SLICES = 7

    # 训练 (L4 优化，EfficientNet-B4 需要更小 batch)
    PATCH = 256
    BATCH_SIZE = 2          # EfficientNet-B4 更大，减小 batch
    LR = 5e-5               # 更小的学习率
    EPOCHS = 30
    PATIENCE = 8

    # 采样
    FG_SAMPLE_RATIO = 0.70  # 提高前景采样比例
    MIN_FG_PIXELS = 16
    MAX_UL_RATIO = 0.70

    # 损失
    POS_WEIGHT = 3.0
    FOCAL_GAMMA = 2.0       # Focal Loss gamma

    # 其他
    TRAIN_SPLIT = 0.85
    STEPS_PER_EPOCH = 300   # 增加每epoch步数
    VAL_STEPS = 60
    SEED = 42

    # 继续训练 (新模型从头开始)
    RESUME = False
    START_EPOCH = 0
    BEST_DICE = 0.0

# ==================== 设备 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)

# ==================== 数据路径 ====================
BASE = Path(CFG.BASE_PATH)
TRAIN_IMG = BASE / "train_images"
TRAIN_LBL = BASE / "train_labels"

if not TRAIN_IMG.exists():
    TRAIN_IMG = BASE / "deprecated_train_images"
    TRAIN_LBL = BASE / "deprecated_train_labels"

OUTPUT_DIR = Path(CFG.OUTPUT_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = OUTPUT_DIR / "smp_best.pt"

print(f"Train images: {TRAIN_IMG}")
print(f"Output: {OUTPUT_DIR}")

# ==================== 数据读取 ====================
from PIL import Image

def read_tif(path):
    """读取 TIF，支持 LZW 压缩"""
    path = str(path)
    # 1) tifffile (需要 imagecodecs)
    try:
        import tifffile as tiff
        return tiff.imread(path)
    except:
        pass
    # 2) PIL fallback
    try:
        im = Image.open(path)
        frames = []
        i = 0
        while True:
            try:
                im.seek(i)
                frames.append(np.array(im))
                i += 1
            except EOFError:
                break
        return np.stack(frames, axis=0) if len(frames) > 1 else frames[0]
    except:
        return None

class LRUCache:
    def __init__(self, max_size=5):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value

cache = LRUCache(max_size=5)

def load_sample(sid):
    cached = cache.get(sid)
    if cached is not None:
        return cached

    img_path = TRAIN_IMG / f"{sid}.tif"
    lbl_path = TRAIN_LBL / f"{sid}.tif"

    if not img_path.exists() or not lbl_path.exists():
        return None, None, None

    vol = read_tif(img_path).astype(np.float32)
    lbl = read_tif(lbl_path).astype(np.uint8)

    # 百分位归一化
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-6), 0, 1)

    fg_flat = np.flatnonzero(lbl == 1)

    cache.put(sid, (vol, lbl, fg_flat))
    return vol, lbl, fg_flat

# ==================== Dataset ====================
class VesuviusDataset(Dataset):
    def __init__(self, ids, is_train=True):
        self.ids = [int(x) for x in ids]
        self.is_train = is_train
        self.k = CFG.K_SLICES
        self.p = CFG.PATCH
        self.steps = CFG.STEPS_PER_EPOCH if is_train else CFG.VAL_STEPS

    def __len__(self):
        return self.steps * CFG.BATCH_SIZE

    def _z_indices(self, z0, D):
        idx = z0 + np.arange(self.k)
        return np.clip(idx, 0, D - 1)

    def _random_crop(self, vol, lbl):
        D, H, W = vol.shape
        z0 = np.random.randint(0, max(1, D - self.k + 1))
        y = np.random.randint(0, max(1, H - self.p + 1))
        x = np.random.randint(0, max(1, W - self.p + 1))

        idx = self._z_indices(z0, D)
        vol_crop = vol[idx, y:y+self.p, x:x+self.p]

        # 中心 slice 的 label
        c = self.k // 2
        lbl_crop = lbl[idx[c], y:y+self.p, x:x+self.p]

        return vol_crop, lbl_crop

    def _foreground_crop(self, vol, lbl, fg_flat):
        if fg_flat is None or len(fg_flat) == 0:
            return self._random_crop(vol, lbl)

        D, H, W = vol.shape
        flat_idx = fg_flat[np.random.randint(len(fg_flat))]
        cz, cy, cx = np.unravel_index(flat_idx, lbl.shape)

        z0 = max(0, min(cz - self.k // 2, D - self.k))
        y = max(0, min(cy - self.p // 2, H - self.p))
        x = max(0, min(cx - self.p // 2, W - self.p))

        idx = self._z_indices(z0, D)
        vol_crop = vol[idx, y:y+self.p, x:x+self.p]

        c = self.k // 2
        lbl_crop = lbl[idx[c], y:y+self.p, x:x+self.p]

        return vol_crop, lbl_crop

    def _augment(self, vol, lbl):
        # 翻转
        if np.random.rand() > 0.5:
            vol = vol[:, ::-1, :].copy()
            lbl = lbl[::-1, :].copy()
        if np.random.rand() > 0.5:
            vol = vol[:, :, ::-1].copy()
            lbl = lbl[:, ::-1].copy()

        # 旋转
        k = np.random.randint(4)
        if k > 0:
            vol = np.rot90(vol, k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k).copy()

        # 亮度/对比度
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-0.1, 0.1)
            vol = np.clip(alpha * vol + beta, 0, 1)

        return vol, lbl

    def __getitem__(self, idx):
        for _ in range(100):
            sid = int(np.random.choice(self.ids))
            vol, lbl, fg_flat = load_sample(sid)
            if vol is None:
                continue

            if self.is_train and np.random.rand() < CFG.FG_SAMPLE_RATIO:
                v, l = self._foreground_crop(vol, lbl, fg_flat)
            else:
                v, l = self._random_crop(vol, lbl)

            if self.is_train:
                v, l = self._augment(v, l)

            # 质量检查
            fg_pixels = (l == 1).sum()
            ul_ratio = (l == 2).mean()

            if fg_pixels < CFG.MIN_FG_PIXELS:
                continue
            if ul_ratio > CFG.MAX_UL_RATIO:
                continue

            # 转换为 tensor
            # vol: (K, H, W) -> (K, H, W) 作为通道
            x = torch.from_numpy(v.astype(np.float32))

            # label: 忽略 unlabeled (2)
            mask = (l < 2).astype(np.float32)
            target = (l == 1).astype(np.float32)

            y = torch.from_numpy(target)
            m = torch.from_numpy(mask)

            return x, y, m

        # fallback
        x = torch.zeros(self.k, self.p, self.p)
        y = torch.zeros(self.p, self.p)
        m = torch.zeros(self.p, self.p)
        return x, y, m

# ==================== 模型 ====================
class SMPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=CFG.ENCODER,
            encoder_weights=CFG.ENCODER_WEIGHTS,
            in_channels=CFG.K_SLICES,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        # x: (B, K, H, W)
        return self.model(x).squeeze(1)  # (B, H, W)

# ==================== 损失函数 ====================
class FocalDiceLoss(nn.Module):
    """Focal Loss + Dice Loss 组合"""
    def __init__(self, pos_weight=3.0, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, pred, target, mask):
        # Focal Loss
        pred_sig = torch.sigmoid(pred)
        bce = -target * torch.log(pred_sig + 1e-6) - (1 - target) * torch.log(1 - pred_sig + 1e-6)

        # Focal weight
        pt = target * pred_sig + (1 - target) * (1 - pred_sig)
        focal_weight = (1 - pt) ** self.gamma

        # Pos weight
        weight = 1.0 + (self.pos_weight - 1.0) * target
        focal = (focal_weight * bce * weight * mask).sum() / (mask.sum() + 1e-6)

        # Dice Loss
        pred_masked = pred_sig * mask
        target_masked = target * mask
        inter = (pred_masked * target_masked).sum()
        union = pred_masked.sum() + target_masked.sum()
        dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)

        return focal + dice

def compute_dice(pred, target, mask, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float() * mask
    target_masked = target * mask
    inter = (pred_bin * target_masked).sum()
    union = pred_bin.sum() + target_masked.sum()
    return (2.0 * inter + 1.0) / (union + 1.0)

# ==================== 训练 ====================
def train():
    # 数据分割
    train_csv = pd.read_csv(BASE / "train.csv")
    csv_ids = set(int(x) for x in train_csv["id"].unique())
    disk_ids = set(int(p.stem) for p in TRAIN_IMG.glob("*.tif"))
    valid_ids = np.array(sorted(csv_ids & disk_ids))

    print(f"Valid samples: {len(valid_ids)}")
    np.random.shuffle(valid_ids)

    split = int(len(valid_ids) * CFG.TRAIN_SPLIT)
    tr_ids = valid_ids[:split]
    va_ids = valid_ids[split:]
    print(f"Train: {len(tr_ids)}, Valid: {len(va_ids)}")

    # DataLoader
    train_ds = VesuviusDataset(tr_ids, is_train=True)
    valid_ds = VesuviusDataset(va_ids, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)

    # 测试数据加载
    print("Testing data loading...")
    x, y, m = train_ds[0]
    print(f"Sample shape: x={x.shape}, y={y.shape}, m={m.shape}")
    print("Data loading OK!")

    # 模型
    model = SMPModel().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 加载 checkpoint（如果继续训练）
    print(f"\nCKPT_PATH: {CKPT_PATH}")
    print(f"CKPT exists: {CKPT_PATH.exists()}")
    print(f"RESUME: {CFG.RESUME}")

    if CFG.RESUME:
        if CKPT_PATH.exists():
            model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
            print(f"Loaded checkpoint: {CKPT_PATH}")
            print(f"Resuming from epoch {CFG.START_EPOCH}, best Dice: {CFG.BEST_DICE}")
        else:
            print("WARNING: RESUME=True but checkpoint not found! Training from scratch.")
            CFG.RESUME = False  # 重置为从头训练

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)

    # 快进学习率调度器到正确位置
    if CFG.RESUME:
        for _ in range(CFG.START_EPOCH):
            scheduler.step()

    criterion = FocalDiceLoss(pos_weight=CFG.POS_WEIGHT, gamma=CFG.FOCAL_GAMMA)

    # 训练循环
    best_dice = CFG.BEST_DICE if CFG.RESUME else 0.0
    patience_counter = 0
    start_epoch = CFG.START_EPOCH if CFG.RESUME else 0

    for epoch in range(start_epoch, CFG.EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS}")
        for i, (x, y, m) in enumerate(train_loader):
            if i >= CFG.STEPS_PER_EPOCH:
                break

            x, y, m = x.to(device), y.to(device), m.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y, m)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += compute_dice(pred, y, m).item()

            if (i + 1) % 50 == 0:
                print(f"  Step {i+1}/{CFG.STEPS_PER_EPOCH} | Loss: {train_loss/(i+1):.4f} | Dice: {train_dice/(i+1):.4f}")

        train_loss /= CFG.STEPS_PER_EPOCH
        train_dice /= CFG.STEPS_PER_EPOCH

        # Validate
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for i, (x, y, m) in enumerate(valid_loader):
                if i >= CFG.VAL_STEPS:
                    break

                x, y, m = x.to(device), y.to(device), m.to(device)
                pred = model(x)

                val_loss += criterion(pred, y, m).item()
                val_dice += compute_dice(pred, y, m).item()

        val_loss /= CFG.VAL_STEPS
        val_dice /= CFG.VAL_STEPS

        scheduler.step()

        # Log
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{CFG.EPOCHS} | LR: {lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f}")

        # Checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  -> Saved best model (Dice: {best_dice:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete! Best Val Dice: {best_dice:.4f}")
    print(f"Checkpoint: {CKPT_PATH}")

if __name__ == "__main__":
    train()
