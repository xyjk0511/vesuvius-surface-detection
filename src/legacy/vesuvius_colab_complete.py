# ============================================================
# Vesuvius Challenge - 完整 Colab 训练脚本
# 直接复制到 Colab 运行
# ============================================================

# ==================== Cell 1: 挂载 Google Drive ====================
from google.colab import drive
drive.mount('/content/drive')

# ==================== Cell 2: 安装依赖 ====================
import os
import warnings
os.environ["KERAS_BACKEND"] = "jax"
warnings.filterwarnings('ignore')

import subprocess
from pathlib import Path

# 智能检测环境并安装依赖
print("Installing dependencies...")
if Path("/kaggle/input").exists():
    # Kaggle 离线环境：从预上传的 wheels 安装
    wheels_path = "/kaggle/input/vesuvius-wheels/wheels/"
    if Path(wheels_path).exists():
        subprocess.run([
            "pip", "install", "--no-index", "--no-deps",
            "--find-links", wheels_path,
            "tifffile", "scikit-image"
        ], check=False)
        print("✓ Installed from local wheels (Kaggle offline)")
    else:
        print("⚠ Warning: vesuvius-wheels dataset not found, trying online install...")
        subprocess.run(["pip", "install", "-q", "tifffile", "scikit-image"], check=False)
else:
    # Colab 在线环境：直接从 PyPI 安装
    subprocess.run(["pip", "install", "-q", "tifffile", "scikit-image"], check=False)
    print("✓ Installed from PyPI (Colab online)")

import numpy as np
import pandas as pd
from pathlib import Path
import keras
from keras import layers, Model, ops
from keras import activations as keras_activations

if not hasattr(keras_activations, "get"):
    def _activations_get(identifier):
        if identifier is None:
            return None
        if isinstance(identifier, str):
            return getattr(keras_activations, identifier)
        return identifier
    keras_activations.get = _activations_get

import tifffile as tiff
from PIL import Image
from collections import OrderedDict

devices = keras.distribution.list_devices()
print(f'Devices: {len(devices)}')

keras.config.disable_flash_attention()
keras.utils.set_random_seed(42)

# ==================== Cell 3: 配置 ====================
class CFG:
    # 路径配置 (修改为你的路径)
    BASE_PATH = "/content/drive/MyDrive/vesuvius-challenge-surface-detection"
    OUTPUT_PATH = "/content/drive/MyDrive/vesuvius_output"

    # 模型配置
    K_SLICES = 7
    SLICE_STRIDE = 1
    PATCH = 256            # 优化: 192→256
    BASE_CH = 32
    USE_ATTENTION = True

    # 训练配置 (优化)
    BATCH_SIZE = 4         # 优化: 2→4 (OOM时改为2)
    STAGE1_LR = 1e-4       # 优化: 3e-5→1e-4
    STAGE2_LR = 3e-5
    STAGE1_EPOCHS = 15     # 优化: 8→15
    STAGE2_EPOCHS = 10
    STAGE1_PATIENCE = 5
    STAGE2_PATIENCE = 4
    STEPS_PER_EPOCH = 250  # 优化: 200→250

    # 采样策略 (优化)
    FG_SAMPLE_RATIO = 0.70   # 优化: 0.50→0.70
    NEG_SAMPLE_RATIO = 0.10
    HARD_NEG_RATIO = 0.10
    MIN_FG_PIXELS = 24       # 优化: 16→24
    MIN_FG_RATIO = 0.001
    MAX_UL_RATIO = 0.70
    Z_OFFSETS = [0]

    # 损失函数 (优化)
    POS_WEIGHT = 4.0         # 优化: 3.0→4.0
    DICE_WEIGHT = 1.0
    LABEL_SMOOTHING = 0.01   # 优化: 0.03→0.01
    SKELETON_WEIGHT = 0.5    # 新增

    # 其他
    SEEDS = [42]
    TRAIN_SPLIT = 0.85
    VAL_STEPS = 60

print("✓ Config OK")

# ==================== Cell 4: 数据路径 ====================
BASE = Path(CFG.BASE_PATH)
TRAIN_IMG = BASE / "train_images"
TRAIN_LBL = BASE / "train_labels"

if not TRAIN_IMG.exists():
    TRAIN_IMG = BASE / "deprecated_train_images"
    TRAIN_LBL = BASE / "deprecated_train_labels"

OUTPUT_DIR = Path(CFG.OUTPUT_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_BEST = OUTPUT_DIR / "best.seed42.weights.h5"
CKPT_LAST = OUTPUT_DIR / "last.seed42.weights.h5"

train_csv = pd.read_csv(BASE / "train.csv")
print(f"✓ Train samples: {len(train_csv)}")
print(f"✓ Train images: {TRAIN_IMG}")
print(f"✓ Output: {OUTPUT_DIR}")

# ==================== Cell 5: 数据读取函数 ====================
def read_tif(path):
    path = str(path)
    try:
        return tiff.imread(path)
    except:
        pass
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

def load_sample(sid):
    img_path = TRAIN_IMG / f"{sid}.tif"
    lbl_path = TRAIN_LBL / f"{sid}.tif"

    if not img_path.exists() or not lbl_path.exists():
        return None, None, None

    vol = read_tif(img_path)
    lbl = read_tif(lbl_path)

    if vol is None or lbl is None:
        return None, None, None

    vol = vol.astype(np.float32)
    lbl = lbl.astype(np.uint8)

    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-6), 0, 1)

    fg_flat = np.flatnonzero(lbl == 1)
    return vol, lbl, fg_flat

def make_surface_target(lbl, idx, y, x, patch, z_offsets):
    p = patch
    k = len(idx)
    c = k // 2
    if not z_offsets:
        z_offsets = (0,)
    sel = []
    for dz in z_offsets:
        j = c + dz
        j = max(0, min(j, k - 1))
        sel.append(idx[j])
    sel = np.unique(sel).astype(int)

    win = lbl[sel, y:y + p, x:x + p]
    if win.ndim == 2:
        known = win < 2
        ink = win == 1
    else:
        known = (win < 2).any(axis=0)
        ink = (win == 1).any(axis=0)

    h, w = known.shape
    out = np.full((h, w), 2, dtype=np.uint8)
    out[known] = 0
    out[ink] = 1
    return out

print("✓ Data loading OK")

# ==================== Cell 6: 模型定义 ====================
def conv_block(x, filters, use_norm=True):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    if use_norm:
        x = layers.GroupNormalization(groups=min(8, filters))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    if use_norm:
        x = layers.GroupNormalization(groups=min(8, filters))(x)
    x = layers.Activation('relu')(x)
    return x

def attention_gate(x, g, filters):
    theta_x = layers.Conv2D(filters, 1, padding='same')(x)
    phi_g = layers.Conv2D(filters, 1, padding='same')(g)
    add_xg = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add_xg)
    psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(act)
    return layers.Multiply()([x, psi])

def build_unet_25d(input_shape, base_ch=32, use_attention=True):
    inputs = layers.Input(shape=input_shape)
    x = layers.Permute((2, 3, 1, 4))(inputs)
    x = layers.Lambda(lambda t: ops.squeeze(t, axis=-1))(x)

    c1 = conv_block(x, base_ch)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, base_ch * 2)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, base_ch * 4)
    p3 = layers.MaxPooling2D(2)(c3)
    c4 = conv_block(p3, base_ch * 8)
    p4 = layers.MaxPooling2D(2)(c4)

    bridge = conv_block(p4, base_ch * 16)

    u4 = layers.UpSampling2D(2)(bridge)
    if use_attention:
        c4 = attention_gate(c4, u4, base_ch * 8)
    u4 = layers.Concatenate()([u4, c4])
    d4 = conv_block(u4, base_ch * 8)

    u3 = layers.UpSampling2D(2)(d4)
    if use_attention:
        c3 = attention_gate(c3, u3, base_ch * 4)
    u3 = layers.Concatenate()([u3, c3])
    d3 = conv_block(u3, base_ch * 4)

    u2 = layers.UpSampling2D(2)(d3)
    if use_attention:
        c2 = attention_gate(c2, u2, base_ch * 2)
    u2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, base_ch * 2)

    u1 = layers.UpSampling2D(2)(d2)
    if use_attention:
        c1 = attention_gate(c1, u1, base_ch)
    u1 = layers.Concatenate()([u1, c1])
    d1 = conv_block(u1, base_ch)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d1)
    return Model(inputs, outputs, name='UNet25D')

print("✓ Model definition OK")

# ==================== Cell 7: 损失函数 ====================
class CombinedLoss(keras.losses.Loss):
    def __init__(self, pos_weight=3.0, dice_weight=0.5, alpha=0.75, gamma=2.0,
                 label_smoothing=0.0, skeleton_weight=0.0, name="combined_loss"):
        super().__init__(name=name)
        self.pos_weight = float(pos_weight)
        self.dice_weight = float(dice_weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.skeleton_weight = float(skeleton_weight)

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        yt = ops.squeeze(y_true, axis=-1)
        p = ops.squeeze(y_pred, axis=-1)

        mask = ops.cast(yt < 1.5, "float32")
        target = ops.cast(yt > 0.5, "float32") * mask
        if self.label_smoothing > 0:
            target = (target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing) * mask

        eps = 1e-7
        p = ops.clip(p, eps, 1.0 - eps)

        pt = target * p + (1.0 - target) * (1.0 - p)
        pt = ops.clip(pt, eps, 1.0 - eps)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        focal = -alpha_t * ops.power(1.0 - pt, self.gamma) * ops.log(pt)
        focal = focal * (1.0 + (self.pos_weight - 1.0) * target)
        focal = ops.sum(focal * mask) / (ops.sum(mask) + 1e-6)

        pred_masked = p * mask
        inter = ops.sum(pred_masked * target)
        union = ops.sum(pred_masked) + ops.sum(target)
        dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)

        total_loss = focal + self.dice_weight * dice

        if self.skeleton_weight > 0:
            dy = target[:, 1:, :] - target[:, :-1, :]
            dx = target[:, :, 1:] - target[:, :, :-1]
            grad_y = ops.pad(ops.abs(dy), [[0,0], [0,1], [0,0]])
            grad_x = ops.pad(ops.abs(dx), [[0,0], [0,0], [0,1]])
            skeleton = ops.cast((grad_y + grad_x) > 0.1, "float32")

            pred_masked_skel = p * skeleton * mask
            target_skeleton = target * skeleton
            recall = ops.sum(pred_masked_skel) / (ops.sum(target_skeleton) + 1e-6)
            skeleton_loss = 1.0 - recall

            total_loss = total_loss + self.skeleton_weight * skeleton_loss

        return total_loss

class MaskedDiceMetric(keras.metrics.Metric):
    def __init__(self, name="masked_dice", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.inter = self.add_weight(name="inter", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = ops.cast(y_true[..., :1], "float32")
        mask = ops.cast(yt < 1.5, "float32")
        target = ops.cast(yt > 0.5, "float32") * mask
        pred = ops.cast(y_pred > self.threshold, "float32") * mask
        self.inter.assign_add(ops.sum(pred * target))
        self.union.assign_add(ops.sum(pred) + ops.sum(target))

    def result(self):
        return (2.0 * self.inter + 1.0) / (self.union + 1.0)

    def reset_state(self):
        self.inter.assign(0.0)
        self.union.assign(0.0)

class SoftMaskedDice(keras.metrics.Metric):
    def __init__(self, name="soft_masked_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.inter = self.add_weight(name="inter", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = ops.cast(y_true[..., :1], "float32")
        p = ops.cast(y_pred[..., :1], "float32")
        mask = ops.cast(yt < 1.5, "float32")
        target = ops.cast(yt > 0.5, "float32") * mask
        pred = p * mask
        self.inter.assign_add(ops.sum(pred * target))
        self.union.assign_add(ops.sum(pred) + ops.sum(target))

    def result(self):
        return (2.0 * self.inter + 1.0) / (self.union + 1.0)

    def reset_state(self):
        self.inter.assign(0.0)
        self.union.assign(0.0)

print("✓ Loss OK")

# ==================== Cell 8: 数据生成器 ====================
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

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ids, batch_size, k_slices, patch_size, steps_per_epoch, is_train=True):
        self.ids = [int(x) for x in list(ids)]
        self.batch_size = batch_size
        self.k = k_slices
        self.patch = patch_size
        self.steps = steps_per_epoch
        self.is_train = is_train
        self.cache = LRUCache(max_size=5)
        self._preload()

    def _preload(self):
        loaded = 0
        for sid in self.ids[:3]:
            vol, lbl, fg_flat = load_sample(sid)
            if vol is not None:
                self.cache.put(sid, (vol, lbl, fg_flat))
                loaded += 1
        print(f"  Preloaded {loaded} samples")

    def __len__(self):
        return self.steps

    def _get_sample(self, sid):
        cached = self.cache.get(sid)
        if cached is not None:
            return cached
        vol, lbl, fg_flat = load_sample(sid)
        if vol is not None:
            self.cache.put(sid, (vol, lbl, fg_flat))
        return (vol, lbl, fg_flat) if vol is not None else (None, None, None)

    def _z_indices(self, z0, D):
        idx = z0 + np.arange(self.k) * CFG.SLICE_STRIDE
        return np.clip(idx, 0, D - 1)

    def _random_crop(self, vol, lbl):
        D, H, W = vol.shape
        k, p = self.k, self.patch

        max_z0 = D - (k - 1) * CFG.SLICE_STRIDE
        z0 = np.random.randint(0, max(1, max_z0))
        y = np.random.randint(0, max(1, H - p + 1))
        x = np.random.randint(0, max(1, W - p + 1))

        idx = self._z_indices(z0, D)
        vol_crop = vol[idx, y:y + p, x:x + p]
        lbl_crop = make_surface_target(lbl, idx, y, x, p, CFG.Z_OFFSETS)

        if vol_crop.shape[0] < k:
            vol_crop = np.pad(vol_crop, ((0, k - vol_crop.shape[0]), (0, 0), (0, 0)))
        if vol_crop.shape[1] < p or vol_crop.shape[2] < p:
            pad_h = max(0, p - vol_crop.shape[1])
            pad_w = max(0, p - vol_crop.shape[2])
            vol_crop = np.pad(vol_crop, ((0, 0), (0, pad_h), (0, pad_w)))
            lbl_crop = np.pad(lbl_crop, ((0, pad_h), (0, pad_w)), constant_values=2)

        return vol_crop, lbl_crop

    def _foreground_crop(self, vol, lbl, fg_flat):
        if fg_flat is None or len(fg_flat) == 0:
            return self._random_crop(vol, lbl)

        D, H, W = vol.shape
        k, p = self.k, self.patch

        flat_idx = fg_flat[np.random.randint(len(fg_flat))]
        cz, cy, cx = np.unravel_index(flat_idx, lbl.shape)

        max_z0 = max(1, D - (k - 1) * CFG.SLICE_STRIDE)
        z0 = int(cz - (k // 2) * CFG.SLICE_STRIDE)
        z0 = max(0, min(z0, max_z0 - 1))
        y = max(0, min(cy - p // 2, H - p))
        x = max(0, min(cx - p // 2, W - p))

        idx = self._z_indices(z0, D)
        vol_crop = vol[idx, y:y + p, x:x + p]
        lbl_crop = make_surface_target(lbl, idx, y, x, p, CFG.Z_OFFSETS)

        return vol_crop, lbl_crop

    def _augment(self, vol, lbl):
        if np.random.rand() > 0.5:
            vol = vol[:, ::-1, :].copy()
            lbl = lbl[::-1, :].copy()
        if np.random.rand() > 0.5:
            vol = vol[:, :, ::-1].copy()
            lbl = lbl[:, ::-1].copy()

        k = np.random.randint(4)
        if k > 0:
            vol = np.rot90(vol, k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k).copy()

        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-0.1, 0.1)
            vol = np.clip(alpha * vol + beta, 0, 1)

        return vol, lbl

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        k, p = self.k, self.patch

        if self.is_train:
            attempts = 0
            while len(batch_x) < self.batch_size and attempts < 1500:
                attempts += 1
                sid = int(np.random.choice(self.ids))
                vol, lbl, fg_flat = self._get_sample(sid)
                if vol is None:
                    continue

                if np.random.rand() < CFG.FG_SAMPLE_RATIO:
                    v, l = self._foreground_crop(vol, lbl, fg_flat)
                else:
                    v, l = self._random_crop(vol, lbl)

                v, l = self._augment(v, l)

                fg_pixels = (l == 1).sum()
                ul_ratio = (l == 2).mean()
                fg_ratio = fg_pixels / float(l.size)

                if fg_pixels < CFG.MIN_FG_PIXELS:
                    continue
                if fg_ratio < CFG.MIN_FG_RATIO:
                    continue
                if ul_ratio > CFG.MAX_UL_RATIO:
                    continue

                batch_x.append(v[..., None])
                batch_y.append(l[..., None])
        else:
            for i in range(self.batch_size):
                sid_idx = int((idx * self.batch_size + i) % len(self.ids))
                sid = int(self.ids[sid_idx])
                seed = (idx * 1000003 + sid % 1000003) & 0xFFFFFFFF
                rng = np.random.RandomState(seed)

                vol, lbl, _ = self._get_sample(sid)
                if vol is None:
                    batch_x.append(np.zeros((k, p, p, 1), dtype=np.float32))
                    batch_y.append(np.full((p, p, 1), 2, dtype=np.float32))
                    continue

                D, H, W = vol.shape
                max_z0 = D - (k - 1) * CFG.SLICE_STRIDE
                z0 = rng.randint(0, max(1, max_z0))
                y = rng.randint(0, max(1, H - p + 1))
                x = rng.randint(0, max(1, W - p + 1))

                idx_z = self._z_indices(z0, D)
                v = vol[idx_z, y:y + p, x:x + p]
                l = make_surface_target(lbl, idx_z, y, x, p, CFG.Z_OFFSETS)

                batch_x.append(v[..., None])
                batch_y.append(l[..., None])

        while len(batch_x) < self.batch_size:
            batch_x.append(np.zeros((k, p, p, 1), dtype=np.float32))
            batch_y.append(np.full((p, p, 1), 2, dtype=np.float32))

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

print("✓ Generator OK")

# ==================== Cell 9: 数据分割 ====================
csv_ids = set(int(x) for x in train_csv["id"].unique())
disk_ids = set(int(p.stem) for p in TRAIN_IMG.glob("*.tif"))
valid_ids = np.array(sorted(csv_ids & disk_ids), dtype=np.int64)

print(f"CSV: {len(csv_ids)}, Disk: {len(disk_ids)}, Valid: {len(valid_ids)}")
np.random.shuffle(valid_ids)

split = int(len(valid_ids) * CFG.TRAIN_SPLIT)
tr_ids = valid_ids[:split].tolist()
va_ids = valid_ids[split:].tolist()
print(f"Train: {len(tr_ids)}, Valid: {len(va_ids)}")

train_gen = DataGenerator(tr_ids, CFG.BATCH_SIZE, CFG.K_SLICES, CFG.PATCH, CFG.STEPS_PER_EPOCH, is_train=True)
valid_gen = DataGenerator(va_ids, CFG.BATCH_SIZE, CFG.K_SLICES, CFG.PATCH, CFG.VAL_STEPS, is_train=False)

x, y = train_gen[0]
print(f"✓ Batch shape: x={x.shape}, y={y.shape}")
print(f"✓ y unique: {np.unique(y)}, foreground: {(y==1).sum()}")

# ==================== Cell 10: 学习率调度器 ====================
class WarmupCosineDecay(keras.callbacks.Callback):
    def __init__(self, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + ops.cos(np.pi * progress))

        self.model.optimizer.learning_rate.assign(float(lr))
        print(f"Epoch {epoch+1}: LR = {float(lr):.6f}")

print("✓ LR Scheduler OK")

# ==================== Cell 11: 构建和编译模型 ====================
model = build_unet_25d(
    input_shape=(CFG.K_SLICES, None, None, 1),
    base_ch=CFG.BASE_CH,
    use_attention=CFG.USE_ATTENTION
)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=CFG.STAGE1_LR, weight_decay=1e-4, clipnorm=1.0),
    loss=CombinedLoss(
        pos_weight=CFG.POS_WEIGHT,
        dice_weight=CFG.DICE_WEIGHT,
        label_smoothing=CFG.LABEL_SMOOTHING,
        skeleton_weight=CFG.SKELETON_WEIGHT,
    ),
    metrics=[MaskedDiceMetric(threshold=0.5), SoftMaskedDice()]
)

print(f"✓ Model params: {model.count_params():,}")

# ==================== Cell 12: 训练 ====================
callbacks = [
    WarmupCosineDecay(
        warmup_epochs=3,
        total_epochs=CFG.STAGE1_EPOCHS,
        base_lr=CFG.STAGE1_LR
    ),
    keras.callbacks.ModelCheckpoint(
        str(CKPT_BEST),
        monitor="val_soft_masked_dice",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        str(CKPT_LAST),
        save_best_only=False,
        save_weights_only=True,
        save_freq="epoch",
        verbose=0
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_soft_masked_dice",
        mode="max",
        patience=CFG.STAGE1_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
]

print("\n" + "="*60)
print("开始训练...")
print("="*60)
print(f"配置: PATCH={CFG.PATCH}, BATCH={CFG.BATCH_SIZE}, LR={CFG.STAGE1_LR}")
print(f"优化: FG_RATIO={CFG.FG_SAMPLE_RATIO}, SKELETON_WEIGHT={CFG.SKELETON_WEIGHT}")
print(f"预期: 验证Dice 0.2646 → 0.35-0.40")
print("="*60 + "\n")

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=CFG.STAGE1_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print("训练完成！")
print("="*60)
print(f"✓ 最佳权重: {CKPT_BEST}")
print(f"✓ 最终验证 Dice: {history.history['val_soft_masked_dice'][-1]:.4f}")
print(f"✓ 最佳验证 Dice: {max(history.history['val_soft_masked_dice']):.4f}")
print("="*60)
