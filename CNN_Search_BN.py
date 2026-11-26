"""
CNN thuần (conv 3x3 + [BN] + ReLU + MaxPool + GAP + FC): VÉT CẠN kiến trúc -> FP32 -> QAT -> INT8,
MỘT kiến trúc dùng chung cho HAI dataset (Hospital_A và Hospital_B).

BẢN SỬA so với bản trước (giữ nguyên conv 3x3 và GAP, golden/RTL không đổi):
  [1] BatchNorm sau mỗi conv (USE_BN). BN được fold vào weight/bias lúc convert => .mem / golden / RTL không đổi.
      QAT dùng fuse_modules_qat (Conv+BN+ReLU) và đóng băng BN cùng lúc với observer.
  [3] Recipe cho mạng nhỏ: dropout 0, AdamW (không decay cho bias/BN, wd nhỏ), warmup + cosine thay cho
      ReduceLROnPlateau, augmentation nhẹ (flip + xoay nhẹ), mặc định tắt blur/jitter/scale.
  [5] Thu hẹp không gian kiến trúc: groups = 1, chỉ dãy kênh KHÔNG GIẢM, depth >= 4, lớp cuối >= 48 kênh.

Bài toán (2 CNN nối tiếp trên FPGA):
  CNN 1 (Hospital_A: 4 bệnh + Unknown): ra bệnh -> xuất kết quả luôn; ra Unknown -> chuyển sang CNN 2.
  CNN 2 (Hospital_B: 4 bệnh + Unknown): ra bệnh -> xuất kết quả; ra Unknown -> kết quả cuối là Unknown.
  Hai CNN PHẢI cùng kiến trúc (cùng số lớp, kênh, groups, pool), chỉ khác trọng số và scale/zero-point.

Quy trình:
  1. Đọc 2 dataset đã chia sẵn 70/15/15 trong HOSPITAL_ROOT/{Hospital_A,Hospital_B}/{train,val,test}.
  2. Liệt kê MỌI kiến trúc thỏa bộ lọc (kênh, groups, depth, không giảm, lớp cuối) và tham số trong
     [MIN_PARAMS, MAX_PARAMS). Kiến trúc ngoài khoảng KHÔNG được train.
  3. Mỗi (kiến trúc, seed): train CÙNG kiến trúc trên từng dataset.
       giai đoạn 1: FP32 cho cả 2 dataset (cổng FP32 đo trên VAL; 1 dataset trượt thì bỏ cả kiến trúc)
       giai đoạn 2: QAT -> INT8 cho cả 2 dataset, đánh giá VAL/TEST bằng fbgemm + golden số nguyên.
  4. Xếp hạng theo VAL (golden) của CẢ HAI dataset. Mỗi khi có kiến trúc tốt nhất mới, xuất NGAY ra
     BEST/{Hospital_A,Hospital_B}/ + BEST/Cascade/.
  5. Chạy lại được (resume): kết quả lưu ở seeds_summary.json, (kiến trúc, seed) đã xong sẽ bị bỏ qua.
     LƯU Ý: nếu đổi recipe / USE_BN thì XÓA seeds_summary.json và BEST/ cũ, kẻo trộn kết quả hai recipe.
"""
import os, sys, random, gc, warnings, copy, json, csv, time, shutil, traceback, itertools
from functools import partial
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from tqdm import tqdm as _tqdm
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.qat as nniqat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHOW_PROGRESS = False               # False: tắt thanh tiến trình tqdm (log của vét cạn không bị rác)
tqdm = partial(_tqdm, disable=not SHOW_PROGRESS)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# CẤU HÌNH
# ============================================================================
TRAIN_SEEDS = [44]                   # vòng vét cạn: 1 seed/kiến trúc. Vòng chốt: list(range(42, 52)) + ARCH_FILTER

# ── Dataset ──
HOSPITAL_ROOT = "/home/khaithien/Work/6.DA/AAZ1/Dataset_Hospital"   # chứa Hospital_A/ và Hospital_B/ (ảnh 512x512)
HOSPITALS     = ("Hospital_A", "Hospital_B")   # THỨ TỰ = thứ tự trên FPGA: CNN 1 (A) -> nếu Unknown -> CNN 2 (B)
UNKNOWN_NAME  = "Unknown"

BASE_DIR    = os.path.dirname(HOSPITAL_ROOT)
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs_arch_search_AB_BN")     # thư mục MỚI để không lẫn kết quả bản cũ
FIG_DIR     = os.path.join(OUTPUT_DIR, "figures")
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")

MODEL_NAME = "CNN_SEARCH_AB"
MODEL_DIR  = os.path.join(OUTPUT_DIR, MODEL_NAME)

# ── Vét cạn kiến trúc [5] ──
MAX_PARAMS, MIN_PARAMS = 35_000, 30_000   # tham số SAU KHI fold BN (weight + 1 bias/kênh)
CHANNEL_CHOICES = (16, 24, 32, 48, 64)
GROUP_CHOICES   = (1,)          # [5] groups=1: trộn kênh đầy đủ. Mã vẫn hỗ trợ (1, 4) nếu muốn thử lại
DEPTH_RANGE     = (4, 5)        # [5] depth >= 4 để receptive field đủ rộng với ảnh 64 px
ONLY_NON_DECREASING = True      # [5] chỉ dãy kênh không giảm (bỏ dãy giảm / xen kẽ: nút cổ chai)
MIN_LAST_CHANNELS   = 48        # [5] lớp conv cuối >= 48 kênh -> vector GAP đủ chiều cho FC
POOL_AFTER      = (0, 1, 2)     # MaxPool sau các lớp conv có chỉ số này (feature map cuối 8x8 khi IMG_SIZE=64)
ARCH_RNG_SEED   = 2026
MAX_ARCH        = None
ARCH_FILTER     = None

# ── Dữ liệu / tiền xử lý ──
IMG_SIZE      = 64
EXPECTED_SIZE = (512, 512)
ENHANCE_MODE  = "none"               # "hist_eq" | "stretch" | "none"  (nên thử lại stretch / hist_eq với mạng nhỏ)
BATCH_SIZE    = 32

# ── Tăng cường [3]: nhẹ hơn cho mạng nhỏ. Đặt None / 0 để tắt từng phép ──
AUG_FLIP_P      = 0.3
AUG_ROTATE_DEG  = 8                  # trước: 12
AUG_SCALE       = None               # trước: (0.85, 1.15). Ví dụ (0.92, 1.08) để bật nhẹ
AUG_JITTER      = None               # trước: brightness 0.2, contrast 0.15. Ví dụ dict(brightness=0.1, contrast=0.1)
AUG_BLUR_P      = 0.0                # trước: 0.15
AUG_REPEAT_FP32 = 3
AUG_REPEAT_QAT  = 1

# ── Mô hình / FP32 [1][3] ──
USE_BN         = True                # [1] BatchNorm sau mỗi conv (fold vào weight/bias khi convert, RTL không đổi)
DROPOUT_P      = 0.0                 # [3] trước: 0.2 (vector GAP chỉ ~48-64 chiều, dropout làm underfit)
EPOCHS, PATIENCE = 60, 15            # [3] cosine cần chạy gần hết lịch LR nên patience rộng hơn
LR, WEIGHT_DECAY = 2e-3, 1e-4        # [3] AdamW; không decay cho bias / BN. (trước: Adam 1e-3, wd 5e-4)
WARMUP_EPOCHS  = 3                   # warmup tuyến tính trước khi vào cosine (FP32)
LR_MIN_RATIO   = 0.01                # LR cuối cosine = LR * ratio
USE_CLASS_WEIGHT = True
FP32_CLASS_WEIGHT_BOOST = {}
QAT_CLASS_WEIGHT_BOOST  = {}

ABORT_EPOCH, ABORT_MIN_VAL_ACC = 10, 0.40

FP32_GATE = True
MIN_FP32_ACC_FOR_INT8 = 0.9
MIN_FP32_CLASS_METRIC_FOR_INT8 = 0.9

# ── QAT ──
QAT_EPOCHS     = 30
QAT_PATIENCE   = 6               # chỉ đếm SAU khi freeze observer + BN
QAT_LR         = 1e-4
QAT_BATCH_SIZE = 32
QUANT_BACKEND  = "fbgemm"
QAT_FREEZE_OBSERVER_RATIO = 0.15
QAT_SELECT_BY  = "val_loss"      # "val_loss" | "val_acc"

# ── Phần cứng ──
RTL_REQUANT_MULT_BITS = 31
GATE_ON_GOLDEN = True
RANKING_METRIC_THRESHOLD = 0.9
RTL_TESTVEC_PER_CLASS = 3
VEC_MAX_PIXELS = 512 * 512
VEC_PER_SIZE   = 1
TOP_PRINT      = 30

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SIZE_TABLE_WH = [(2004, 1690), (512, 512), (256, 256), (2592, 1728), (2464, 1632), (1848, 1224)]


# ============================================================================
# TIỀN XỬ LÝ SỐ NGUYÊN (khớp bit với RTL)
# ============================================================================
RESIZE_SHIFT = 24


def block_params(w, h, out_size):
    if w < out_size or h < out_size:
        raise ValueError(f"Ảnh {w}x{h} nhỏ hơn kích thước đầu ra {out_size}x{out_size}")
    bw, bh = w // out_size, h // out_size
    m = int(round((1 << RESIZE_SHIFT) / (bw * bh)))
    return bw, bh, m


def block_average_resize(arr, out_size):
    h, w, c = arr.shape
    bw, bh, m = block_params(w, h, out_size)
    crop = arr[:bh * out_size, :bw * out_size]
    s = crop.reshape(out_size, bh, out_size, bw, c).sum(axis=(1, 3), dtype=np.uint64)
    out = (s * np.uint64(m) + np.uint64(1 << (RESIZE_SHIFT - 1))) >> np.uint64(RESIZE_SHIFT)
    return np.minimum(out, 255).astype(np.uint8)


def hist_equalize_int(arr):
    h, w, c = arr.shape
    n = h * w
    shift = n.bit_length() - 1
    assert n == (1 << shift), "H*W phải là lũy thừa của 2 để không cần phép chia"
    out = np.empty_like(arr)
    for k in range(c):
        ch = arr[..., k]
        cdf = np.cumsum(np.bincount(ch.ravel(), minlength=256)).astype(np.int64)
        lut = ((cdf * 255 + (n >> 1)) >> shift).astype(np.uint8)
        out[..., k] = lut[ch]
    return out


def contrast_stretch_int(arr):
    out = np.empty_like(arr)
    for k in range(arr.shape[2]):
        ch = arr[..., k].astype(np.int64)
        lo, hi = int(ch.min()), int(ch.max())
        if hi == lo:
            out[..., k] = arr[..., k]
            continue
        d = hi - lo
        scale = ((255 << 16) + (d >> 1)) // d
        out[..., k] = np.clip(((ch - lo) * scale + (1 << 15)) >> 16, 0, 255)
    return out


def enhance_int(arr, mode):
    if mode == "hist_eq":
        return hist_equalize_int(arr)
    if mode == "stretch":
        return contrast_stretch_int(arr)
    if mode in (None, "none"):
        return arr
    raise ValueError(f"ENHANCE_MODE không hợp lệ: {mode}")


def preprocess_desc():
    return {"resize": "block_average_int", "resize_shift": RESIZE_SHIFT, "enhance": ENHANCE_MODE,
            "normalize_quantize": "LUT 256 entries / channel (input_lut_*.mem)"}


def load_resize_enhance(path, img_size, enhance_mode):
    img = Image.open(path).convert("RGB")
    size_wh = img.size
    out = enhance_int(block_average_resize(np.asarray(img), img_size), enhance_mode)
    return Image.fromarray(out), size_wh


def preprocess_and_cache(samples, desc):
    print(f"  {desc}: block-average resize + enhance[{ENHANCE_MODE}] cho {len(samples)} ảnh...")
    res = Parallel(n_jobs=-1)(delayed(load_resize_enhance)(p, IMG_SIZE, ENHANCE_MODE) for p, _ in samples)
    sizes = Counter(r[1] for r in res)
    if set(sizes) != {EXPECTED_SIZE}:
        print(f"  [CẢNH BÁO] {desc}: có ảnh KHÔNG phải {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}: "
              f"{dict(sizes)} (resize sẽ cắt phần dư ở rìa phải/dưới)")
    return list(zip([r[0] for r in res], [l for _, l in samples]))


# ============================================================================
# DỮ LIỆU
# ============================================================================
class EyeDataset(Dataset):
    def __init__(self, cached_samples, transform=None):
        self.cached_samples, self.transform = cached_samples, transform

    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, idx):
        img, label = self.cached_samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class RepeatDataset(Dataset):
    """Lặp dataset k lần mỗi epoch (mỗi lần lấy ảnh, transform ngẫu nhiên chạy lại)."""
    def __init__(self, base, k):
        self.base, self.k = base, max(1, int(k))

    def __len__(self):
        return len(self.base) * self.k

    def __getitem__(self, i):
        return self.base[i % len(self.base)]


def htag(h):
    return h.split("_")[-1]


def collect_split_samples(h_dir):
    exts = (".png", ".jpg", ".jpeg")

    def subdirs(d):
        if not os.path.isdir(d):
            sys.exit(f"[LỖI] Không thấy folder {d}. Chạy tao_hospital_AB.py trước.")
        return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))

    classes = subdirs(os.path.join(h_dir, "train"))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    splits = {}
    for sp in ("train", "val", "test"):
        sp_dir = os.path.join(h_dir, sp)
        if subdirs(sp_dir) != classes:
            sys.exit(f"[LỖI] {sp_dir} có lớp khác với train: {subdirs(sp_dir)} != {classes}")
        samples = []
        for c in classes:
            cdir = os.path.join(sp_dir, c)
            samples += [(os.path.join(cdir, f), class_to_idx[c])
                        for f in sorted(os.listdir(cdir)) if f.lower().endswith(exts)]
        splits[sp] = samples
    return splits, class_to_idx, classes


def get_transforms():
    """[3] Augmentation nhẹ, cấu hình bằng AUG_*. Val/test chỉ ToTensor + Normalize."""
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    val_t = transforms.Compose([transforms.ToTensor(), norm])
    aug = []
    if AUG_FLIP_P:
        aug.append(transforms.RandomHorizontalFlip(p=AUG_FLIP_P))
    if AUG_ROTATE_DEG:
        aug.append(transforms.RandomRotation(degrees=AUG_ROTATE_DEG))
    if AUG_SCALE:
        aug.append(transforms.RandomAffine(degrees=0, scale=AUG_SCALE))
    if AUG_JITTER:
        aug.append(transforms.ColorJitter(**AUG_JITTER))
    if AUG_BLUR_P:
        aug.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=AUG_BLUR_P))
    train_t = transforms.Compose(aug + [transforms.ToTensor(), norm])
    return train_t, val_t


def prepare_hospital_data(h):
    h_dir = os.path.join(HOSPITAL_ROOT, h)
    print(f"\n[DATA] {h}: đọc {h_dir} (đã chia sẵn train/val/test)")
    splits, class_to_idx, classes = collect_split_samples(h_dir)
    if UNKNOWN_NAME not in class_to_idx:
        sys.exit(f"[LỖI] {h}: không có lớp '{UNKNOWN_NAME}' (các lớp: {classes})")
    train_s, val_s, test_s = splits["train"], splits["val"], splits["test"]
    total = len(train_s) + len(val_s) + len(test_s)
    print(f"  {len(classes)} lớp (Unknown = chỉ số {class_to_idx[UNKNOWN_NAME]}): {classes}")
    print(f"  Train/Val/Test: {len(train_s)}/{len(val_s)}/{len(test_s)} "
          f"({100 * len(train_s) / total:.1f}% / {100 * len(val_s) / total:.1f}% / {100 * len(test_s) / total:.1f}%)")
    cnt = {sp: Counter(l for _, l in s) for sp, s in splits.items()}
    print(f"  {'':24s}" + "".join(f"{sp:>7}" for sp in splits))
    for c in classes:
        print(f"  {c:24s}" + "".join(f"{cnt[sp][class_to_idx[c]]:>7}" for sp in splits))

    print(f"  Tiền xử lý + cache ({IMG_SIZE}x{IMG_SIZE})...")
    train_c = preprocess_and_cache(train_s, f"{h} Train")
    val_c   = preprocess_and_cache(val_s, f"{h} Val")
    test_c  = preprocess_and_cache(test_s, f"{h} Test")

    train_t, val_t = get_transforms()
    return dict(
        name=h, classes=classes, num_classes=len(classes), unknown_idx=class_to_idx[UNKNOWN_NAME],
        train_samples=train_s, val_samples=val_s, test_samples=test_s,
        test_cache=test_c,
        train_ds=EyeDataset(train_c, train_t),
        val_ds=EyeDataset(val_c, val_t),
        test_ds=EyeDataset(test_c, val_t),
        train_labels=[l for _, l in train_s],
        src_wh=get_raw_size(test_s) or (0, 0),
    )


def make_shared_loaders(data):
    data["fp32_train_ld"] = make_loader(RepeatDataset(data["train_ds"], AUG_REPEAT_FP32), BATCH_SIZE, True)
    data["qat_train_ld"]  = make_loader(RepeatDataset(data["train_ds"], AUG_REPEAT_QAT), QAT_BATCH_SIZE, True)
    data["gpu_val_ld"]    = make_loader(data["val_ds"], BATCH_SIZE, False, workers=2)
    data["gpu_test_ld"]   = make_loader(data["test_ds"], BATCH_SIZE, False, workers=2)
    data["cpu_val_ld"]    = DataLoader(data["val_ds"], batch_size=BATCH_SIZE, shuffle=False)
    data["cpu_test_ld"]   = DataLoader(data["test_ds"], batch_size=BATCH_SIZE, shuffle=False)


def make_class_weights(train_labels, classes, boost):
    if not USE_CLASS_WEIGHT:
        return None
    cw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    for name, b in boost.items():
        if name in classes:
            cw[list(classes).index(name)] *= b
        else:
            print(f"[CẢNH BÁO] CLASS_WEIGHT_BOOST: không có lớp '{name}'")
    return torch.tensor(cw, dtype=torch.float32).to(device)


def make_loader(ds, bs, shuffle, workers=4):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=True,
                      persistent_workers=True)


# ============================================================================
# KIẾN TRÚC CNN THUẦN + QAT
#   layers = [(c_out, groups, pool), ...]: conv3x3(pad=1) [+BN] + ReLU (+ MaxPool2x2 nếu pool) -> GAP -> FC
# ============================================================================
def arch_name(layers):
    return "-".join(f"{c}{'g%d' % g if g > 1 else ''}{'p' if p else ''}" for c, g, p in layers)


def count_arch_params(layers, num_classes):
    """Tham số SAU KHI fold BN. Conv: 9*(cin/groups)*cout + cout.  FC: last_c*num_classes + num_classes."""
    total, c_in = 0, 3
    for c, g, _ in layers:
        total += 9 * (c_in // g) * c + c
        c_in = c
    return total + c_in * num_classes + num_classes


def count_arch_macs(layers, num_classes, img_size=IMG_SIZE):
    macs, c_in, size = 0, 3, img_size
    for c, g, pool in layers:
        macs += size * size * 9 * (c_in // g) * c
        c_in = c
        if pool:
            size //= 2
    return macs + c_in * num_classes


def make_features(layers, in_channels=3):
    """[1] Conv(bias=False nếu có BN) -> BN -> ReLU -> (MaxPool)."""
    mods, c_in = [], in_channels
    for c, g, pool in layers:
        mods.append(nn.Conv2d(c_in, c, kernel_size=3, padding=1, groups=g, bias=not USE_BN))
        if USE_BN:
            mods.append(nn.BatchNorm2d(c))
        mods.append(nn.ReLU(inplace=True))
        if pool:
            mods.append(nn.MaxPool2d(kernel_size=2, stride=2))
        c_in = c
    return nn.Sequential(*mods), c_in


class CNNCustom(nn.Module):
    def __init__(self, features, last_c, num_classes, dropout_p=0.0):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Giữ Dropout (kể cả p=0) để chỉ số classifier[2] = Linear không đổi (golden/QAT tham chiếu chỉ số này)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout_p), nn.Linear(last_c, num_classes))

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


def build_cnn_model(num_classes, layers, dropout_p):
    features, last_c = make_features(layers)
    return CNNCustom(features, last_c, num_classes, dropout_p)


def enumerate_architectures(num_classes):
    """Vét cạn kênh x groups, giữ kiến trúc thỏa: depth, dãy kênh không giảm [5], lớp cuối >= MIN_LAST_CHANNELS [5],
    tham số trong [MIN_PARAMS, MAX_PARAMS). Thứ tự: dãy tăng -> giảm -> xen kẽ (nếu được phép)."""
    cand, total = [], 0
    for depth in range(DEPTH_RANGE[0], DEPTH_RANGE[1] + 1):
        for chans in itertools.product(CHANNEL_CHOICES, repeat=depth):
            if ONLY_NON_DECREASING and any(x > y for x, y in zip(chans, chans[1:])):
                continue
            if chans[-1] < MIN_LAST_CHANNELS:
                continue
            for gs in itertools.product(GROUP_CHOICES, repeat=depth - 1):
                total += 1
                gs, layers, c_in, ok = (1,) + gs, [], 3, True
                for i, (c, g) in enumerate(zip(chans, gs)):
                    if c_in % g or c % g:
                        ok = False; break
                    layers.append((c, g, i in POOL_AFTER)); c_in = c
                if not ok:
                    continue
                n = count_arch_params(layers, num_classes)
                if MIN_PARAMS <= n < MAX_PARAMS:
                    cand.append(dict(name=arch_name(layers), layers=layers, params=n,
                                     macs=count_arch_macs(layers, num_classes), chans=chans, gs=gs))

    def kind(a):   # 0 = tăng, 1 = giảm, 2 = xen kẽ
        c = a["chans"]
        if all(x <= y for x, y in zip(c, c[1:])): return 0
        if all(x >= y for x, y in zip(c, c[1:])): return 1
        return 2

    order = lambda a: (len(a["chans"]), a["chans"], a["gs"])
    inc = sorted([a for a in cand if kind(a) == 0], key=order)
    dec = sorted([a for a in cand if kind(a) == 1], key=order)
    mix = [a for a in cand if kind(a) == 2]
    random.Random(ARCH_RNG_SEED).shuffle(mix)
    archs = inc + dec + mix
    print(f"[ARCH] {total} tổ hợp (sau lọc không giảm/lớp cuối) -> {len(archs)} kiến trúc trong "
          f"[{MIN_PARAMS:,}, {MAX_PARAMS:,}) tham số | tăng: {len(inc)} | giảm: {len(dec)} | xen kẽ: {len(mix)}")
    if not archs:
        print("[CẢNH BÁO] Không có kiến trúc nào: nới MIN/MAX_PARAMS, CHANNEL_CHOICES hoặc MIN_LAST_CHANNELS.")
    if ARCH_FILTER is not None:
        keep = set(ARCH_FILTER)
        archs = [a for a in archs if a["name"] in keep]
        print(f"[ARCH] ARCH_FILTER: giữ {len(archs)}/{len(keep)} kiến trúc")
    return archs[:MAX_ARCH] if MAX_ARCH else archs


def count_hw_params(model):
    """Tham số phần cứng SAU fold BN: weight + 1 bias/kênh (khớp count_arch_params). Không đếm gamma/beta/mean/var."""
    t = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            t += m.weight.numel() + m.out_channels
        elif isinstance(m, nn.Linear):
            t += m.weight.numel() + m.out_features
    return t


class QuantWrappedCNN(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        self.quant = tq.QuantStub()
        self.vgg = vgg_model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.vgg(self.quant(x)))


def fuse_cnn_modules(wrapped):
    """[1] Fuse Conv(+BN)(+ReLU) ở chế độ TRAIN (fuse_modules_qat) để QAT mô phỏng việc fold BN vào weight."""
    feats = wrapped.vgg.features._modules
    names, to_fuse, i = list(feats.keys()), [], 0
    while i < len(names):
        if isinstance(feats[names[i]], nn.Conv2d):
            grp, j = [f"vgg.features.{names[i]}"], i + 1
            if j < len(names) and isinstance(feats[names[j]], nn.BatchNorm2d):
                grp.append(f"vgg.features.{names[j]}"); j += 1
            if j < len(names) and isinstance(feats[names[j]], nn.ReLU):
                grp.append(f"vgg.features.{names[j]}"); j += 1
            to_fuse.append(grp); i = j
        else:
            i += 1
    print(f"  Fuse {len(to_fuse)} nhóm Conv{'+BN' if USE_BN else ''}+ReLU (QAT)...")
    tq.fuse_modules_qat(wrapped, to_fuse, inplace=True)
    return wrapped


def _act_fq(qscheme):
    return tq.FakeQuantize.with_args(observer=tq.MovingAverageMinMaxObserver,
                                     quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=qscheme)


def prepare_qat_model(fp32_model):
    """Activation sau Conv+ReLU: per_tensor_AFFINE (zp=0). Input và logits có số âm -> symmetric (zp=127).
    Weight: int8 per-tensor symmetric (zp=0). Fuse ở chế độ train (cần cho Conv+BN)."""
    wrapped = QuantWrappedCNN(copy.deepcopy(fp32_model))
    wrapped.train()
    fuse_cnn_modules(wrapped)
    w_fq = tq.FakeQuantize.with_args(observer=tq.MovingAverageMinMaxObserver,
                                     quant_min=-128, quant_max=127, dtype=torch.qint8,
                                     qscheme=torch.per_tensor_symmetric)
    hidden = tq.QConfig(activation=_act_fq(torch.per_tensor_affine), weight=w_fq)
    edge = tq.QConfig(activation=_act_fq(torch.per_tensor_symmetric), weight=w_fq)
    wrapped.qconfig = hidden
    wrapped.quant.qconfig = edge
    wrapped.vgg.classifier[2].qconfig = edge
    tq.prepare_qat(wrapped, inplace=True)
    return wrapped.to(device)


def convert_qat_to_int8(qat_model):
    torch.backends.quantized.engine = QUANT_BACKEND
    qat_model.eval()
    return tq.convert(qat_model.to("cpu"), inplace=False)


# ============================================================================
# OPTIMIZER / SCHEDULER [3]
# ============================================================================
def make_optimizer(model, lr, wd):
    """AdamW; weight decay chỉ áp cho tensor >= 2 chiều (weight conv/fc). Bias và BN gamma/beta không decay."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def make_scheduler(optimizer, epochs, warmup=0):
    """Warmup tuyến tính (tùy chọn) + cosine về LR*LR_MIN_RATIO. Gọi scheduler.step() mỗi epoch."""
    base_lr = optimizer.param_groups[0]["lr"]
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup), eta_min=base_lr * LR_MIN_RATIO)
    if warmup > 0:
        wu = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [wu, cos], milestones=[warmup])
    return cos


# ============================================================================
# TRAIN / VALIDATE / EVALUATE
# ============================================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, preds, labels = 0.0, [], []
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        preds.extend(out.argmax(1).cpu().numpy()); labels.extend(y.cpu().numpy())
    return loss_sum / len(loader.dataset), accuracy_score(labels, preds)


def validate_epoch(model, loader, criterion, dev):
    model.eval()
    loss_sum, preds, labels = 0.0, [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            preds.extend(out.argmax(1).cpu().numpy()); labels.extend(y.cpu().numpy())
    return loss_sum / len(loader.dataset), accuracy_score(labels, preds), preds, labels


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience,
                select_by="val_acc", is_qat=False, freeze_observer_epoch=None,
                abort_epoch=None, abort_min_val_acc=0.0):
    """QAT: observer + BN stats chạy tới freeze_observer_epoch rồi khóa hẳn. Trước khi freeze KHÔNG chọn
    checkpoint / không đếm patience (scale còn trôi) -> scale trong checkpoint là scale cuối cùng.
    Scheduler: ReduceLROnPlateau (step theo val_loss) hoặc loại khác (cosine...: step() mỗi epoch)."""
    best_metric, best_val_acc, best_state, bad = float("inf"), 0.0, None, 0
    best_epoch, frozen, aborted = 0, False, False
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    for epoch in range(epochs):
        if is_qat:
            if freeze_observer_epoch is not None and epoch >= freeze_observer_epoch:
                if not frozen:
                    print(f"  [QAT] Freeze observer + BN từ epoch {epoch + 1}")
                    model.apply(tq.disable_observer)
                    model.apply(nniqat.freeze_bn_stats)          # [1] khóa mean/var của BN
                    frozen = True
            else:
                model.apply(tq.enable_observer)

        tl, ta = train_epoch(model, train_loader, criterion, optimizer)
        if is_qat:
            model.apply(tq.disable_observer)       # tránh observer "nhìn thấy" val
        vl, va, *_ = validate_epoch(model, val_loader, criterion, device)
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(vl)
            else:
                scheduler.step()
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(v)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch + 1}/{epochs} | lr={lr_now:.2e} | train_loss={tl:.4f} train_acc={ta:.4f} "
              f"| val_loss={vl:.4f} val_acc={va:.4f}")

        if abort_epoch is not None and epoch + 1 == abort_epoch and va < abort_min_val_acc:
            print(f"  [DỪNG SỚM] epoch {abort_epoch}: val_acc={va:.4f} < {abort_min_val_acc} -> bỏ kiến trúc này")
            aborted = True
            break
        if is_qat and freeze_observer_epoch is not None and not frozen:
            continue
        metric = vl if select_by == "val_loss" else -va
        if metric < best_metric:
            best_metric, best_val_acc, bad, best_epoch = metric, va, 0, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print(f"  Early stopping sau {epoch + 1} epochs"); break
    if best_state:
        model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    hist["aborted"] = aborted
    return model, hist, best_val_acc


def metrics_from_preds(labels, preds):
    labels, preds = np.asarray(labels), np.asarray(preds)
    return {"accuracy": accuracy_score(labels, preds),
            "balanced_accuracy": balanced_accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average=None, zero_division=0),
            "recall": recall_score(labels, preds, average=None, zero_division=0),
            "f1": f1_score(labels, preds, average=None, zero_division=0),
            "preds": preds, "labels": labels}


def evaluate_model(model, loader, dev):
    _, _, preds, labels = validate_epoch(model, loader, nn.CrossEntropyLoss(), dev)
    return metrics_from_preds(labels, preds)


def print_report(res, classes, title=""):
    print(f"\nClassification Report {title}:")
    print(classification_report(res["labels"], res["preds"], target_names=classes, zero_division=0))
    cm = confusion_matrix(res["labels"], res["preds"])
    print(f"Confusion matrix {title} (hàng = thật, cột = dự đoán):")
    print(" " * 22 + " ".join(f"{c[:6]:>6}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<22}" + " ".join(f"{v:>6}" for v in cm[i]))


def print_per_class(res, classes, title="", threshold=RANKING_METRIC_THRESHOLD):
    n = len(classes)
    cm = confusion_matrix(res["labels"], res["preds"], labels=np.arange(n))
    print(f"\n[TỪNG LỚP] {title} (ngưỡng > {threshold})")
    print(f"  {'Lớp':<24}{'Prec':>7}{'Recall':>8}{'F1':>7}{'N':>6}   Ghi chú")
    for i, c in enumerate(classes):
        p, r, f1 = float(res["precision"][i]), float(res["recall"][i]), float(res["f1"][i])
        note = []
        if r <= threshold:
            row = cm[i].copy(); row[i] = 0
            if row.sum() > 0:
                j = int(row.argmax())
                note.append(f"recall thấp: {int(row[j])}/{int(cm[i].sum())} ảnh bị đoán thành '{classes[j]}'")
        if p <= threshold:
            col = cm[:, i].copy(); col[i] = 0
            if col.sum() > 0:
                j = int(col.argmax())
                note.append(f"precision thấp: {int(col[j])} ảnh '{classes[j]}' bị đoán nhầm vào đây")
        flag = "<<" if (r <= threshold or p <= threshold) else "  "
        print(f"{flag}{c:<24}{p:>7.4f}{r:>8.4f}{f1:>7.4f}{int(cm[i].sum()):>6}   " + "; ".join(note))
    print(f"  Lớp recall thấp nhất : {classes[int(np.argmin(res['recall']))]} "
          f"({float(np.min(res['recall'])):.4f})")
    print(f"  Lớp precision thấp nhất: {classes[int(np.argmin(res['precision']))]} "
          f"({float(np.min(res['precision'])):.4f})")


def export_misclassified(res, samples, classes, out_csv, title="", threshold=RANKING_METRIC_THRESHOLD,
                         print_all=False):
    labels, preds = np.asarray(res["labels"]), np.asarray(res["preds"])
    assert len(samples) == len(labels), "số mẫu không khớp số dự đoán (loader có shuffle?)"
    wrong = np.where(labels != preds)[0]
    rows = [[os.path.basename(samples[i][0]), classes[labels[i]], classes[preds[i]], samples[i][0]]
            for i in wrong]
    rows.sort(key=lambda r: (r[1], r[2], r[0]))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_class", "pred_class", "source_path"])
        w.writerows(rows)

    weak = {i for i in range(len(classes))
            if res["recall"][i] <= threshold or res["precision"][i] <= threshold}
    weak_names = {classes[i] for i in weak}
    shown = [r for r in rows if print_all or r[1] in weak_names or r[2] in weak_names]
    print(f"\n[ẢNH SAI] {title}: {len(rows)}/{len(labels)} ảnh sai | đã lưu: {out_csv}")
    cur = None
    for fname, t, p, path in shown:
        if (t, p) != cur:
            cur = (t, p)
            print(f"  thật '{t}' -> đoán '{p}':")
        print(f"    {fname}   ({path})")
    if not shown:
        print("  (không có lớp nào dưới ngưỡng)")
    return rows


def check_int8_threshold(res, threshold=RANKING_METRIC_THRESHOLD):
    p, r = np.asarray(res["precision"], float), np.asarray(res["recall"], float)
    return bool(np.all(p > threshold) and np.all(r > threshold)), float(p.min()), float(r.min())


def brief(res, threshold=RANKING_METRIC_THRESHOLD):
    ok, p, r = check_int8_threshold(res, threshold)
    return f"acc={res['accuracy']:.4f} | min_precision={p:.4f} | min_recall={r:.4f} | đạt>{threshold}: {ok}"


def plot_training_curves(hist, save_path, title):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, (k1, k2, name) in zip(ax, (("train_loss", "val_loss", "Loss"), ("train_acc", "val_acc", "Accuracy"))):
        a.plot(ep, hist[k1], label=k1); a.plot(ep, hist[k2], label=k2)
        a.set_xlabel("Epoch"); a.set_ylabel(name); a.set_title(f"{title} — {name}")
        a.legend(); a.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()


CLASS_NAMES_VI = {
    "Cataract": "Đục thủy tinh thể", "Diabetic Retinopathy": "Võng mạc tiểu đường",
    "Disc Edema": "Phù gai thị", "Glaucoma": "Tăng nhãn áp", "Macular Scar": "Sẹo hoàng điểm",
    "Myopia": "Cận thị", "Normal": "Bình thường", "Retinal Detachment": "Bong võng mạc",
    "Retinitis Pigmentosa": "Viêm võng mạc sắc tố", "Unknown": "Không xác định",
}
FIG_TEXT = {
    "en": dict(acc_title="Accuracy Over Epochs", loss_title="Loss Over Epochs", acc="Accuracy", loss="Loss",
               train_acc="Train Acc", val_acc="Val Acc", train_loss="Train Loss", val_loss="Val Loss",
               best="Best model (epoch {e})", epoch="Epoch", cm_title="Confusion Matrix - {m}",
               true="True Label", pred="Predicted Label"),
    "vi": dict(acc_title="Độ chính xác qua các epoch", loss_title="Hàm mất mát qua các epoch",
               acc="Độ chính xác", loss="Mất mát", train_acc="Acc huấn luyện", val_acc="Acc kiểm định",
               train_loss="Loss huấn luyện", val_loss="Loss kiểm định",
               best="Mô hình tốt nhất (epoch {e})", epoch="Epoch", cm_title="Ma trận nhầm lẫn - {m}",
               true="Nhãn thật", pred="Nhãn dự đoán"),
}


def plot_best_curves(hist, lang, save_path):
    t, ep, be = FIG_TEXT[lang], range(1, len(hist["train_loss"]) + 1), hist.get("best_epoch")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = (("train_acc", "val_acc", "acc_title", "acc"), ("train_loss", "val_loss", "loss_title", "loss"))
    for ax, (k_tr, k_va, k_title, k_y) in zip(axes, panels):
        ax.plot(ep, hist[k_tr], label=t[k_tr]); ax.plot(ep, hist[k_va], label=t[k_va])
        if be:
            ax.axvline(be, color="gray", ls="--", lw=1, label=t["best"].format(e=be))
        ax.set_title(t[k_title]); ax.set_xlabel(t["epoch"]); ax.set_ylabel(t[k_y])
        ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_confusion_pair(res_fp32, res_int8, classes, lang, save_path):
    t, n = FIG_TEXT[lang], len(classes)
    names = [CLASS_NAMES_VI.get(c, c) for c in classes] if lang == "vi" else list(classes)
    cms = [confusion_matrix(r["labels"], r["preds"], labels=np.arange(n)) for r in (res_fp32, res_int8)]
    vmax = max(int(cm.max()) for cm in cms)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, cm, tag in zip(axes, cms, ("FP32", "INT8")):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90)
        ax.set_yticks(range(n)); ax.set_yticklabels(names)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > vmax / 2 else "black")
        ax.set_title(t["cm_title"].format(m=tag)); ax.set_xlabel(t["pred"]); ax.set_ylabel(t["true"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()


def export_best_figures(art, classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for lang in ("vi", "en"):
        plot_best_curves(art["fp32_hist"], lang, os.path.join(out_dir, f"training_curves_fp32_{lang}.png"))
        plot_confusion_pair(art["fp32_test"], art["int8_test"], classes, lang,
                            os.path.join(out_dir, f"confusion_matrix_{lang}.png"))
    print(f"  Đã xuất hình (vi + en) vào: {out_dir}")


# ============================================================================
# GOLDEN MODEL SỐ NGUYÊN — mô phỏng phép tính của FPGA (KHÔNG ĐỔI: BN đã fold vào weight/bias)
#   conv/linear : acc_int32 = sum((x_q - zp_in) * w_int8) + bias_int32   (padding = zp_in => 0)
#   requant     : y = ((acc * M0 + 2^(n-1)) >> n) + zp_out,  clamp [zp_out (ReLU) | 0, 255]
#   maxpool     : max trên uint8 | GAP: (sum + N/2) // N
# ============================================================================
def _fixed_point_multiplier(m, bits):
    assert m > 0, "requant multiplier phải > 0"
    n = bits - 1 - int(np.floor(np.log2(m)))
    m0 = int(round(m * (2 ** n)))
    if m0 >= (1 << bits):
        m0 >>= 1
        n -= 1
    return m0, n


def _extract_bias_int32(module, bias_scale):
    """Lấy bias int32 từ module đã lượng tử hóa.
    Bias sau khi fold BN có thể lớn (> 100), nên KHÔNG đoán dạng chỉ bằng độ lớn: chỉ coi là 'int32 lưu dạng float'
    khi MỌI phần tử đều là số nguyên và max > 100; còn lại chia cho bias_scale (đúng cách fbgemm làm)."""
    bias = module.bias()
    if bias is None:
        return None
    if bias.dtype in (torch.int32, torch.int64):
        return bias.flatten().to(torch.int64)
    if getattr(bias, "is_quantized", False):
        return bias.int_repr().flatten().to(torch.int64)
    bias = bias.detach().cpu().flatten().to(torch.float64)
    if bias.numel() > 0 and float(bias.abs().max()) > 100.0 and bool(torch.all(bias == bias.round())):
        return bias.round().to(torch.int64)
    return torch.round(bias / bias_scale).to(torch.int64)


def extract_golden_layers(int8_model):
    in_scale, in_zp = float(int8_model.quant.scale), int(int8_model.quant.zero_point)
    prev_scale, prev_zp = in_scale, in_zp
    layers = []

    def conv_like(m, relu):
        nonlocal prev_scale, prev_zp
        wq = m.weight()
        assert wq.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric) \
            and int(wq.q_zero_point()) == 0, "golden model giả định weight per-tensor, zero_point = 0"
        w_scale = float(wq.q_scale())
        out_scale, out_zp = float(m.scale), int(m.zero_point)
        bias = _extract_bias_int32(m, prev_scale * w_scale)
        m0, n = _fixed_point_multiplier(prev_scale * w_scale / out_scale, RTL_REQUANT_MULT_BITS)
        d = dict(w=torch.int_repr(wq).to(torch.float64),
                 bias=bias.to(torch.int64) if bias is not None else None,
                 in_zp=prev_zp, out_zp=out_zp, m0=m0, shift=n, relu=relu)
        prev_scale, prev_zp = out_scale, out_zp
        return d

    for m in int8_model.vgg.features:
        if isinstance(m, nnq.Conv2d):
            assert tuple(m.kernel_size) == (3, 3) and tuple(m.padding) == (1, 1) \
                and tuple(m.stride) == (1, 1), "golden model giả định conv 3x3 pad=1 stride=1"
            d = conv_like(m, relu=type(m).__name__.endswith("ReLU2d"))
            d["kind"] = "conv"
            d["groups"] = int(m.groups)
            layers.append(d)
        elif isinstance(m, nn.MaxPool2d):
            layers.append(dict(kind="pool"))
    lin = int8_model.vgg.classifier[2]
    assert isinstance(lin, nnq.Linear)
    d = conv_like(lin, relu=False)
    d["kind"] = "linear"
    layers.append(d)
    return dict(in_scale=in_scale, in_zp=in_zp, layers=layers)


def _requant(acc, d):
    y = (acc * d["m0"] + (1 << (d["shift"] - 1))) >> d["shift"]
    y = y + d["out_zp"]
    return torch.clamp(y, d["out_zp"] if d["relu"] else 0, 255)


@torch.no_grad()
def golden_run(g, xq, trace=False):
    xq = xq.to(torch.int64)
    tr = [("input", xq.clone())] if trace else []
    ci = pi = 0
    for d in g["layers"]:
        if d["kind"] == "conv":
            acc = F.conv2d((xq - d["in_zp"]).to(torch.float64), d["w"], padding=1, groups=d["groups"])
            if d["bias"] is not None:
                acc = acc + d["bias"].to(torch.float64).view(1, -1, 1, 1)
            acc = acc.round().to(torch.int64)
            assert int(acc.abs().max()) < (1 << 30), "acc vượt 2^30: kiểm tra độ rộng accumulator RTL"
            xq = _requant(acc, d)
            ci += 1
            if trace: tr.append((f"conv{ci}", xq.clone()))
        elif d["kind"] == "pool":
            xq = F.max_pool2d(xq.to(torch.float64), 2).to(torch.int64)
            pi += 1
            if trace: tr.append((f"pool{pi}", xq.clone()))
        else:  # GAP (half-up) + Linear
            n = xq.shape[2] * xq.shape[3]
            xq = (xq.sum(dim=(2, 3)) + n // 2) // n
            if trace: tr.append(("gap", xq.clone()))
            acc = (xq - d["in_zp"]).to(torch.float64) @ d["w"].t()
            if d["bias"] is not None:
                acc = acc + d["bias"].to(torch.float64).view(1, -1)
            acc = acc.round().to(torch.int64)
            lg = _requant(acc, d)
            if trace: tr.append(("fc", lg.clone()))
            return lg, acc, tr
    raise RuntimeError("model không có lớp Linear cuối")


@torch.no_grad()
def golden_forward(g, x_float):
    xq = torch.quantize_per_tensor(x_float.float(), g["in_scale"], g["in_zp"],
                                   torch.quint8).int_repr().to(torch.int64)
    lg, acc, _ = golden_run(g, xq)
    return lg, acc


@torch.no_grad()
def golden_forward_from_u8(g, lut, imgs_u8, trace=False):
    imgs = np.asarray(imgs_u8, dtype=np.uint8)
    xq = np.empty(imgs.shape, dtype=np.int64)
    for c in range(3):
        xq[..., c] = lut[c][imgs[..., c]]
    return golden_run(g, torch.from_numpy(xq).permute(0, 3, 1, 2).contiguous(), trace=trace)


def golden_u8_predict(g, lut, pil_imgs, bs=64):
    rq, ac = [], []
    for i in range(0, len(pil_imgs), bs):
        batch = np.stack([np.asarray(im) for im in pil_imgs[i:i + bs]])
        lg, acc, _ = golden_forward_from_u8(g, lut, batch)
        rq.append(lg.numpy()); ac.append(acc.numpy())
    return np.concatenate(rq), np.concatenate(ac)


def evaluate_golden(int8_model, loader):
    g = extract_golden_layers(int8_model)
    p_rq, p_acc, labels = [], [], []
    for x, y in tqdm(loader, desc="Golden", leave=False):
        lg_rq, lg_acc = golden_forward(g, x)
        p_rq.extend(lg_rq.argmax(1).tolist()); p_acc.extend(lg_acc.argmax(1).tolist())
        labels.extend(y.tolist())
    res = metrics_from_preds(labels, p_rq)
    res["preds_argmax_acc"] = np.asarray(p_acc)
    return res


def compare_golden_vs_pytorch(res_fb, res_gold, tag=""):
    agree = float((res_fb["preds"] == res_gold["preds"]).mean())
    agree_acc = float((res_fb["preds"] == res_gold["preds_argmax_acc"]).mean())
    print(f"[SO KHỚP {tag}] acc fbgemm={res_fb['accuracy']:.4f} | acc golden={res_gold['accuracy']:.4f} "
          f"| trùng dự đoán: golden(requant)={agree:.4f} | golden(argmax acc)={agree_acc:.4f}")
    if agree < 0.995:
        print("  *** Golden lệch fbgemm > 0.5%: nghi bão hòa int16 của fbgemm hoặc khác cách làm tròn. "
              "Số của GOLDEN mới sát FPGA. ***")


def check_zero_points(int8_model):
    mods = [(n, m) for n, m in int8_model.named_modules() if isinstance(m, (nnq.Conv2d, nnq.Linear))]
    ok = True
    for i, (name, m) in enumerate(mods):
        expect = 127 if i == len(mods) - 1 else 0
        if int(m.zero_point) != expect:
            ok = False
            print(f"  *** SAI zero-point {name}: out_zp={int(m.zero_point)} (mong đợi {expect}) ***")
    if int(int8_model.quant.zero_point) != 127:
        ok = False
        print(f"  *** SAI zero-point input: {int(int8_model.quant.zero_point)} (mong đợi 127) ***")
    print(f"[ZERO-POINT] {'OK' if ok else 'CÓ LỖI, kiểm tra qconfig trong prepare_qat_model'}")
    return ok


def check_no_bn_left(int8_model):
    """[1] Sau convert không được còn BatchNorm (phải fold hết vào conv)."""
    left = [n for n, m in int8_model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    print(f"[BN] {'OK: đã fold hết BN vào conv' if not left else '*** CÒN BN CHƯA FOLD: ' + str(left) + ' ***'}")
    return not left


def dump_quantization_info(int8_model):
    print("\n=== QUANTIZATION INFO ===")
    print(f"QuantStub (input): scale={float(int8_model.quant.scale):.10f}, "
          f"zero_point={int(int8_model.quant.zero_point)}")
    for name, m in int8_model.named_modules():
        if isinstance(m, (nnq.Conv2d, nnq.Linear)):
            wq = m.weight()
            print(f"{name}: w_scale={float(wq.q_scale()):.8f} w_zp={int(wq.q_zero_point())} "
                  f"| out_scale={float(m.scale):.8f} out_zp={int(m.zero_point)}")


# ============================================================================
# XUẤT PHẦN CỨNG (giữ nguyên tên file)
# ============================================================================
def export_int8_mem(int8_model, memory_dir):
    os.makedirs(memory_dir, exist_ok=True)
    input_scale, input_zp = float(int8_model.quant.scale), int(int8_model.quant.zero_point)
    quant_params = {"input_quant": {"scale": input_scale, "zero_point": input_zp}}
    prev_scale, prev_zp = input_scale, input_zp
    n_layers = n_w = n_b = 0

    for name, m in int8_model.named_modules():
        if not isinstance(m, (nnq.Conv2d, nnq.Linear)):
            continue
        safe = name.replace(".", "_")
        wq = m.weight()
        w_int8 = wq.int_repr().flatten()
        w_scale, w_zp = float(wq.q_scale()), int(wq.q_zero_point())
        with open(os.path.join(memory_dir, f"{safe}_weight_int8.mem"), "w") as f:
            for v in w_int8.tolist():
                f.write(f"{v & 0xFF:02x}\n")
        n_w += w_int8.numel()

        out_scale, out_zp = float(m.scale), int(m.zero_point)
        bias_scale = prev_scale * w_scale
        b_int32 = _extract_bias_int32(m, bias_scale)
        nb = 0
        if b_int32 is not None:
            nb = b_int32.numel(); n_b += nb
            with open(os.path.join(memory_dir, f"{safe}_bias_int32.mem"), "w") as f:
                for v in b_int32.tolist():
                    f.write(f"{v & 0xFFFFFFFF:08x}\n")

        requant = (prev_scale * w_scale / out_scale) if out_scale > 0 else 0.0
        m0, shift = _fixed_point_multiplier(requant, RTL_REQUANT_MULT_BITS)
        quant_params[name] = {
            "layer_type": type(m).__name__, "weight_shape": list(wq.shape),
            "weight_scale": w_scale, "weight_zero_point": w_zp,
            "input_scale": prev_scale, "input_zero_point": prev_zp,
            "output_scale": out_scale, "output_zero_point": out_zp,
            "bias_scale": bias_scale,
            "bias_dtype": "int32 (bias_q = round(bias_fp32 / (input_scale * weight_scale))), BN đã fold",
            "num_bias": int(nb), "requantize_multiplier": requant,
            "requant_M0": m0, "requant_shift": shift, "requant_mult_bits": RTL_REQUANT_MULT_BITS,
            "requant_formula": "y = ((acc*M0 + 2^(shift-1)) >> shift) + output_zero_point; clamp [relu? out_zp : 0, 255]",
            "fused_relu": type(m).__name__.endswith("ReLU2d"),
        }
        prev_scale, prev_zp = out_scale, out_zp
        n_layers += 1

    with open(os.path.join(memory_dir, "quant_params.json"), "w", encoding="utf-8") as f:
        json.dump(quant_params, f, indent=2)
    mem_files = [f for f in os.listdir(memory_dir) if f.endswith(".mem")]
    total_mb = sum(os.path.getsize(os.path.join(memory_dir, f)) for f in mem_files) / (1024 * 1024)
    hw_mb = (n_w + n_b * 4) / (1024 * 1024)
    with open(os.path.join(memory_dir, "mem_size_report.json"), "w", encoding="utf-8") as f:
        json.dump({"num_layers_exported": n_layers, "num_mem_files": len(mem_files),
                   "num_weight_elements_int8": int(n_w), "num_bias_elements_int32": int(n_b),
                   "total_mem_file_size_MB": round(total_mb, 6), "hw_memory_size_MB": round(hw_mb, 6),
                   "hw_memory_size_Mbit": round((n_w + n_b * 4) * 8 / 1e6, 6)}, f, indent=2)
    print(f"  Đã xuất {n_layers} lớp INT8 vào {memory_dir} | {len(mem_files)} file .mem | HW: {hw_mb:.6f} MB")
    return total_mb, hw_mb


def build_input_lut(scale, zero_point):
    lut = np.empty((3, 256), dtype=np.uint8)
    p = np.arange(256, dtype=np.float32) / np.float32(255.0)
    for c in range(3):
        x = ((p - np.float32(IMAGENET_MEAN[c])) / np.float32(IMAGENET_STD[c])).astype(np.float32)
        q = torch.quantize_per_tensor(torch.from_numpy(x), float(scale), int(zero_point), torch.quint8)
        lut[c] = torch.int_repr(q).numpy()
    return lut


def export_input_lut(int8_model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    scale, zp = float(int8_model.quant.scale), int(int8_model.quant.zero_point)
    lut = build_input_lut(scale, zp)
    for c, name in enumerate("RGB"):
        with open(os.path.join(out_dir, f"input_lut_{name}.mem"), "w") as f:
            for v in lut[c].tolist():
                f.write(f"{v:02x}\n")
    with open(os.path.join(out_dir, "input_lut.json"), "w", encoding="utf-8") as f:
        json.dump({"input_scale": scale, "input_zero_point": zp,
                   "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD,
                   "enhance_mode": ENHANCE_MODE, "preprocess": preprocess_desc(),
                   "note": "index = pixel 0..255 (sau resize + tăng cường); value = quint8 đưa vào conv1"},
                  f, indent=2)
    print(f"  Đã xuất LUT Normalize+Quantize (scale={scale:.8g}, zp={zp}) vào: {out_dir}")


def get_raw_size(samples):
    sizes = {Image.open(p).size for p, _ in samples}
    return next(iter(sizes)) if len(sizes) == 1 else None


def export_rtl_package(int8_model, out_dir, img_size, src_wh):
    """Sinh Package_VGG9.sv từ CHÍNH golden (M0/shift/zp khớp 100% với Python). Không sửa tay file này."""
    os.makedirs(out_dir, exist_ok=True)
    g = extract_golden_layers(int8_model)
    names = [n for n, m in int8_model.named_modules() if isinstance(m, (nnq.Conv2d, nnq.Linear))]
    size, cin, convs, fc = img_size, 3, [], None
    for d in g["layers"]:
        if d["kind"] == "conv":
            cout = int(d["w"].shape[0])
            convs.append(dict(cin=cin, cout=cout, size=size, pool=False, d=d)); cin = cout
        elif d["kind"] == "pool":
            convs[-1]["pool"] = True; size //= 2
        else:
            fc = d
    assert len(convs) >= 1 and fc is not None, "Cần ít nhất 1 conv + 1 FC"
    nconv = len(convs)
    for d in [c["d"] for c in convs] + [fc]:
        assert 0 < d["m0"] < (1 << 31), "M0 phải < 2^31 (đặt RTL_REQUANT_MULT_BITS=31)"
        assert d["shift"] <= 62

    L = ["package Package_VGG9;", "",
         f"    // FILE TU SINH boi CNN_Search_BN.py ({nconv} conv + GAP + FC) - KHONG SUA TAY",
         f"    // input_scale={g['in_scale']:.10f}  input_zp={g['in_zp']}  requant: y=((acc*M0 + 2^(S-1))>>>S)+ZP_OUT",
         "    localparam int KERNEL = 3*3;", "",
         f"    // Tien xu ly tren PL: block-average resize -> enhance[{ENHANCE_MODE}] -> LUT",
         f"    localparam int SRC_W = {src_wh[0]}, SRC_H = {src_wh[1]};",
         f"    localparam int BLOCK_W = {src_wh[0] // img_size}, BLOCK_H = {src_wh[1] // img_size};",
         f"    localparam int RESIZE_M = {int(round((1 << RESIZE_SHIFT) / max(1, (src_wh[0] // img_size) * (src_wh[1] // img_size))))}, RESIZE_SHIFT = {RESIZE_SHIFT};",
         *([f"    localparam int HIST_SHIFT = {(img_size * img_size).bit_length() - 1};  // N = {img_size}*{img_size}"]
           if ENHANCE_MODE == "hist_eq" else []),
         '    localparam string LUT_R = "input_lut_R.mem", LUT_G = "input_lut_G.mem", LUT_B = "input_lut_B.mem";', ""]
    for i, c in enumerate(convs, 1):
        d = c["d"]
        feat = "SIZE_IMAGE" if i == 1 else f"SIZE_FEATURE{i-1}"
        L += [f"    // Conv {i}: {c['size']}x{c['size']}, {c['cin']} -> {c['cout']}, {'MaxPool' if c['pool'] else 'khong pool'}",
              f"    localparam int {feat} = {c['size']}, CIN_CONV{i} = {c['cin']}, COUT_CONV{i} = {c['cout']};",
              f"    localparam int GROUPS_CONV{i} = {d['groups']};  // kenh ra thuoc nhom g chi doc {c['cin'] // d['groups']} kenh vao [g*{c['cin'] // d['groups']} ..]",
              f"    localparam int MULT_CONV{i} = {d['m0']}, SHIFT_CONV{i} = {d['shift']};",
              f"    localparam int ZP_IN_CONV{i} = {d['in_zp']}, ZP_OUT_CONV{i} = {d['out_zp']};",
              f'    localparam string WEIGHT_CONV{i} = "{names[i-1].replace(".", "_")}_weight_int8.mem";',
              f'    localparam string BIAS_CONV{i}   = "{names[i-1].replace(".", "_")}_bias_int32.mem";', ""]
    L += [f"    localparam int SIZE_FEATURE{nconv} = {size}, CIN_GAP = {cin};", "",
          f"    localparam int CIN_FC = {int(fc['w'].shape[1])}, COUT_FC = {int(fc['w'].shape[0])};",
          f"    localparam int MULT_FC = {fc['m0']}, SHIFT_FC = {fc['shift']};",
          f"    localparam int ZP_IN_FC = {fc['in_zp']}, ZP_OUT_FC = {fc['out_zp']};",
          f'    localparam string WEIGHT_FC = "{names[nconv].replace(".", "_")}_weight_int8.mem";',
          f'    localparam string BIAS_FC   = "{names[nconv].replace(".", "_")}_bias_int32.mem";', "",
          "endpackage", ""]
    path = os.path.join(out_dir, "Package_VGG9.sv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  Đã sinh {path}")


def _write_hex(path, values, width=2):
    with open(path, "w") as f:
        for v in np.asarray(values).reshape(-1).tolist():
            f.write(f"{int(v) & ((1 << (4 * width)) - 1):0{width}x}\n")


def export_rtl_testvectors(int8_model, test_samples, test_cache, out_dir, n_trace=2):
    os.makedirs(out_dir, exist_ok=True)
    wh = get_raw_size(test_samples)
    if wh is None:
        print("  [CẢNH BÁO] Ảnh test không cùng kích thước -> không xuất được ảnh thô.")
        return None
    g = extract_golden_layers(int8_model)
    lut = build_input_lut(g["in_scale"], g["in_zp"])

    with open(os.path.join(out_dir, "test_raw_images.mem"), "w") as raw_f:
        for i, (p, _) in enumerate(tqdm(test_samples, desc="raw .mem", leave=False)):
            raw = np.asarray(Image.open(p).convert("RGB"))
            hw = enhance_int(block_average_resize(raw, IMG_SIZE), ENHANCE_MODE)
            assert np.array_equal(hw, np.asarray(test_cache[i][0])), f"tiền xử lý ảnh {i} lệch cache"
            for px in raw.reshape(-1, 3).tolist():
                raw_f.write(f"{px[2]:02x}{px[1]:02x}{px[0]:02x}\n")

    lg_rq, _ = golden_u8_predict(g, lut, [im for im, _ in test_cache])
    preds = np.argmax(lg_rq, axis=1)
    labels = np.array([l for _, l in test_samples])
    np.savetxt(os.path.join(out_dir, "golden_pred.txt"), preds, fmt="%d")
    np.savetxt(os.path.join(out_dir, "test_labels.txt"), labels, fmt="%d")
    np.savetxt(os.path.join(out_dir, "golden_logits.txt"), lg_rq, fmt="%d")

    for i in range(min(n_trace, len(test_cache))):
        _, _, tr = golden_forward_from_u8(g, lut, np.asarray(test_cache[i][0])[None], trace=True)
        for name, t in tr:
            t = t[0].numpy()
            if t.ndim == 3:
                t = t.transpose(1, 2, 0)
            _write_hex(os.path.join(out_dir, f"dbg_img{i}_{name}.mem"), t)
    acc = float((preds == labels).mean())
    with open(os.path.join(out_dir, "testvec_info.json"), "w") as f:
        json.dump({"num_images": len(test_samples), "src_w": wh[0], "src_h": wh[1],
                   "pixel_format": "tdata[23:0]={B,G,R}", "golden_acc": acc}, f, indent=2)
    print(f"  Đã xuất test vector: {len(test_samples)} ảnh {wh[0]}x{wh[1]} | golden acc={acc:.4f} -> {out_dir}")
    return preds


def export_size_rom(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with open(os.path.join(out_dir, "size_rom.mem"), "w") as f:
        f.write("// size_sel -> {W[15:0], H[15:0], bw[7:0], bh[7:0], M[23:0]} (72 bit)\n")
        f.write(f"// out = (sum_block * M + 2^{RESIZE_SHIFT - 1}) >> {RESIZE_SHIFT}\n")
        for i, (w, h) in enumerate(SIZE_TABLE_WH):
            bw, bh, m = block_params(w, h, IMG_SIZE)
            f.write(f"{w:04x}{h:04x}{bw:02x}{bh:02x}{m:06x}\n")
            rows.append(dict(size_sel=i, width=w, height=h, bw=bw, bh=bh, M=m,
                             cols_used=bw * IMG_SIZE, rows_used=bh * IMG_SIZE))
    with open(os.path.join(out_dir, "size_rom.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"  Đã xuất ROM kích thước ({len(rows)} dòng) vào: {out_dir}")


def _write_rgb_mem(path, arr):
    flat = arr.reshape(-1, 3).astype(np.uint32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    with open(path, "w") as f:
        f.write("\n".join(f"{v:06x}" for v in packed.tolist()) + "\n")


def export_preprocess_vectors(test_samples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    picked = {}
    for path, _ in test_samples:
        with Image.open(path) as im:
            size = im.size
        if size[0] * size[1] > VEC_MAX_PIXELS:
            continue
        lst = picked.setdefault(size, [])
        if len(lst) < VEC_PER_SIZE:
            lst.append(path)
    meta = []
    for size, paths in picked.items():
        for j, path in enumerate(paths):
            arr = np.asarray(Image.open(path).convert("RGB"))
            resized = block_average_resize(arr, IMG_SIZE)
            enhanced = enhance_int(resized, ENHANCE_MODE)
            tag = f"vec_{size[0]}x{size[1]}_{j}"
            _write_rgb_mem(os.path.join(out_dir, f"{tag}_in.mem"), arr)
            _write_rgb_mem(os.path.join(out_dir, f"{tag}_resized.mem"), resized)
            _write_rgb_mem(os.path.join(out_dir, f"{tag}_enhanced.mem"), enhanced)
            bw, bh, m = block_params(size[0], size[1], IMG_SIZE)
            meta.append(dict(tag=tag, source=path, width=size[0], height=size[1],
                             size_sel=(SIZE_TABLE_WH.index(size) if size in SIZE_TABLE_WH else -1),
                             bw=bw, bh=bh, M=m, enhance_mode=ENHANCE_MODE))
    with open(os.path.join(out_dir, "vectors_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Đã xuất {len(meta)} bộ vector tiền xử lý vào: {out_dir}" if meta else
          f"  [CẢNH BÁO] Không có ảnh test nào có W*H <= {VEC_MAX_PIXELS}")


# ============================================================================
# MỘT (KIẾN TRÚC, SEED): FP32 -> cổng -> QAT -> INT8 -> đánh giá
# ============================================================================
def write_csv_row(path, record):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(record.keys()))
        w.writeheader(); w.writerow(record)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summary_key(s):
    acc = s["int8_val_acc_golden"] if GATE_ON_GOLDEN else s["int8_val_acc_fbgemm"]
    return (bool(s["meets_threshold_val"]),
            min(s["int8_val_min_precision"], s["int8_val_min_recall"]), acc, -s["num_params"])


def fp32_stage(arch, seed, data, h):
    classes, nc = data["classes"], data["num_classes"]
    name, layers = arch["name"], arch["layers"]
    sub = {}

    set_seed(seed)
    model = build_cnn_model(nc, layers, DROPOUT_P).to(device)
    n_params = count_hw_params(model)          # [1] tham số phần cứng (sau fold BN)
    assert n_params == arch["params"], f"đếm tham số sai: công thức {arch['params']} != thật {n_params}"
    n_train = len(data["train_ds"])
    print(f"[FP32][{h}] Params (HW, sau fold BN): {n_params:,} | MACs: {arch['macs'] / 1e6:.2f}M | BN={USE_BN} | "
          f"train mỗi epoch: {n_train} x {AUG_REPEAT_FP32} = {n_train * AUG_REPEAT_FP32} mẫu")
    tr_ld, va_ld, te_ld = data["fp32_train_ld"], data["gpu_val_ld"], data["gpu_test_ld"]
    cw = make_class_weights(data["train_labels"], classes, FP32_CLASS_WEIGHT_BOOST)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = make_optimizer(model, LR, WEIGHT_DECAY)                       # [3] AdamW
    scheduler = make_scheduler(optimizer, EPOCHS, warmup=WARMUP_EPOCHS)       # [3] warmup + cosine
    t0 = time.time()
    model, hist, best_val_acc = train_model(model, tr_ld, va_ld, criterion, optimizer, scheduler,
                                            EPOCHS, PATIENCE, select_by="val_acc",
                                            abort_epoch=ABORT_EPOCH, abort_min_val_acc=ABORT_MIN_VAL_ACC)
    fp32_time = time.time() - t0
    sub.update(num_params=int(n_params), aborted_early=bool(hist["aborted"]),
               fp32_train_time_min=round(fp32_time / 60, 2), epochs_run=len(hist["train_loss"]),
               fp32_final_train_acc=round(float(hist["train_acc"][-1]), 4))
    if hist["aborted"]:
        sub.update(fp32_val_acc=round(float(hist["val_acc"][-1]), 4), int8_skipped=True)
        del model, criterion, optimizer, scheduler; cleanup()
        return sub, None

    fp_val, fp_test = evaluate_model(model, va_ld, device), evaluate_model(model, te_ld, device)
    print(f"[FP32][{h}] VAL : {brief(fp_val, MIN_FP32_CLASS_METRIC_FOR_INT8)}")
    print(f"[FP32][{h}] TEST: {brief(fp_test, MIN_FP32_CLASS_METRIC_FOR_INT8)} | {fp32_time / 60:.1f} phút")
    print(f"[FP32][{h}] train_acc cuối={hist['train_acc'][-1]:.4f} vs val_acc={fp_val['accuracy']:.4f} "
          f"(train thấp = underfit, train cao & val thấp = overfit/nhiễu nhãn)")
    fp_ok_cls, fp_min_p, fp_min_r = check_int8_threshold(fp_val, MIN_FP32_CLASS_METRIC_FOR_INT8)
    fp_pass = fp_val["accuracy"] >= MIN_FP32_ACC_FOR_INT8 and fp_ok_cls
    sub.update(fp32_val_acc=round(float(fp_val["accuracy"]), 4), fp32_test_acc=round(float(fp_test["accuracy"]), 4),
               fp32_val_min_precision=round(fp_min_p, 4), fp32_val_min_recall=round(fp_min_r, 4),
               fp32_passed_gate=bool(fp_pass))
    del criterion, optimizer, scheduler

    if FP32_GATE and not fp_pass:
        print(f"[BỎ QUA INT8][{h}] FP32 không qua cổng trên VAL "
              f"(acc>={MIN_FP32_ACC_FOR_INT8} và precision/recall mọi lớp > {MIN_FP32_CLASS_METRIC_FOR_INT8})")
        sub["int8_skipped"] = True
        del model; cleanup()
        return sub, None

    fp32_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model = model.to("cpu"); cleanup()
    return sub, dict(model=model, fp32_state=fp32_state, fp32_hist=hist, fp_val=fp_val, fp_test=fp_test,
                     fp32_time=fp32_time)


def qat_stage(arch, seed, data, h, st, sub):
    classes = data["classes"]
    va_ld = data["gpu_val_ld"]
    model = st.pop("model")
    print(f"[QAT][{h}] backend={QUANT_BACKEND}")
    set_seed(seed)
    qat_model = prepare_qat_model(model)
    del model
    qcw = make_class_weights(data["train_labels"], classes, QAT_CLASS_WEIGHT_BOOST)
    qcriterion = nn.CrossEntropyLoss(weight=qcw)
    qopt = make_optimizer(qat_model, QAT_LR, WEIGHT_DECAY)
    qsched = make_scheduler(qopt, QAT_EPOCHS, warmup=0)
    t0 = time.time()
    qat_model, qhist, qat_val_acc = train_model(
        qat_model, data["qat_train_ld"], va_ld, qcriterion, qopt, qsched, QAT_EPOCHS, QAT_PATIENCE,
        select_by=QAT_SELECT_BY, is_qat=True,
        freeze_observer_epoch=max(1, int(QAT_EPOCHS * QAT_FREEZE_OBSERVER_RATIO)))
    qat_time = time.time() - t0
    del qcriterion, qopt, qsched

    int8_model = convert_qat_to_int8(qat_model)
    del qat_model; cleanup()
    check_no_bn_left(int8_model)

    cpu = torch.device("cpu")
    val_ld, test_ld = data["cpu_val_ld"], data["cpu_test_ld"]
    val_fb, val_gold = evaluate_model(int8_model, val_ld, cpu), evaluate_golden(int8_model, val_ld)
    test_fb, test_gold = evaluate_model(int8_model, test_ld, cpu), evaluate_golden(int8_model, test_ld)
    compare_golden_vs_pytorch(val_fb, val_gold, f"{h} VAL")
    compare_golden_vs_pytorch(test_fb, test_gold, f"{h} TEST")
    print(f"[VAL  golden][{h}] {brief(val_gold)}\n[TEST golden][{h}] {brief(test_gold)}")

    gate_res = val_gold if GATE_ON_GOLDEN else val_fb
    passed, min_p, min_r = check_int8_threshold(gate_res)
    tp_ok, tp, tr_ = check_int8_threshold(test_gold)
    sub.update(
        int8_skipped=False, qat_val_acc=round(float(qat_val_acc), 4),
        int8_val_acc_golden=round(float(val_gold["accuracy"]), 4), int8_val_acc_fbgemm=round(float(val_fb["accuracy"]), 4),
        int8_val_min_precision=round(min_p, 4), int8_val_min_recall=round(min_r, 4), meets_threshold_val=bool(passed),
        int8_test_acc_golden=round(float(test_gold["accuracy"]), 4), int8_test_acc_fbgemm=round(float(test_fb["accuracy"]), 4),
        int8_test_min_precision=round(tp, 4), int8_test_min_recall=round(tr_, 4), meets_threshold_test=bool(tp_ok),
        qat_time_min=round(qat_time / 60, 2))
    for i, c in enumerate(classes):
        sub[f"int8_precision_{c}"] = round(float(test_gold["precision"][i]), 4)
        sub[f"int8_recall_{c}"] = round(float(test_gold["recall"][i]), 4)
    art = dict(arch=arch, seed=seed, int8_model=int8_model, fp32_state=st["fp32_state"], fp32_hist=st["fp32_hist"],
               qat_hist=qhist, fp_val=st["fp_val"], fp_test=st["fp_test"], val_fb=val_fb, val_gold=val_gold,
               test_fb=test_fb, test_gold=test_gold, qat_val_acc=qat_val_acc, fp32_time=st["fp32_time"],
               qat_time=qat_time, meets_val=bool(passed), meets_test=bool(tp_ok))
    cleanup()
    return art


def _merge_hospital_results(summary, subs):
    for h, sub in subs.items():
        for k, v in sub.items():
            summary[f"{htag(h)}_{k}"] = v


def run_one(arch, seed, datas):
    summary = {"arch": arch["name"], "seed": seed, "params_est": arch["params"], "macs": arch["macs"],
               "num_params": int(arch["params"]), "aug_repeat_fp32": AUG_REPEAT_FP32,
               "aug_repeat_qat": AUG_REPEAT_QAT, "use_bn": USE_BN}
    subs, sts = {}, {}

    for h in HOSPITALS:
        print(f"\n--- [{h}] FP32 ---")
        sub, st = fp32_stage(arch, seed, datas[h], h)
        subs[h] = sub
        if st is None:
            _merge_hospital_results(summary, subs)
            summary.update(int8_skipped=True, aborted_early=bool(sub["aborted_early"]),
                           fp32_val_acc=sub["fp32_val_acc"], failed_hospital=h)
            print(f"[BỎ KIẾN TRÚC] {arch['name']}: {h} không qua FP32 -> bỏ cả kiến trúc "
                  f"(không train các dataset còn lại, không QAT)")
            sts.clear(); cleanup()
            return summary, None
        sts[h] = st

    arts = {}
    for h in HOSPITALS:
        print(f"\n--- [{h}] QAT -> INT8 ---")
        arts[h] = qat_stage(arch, seed, datas[h], h, sts[h], subs[h])
        cleanup()
    _merge_hospital_results(summary, subs)

    def mean(k):
        return round(float(np.mean([subs[h][k] for h in HOSPITALS])), 4)

    def low(k):
        return round(float(min(subs[h][k] for h in HOSPITALS)), 4)

    summary.update(
        int8_skipped=False, aborted_early=False,
        epochs_run=int(sum(subs[h]["epochs_run"] for h in HOSPITALS)),
        fp32_train_time_min=round(sum(subs[h]["fp32_train_time_min"] for h in HOSPITALS), 2),
        qat_time_min=round(sum(subs[h]["qat_time_min"] for h in HOSPITALS), 2),
        fp32_val_acc=mean("fp32_val_acc"), fp32_test_acc=mean("fp32_test_acc"), qat_val_acc=mean("qat_val_acc"),
        int8_val_acc_golden=mean("int8_val_acc_golden"), int8_val_acc_fbgemm=mean("int8_val_acc_fbgemm"),
        int8_val_min_precision=low("int8_val_min_precision"), int8_val_min_recall=low("int8_val_min_recall"),
        meets_threshold_val=all(bool(subs[h]["meets_threshold_val"]) for h in HOSPITALS),
        int8_test_acc_golden=mean("int8_test_acc_golden"), int8_test_acc_fbgemm=mean("int8_test_acc_fbgemm"),
        int8_test_min_precision=low("int8_test_min_precision"), int8_test_min_recall=low("int8_test_min_recall"),
        meets_threshold_test=all(bool(subs[h]["meets_threshold_test"]) for h in HOSPITALS))
    return summary, arts


# ============================================================================
# XUẤT MÔ HÌNH TỐT NHẤT -> BEST/
# ============================================================================
def recipe_desc():
    return {"use_bn": USE_BN, "dropout_p": DROPOUT_P, "optimizer": "AdamW (no decay on bias/BN)",
            "lr": LR, "weight_decay": WEIGHT_DECAY, "scheduler": f"warmup {WARMUP_EPOCHS} + cosine",
            "epochs_cfg": EPOCHS, "augment": {"flip_p": AUG_FLIP_P, "rotate_deg": AUG_ROTATE_DEG,
                                              "scale": AUG_SCALE, "jitter": AUG_JITTER, "blur_p": AUG_BLUR_P}}


def export_best_one(art, data, h, best_dir):
    classes, nc = data["classes"], data["num_classes"]
    arch, seed, int8_model = art["arch"], art["seed"], art["int8_model"]
    layers, name = arch["layers"], arch["name"]
    P = dict(FP32_DIR=os.path.join(best_dir, "FP32"), INT8_DIR=os.path.join(best_dir, "INT8"))
    P["MEMORY_DIR"] = os.path.join(P["INT8_DIR"], "Memory")
    P["PREPROCESS_DIR"] = os.path.join(P["INT8_DIR"], "Preprocess")
    P["RTL_DIR"] = os.path.join(P["INT8_DIR"], "RTL")
    P["FIG_DIR"] = os.path.join(best_dir, "Figures")
    for d in P.values():
        os.makedirs(d, exist_ok=True)
    print("\n" + "-" * 70 + f"\n[{h}] {name} seed {seed} | {arch['params']:,} tham số | "
          f"đạt ngưỡng VAL: {art['meets_val']}\n" + "-" * 70)

    # ---- FP32 ----
    fp_val, fp_test = art["fp_val"], art["fp_test"]
    print_per_class(fp_val, classes, f"FP32 VAL, {name}", MIN_FP32_CLASS_METRIC_FOR_INT8)
    print_per_class(fp_test, classes, f"FP32 TEST, {name}", MIN_FP32_CLASS_METRIC_FOR_INT8)
    export_misclassified(fp_val, data["val_samples"], classes, os.path.join(P["FP32_DIR"], "misclassified_val.csv"),
                         f"FP32 VAL, {name}", MIN_FP32_CLASS_METRIC_FOR_INT8)
    export_misclassified(fp_test, data["test_samples"], classes, os.path.join(P["FP32_DIR"], "misclassified_test.csv"),
                         f"FP32 TEST, {name}", MIN_FP32_CLASS_METRIC_FOR_INT8)
    torch.save(art["fp32_state"], os.path.join(P["FP32_DIR"], "vgg9_fp32_state_dict.pth"))
    fp32_record = {"model": f"{name}-FP32", "seed": seed, "layers": arch["name"], "img_size": IMG_SIZE,
                   "aug_repeat_fp32": AUG_REPEAT_FP32, "epochs_cfg": EPOCHS, "patience_cfg": PATIENCE,
                   "num_params": int(arch["params"]), "macs": int(arch["macs"]),
                   "val_acc": float(fp_val["accuracy"]), "test_acc": float(fp_test["accuracy"]),
                   "test_balanced_acc": float(fp_test["balanced_accuracy"]),
                   "test_f1_macro": float(np.mean(fp_test["f1"])),
                   "epochs_run": len(art["fp32_hist"]["train_loss"]), "train_time_sec": round(art["fp32_time"], 1)}
    for i, c in enumerate(classes):
        fp32_record[f"precision_{c}"] = float(fp_test["precision"][i])
        fp32_record[f"recall_{c}"] = float(fp_test["recall"][i])
    with open(os.path.join(P["FP32_DIR"], "vgg9_fp32_config.json"), "w", encoding="utf-8") as f:
        json.dump({"num_classes": nc, "classes": classes, "img_size": IMG_SIZE, "preprocess": preprocess_desc(),
                   "dropout_p": DROPOUT_P, "use_bn": USE_BN, "recipe": recipe_desc(),
                   "layers": [list(l) for l in layers], "seed": seed,
                   "result": fp32_record}, f, indent=2, ensure_ascii=False)
    write_csv_row(os.path.join(P["FP32_DIR"], "vgg9_fp32_result.csv"), fp32_record)

    # ---- INT8 ----
    val_gold, test_gold, val_fb, test_fb = art["val_gold"], art["test_gold"], art["val_fb"], art["test_fb"]
    print_report(val_gold, classes, f"(INT8 golden VAL, {name})")
    print_per_class(val_gold, classes, f"INT8 golden VAL, {name}")
    print_per_class(test_gold, classes, f"INT8 golden TEST, {name}")
    print_report(test_gold, classes, f"({name} INT8 GOLDEN, TEST, seed {seed})")
    export_misclassified(val_gold, data["val_samples"], classes, os.path.join(P["INT8_DIR"], "misclassified_val.csv"),
                         f"INT8 golden VAL, {name}")
    export_misclassified(test_gold, data["test_samples"], classes, os.path.join(P["INT8_DIR"], "misclassified_test.csv"),
                         f"INT8 golden TEST, {name}")
    check_zero_points(int8_model)
    check_no_bn_left(int8_model)
    dump_quantization_info(int8_model)
    print("\n[HW] Xuất .mem / LUT / Package_VGG9.sv...")
    total_mb, hw_mb = export_int8_mem(int8_model, P["MEMORY_DIR"])
    export_input_lut(int8_model, P["PREPROCESS_DIR"])
    export_rtl_package(int8_model, P["RTL_DIR"], IMG_SIZE, data["src_wh"])
    torch.save(int8_model.state_dict(), os.path.join(P["INT8_DIR"], "vgg9_int8_state_dict.pth"))
    torch.save(int8_model, os.path.join(P["INT8_DIR"], "vgg9_int8_quantized.pth"))
    int8_record = {"model": f"{name}-INT8-QAT", "seed": seed, "layers": name, "quant_backend": QUANT_BACKEND,
                   "num_params": int(arch["params"]), "macs": int(arch["macs"]),
                   "qat_epochs_run": len(art["qat_hist"]["train_loss"]), "qat_val_acc": float(art["qat_val_acc"]),
                   "val_acc_golden": float(val_gold["accuracy"]), "val_acc_fbgemm": float(val_fb["accuracy"]),
                   "test_acc_golden": float(test_gold["accuracy"]), "test_acc_fbgemm": float(test_fb["accuracy"]),
                   "test_f1_macro_golden": float(np.mean(test_gold["f1"])),
                   "meets_threshold_val": bool(art["meets_val"]), "qat_train_time_sec": round(art["qat_time"], 1),
                   "mem_file_size_MB": round(total_mb, 6), "hw_memory_size_MB": round(hw_mb, 6)}
    with open(os.path.join(P["INT8_DIR"], "vgg9_int8_config.json"), "w", encoding="utf-8") as f:
        json.dump({"num_classes": nc, "classes": classes, "img_size": IMG_SIZE, "preprocess": preprocess_desc(),
                   "layers": [list(l) for l in layers], "quant_backend": QUANT_BACKEND, "enhance_mode": ENHANCE_MODE,
                   "use_bn": USE_BN, "bn_folded": True, "recipe": recipe_desc(),
                   "activation_qscheme": "hidden: per_tensor_affine (zp=0) | input & logits: per_tensor_symmetric (zp=127)",
                   "rtl_requant_mult_bits": RTL_REQUANT_MULT_BITS, "seed": seed, "result": int8_record},
                  f, indent=2, ensure_ascii=False)
    write_csv_row(os.path.join(P["INT8_DIR"], "vgg9_int8_result.csv"), int8_record)

    # ---- hình (vi + en) ----
    plot_training_curves(art["fp32_hist"], os.path.join(P["FIG_DIR"], "fp32_training_curves.png"), f"{name} FP32")
    plot_training_curves(art["qat_hist"], os.path.join(P["FIG_DIR"], "int8_qat_training_curves.png"), f"{name} QAT")
    export_best_figures(dict(fp32_hist=art["fp32_hist"], fp32_test=fp_test, int8_test=test_gold), classes, P["FIG_DIR"])

    # ---- kiểm tra LUT + test vector RTL ----
    print("\n[RTL] Golden đường u8 + test vector...")
    g = extract_golden_layers(int8_model)
    lut = build_input_lut(g["in_scale"], g["in_zp"])
    lg_rq, _ = golden_u8_predict(g, lut, [im for im, _ in data["test_cache"]])
    same = float((np.argmax(lg_rq, axis=1) == test_gold["preds"]).mean())
    print(f"  [KIỂM TRA LUT] đường u8 vs đường float (golden, toàn bộ TEST): trùng {same:.4f}")
    if same < 1.0:
        print("  *** Không trùng 100%: LUT/đường float lệch nhau -> sửa trước khi chạy RTL. ***")
    cnt, idx = Counter(), []
    for i, (_, l) in enumerate(data["test_samples"]):
        if cnt[l] < RTL_TESTVEC_PER_CLASS:
            idx.append(i); cnt[l] += 1
    export_rtl_testvectors(int8_model, [data["test_samples"][i] for i in idx],
                           [data["test_cache"][i] for i in idx], os.path.join(P["INT8_DIR"], "TestVectors"))


def architecture_signature(int8_model):
    sig = []
    for d in extract_golden_layers(int8_model)["layers"]:
        if d["kind"] == "pool":
            sig.append(("pool",))
        else:
            sig.append((d["kind"], tuple(int(x) for x in d["w"].shape), int(d.get("groups", 1))))
    return sig


def load_hospital_test_union():
    seen, rows = set(), []
    for h in HOSPITALS:
        path = os.path.join(HOSPITAL_ROOT, h, "manifest.csv")
        if not os.path.isfile(path):
            print(f"  [CẢNH BÁO] Không thấy {path}")
            return None
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] != "test" or r["source_path"] in seen:
                    continue
                seen.add(r["source_path"])
                rows.append((os.path.join(HOSPITAL_ROOT, h, r["file"]), r["source_class"]))
    return rows


def evaluate_cascade(arts, datas, out_dir):
    rows = load_hospital_test_union()
    if not rows:
        return None
    os.makedirs(out_dir, exist_ok=True)
    diseases = sorted({c for h in HOSPITALS for c in datas[h]["classes"] if c != UNKNOWN_NAME})
    final_classes = diseases + [UNKNOWN_NAME]
    truth = [c if c in diseases else UNKNOWN_NAME for _, c in rows]
    print(f"\n[CASCADE] TEST gộp: {len(rows)} ảnh | " + ", ".join(f"{c}={truth.count(c)}" for c in final_classes))

    cache = preprocess_and_cache([(p, 0) for p, _ in rows], "Cascade TEST")
    imgs = [im for im, _ in cache]
    preds = {}
    for h in HOSPITALS:
        g = extract_golden_layers(arts[h]["int8_model"])
        lut = build_input_lut(g["in_scale"], g["in_zp"])
        lg, _ = golden_u8_predict(g, lut, imgs)
        preds[h] = np.argmax(lg, axis=1)

    final, route = [], []
    for i in range(len(imgs)):
        out, via = UNKNOWN_NAME, "none"
        for h in HOSPITALS:
            c = datas[h]["classes"][int(preds[h][i])]
            if c != UNKNOWN_NAME:
                out, via = c, h
                break
        final.append(out); route.append(via)

    n = len(final_classes)
    y_true = np.array([final_classes.index(t) for t in truth])
    y_pred = np.array([final_classes.index(p) for p in final])
    acc, bacc = accuracy_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred)
    labels = np.arange(n)
    print(f"[CASCADE] accuracy={acc:.4f} | balanced_accuracy={bacc:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, target_names=final_classes, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix cascade (hàng = thật, cột = dự đoán):")
    print(" " * 22 + " ".join(f"{c[:6]:>6}" for c in final_classes))
    for i, c in enumerate(final_classes):
        print(f"{c:<22}" + " ".join(f"{v:>6}" for v in cm[i]))
    rc = Counter(route)
    print("Số ảnh được quyết định bởi: " + " | ".join(f"{k}={v}" for k, v in sorted(rc.items())))

    with open(os.path.join(out_dir, "cascade_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_class", "final_pred", "decided_by"]
                   + [f"{htag(h)}_pred" for h in HOSPITALS])
        for i, (p, _) in enumerate(rows):
            w.writerow([p, truth[i], final[i], route[i]]
                       + [datas[h]["classes"][int(preds[h][i])] for h in HOSPITALS])
    p_cls = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r_cls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    with open(os.path.join(out_dir, "cascade_report.json"), "w", encoding="utf-8") as f:
        json.dump({"num_images": len(rows), "accuracy": float(acc), "balanced_accuracy": float(bacc),
                   "classes": final_classes, "decided_by": dict(rc),
                   "precision": {c: float(v) for c, v in zip(final_classes, p_cls)},
                   "recall": {c: float(v) for c, v in zip(final_classes, r_cls)},
                   "confusion_matrix": cm.tolist(),
                   "note": "Diabetic Retinopathy được tính là Unknown; chỉ gồm phần DR nằm trong TEST của 2 dataset"},
                  f, indent=2, ensure_ascii=False)
    print(f"  Đã lưu cascade vào: {out_dir}")
    return acc


def export_best(arts, summary, datas):
    h0 = HOSPITALS[0]
    arch, seed = arts[h0]["arch"], arts[h0]["seed"]
    layers, name = arch["layers"], arch["name"]
    best_dir = os.path.join(MODEL_DIR, "BEST")
    if os.path.isdir(best_dir):
        shutil.rmtree(best_dir)
    os.makedirs(best_dir, exist_ok=True)
    key = summary_key(summary)
    print("\n" + "=" * 70 + f"\nKIẾN TRÚC TỐT NHẤT MỚI: {name} seed {seed} | {arch['params']:,} tham số | "
          f"đạt ngưỡng VAL (cả 2): {key[0]} | minP/R={key[1]:.4f} | VAL acc TB={key[2]:.4f}\n" + "=" * 70)

    for h in HOSPITALS:
        export_best_one(arts[h], datas[h], h, os.path.join(best_dir, h))

    sigs = {h: architecture_signature(arts[h]["int8_model"]) for h in HOSPITALS}
    same = all(sigs[h] == sigs[h0] for h in HOSPITALS)
    print(f"\n[KIẾN TRÚC] {len(HOSPITALS)} CNN: " + ("CÙNG kiến trúc (cấu trúc lớp/kênh/groups/pool trùng khớp)."
                                                    if same else "*** KHÁC NHAU, kiểm tra lại code! ***"))
    with open(os.path.join(best_dir, "architecture_check.json"), "w", encoding="utf-8") as f:
        json.dump({"same_architecture": bool(same), "arch": name, "layers": [list(l) for l in layers],
                   "signature": {h: [list(s) for s in sigs[h]] for h in HOSPITALS}}, f, indent=2)

    try:
        evaluate_cascade(arts, datas, os.path.join(best_dir, "Cascade"))
    except Exception as e:
        traceback.print_exc()
        print(f"[CẢNH BÁO] Không đánh giá được cascade: {type(e).__name__}: {e}")

    with open(os.path.join(best_dir, "best_seed.json"), "w", encoding="utf-8") as f:
        json.dump({"arch": name, "layers": [list(l) for l in layers], "seed": seed, "params": arch["params"],
                   "macs": arch["macs"], "hospitals": list(HOSPITALS), "use_bn": USE_BN,
                   "selected_by": "VAL " + ("golden" if GATE_ON_GOLDEN else "fbgemm")
                                  + " (min precision/recall: thấp nhất của 2 dataset; acc: trung bình)",
                   "key": [bool(key[0]), key[1], key[2], key[3]]}, f, indent=2)
    print(f"\nĐã lưu bản tốt nhất vào: {best_dir}")


# ============================================================================
# TỔNG HỢP / XẾP HẠNG / RESUME
# ============================================================================
def save_summaries(summaries):
    os.makedirs(MODEL_DIR, exist_ok=True)
    fields = []
    for s in summaries:
        fields += [k for k in s if k not in fields]
    with open(os.path.join(MODEL_DIR, "seeds_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summaries)
    tmp = os.path.join(MODEL_DIR, "seeds_summary.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(MODEL_DIR, "seeds_summary.json"))


def load_summaries():
    path = os.path.join(MODEL_DIR, "seeds_summary.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        old = json.load(f)
    ok = [s for s in old if "error" not in s]
    print(f"[RESUME] Đã có {len(ok)} kết quả trong {path}" + (f" (bỏ {len(old) - len(ok)} bản ghi lỗi để chạy lại)"
                                                           if len(ok) != len(old) else ""))
    return ok


def load_best_key():
    path = os.path.join(MODEL_DIR, "BEST", "best_seed.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        b = json.load(f)
    print(f"[RESUME] BEST hiện tại: {b['arch']} seed {b['seed']}")
    return tuple(b["key"])


def print_ranking(summaries):
    done = [s for s in summaries if s.get("int8_skipped") is False]
    ranked = sorted(done, key=summary_key, reverse=True)
    n_abort = sum(1 for s in summaries if s.get("aborted_early"))
    n_gate = sum(1 for s in summaries if s.get("int8_skipped") and not s.get("aborted_early"))
    vcols = [f"{htag(h)}_int8_val_acc_golden" for h in HOSPITALS]
    tcols = [f"{htag(h)}_int8_test_acc_golden" for h in HOSPITALS]
    print("\n" + "=" * 132 + f"\nXẾP HẠNG (theo VAL {'golden' if GATE_ON_GOLDEN else 'fbgemm'}, gộp {len(HOSPITALS)} dataset) | "
          f"đã chạy {len(summaries)} | INT8 xong {len(done)} | dừng sớm {n_abort} | không qua cổng FP32 {n_gate} | "
          f"đạt ngưỡng VAL (cả 2) {sum(1 for s in done if s['meets_threshold_val'])}\n" + "=" * 132)
    print(f"{'Hạng':<5}{'Kiến trúc':<36}{'Seed':>5}{'Params':>8}{'MMAC':>7}{'FP32 val':>9}{'INT8 val':>9}"
          + "".join(f"{'val' + htag(h):>7}" for h in HOSPITALS)
          + f"{'minP':>7}{'minR':>7}{'INT8 test':>10}{'Đạt':>6}")
    for r, s in enumerate(ranked[:TOP_PRINT], 1):
        print(f"{r:<5}{s['arch']:<36}{s['seed']:>5}{s['num_params']:>8,}{s['macs'] / 1e6:>7.2f}{s['fp32_val_acc']:>9.4f}"
              f"{s['int8_val_acc_golden']:>9.4f}" + "".join(f"{s[c]:>7.4f}" for c in vcols)
              + f"{s['int8_val_min_precision']:>7.4f}{s['int8_val_min_recall']:>7.4f}"
              f"{s['int8_test_acc_golden']:>10.4f}{str(s['meets_threshold_val']):>6}")
    if ranked:
        with open(os.path.join(MODEL_DIR, "ranking.csv"), "w", newline="", encoding="utf-8") as f:
            cols = ["rank", "arch", "seed", "num_params", "macs", "fp32_val_acc", "fp32_test_acc", "int8_val_acc_golden"] \
                   + vcols + ["int8_val_min_precision", "int8_val_min_recall", "meets_threshold_val",
                              "int8_test_acc_golden"] + tcols \
                   + ["int8_test_min_precision", "int8_test_min_recall", "meets_threshold_test"]
            w = csv.writer(f); w.writerow(cols)
            for r, s in enumerate(ranked, 1):
                w.writerow([r] + [s[c] for c in cols[1:]])
        print(f"\nRanking đầy đủ: {os.path.join(MODEL_DIR, 'ranking.csv')}")


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for x in self.streams: x.write(s); x.flush()
    def flush(self):
        for x in self.streams: x.flush()


def main():
    for d in (OUTPUT_DIR, FIG_DIR, LOG_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{MODEL_NAME}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, log_f), _Tee(sys.__stderr__, log_f)
    print(f"[LOG] {log_path}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | "
          f"PyTorch {torch.__version__} | seeds={TRAIN_SEEDS} | kênh={CHANNEL_CHOICES} groups={GROUP_CHOICES} "
          f"depth={DEPTH_RANGE} | không giảm={ONLY_NON_DECREASING} lớp cuối>={MIN_LAST_CHANNELS} "
          f"| params [{MIN_PARAMS:,}, {MAX_PARAMS:,}) | enhance={ENHANCE_MODE}")
    print(f"Dataset: {' -> '.join(HOSPITALS)} (CNN 1 -> nếu Unknown -> CNN 2) | CÙNG 1 kiến trúc cho cả hai")
    print(f"Recipe: BN={USE_BN} dropout={DROPOUT_P} AdamW lr={LR} wd={WEIGHT_DECAY} warmup={WARMUP_EPOCHS}+cosine | "
          f"aug flip={AUG_FLIP_P} rot={AUG_ROTATE_DEG} scale={AUG_SCALE} jitter={AUG_JITTER} blur={AUG_BLUR_P}")
    print(f"FP32 x{AUG_REPEAT_FP32} (EPOCHS={EPOCHS}, PATIENCE={PATIENCE}) | "
          f"QAT x{AUG_REPEAT_QAT} (QAT_EPOCHS={QAT_EPOCHS}, QAT_PATIENCE={QAT_PATIENCE}) | "
          f"dừng sớm: {'tắt' if ABORT_EPOCH is None else f'epoch {ABORT_EPOCH} val_acc<{ABORT_MIN_VAL_ACC}'}")

    datas = {h: prepare_hospital_data(h) for h in HOSPITALS}
    num_classes = {datas[h]["num_classes"] for h in HOSPITALS}
    if len(num_classes) != 1:
        sys.exit(f"[LỖI] Các dataset phải có cùng số lớp để dùng chung 1 kiến trúc: "
                 f"{ {h: datas[h]['num_classes'] for h in HOSPITALS} }")
    for h in HOSPITALS:
        make_shared_loaders(datas[h])

    print("\n[HW] Xuất ROM kích thước + vector tiền xử lý (1 lần, không phụ thuộc kiến trúc / dataset)...")
    pre_dir = os.path.join(OUTPUT_DIR, "preprocess_hw")
    export_size_rom(pre_dir)
    export_preprocess_vectors(datas[HOSPITALS[0]]["test_samples"], pre_dir)

    archs = enumerate_architectures(num_classes.pop())
    summaries = load_summaries()
    best_key = load_best_key()
    done = {(s["arch"], s["seed"]) for s in summaries}
    todo = [(a, sd) for a in archs for sd in TRAIN_SEEDS if (a["name"], sd) not in done]
    print(f"[PLAN] {len(archs)} kiến trúc x {len(TRAIN_SEEDS)} seed x {len(HOSPITALS)} dataset | "
          f"đã xong {len(archs) * len(TRAIN_SEEDS) - len(todo)} | còn {len(todo)}")

    t_start, n_run = time.time(), 0
    try:
        for k, (arch, seed) in enumerate(todo, 1):
            elapsed = time.time() - t_start
            eta = f" | ETA ~{elapsed / n_run * (len(todo) - n_run) / 3600:.1f} giờ" if n_run else ""
            print("\n" + "#" * 78 + f"\n# [{k}/{len(todo)}] {arch['name']} | {arch['params']:,} tham số | seed {seed}{eta}\n"
                  + "#" * 78)
            try:
                summary, arts = run_one(arch, seed, datas)
                if arts is not None and (best_key is None or summary_key(summary) > best_key):
                    try:
                        export_best(arts, summary, datas)
                        best_key = summary_key(summary)
                    except Exception as e:
                        traceback.print_exc()
                        print(f"[LỖI XUẤT BEST] {arch['name']}: {type(e).__name__}: {e} (BEST có thể chưa đầy đủ)")
                summaries.append(summary)
                if arts is not None:
                    per_h = ", ".join(f"{htag(h)}={summary[htag(h) + '_int8_val_acc_golden']:.4f}" for h in HOSPITALS)
                    print(f"[KẾT QUẢ] {arch['name']} | INT8 val TB={summary['int8_val_acc_golden']:.4f} ({per_h}) "
                          f"test TB={summary['int8_test_acc_golden']:.4f} | đạt ngưỡng VAL (cả 2): {summary['meets_threshold_val']}")
                del arts
            except Exception as e:
                traceback.print_exc()
                print(f"\n[LỖI] {arch['name']} seed {seed}: {type(e).__name__}: {e}")
                summaries.append({"arch": arch["name"], "seed": seed, "error": str(e)})
            save_summaries(summaries)
            n_run += 1
            cleanup()
    except KeyboardInterrupt:
        print("\n[DỪNG] Bạn ngắt tay: xếp hạng phần đã chạy (chạy lại script để làm tiếp).")
    print_ranking([s for s in summaries if "error" not in s])


if __name__ == "__main__":
    main()
