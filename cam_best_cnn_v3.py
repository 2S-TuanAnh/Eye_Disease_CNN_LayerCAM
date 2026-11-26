"""
cam_best_cnn_v3.py — LayerCAM khối 3 (FP32 + INT8) cho CNN BEST, skip Normal/Unknown.

Sửa 13.1 (bản này): max-pool argmax lấy trên acc int32 TRƯỚC requant (như phần cứng),
không lấy trên giá trị sau requant. --calib-n default 128 theo spec.

Mặc định:
  - LayerCAM FP32 LUÔN ở khối 3 (hằng số LCAM_LAYER = 3).
  - --skip-classes Unknown Normal  (bỏ ảnh thuộc 2 lớp này, theo tên thư mục cha).
  - --gcpp-layer 3 (Grad-CAM++ chỉ để tham chiếu).
  - --hw-block 3 (INT8 LayerCAM cùng khối 3).
  - --calib-n 128 (số ảnh hiệu chuẩn M0/shift).

Ví dụ:
  python cam_best_cnn_v3.py --images Dataset_Hospital/Hospital_A/test \\
      --calib-images Dataset_Hospital/Hospital_A/train --mode A --out cam_out/A \\
      --hw --export-hw --eval --per-folder 30
"""
import os, sys, json, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CẤU HÌNH
# ============================================================================
BEST_ROOT_DEFAULT = "/home/khaithien/Work/6.DA/AAZ1/outputs_arch_search_AB_BN/CNN_SEARCH_AB/BEST"
HOSPITALS = ("Hospital_A", "Hospital_B")
UNKNOWN_NAME = "Unknown"
IMG_SIZE = 64
RESIZE_SHIFT = 24
RTL_REQUANT_MULT_BITS = 31
IMAGENET_MEAN_L = [0.485, 0.456, 0.406]
IMAGENET_STD_L = [0.229, 0.224, 0.225]
IMAGENET_MEAN = torch.tensor(IMAGENET_MEAN_L).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor(IMAGENET_STD_L).view(1, 3, 1, 1)
IMG_EXTS = (".png", ".jpg", ".jpeg")
CLIP_Q = 0.99
GRAD_MAX = 127

LCAM_LAYER = 3          # LayerCAM FP32/INT8 cố định khối 3


# ============================================================================
# TIỀN XỬ LÝ
# ============================================================================
def block_average_resize(arr, out_size):
    h, w, c = arr.shape
    if w < out_size or h < out_size:
        raise ValueError(f"Ảnh {w}x{h} nhỏ hơn {out_size}x{out_size}")
    bw, bh = w // out_size, h // out_size
    m = int(round((1 << RESIZE_SHIFT) / (bw * bh)))
    crop = arr[:bh * out_size, :bw * out_size]
    s = crop.reshape(out_size, bh, out_size, bw, c).sum(axis=(1, 3), dtype=np.uint64)
    out = (s * np.uint64(m) + np.uint64(1 << (RESIZE_SHIFT - 1))) >> np.uint64(RESIZE_SHIFT)
    return np.minimum(out, 255).astype(np.uint8)


def hist_equalize_int(arr):
    h, w, c = arr.shape
    n = h * w
    shift = n.bit_length() - 1
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
    return arr


def fov_mask(raw, erode=2, thr=12):
    from scipy import ndimage
    bh, bw = raw.shape[0] // IMG_SIZE, raw.shape[1] // IMG_SIZE
    g = raw[:bh * IMG_SIZE, :bw * IMG_SIZE].max(axis=2)
    g = g.reshape(IMG_SIZE, bh, IMG_SIZE, bw).mean(axis=(1, 3))
    m = ndimage.binary_fill_holes(g > thr)
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
    if erode > 0:
        m2 = ndimage.binary_erosion(m, iterations=erode)
        m = m2 if m2.any() else m
    return m if m.any() else np.ones((IMG_SIZE, IMG_SIZE), bool)


def load_raw(path):
    return np.asarray(Image.open(path).convert("RGB"))


def make_input(raw, enhance_mode):
    small = enhance_int(block_average_resize(raw, IMG_SIZE), enhance_mode)
    x = torch.from_numpy(small).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD, small


def make_input_from_small(small):
    x = torch.from_numpy(small).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD, small


# ============================================================================
# MODEL FP32
# ============================================================================
class CNNCustom(nn.Module):
    def __init__(self, layers, num_classes, use_bn=True, dropout_p=0.0):
        super().__init__()
        mods, c_in = [], 3
        for c, g, pool in layers:
            mods.append(nn.Conv2d(c_in, c, 3, padding=1, groups=g, bias=not use_bn))
            if use_bn:
                mods.append(nn.BatchNorm2d(c))
            mods.append(nn.ReLU(inplace=False))
            if pool:
                mods.append(nn.MaxPool2d(2, 2))
            c_in = c
        self.features = nn.Sequential(*mods)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout_p), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


def load_best_model(best_root, hospital, device):
    fp32_dir = os.path.join(best_root, hospital, "FP32")
    cfg_path = os.path.join(fp32_dir, "vgg9_fp32_config.json")
    sd_path = os.path.join(fp32_dir, "vgg9_fp32_state_dict.pth")
    for p in (cfg_path, sd_path):
        if not os.path.isfile(p):
            sys.exit(f"[LỖI] Không thấy {p}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    layers = [tuple(l) for l in cfg["layers"]]
    model = CNNCustom(layers, cfg["num_classes"], cfg.get("use_bn", True), cfg.get("dropout_p", 0.0))
    try:
        sd = torch.load(sd_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(sd_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()
    enhance = (cfg.get("preprocess") or {}).get("enhance", "none")
    return dict(model=model, classes=cfg["classes"], layers=layers, enhance=enhance, name=hospital)


# ============================================================================
# CAM FP32
# ============================================================================
class CAMExtractor:
    def __init__(self, model, block_ids):
        self.model, self.acts, self.grads, self.handles = model, {}, {}, []
        relus = [m for m in model.features if isinstance(m, nn.ReLU)]
        for k in block_ids:
            if not 1 <= k <= len(relus):
                raise ValueError(f"Khối conv {k} không tồn tại (mạng có {len(relus)} khối)")
            self.handles.append(relus[k - 1].register_forward_hook(self._make_hook(k)))

    def _make_hook(self, k):
        def hook(_mod, _inp, out):
            self.acts[k] = out.detach()
            out.register_hook(lambda g: self.grads.__setitem__(k, g.detach()))
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

    def run(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(1)
            elif isinstance(class_idx, int):
                class_idx = torch.full((x.size(0),), class_idx, device=x.device, dtype=torch.long)
            logits.gather(1, class_idx.view(-1, 1)).sum().backward()
        return logits.detach(), class_idx


def grad_cam_pp(A, G):
    g2, g3 = G ** 2, (G ** 3)
    sum_a = A.sum(dim=(2, 3), keepdim=True)
    denom = 2 * g2 + sum_a * g3
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    w = ((g2 / denom) * F.relu(G)).sum(dim=(2, 3), keepdim=True)
    return F.relu((w * A).sum(dim=1, keepdim=True))


def layer_cam(A, G):
    return F.relu((F.relu(G) * A).sum(dim=1, keepdim=True))


def upsample(cam):
    return F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]


def norm_fov(cam, mask):
    cam = cam * mask
    v = cam[mask]
    if v.numel() == 0:
        return cam
    lo, hi = v.min(), torch.quantile(v, CLIP_Q)
    return ((cam - lo) / (hi - lo + 1e-8)).clamp(0, 1) * mask


def compute_maps(model, x, mask, blocks, class_idx=None):
    maps = {}
    with CAMExtractor(model, blocks) as ext:
        logits, cls = ext.run(x, class_idx)
        for k in blocks:
            A, G = ext.acts[k], ext.grads[k]
            maps[("gc", k)] = norm_fov(upsample(grad_cam_pp(A, G)), mask)
            maps[("lc", k)] = norm_fov(upsample(layer_cam(A, G)), mask)
    return F.softmax(logits, dim=1)[0], int(cls[0]), maps


# ============================================================================
# INT8 (FPGA golden) — SỬA 13.1: argmax pool trên acc int32
# ============================================================================
def fixed_point_multiplier(m, bits=RTL_REQUANT_MULT_BITS):
    assert m > 0
    n = bits - 1 - int(np.floor(np.log2(m)))
    m0 = int(round(m * (2 ** n)))
    if m0 >= (1 << bits):
        m0 >>= 1
        n -= 1
    assert n >= 1
    return m0, n


def requant_int(acc, m0, shift, zp, lo, hi):
    y = (acc * int(m0) + (1 << (int(shift) - 1))) >> int(shift)
    return torch.clamp(y + int(zp), lo, hi)


def read_mem(path, width, signed):
    bits = 4 * width
    vals = []
    with open(path) as f:
        for ln in f:
            ln = ln.split("//")[0].strip()
            if ln:
                vals.append(int(ln, 16))
    a = np.asarray(vals, dtype=np.int64)
    if signed:
        a = np.where(a >= (1 << (bits - 1)), a - (1 << bits), a)
    return a


def write_hex(path, values, width=2):
    mask = (1 << (4 * width)) - 1
    with open(path, "w") as f:
        for v in np.asarray(values).reshape(-1).tolist():
            f.write(f"{int(v) & mask:0{width}x}\n")


def build_input_lut(scale, zero_point):
    lut = np.empty((3, 256), dtype=np.uint8)
    p = np.arange(256, dtype=np.float32) / np.float32(255.0)
    for c in range(3):
        x = ((p - np.float32(IMAGENET_MEAN_L[c])) / np.float32(IMAGENET_STD_L[c])).astype(np.float32)
        q = torch.quantize_per_tensor(torch.from_numpy(x), float(scale), int(zero_point), torch.quint8)
        lut[c] = torch.int_repr(q).numpy()
    return lut


def load_int8_hw(best_root, hospital, pools):
    int8_dir = os.path.join(best_root, hospital, "INT8")
    mem_dir = os.path.join(int8_dir, "Memory")
    qp_path = os.path.join(mem_dir, "quant_params.json")
    if not os.path.isfile(qp_path):
        sys.exit(f"[LỖI] Không thấy {qp_path}")
    with open(qp_path, encoding="utf-8") as f:
        qp = json.load(f)
    in_scale, in_zp = float(qp["input_quant"]["scale"]), int(qp["input_quant"]["zero_point"])
    convs, fc, prev_c = [], None, 3
    for name, q in qp.items():
        if name == "input_quant":
            continue
        safe = name.replace(".", "_")
        shape = list(q["weight_shape"])
        w = read_mem(os.path.join(mem_dir, f"{safe}_weight_int8.mem"), 2, True)
        if w.size != int(np.prod(shape)):
            sys.exit(f"[LỖI] {safe}_weight_int8.mem kích thước sai")
        w = torch.from_numpy(w.reshape(shape))
        b_path = os.path.join(mem_dir, f"{safe}_bias_int32.mem")
        bias = torch.from_numpy(read_mem(b_path, 8, True)) if q.get("num_bias", 0) > 0 and os.path.isfile(b_path) else None
        d = dict(name=name, w=w, bias=bias, in_zp=int(q["input_zero_point"]), out_zp=int(q["output_zero_point"]),
                 m0=int(q["requant_M0"]), shift=int(q["requant_shift"]), relu=bool(q["fused_relu"]),
                 w_scale=float(q["weight_scale"]), in_scale=float(q["input_scale"]), out_scale=float(q["output_scale"]))
        if q["layer_type"] == "Linear":
            fc = d
        else:
            d["groups"] = prev_c // shape[1]
            prev_c = shape[0]
            convs.append(d)
    if fc is None or len(convs) != len(pools):
        sys.exit(f"[LỖI] {hospital}: số conv INT8 != số khối FP32")
    lut = build_input_lut(in_scale, in_zp)
    lut_files = [os.path.join(int8_dir, "Preprocess", f"input_lut_{c}.mem") for c in "RGB"]
    if all(os.path.isfile(p) for p in lut_files):
        lut_file = np.stack([read_mem(p, 2, False) for p in lut_files]).astype(np.uint8)
        if not np.array_equal(lut, lut_file):
            print(f"  [CẢNH BÁO] {hospital}: LUT tính lại khác file .mem -> dùng file .mem")
        lut = lut_file
    return dict(name=hospital, conv=convs, fc=fc, pools=[bool(p) for p in pools], lut=lut,
                in_scale=in_scale, in_zp=in_zp, mem_dir=mem_dir)


def quantize_input(hw, small):
    xq = np.empty(small.shape, dtype=np.int64)
    for c in range(3):
        xq[..., c] = hw["lut"][c][small[..., c]]
    return torch.from_numpy(xq).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def int_forward(hw, xq):
    """Forward số nguyên. SỬA 13.1: argmax MaxPool lấy trên acc int32 (như phần cứng),
    rồi requant giá trị max. Không lấy argmax trên giá trị đã requant (uint8, đã bão hoà)."""
    tr, x = {}, xq.to(torch.int64)
    for i, L in enumerate(hw["conv"], 1):
        acc = F.conv2d((x - L["in_zp"]).double(), L["w"].double(), padding=1, groups=L["groups"])
        if L["bias"] is not None:
            acc = acc + L["bias"].double().view(1, -1, 1, 1)
        acc = acc.round().long()
        assert int(acc.abs().max()) < (1 << 30)
        if hw["pools"][i - 1]:
            # ---- FIX 13.1 ----
            xp_acc, idx = F.max_pool2d(acc.double(), 2, return_indices=True)
            tr[f"pidx{i}"] = idx
            x = requant_int(xp_acc.long(), L["m0"], L["shift"], L["out_zp"],
                            L["out_zp"] if L["relu"] else 0, 255)
            tr[f"pool{i}"] = x
            tr[f"relu{i}"] = requant_int(acc, L["m0"], L["shift"], L["out_zp"],
                                         L["out_zp"] if L["relu"] else 0, 255)
        else:
            x = requant_int(acc, L["m0"], L["shift"], L["out_zp"],
                            L["out_zp"] if L["relu"] else 0, 255)
            tr[f"relu{i}"] = x
    n = x.shape[2] * x.shape[3]
    gap = (x.sum(dim=(2, 3)) + n // 2) // n
    tr["gap"] = gap
    f = hw["fc"]
    acc = (gap - f["in_zp"]).double() @ f["w"].double().t()
    if f["bias"] is not None:
        acc = acc + f["bias"].double().view(1, -1)
    lg = requant_int(acc.round().long(), f["m0"], f["shift"], f["out_zp"], 0, 255)
    return lg, tr


def trim_trace(tr, block):
    keep = {}
    for k, v in tr.items():
        if k.startswith("relu") and int(k[4:]) >= block:
            keep[k] = v
        elif k.startswith("pidx") and int(k[4:]) >= block:
            keep[k] = v
        elif k.startswith("pool") and int(k[4:]) >= block:
            keep[k] = v
    return keep


def _chain(hw, tr, cls, block, steps, probe=False):
    n = len(hw["conv"])
    row = hw["fc"]["w"][cls].to(torch.int64)
    A_n = tr[f"pool{n}"] if n in [i for i in range(1, n + 1) if hw["pools"][i - 1]] else tr[f"relu{n}"]
    bt = {}
    if block == n:
        return row.view(1, -1, 1, 1).expand_as(A_n).clone(), bt
    d = (A_n > 0).long() * row.view(1, -1, 1, 1)
    bt[f"dz{n}"] = d
    for s, j in enumerate(range(n, block, -1)):
        L = hw["conv"][j - 1]
        acc = F.conv_transpose2d(d.double(), L["w"].double(), padding=1, groups=L["groups"]).round().long()
        bt[f"acc{j}"] = acc
        if s >= len(steps):
            if probe:
                return acc, bt
            raise ValueError("thiếu tham số requant gradient")
        g = requant_int(acc, steps[s]["m0"], steps[s]["shift"], 0, -GRAD_MAX, GRAD_MAX)
        bt[f"g{j}"] = g
        if hw["pools"][j - 2]:
            g = F.max_unpool2d(g.double(), tr[f"pidx{j - 1}"], 2).long()
        if j - 1 > block:
            g = g * (tr[f"relu{j - 1}"] > 0).long()
            bt[f"dz{j - 1}"] = g
        d = g
    return d, bt


def calibrate_hw(hw, items, block, grad_pct=99.9, cam_pct=99.5):
    n = len(hw["conv"])
    steps = []
    for s in range(n - block):
        vals = []
        for tr, c in items:
            acc, _ = _chain(hw, tr, c, block, steps, probe=True)
            v = np.abs(acc.numpy()).ravel()
            vals.append(v[v > 0])
        v = np.concatenate(vals) if vals else np.zeros(0)
        p = float(np.percentile(v, grad_pct)) if v.size else 1.0
        p = max(p, 1.0)
        m = GRAD_MAX / p
        m0, sh = fixed_point_multiplier(m)
        steps.append(dict(conv=n - s, m0=m0, shift=sh, acc_pct=p, acc_max=float(v.max()) if v.size else 0.0,
                          sat_frac=float((v * m > GRAD_MAX).mean()) if v.size else 0.0, ratio=m))
    cams = []
    for tr, c in items:
        G, _ = _chain(hw, tr, c, block, steps)
        cams.append((G.clamp_min(0) * tr[f"relu{block}"]).sum(dim=1).numpy().ravel())
    cv = np.concatenate(cams)
    cv = cv[cv > 0]
    p = max(float(np.percentile(cv, cam_pct)) if cv.size else 1.0, 1.0)
    cm0, csh = fixed_point_multiplier(255.0 / p)
    cam = dict(m0=cm0, shift=csh, cam_pct=p, cam_max=float(cv.max()) if cv.size else 0.0,
               sat_frac=float((cv * (255.0 / p) > 255).mean()) if cv.size else 0.0, ratio=255.0 / p)
    return dict(steps=steps, cam=cam, n_items=len(items), grad_pct=grad_pct, cam_pct_q=cam_pct)


def hw_infer(hw, small, cls, block):
    xq = quantize_input(hw, small)
    lg, tr = int_forward(hw, xq)
    G, bt = _chain(hw, tr, cls, block, hw["lcam"]["steps"])
    A = tr[f"relu{block}"]
    assert hw["conv"][block - 1]["out_zp"] == 0
    cam_acc = (G.clamp_min(0) * A).sum(dim=1)[0]
    c = hw["lcam"]["cam"]
    cam_u8 = requant_int(cam_acc, c["m0"], c["shift"], 0, 0, 255)
    return dict(xq=xq, logits=lg, tr=tr, bt=bt, G=G, cam_acc=cam_acc, cam_u8=cam_u8,
                int_pred=int(lg.argmax(1)[0]))


def hw_heat(cam_u8, mask):
    c = cam_u8.float().to(mask.device)[None, None]
    return norm_fov(upsample(c), mask)


def map_agreement(h_ref, h_hw, mask_np, top_frac):
    a, b = h_ref[mask_np], h_hw[mask_np]
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")
    ta, tb = a >= np.quantile(a, 1 - top_frac), b >= np.quantile(b, 1 - top_frac)
    return corr, float((ta & tb).sum() / max(1, (ta | tb).sum()))


# ============================================================================
# XUẤT FPGA
# ============================================================================
def export_hw_params(hw, classes, out_dir):
    blk, n, cal = hw["block"], len(hw["conv"]), hw["lcam"]
    d = os.path.join(out_dir, hw["name"], f"LayerCAM_B{blk}")
    os.makedirs(os.path.join(d, "vectors"), exist_ok=True)
    steps_meta = []
    for st in cal["steps"]:
        j = st["conv"]
        L = hw["conv"][j - 1]
        assert L["groups"] == 1
        wb = torch.flip(L["w"].permute(1, 0, 2, 3), dims=(2, 3)).contiguous()
        fn = f"lcam_w{j}_bwd_int8.mem"
        write_hex(os.path.join(d, fn), wb.numpy(), 2)
        steps_meta.append(dict(
            conv_index=j, weight_file=fn, weight_shape_out_in_kh_kw=list(wb.shape),
            requant_M0=st["m0"], requant_shift=st["shift"], requant_ratio=st["ratio"],
            acc_percentile_value=st["acc_pct"], acc_abs_max=st["acc_max"],
            saturation_fraction=st["sat_frac"],
            note=f"dZ_{j-1}in = clamp(((conv3x3(dZ_{j}, W_bwd)*M0 + 2^(S-1)) >> S), -127, 127)"))
    fc = hw["fc"]
    write_hex(os.path.join(d, "lcam_fc_weight_int8.mem"), fc["w"].numpy(), 2)
    c = cal["cam"]
    meta = dict(
        hospital=hw["name"], classes=classes, block=blk, n_conv=n, pools=hw["pools"],
        input_scale=hw["in_scale"], input_zero_point=hw["in_zp"],
        forward_params_dir=hw["mem_dir"],
        grad_range=[-GRAD_MAX, GRAD_MAX],
        fc=dict(weight_file="lcam_fc_weight_int8.mem", shape=list(fc["w"].shape),
                row_offset_formula="cls * C_last",
                note="dZ_n = (A_n > 0) * Wfc[cls,:]"),
        steps=steps_meta,
        cam=dict(requant_M0=c["m0"], requant_shift=c["shift"], requant_ratio=c["ratio"],
                 acc_percentile_value=c["cam_pct"], acc_max=c["cam_max"],
                 saturation_fraction=c["sat_frac"],
                 formula="cam_acc = sum_c ReLU(G[c]) * A_k[c]; cam_u8 = clamp(((cam_acc*M0+2^(S-1))>>S), 0, 255)"),
        rules=dict(
            relu_mask="mask_j = (A_j > 0)",
            maxpool_argmax="pidx{j}: argmax trên ACC int32 TRƯỚC requant (sửa 13.1); "
                           "mã (row%2)*2 + (col%2); hoà chọn pixel đầu (raster TL,TR,BL,BR)",
            layerCAM_gradient="G tại đầu ra ReLU khối k, KHÔNG mask ReLU khối k",
            tensor_layout_in_mem_traces="HWC, kênh chạy nhanh nhất, số có dấu lưu bù 2",
            bwd_weight_layout="idx = ((o*C_in_bwd + i)*3 + ky)*3 + kx"),
        calibration=dict(n_images=cal["n_items"], grad_percentile=cal["grad_pct"],
                         cam_percentile=cal["cam_pct_q"]),
        mult_bits=RTL_REQUANT_MULT_BITS)
    with open(os.path.join(d, "lcam_params.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    lines = [f"// Tự sinh bởi cam_best_cnn_v3.py ({hw['name']}, LayerCAM khối {blk}) - v3+fix13.1",
             f"localparam integer LCAM_BLOCK = {blk};",
             f"localparam integer LCAM_NSTEP = {len(cal['steps'])};",
             f"localparam integer LCAM_GRAD_MAX = {GRAD_MAX};"]
    for s, st in enumerate(cal["steps"]):
        lines += [f"localparam integer LCAM_CONV_{s} = {st['conv']};",
                  f"localparam integer LCAM_M0_{s} = {st['m0']};",
                  f"localparam integer LCAM_SHIFT_{s} = {st['shift']};"]
    lines += [f"localparam integer LCAM_CAM_M0 = {c['m0']};",
              f"localparam integer LCAM_CAM_SHIFT = {c['shift']};", ""]
    with open(os.path.join(d, "lcam_params.vh"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [XUẤT] {hw['name']}: {len(cal['steps'])} bước gradient + CAM -> {d}")
    return d


def export_hw_vector(hw, hr, vec_dir, tag):
    os.makedirs(vec_dir, exist_ok=True)
    blk, n = hw["block"], len(hw["conv"])
    hwc = lambda t: t[0].permute(1, 2, 0).numpy()
    p = lambda name: os.path.join(vec_dir, f"{tag}_{name}.mem")
    write_hex(p("input"), hwc(hr["xq"]), 2)
    write_hex(p(f"a{blk}"), hwc(hr["tr"][f"relu{blk}"]), 2)
    for j in range(blk, n):
        if hw["pools"][j - 1]:
            idx = hr["tr"][f"pidx{j}"]
            W = hr["tr"][f"relu{j}"].shape[3]
            code = ((idx // W) % 2) * 2 + (idx % W) % 2
            write_hex(p(f"pidx{j}"), hwc(code), 1)
    for k, v in hr["bt"].items():
        write_hex(p(k), hwc(v), 8 if k.startswith("acc") else 2)
    write_hex(p(f"G{blk}"), hwc(hr["G"]), 2)
    write_hex(p("cam_acc"), hr["cam_acc"].numpy(), 8)
    write_hex(p("cam_u8"), hr["cam_u8"].numpy(), 2)
    with open(p("info").replace(".mem", ".json"), "w") as f:
        json.dump(dict(int_logits=hr["logits"][0].tolist(), int_pred=hr["int_pred"]), f)


# ============================================================================
# ĐÁNH GIÁ
# ============================================================================
@torch.no_grad()
def deletion_insertion(model, x, mask, heat, cls, steps=16, blur_k=15):
    H = W = IMG_SIZE
    m4 = mask[None, None]
    mf = m4.float()
    pool = lambda t: F.avg_pool2d(t, blur_k, 1, blur_k // 2, count_include_pad=False)
    base = torch.where(m4, pool(x * mf) / pool(mf).clamp_min(1e-6), x)
    idx = torch.nonzero(mask.flatten()).squeeze(1)
    order = idx[torch.argsort(heat.flatten()[idx], descending=True)]
    rank = torch.full((H * W,), H * W, device=x.device)
    rank[order] = torch.arange(order.numel(), device=x.device)
    fr = torch.linspace(0, 1, steps + 1, device=x.device)
    ks = (fr * order.numel()).round()
    top = (rank[None, :] < ks[:, None]).view(-1, 1, H, W)
    p_ins = F.softmax(model(torch.where(top, x, base)), dim=1)[:, cls]
    p_del = F.softmax(model(torch.where(top, base, x)), dim=1)[:, cls]
    return float(torch.trapz(p_ins, fr)), float(torch.trapz(p_del, fr))


def random_heat(mask, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(IMG_SIZE, IMG_SIZE, generator=g).to(mask.device) * mask


def heat_to_boxes(heat, mask, top_frac, max_boxes, W, H):
    from scipy import ndimage
    if not mask.any():
        return []
    thr = np.quantile(heat[mask], 1 - top_frac)
    lab, n = ndimage.label((heat >= thr) & mask)
    if n == 0:
        return []
    mass = np.asarray(ndimage.sum(heat, lab, range(1, n + 1)))
    objs = ndimage.find_objects(lab)
    sx, sy = W / heat.shape[1], H / heat.shape[0]
    out = []
    for i in np.argsort(-mass)[:max_boxes]:
        ys, xs = objs[i]
        reg = lab == i + 1
        out.append(dict(box=[int(xs.start * sx), int(ys.start * sy), int(xs.stop * sx), int(ys.stop * sy)],
                        area_frac=round(float(reg.sum() / mask.sum()), 4),
                        score=round(float(heat[reg].mean()), 4)))
    return out


# ============================================================================
# CASCADE + VẼ
# ============================================================================
@torch.no_grad()
def predict(info, x, device):
    p = F.softmax(info["model"](x.to(device)), dim=1)[0].cpu().numpy()
    return int(p.argmax()), p


def pick_model(infos, get_x, device, mode):
    order = HOSPITALS if mode == "cascade" else (f"Hospital_{mode}",)
    trace = []
    for h in order:
        x, small = get_x(infos[h]["enhance"])
        c, p = predict(infos[h], x, device)
        name = infos[h]["classes"][c]
        trace.append(f"{h}:{name}({p[c]:.2f})")
        if name != UNKNOWN_NAME or mode != "cascade":
            return infos[h], c, p, trace, x, small
    return None, None, None, trace, None, None


def draw_panel(ax, crop, heat, boxes, title, mask=None):
    H, W = crop.shape[:2]
    ax.imshow(crop, extent=(0, W, H, 0))
    if heat is not None:
        if mask is not None:
            heat = np.ma.masked_where(~mask, heat)
        ax.imshow(heat, extent=(0, W, H, 0), cmap="jet", alpha=0.45, vmin=0, vmax=1,
                  interpolation="bilinear")
    for b in boxes or []:
        x0, y0, x1, y1 = b["box"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.6))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_figure(path, crop, mask, maps, cls_name, p_cls, who, args, boxes_g, boxes_l, boxes_h, sc, agree=None):
    has_hw = "hw" in maps
    ncols = 4 if has_hw else 3
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.6), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    sfx = lambda k: f"\nins={sc[k][0]:.2f} del={sc[k][1]:.2f}" if k in sc else ""
    draw_panel(axes[0, 0], crop, None, None, f"{who}: {cls_name} ({p_cls:.2f})")
    draw_panel(axes[0, 1], crop, maps["gc"], boxes_g,
               f"Grad-CAM++ (khối {args.gcpp_layer})" + sfx("gc"), mask)
    draw_panel(axes[0, 2], crop, maps["lc"], boxes_l,
               f"LayerCAM khối {LCAM_LAYER} (top {int(args.top_frac * 100)}%)" + sfx("lc"), mask)
    if has_hw:
        extra = f"\nr={agree[0]:.2f} IoU={agree[1]:.2f}" if agree else ""
        draw_panel(axes[0, 3], crop, maps["hw"], boxes_h,
                   f"INT8 LayerCAM khối {args.hw_block} (FPGA)" + extra + sfx("hw"), mask)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def collect_images(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(IMG_EXTS)]
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"[CẢNH BÁO] Bỏ qua (không tồn tại): {p}")
    return out


def even_pick(items, n):
    if n <= 0 or n >= len(items):
        return list(items)
    return [items[i] for i in np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))]


def sample_images(imgs, limit, per_folder):
    if per_folder > 0:
        groups = defaultdict(list)
        for p in sorted(imgs):
            groups[os.path.dirname(p)].append(p)
        imgs = [p for g in sorted(groups) for p in even_pick(groups[g], per_folder)]
    if limit > 0:
        imgs = even_pick(imgs, limit)
    return imgs


def filter_skip(imgs, skip_classes):
    if not skip_classes:
        return imgs
    skip = set(skip_classes)
    before = len(imgs)
    out = [p for p in imgs if os.path.basename(os.path.dirname(p)) not in skip]
    if before - len(out) > 0:
        print(f"[SKIP] Bỏ {before - len(out)} ảnh thuộc lớp {skip_classes}, còn {len(out)}")
    return out


# ============================================================================
# SETUP INT8
# ============================================================================
def setup_hw(args, infos, calib_paths):
    blk, hw_all = args.hw_block, {}
    for h, inf in infos.items():
        hw = load_int8_hw(args.best_root, h, [l[2] for l in inf["layers"]])
        if not 1 <= blk <= len(hw["conv"]):
            sys.exit(f"[LỖI] --hw-block {blk} không hợp lệ (mạng có {len(hw['conv'])} khối)")
        items = []
        for p in calib_paths:
            _, small = make_input(load_raw(p), inf["enhance"])
            lg, tr = int_forward(hw, quantize_input(hw, small))
            c = int(lg.argmax(1)[0])
            if inf["classes"][c] != UNKNOWN_NAME:
                items.append((trim_trace(tr, blk), c))
        if not items:
            sys.exit(f"[LỖI] {h}: không ảnh calib nào được dự đoán ra bệnh (toàn Unknown).")
        if len(items) < 10:
            print(f"  [CẢNH BÁO] {h}: chỉ {len(items)} ảnh calib -> M0/shift kém ổn định")
        hw["block"] = blk
        hw["lcam"] = calibrate_hw(hw, items, blk, args.grad_pct, args.cam_pct)
        cal = hw["lcam"]
        print(f"[HW] {h}: LayerCAM INT8 khối {blk}, hiệu chuẩn {cal['n_items']} ảnh")
        for st in cal["steps"]:
            print(f"     gradient conv{st['conv']}: M0={st['m0']} shift={st['shift']} "
                  f"(acc p{args.grad_pct}={st['acc_pct']:.0f}, max={st['acc_max']:.0f}, "
                  f"bão hoà={st['sat_frac']:.4f})")
        c = cal["cam"]
        print(f"     CAM: M0={c['m0']} shift={c['shift']} (cam_acc p{args.cam_pct}={c['cam_pct']:.0f}, "
              f"max={c['cam_max']:.0f}, bão hoà={c['sat_frac']:.4f})")
        if args.export_hw:
            export_hw_params(hw, inf["classes"], os.path.join(args.out, "hw_lcam"))
        hw_all[h] = hw
    return hw_all


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="file ảnh hoặc thư mục (đệ quy)")
    ap.add_argument("--out", default="cam_out")
    ap.add_argument("--best-root", default=BEST_ROOT_DEFAULT)
    ap.add_argument("--mode", choices=["cascade", "A", "B"], default="cascade")
    ap.add_argument("--target-class", default=None)
    ap.add_argument("--skip-classes", nargs="+", default=["Unknown", "Normal"])
    ap.add_argument("--gcpp-layer", type=int, default=3)
    ap.add_argument("--top-frac", type=float, default=0.15)
    ap.add_argument("--max-boxes", type=int, default=1)
    ap.add_argument("--erode", type=int, default=2)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-folder", type=int, default=0)
    ap.add_argument("--hw", action="store_true")
    ap.add_argument("--export-hw", action="store_true")
    ap.add_argument("--hw-block", type=int, default=LCAM_LAYER)
    ap.add_argument("--hw-vectors", type=int, default=2)
    ap.add_argument("--calib-images", nargs="+", default=None)
    ap.add_argument("--calib-n", type=int, default=128,
                    help="số ảnh hiệu chuẩn M0/shift (mặc định 128 theo spec)")
    ap.add_argument("--grad-pct", type=float, default=99.9)
    ap.add_argument("--cam-pct", type=float, default=99.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.hw = args.hw or args.export_hw

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    need = HOSPITALS if args.mode == "cascade" else (f"Hospital_{args.mode}",)
    infos = {h: load_best_model(args.best_root, h, device) for h in need}
    for h, inf in infos.items():
        print(f"[MODEL] {h}: {len(inf['layers'])} khối conv, lớp = {inf['classes']}, "
              f"enhance = {inf['enhance']}")
    n_blocks = len(next(iter(infos.values()))["layers"])
    for k in {LCAM_LAYER, args.gcpp_layer, args.hw_block}:
        if not 1 <= k <= n_blocks:
            sys.exit(f"[LỖI] Khối {k} không hợp lệ (mạng có {n_blocks} khối)")

    imgs = filter_skip(collect_images(args.images), args.skip_classes)
    imgs = sample_images(imgs, args.limit, args.per_folder)
    if not imgs:
        sys.exit("[LỖI] Không còn ảnh nào sau khi lọc")

    hw_all = {}
    if args.hw:
        calib = collect_images(args.calib_images) if args.calib_images else collect_images(args.images)
        calib = filter_skip(calib, args.skip_classes)
        calib = even_pick(sorted(calib), args.calib_n)
        if not calib:
            sys.exit("[LỖI] Không còn ảnh calib sau khi lọc")
        hw_all = setup_hw(args, infos, calib)

    results = []
    fid = defaultdict(list)
    n_vec = defaultdict(int)

    for i, path in enumerate(imgs):
        raw = load_raw(path)
        H0, W0 = raw.shape[:2]
        crop = raw[:(H0 // IMG_SIZE) * IMG_SIZE, :(W0 // IMG_SIZE) * IMG_SIZE]
        mask_np = fov_mask(raw, args.erode)
        prep = {}

        def get_x(enh, raw=raw, prep=prep):
            if enh not in prep:
                prep[enh] = make_input(raw, enh)
            return prep[enh]

        info, c, p, trace, x, small = pick_model(infos, get_x, device, args.mode)
        x = x.to(device) if x is not None else None
        mask = torch.from_numpy(mask_np).to(device)
        rec = dict(file=path, trace=trace)
        if info is None:
            rec.update(final_pred=UNKNOWN_NAME, note="Unknown ở mọi CNN -> không vẽ CAM")
            print(f"[{i}] {os.path.basename(path)}: {' -> '.join(trace)} => {UNKNOWN_NAME} (bỏ qua)")
            results.append(rec)
            continue
        if args.target_class is not None:
            if args.target_class not in info["classes"]:
                sys.exit(f"[LỖI] {info['name']} không có lớp '{args.target_class}'")
            c = info["classes"].index(args.target_class)
        cls_name = info["classes"][c]
        model = info["model"]
        hw = hw_all.get(info["name"])

        if cls_name == UNKNOWN_NAME:
            print(f"  [LƯU Ý] CAM của lớp '{UNKNOWN_NAME}' ít ý nghĩa")

        blocks = sorted({LCAM_LAYER, args.gcpp_layer})
        probs, c, maps = compute_maps(model, x, mask, blocks, c)
        maps["gc"] = maps[("gc", args.gcpp_layer)]
        maps["lc"] = maps[("lc", LCAM_LAYER)]

        hr, agree = None, None
        if hw is not None:
            hr = hw_infer(hw, small, c, args.hw_block)
            maps["hw"] = hw_heat(hr["cam_u8"], mask)
            ref = maps.get(("lc", args.hw_block))
            if ref is None:
                _, _, extra_maps = compute_maps(model, x, mask, [args.hw_block], c)
                ref = extra_maps[("lc", args.hw_block)]
            agree = map_agreement(ref.cpu().numpy(), maps["hw"].cpu().numpy(), mask_np, args.top_frac)
            fid[info["name"]].append((agree[0], agree[1], int(hr["int_pred"] == c)))
            if args.export_hw and n_vec[info["name"]] < args.hw_vectors:
                stem0 = f"{i:04d}_{os.path.splitext(os.path.basename(path))[0]}".replace(" ", "_")
                export_hw_vector(hw, hr, os.path.join(args.out, "hw_lcam", info["name"],
                                                      f"LayerCAM_B{args.hw_block}", "vectors"), stem0)
                n_vec[info["name"]] += 1

        sc = {}
        if args.eval:
            keys = ["gc", "lc"] + (["hw"] if hr is not None else [])
            for key in keys:
                sc[key] = deletion_insertion(model, x, mask, maps[key], c, args.steps)
            sc["random"] = deletion_insertion(model, x, mask, random_heat(mask, i), c, args.steps)

        np_maps = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in maps.items()}
        H, W = crop.shape[:2]
        boxes_g = heat_to_boxes(np_maps["gc"], mask_np, args.top_frac, args.max_boxes, W, H)
        boxes_l = heat_to_boxes(np_maps["lc"], mask_np, args.top_frac, args.max_boxes, W, H)
        boxes_h = heat_to_boxes(np_maps["hw"], mask_np, args.top_frac, args.max_boxes, W, H) if hr is not None else []

        stem = f"{i:04d}_{os.path.splitext(os.path.basename(path))[0]}"
        save_figure(os.path.join(args.out, stem + "_cam.png"), crop, mask_np, np_maps, cls_name,
                    float(probs[c]), info["name"], args, boxes_g, boxes_l, boxes_h, sc, agree)
        np.save(os.path.join(args.out, stem + "_gradcampp.npy"), np_maps["gc"])
        np.save(os.path.join(args.out, stem + "_layercam.npy"), np_maps["lc"])
        if hr is not None:
            np.save(os.path.join(args.out, stem + "_layercam_int8.npy"), np_maps["hw"])
            np.save(os.path.join(args.out, stem + "_layercam_int8_u8_16x16.npy"),
                    hr["cam_u8"].numpy().astype(np.uint8))

        rec.update(decided_by=info["name"], final_pred=cls_name, cam_class=cls_name,
                   prob=round(float(probs[c]), 4),
                   boxes_gradcampp=boxes_g, boxes_layercam=boxes_l,
                   eval={k: dict(ins=round(v[0], 4), dele=round(v[1], 4)) for k, v in sc.items()})
        if hr is not None:
            rec.update(boxes_layercam_int8=boxes_h, int8_pred=hr["int_pred"],
                       int8_agrees_class=bool(hr["int_pred"] == c),
                       int8_vs_fp32=dict(pearson=round(agree[0], 4), iou_top=round(agree[1], 4)))
        results.append(rec)

        extra = (f" | ins/del LayerCAM={sc['lc'][0]:.2f}/{sc['lc'][1]:.2f}, "
                 f"random={sc['random'][0]:.2f}/{sc['random'][1]:.2f}") if sc else ""
        hwtxt = f" | INT8 vs FP32: r={agree[0]:.2f} IoU={agree[1]:.2f}" if agree else ""
        print(f"[{i}] {os.path.basename(path)}: {' -> '.join(trace)} => {cls_name} | "
              f"khung Grad-CAM++={len(boxes_g)} LayerCAM={len(boxes_l)}{extra}{hwtxt}")

    with open(os.path.join(args.out, "cam_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if fid:
        rep = {}
        print("\n=== ĐỘ KHỚP LayerCAM INT8 so với FP32 (trong FOV) ===")
        for h, v in fid.items():
            a = np.asarray(v, dtype=float)
            rep[h] = dict(n=len(v), pearson_mean=float(np.nanmean(a[:, 0])),
                          iou_top_mean=float(a[:, 1].mean()),
                          int8_pred_agrees=float(a[:, 2].mean()))
            print(f"{h}: n={len(v)} | Pearson={rep[h]['pearson_mean']:.3f} | "
                  f"IoU top-{int(args.top_frac * 100)}%={rep[h]['iou_top_mean']:.3f} | "
                  f"INT8 trùng lớp FP32={rep[h]['int8_pred_agrees']:.3f}")
        with open(os.path.join(args.out, "hw_fidelity.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)

    print(f"Xong. Kết quả trong: {args.out}")


if __name__ == "__main__":
    main()