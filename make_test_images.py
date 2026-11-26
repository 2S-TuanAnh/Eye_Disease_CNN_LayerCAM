#!/usr/bin/env python3
"""Tao anh test 64x64 va ket qua mong doi (cascade 2 CNN + LayerCAM) cho FPGA.

Dau ra (trong --out-dir):
  image{i}_raw64.mem        12288 dong hex, thu tu pixel R,G,B (giong luong FPGA nhan qua UART)
  ref_img{i}_{ten}.mem      tung lop cua CNN 1 va CNN 2: input, pool1, pool2, pool3, pidx3, conv4, conv5, gap, fc
                            (hex, 1 byte/dong, thu tu [row][col][channel]); ten file co hau to _cnn1 / _cnn2
  ref_img{i}_{dz5,dz4,g5,g4,cam_u8}_{cnn}.mem
                            chi khi cam_flag = 1: gradient/CAM cua CNN quyet dinh (dz5, dz4, g5, g4 la (v & 0xFF),
                            cam_u8 la 256 byte 16x16 theo hang)
  expected_cascade.json     logit CNN 1, CNN 2, nguon quyet dinh, lop cuoi, cam_flag, cam_u8 (256 so hoac []),
                            source_path (duong dan tuyet doi cua anh goc) cua tung anh

Gia tri mong doi duoc tinh bang mo hinh numpy bit-exact theo SEPC.md muc 4 va SEPC_LayerCAM.md, doc thang tu
sd_image.bin (nen cung kiem luon viec dong goi). De doi chieu voi golden_run trong CNN_Search_BN.py, so sanh file
ref_img*_*.mem voi TestVectors/dbg_img*_*.mem. cam_u8 phai bang *_cam_u8.mem trong hw_lcam/.../vectors/ cua
cam_best_cnn_v3.py (lech thi sai o cam_best_cnn_v3.py hoac o run_layercam ben duoi).

Doc anh: dung load_resize_enhance trong CNN_Search_BN.py (ENHANCE_MODE = "none", resize block_average_resize).
Chu ky that: load_resize_enhance(path, img_size, enhance_mode) -> (PIL Image, size_wh); dung search.IMG_SIZE va search.ENHANCE_MODE.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

LAYER_SHAPES = [(3, 16), (16, 24), (24, 24), (24, 32), (32, 48), (48, 5)]
LAYER_OFFSETS = [0x0600, 0x07F0, 0x15D0, 0x2A70, 0x45F0, 0x7CB0]
GROUP_BYTES = [248, 1184, 1760, 1760, 2336, 416]
GROUP_COUNTS = [2, 3, 3, 4, 6, 1]
CNN_BLOCK_BYTES = 0x10000

# LayerCAM
BWD_SHAPES = [(48, 32), (32, 24)]          # (vao, ra) cua lop 6 (BWD1), lop 7 (BWD2)
BWD_OFFSETS = [0x8100, 0xB780]
BWD_GROUP_COUNTS = [4, 3]
LCAM_TABLE_OFFSET = 0x8000


# ----------------------------------------------------------------------------
# Doc anh (diem can chinh neu chu ky ham trong CNN_Search_BN.py khac)
# ----------------------------------------------------------------------------
def load_image_64(path, module_dir):
    """Tra ve mang uint8 (64, 64, 3) thu tu RGB, giong het anh PC se gui."""
    if module_dir:
        sys.path.insert(0, module_dir)
    import CNN_Search_BN as search  # noqa: E402

    result = search.load_resize_enhance(path, search.IMG_SIZE, search.ENHANCE_MODE)   # tra ve (PIL Image, size_wh)
    if isinstance(result, tuple):
        result = result[0]
    array = np.asarray(result)
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    if array.shape != (64, 64, 3):
        raise SystemExit("load_resize_enhance tra ve shape %s, can (64, 64, 3). Chinh load_image_64()." % (array.shape,))
    return array


# ----------------------------------------------------------------------------
# Mo hinh tham chieu (muc 4 cua SEPC.md + LayerCAM)
# ----------------------------------------------------------------------------
def read_layer_groups(block, offset, groups, in_ch, out_ch, kernel_area):
    """Doc 'groups' khoi nhom (bias 8 x int32 + trong so PE0..PE7). Tra ve (weights, biases) da bo phan dem.

    weights: (out, in, 3, 3) neu kernel_area == 9, (out, in) neu kernel_area == 1. biases: (out,).
    """
    pe_bytes = in_ch * kernel_area
    padded_out = groups * 8
    shape = (padded_out, in_ch, 3, 3) if kernel_area == 9 else (padded_out, in_ch)
    weights = np.zeros(shape, dtype=np.int64)
    biases = np.zeros(padded_out, dtype=np.int64)
    position = offset
    for group in range(groups):
        for pe in range(8):
            biases[group * 8 + pe] = struct.unpack_from("<i", block, position + 4 * pe)[0]
        position += 32
        for pe in range(8):
            raw = np.frombuffer(block[position:position + pe_bytes], dtype=np.int8)
            weights[group * 8 + pe] = raw.reshape(shape[1:]).astype(np.int64)
            position += pe_bytes
    return weights[:out_ch], biases[:out_ch]


class CnnBlock:
    def __init__(self, block):
        if block[0:4] != b"FCNN":
            raise SystemExit("Sai magic trong khoi CNN")
        self.cnn_id = block[4]
        self.unknown_class_index = block[5]
        self.num_classes = block[6]
        self.normal_class_index = block[7]            # 0xFF = khong co Normal
        self.layer_params = []
        for layer in range(6):
            m0, shift, zp_in, zp_out, relu = struct.unpack_from("<IBBBB", block, 8 + 8 * layer)
            self.layer_params.append((m0, shift, zp_in, zp_out, relu))
        self.lut = np.frombuffer(block[0x200:0x500], dtype=np.uint8).reshape(3, 256)

        self.weights = []
        self.biases = []
        for layer, (in_ch, out_ch) in enumerate(LAYER_SHAPES):
            kernel_area = 1 if layer == 5 else 9      # FC: 48 trong so moi PE
            weights, biases = read_layer_groups(block, LAYER_OFFSETS[layer], GROUP_COUNTS[layer],
                                                in_ch, out_ch, kernel_area)
            self.weights.append(weights)
            self.biases.append(biases)

        # LayerCAM: bang 0x8000 + trong so BWD
        self.bwd_m0 = []
        self.bwd_shift = []
        for k in range(2):
            m0, shift, _zp_in, _zp_out, _relu = struct.unpack_from("<IBBBB", block, LCAM_TABLE_OFFSET + 8 * k)
            self.bwd_m0.append(m0)
            self.bwd_shift.append(shift)
        self.cam_m0, self.cam_shift = struct.unpack_from("<IB", block, LCAM_TABLE_OFFSET + 0x10)
        self.bwd_weights = []
        for k, (in_ch, out_ch) in enumerate(BWD_SHAPES):
            weights, _biases = read_layer_groups(block, BWD_OFFSETS[k], BWD_GROUP_COUNTS[k], in_ch, out_ch, 9)
            self.bwd_weights.append(weights)          # (out, in, 3, 3)


def requantize(acc, m0, shift, zp_out, relu):
    product = acc.astype(np.int64) * np.int64(m0)
    value = (product + (np.int64(1) << (shift - 1))) >> shift      # dich so hoc tren int64
    y = value + zp_out
    lower = zp_out if relu else 0
    return np.clip(y, lower, 255).astype(np.uint8)


def requant_signed(acc, m0, shift):
    """(acc*M0 + 2^(S-1)) >> S, khong cong zp (dung cho lop nguoc va CAM)."""
    product = np.asarray(acc, dtype=np.int64) * np.int64(m0)
    return (product + (np.int64(1) << (shift - 1))) >> shift


def conv3x3_accumulate(x, weights, biases, zp_in):
    height, width, in_ch = x.shape
    shifted = x.astype(np.int64) - zp_in
    padded = np.zeros((height + 2, width + 2, in_ch), dtype=np.int64)   # padding dong gop 0
    padded[1:-1, 1:-1, :] = shifted
    acc = np.zeros((height, width, weights.shape[0]), dtype=np.int64) + biases
    for kernel_row in range(3):
        for kernel_col in range(3):
            window = padded[kernel_row:kernel_row + height, kernel_col:kernel_col + width, :]
            acc += window @ weights[:, :, kernel_row, kernel_col].T
    return acc


def max_pool_2x2(acc):
    height, width, channels = acc.shape
    blocks = acc.reshape(height // 2, 2, width // 2, 2, channels)
    return blocks.max(axis=(1, 3))


def pool_argmax_code(acc):
    """Ma vi tri thang trong moi o 2x2, tinh tren acc (truoc requant): (r%2)*2 + (c%2).

    np.argmax lay phan tu dau khi hoa, khop voi dieu kien '>' nghiem ngat cua RTL."""
    height, width, channels = acc.shape
    blocks = acc.reshape(height // 2, 2, width // 2, 2, channels)
    flat = blocks.transpose(0, 2, 4, 1, 3).reshape(height // 2, width // 2, channels, 4)
    return flat.argmax(axis=-1).astype(np.uint8)


def run_cnn(image, cnn):
    """image: (64, 64, 3) uint8. Tra ve (logits[5], class_index, dict lop trung gian)."""
    layers = {}
    x = np.stack([cnn.lut[c][image[:, :, c]] for c in range(3)], axis=-1)     # LUT: R, G, B
    layers["input"] = x
    pool_flags = [True, True, True, False, False]
    layer_names = ["pool1", "pool2", "pool3", "conv4", "conv5"]
    for layer in range(5):
        m0, shift, zp_in, zp_out, relu = cnn.layer_params[layer]
        acc = conv3x3_accumulate(x, cnn.weights[layer], cnn.biases[layer], zp_in)
        if pool_flags[layer]:
            if layer == 2:
                layers["pidx3"] = pool_argmax_code(acc)     # (8, 8, 24), cho LayerCAM
            acc = max_pool_2x2(acc)                 # max tren acc co dau, truoc requant
        x = requantize(acc, m0, shift, zp_out, relu)
        layers[layer_names[layer]] = x
    gap = ((x.astype(np.int64).reshape(-1, x.shape[-1]).sum(axis=0) + 32) >> 6).astype(np.uint8)
    layers["gap"] = gap
    m0, shift, zp_in, zp_out, relu = cnn.layer_params[5]
    acc = (cnn.weights[5] @ (gap.astype(np.int64) - zp_in)) + cnn.biases[5]
    logits = requantize(acc, m0, shift, zp_out, 0)
    layers["fc"] = logits
    return logits, int(np.argmax(logits)), layers


def run_layercam(layers, cnn, cls):
    """LayerCAM cua CNN quyet dinh cho lop 'cls'. layers: dict tu run_cnn cua chinh CNN do.

    Tra ve dict: dz5 (8,8,48), g5 (8,8,32), dz4 (8,8,32), g4 (8,8,24) (so nguyen co dau),
    cam_u8 (16,16) uint8."""
    a5 = layers["conv5"].astype(np.int64)             # (8, 8, 48)
    a4 = layers["conv4"].astype(np.int64)             # (8, 8, 32)
    pool3 = layers["pool3"].astype(np.int64)          # (8, 8, 24)
    pidx3 = layers["pidx3"].astype(np.int64)          # (8, 8, 24)

    dz5 = (a5 > 0) * cnn.weights[5][cls].astype(np.int64)                   # (8, 8, 48)
    acc5 = conv3x3_accumulate(dz5 + 128, cnn.bwd_weights[0], np.zeros(32, dtype=np.int64), 128)
    g5 = np.clip(requant_signed(acc5, cnn.bwd_m0[0], cnn.bwd_shift[0]), -127, 127)

    dz4 = (a4 > 0) * g5                                                     # mask ReLU chi o BWD1
    acc4 = conv3x3_accumulate(dz4 + 128, cnn.bwd_weights[1], np.zeros(24, dtype=np.int64), 128)
    g4 = np.clip(requant_signed(acc4, cnn.bwd_m0[1], cnn.bwd_shift[1]), -127, 127)

    code_up = np.repeat(np.repeat(pidx3, 2, axis=0), 2, axis=1)             # (16, 16, 24)
    rows = np.arange(16).reshape(16, 1)
    cols = np.arange(16).reshape(1, 16)
    expect = (rows % 2) * 2 + (cols % 2)                                    # (16, 16)
    weighted = np.maximum(g4, 0) * pool3                                    # (8, 8, 24)
    weighted_up = np.repeat(np.repeat(weighted, 2, axis=0), 2, axis=1)      # (16, 16, 24)
    cam_acc = ((code_up == expect[..., None]) * weighted_up).sum(axis=-1)   # (16, 16) int64
    cam_u8 = np.clip((cam_acc * np.int64(cnn.cam_m0) + (np.int64(1) << (cnn.cam_shift - 1))) >> cnn.cam_shift,
                     0, 255).astype(np.uint8)
    return {"dz5": dz5, "g5": g5, "dz4": dz4, "g4": g4, "cam_u8": cam_u8}


# ----------------------------------------------------------------------------
def write_mem(path, array):
    with open(path, "w", encoding="utf-8") as handle:
        for value in np.asarray(array).astype(np.int64).reshape(-1):
            handle.write("%02x\n" % (int(value) & 0xFF))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="duong dan anh goc")
    parser.add_argument("--sd-image", default="sd_image.bin")
    parser.add_argument("--module-dir", default=".", help="thu muc chua CNN_Search_BN.py")
    parser.add_argument("--out-dir", default="test_images")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.sd_image, "rb") as handle:
        sd_image = handle.read()
    if len(sd_image) != 2 * CNN_BLOCK_BYTES:
        raise SystemExit("%s phai dai %d byte" % (args.sd_image, 2 * CNN_BLOCK_BYTES))
    cnn1 = CnnBlock(sd_image[:CNN_BLOCK_BYTES])
    cnn2 = CnnBlock(sd_image[CNN_BLOCK_BYTES:])

    expected = []
    for index, path in enumerate(args.images):
        image = load_image_64(path, args.module_dir)
        write_mem(os.path.join(args.out_dir, "image%d_raw64.mem" % index), image)   # [row][col][R,G,B]

        logits1, class1, layers1 = run_cnn(image, cnn1)
        logits2 = np.zeros(5, dtype=np.uint8)
        layers2 = None
        if class1 != cnn1.unknown_class_index:
            source, final_class = 1, class1
        else:
            logits2, class2, layers2 = run_cnn(image, cnn2)
            if class2 != cnn2.unknown_class_index:
                source, final_class = 2, class2
            else:
                source, final_class = 0, 0xFF

        for tag, layers in (("cnn1", layers1), ("cnn2", layers2)):
            if layers is None:
                continue
            for name, array in layers.items():
                write_mem(os.path.join(args.out_dir, "ref_img%d_%s_%s.mem" % (index, name, tag)), array)

        # LayerCAM: CNN quyet dinh la CNN cho ra final_class (khi source = 0 khong chay CAM)
        if source == 1:
            decider, decider_layers, decider_tag = cnn1, layers1, "cnn1"
        elif source == 2:
            decider, decider_layers, decider_tag = cnn2, layers2, "cnn2"
        else:
            decider, decider_layers, decider_tag = None, None, None
        cam_flag = decider is not None and final_class != decider.normal_class_index
        cam_u8 = []
        if cam_flag:
            cam = run_layercam(decider_layers, decider, final_class)
            for name in ("dz5", "g5", "dz4", "g4", "cam_u8"):
                write_mem(os.path.join(args.out_dir, "ref_img%d_%s_%s.mem" % (index, name, decider_tag)), cam[name])
            cam_u8 = [int(v) for v in cam["cam_u8"].reshape(-1)]      # 256 so, theo hang

        expected.append({
            "image": os.path.basename(path),
            "index": index,
            "raw_file": "image%d_raw64.mem" % index,
            "source_path": os.path.abspath(path),
            "logits_cnn1": [int(v) for v in logits1],
            "logits_cnn2": [int(v) for v in logits2],
            "final_source": source,
            "final_class": final_class,
            "cam_flag": int(cam_flag),
            "cam_u8": cam_u8,
        })
        print("%s -> source=%d class=%d cam_flag=%d logits1=%s logits2=%s" %
              (os.path.basename(path), source, final_class, int(cam_flag),
               list(map(int, logits1)), list(map(int, logits2))))

    with open(os.path.join(args.out_dir, "expected_cascade.json"), "w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
