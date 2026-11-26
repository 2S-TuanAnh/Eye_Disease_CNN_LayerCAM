#!/usr/bin/env python3
"""Dong goi tham so 2 CNN (Hospital_A, Hospital_B) + LayerCAM thanh sd_image.bin (131.072 byte) + sd_image.hex.

Bo cuc (moi CNN 64 KB = 65.536 byte, little-endian; CNN 1 o 0x00000, CNN 2 o 0x10000):
  0x0000  magic "FCNN", CNN_ID, UNKNOWN_CLASS_INDEX, NUM_CLASSES, NORMAL_CLASS_INDEX (0xFF neu khong co Normal)
  0x0008  6 lop x 8 byte: M0 (u32), SHIFT (u8), ZP_IN (u8), ZP_OUT (u8), RELU_ENABLE (u8)
  0x0200  LUT dau vao 3x256 byte (R, G, B)
  0x0600  du lieu 6 lop forward; moi khoi nhom (8 kenh ra) = bias 8 x int32 + trong so PE0..PE7 [in][kr][kc]
  --- vung LayerCAM ---
  0x8000  bang LCAM 24 byte:
            +0x00 lop 6 (BWD1): M0 (u32), SHIFT (u8), ZP_IN=128, ZP_OUT=128, RELU=0
            +0x08 lop 7 (BWD2): nhu tren
            +0x10 CAM: M0 (u32), SHIFT (u8), 3 byte 0
  0x8100  trong so BWD1 (48 vao -> 32 ra, 4 nhom x 3488 byte, bias = 0)
  0xB780  trong so BWD2 (32 vao -> 24 ra, 3 nhom x 2336 byte, bias = 0), ket thuc 0xD2E0
  (hang FC cua LayerCAM nam san trong khoi FC 0x7CB0 + 32 + cls*48, khong ghi them)

Thu muc dau vao (moi CNN):
  <Hospital>/INT8/Memory/quant_params.json
  <Hospital>/INT8/Memory/vgg_features_{0,4,8,12,15}_{weight_int8,bias_int32}.mem
  <Hospital>/INT8/Memory/vgg_classifier_2_{weight_int8,bias_int32}.mem
  <Hospital>/INT8/Preprocess/input_lut_{R,G,B}.mem
  <Hospital>/INT8/*.json co khoa "classes" (vd vgg9_int8_config.json), trong do co "Unknown" (va "Normal" neu co)
  <lcam-root>/A/hw_lcam/Hospital_A/LayerCAM_B3/{lcam_params.json, lcam_w5_bwd_int8.mem, lcam_w4_bwd_int8.mem}
  <lcam-root>/B/hw_lcam/Hospital_B/LayerCAM_B3/...   (tuong tu)

Vi du:
  python pack_sd_image.py --root ./BEST --lcam-root cam_out
  python pack_sd_image.py --a-dir ./BEST/Hospital_A --b-dir ./BEST/Hospital_B --a-lcam-dir X --b-lcam-dir Y
  python pack_sd_image.py --root ./BEST --unknown-index 4     (neu khong tim thay file classes)
  python pack_sd_image.py --root ./BEST --a-lcam-json lcam_params_e59f3b.json --b-lcam-json lcam_params_a6b4f9.json

Ghi the (CANH BAO: xoa du lieu the): Linux `sudo dd if=sd_image.bin of=/dev/sdX bs=512 conv=fsync`,
Windows dung Win32DiskImager. Dung the SDHC.
"""
import argparse
import json
import os
import struct
import sys

CNN_BLOCK_BYTES = 0x10000
LAYERS = [  # (in_channels, out_channels)
    (3, 16), (16, 24), (24, 24), (24, 32), (32, 48), (48, 5),
]
LAYER_OFFSETS = [0x0600, 0x07F0, 0x15D0, 0x2A70, 0x45F0, 0x7CB0]
GROUP_BYTES = [248, 1184, 1760, 1760, 2336, 416]
GROUP_COUNTS = [2, 3, 3, 4, 6, 1]

# LayerCAM: 2 lop nguoc (vao, ra)
BWD_LAYERS = [(48, 32), (32, 24)]
BWD_OFFSETS = [0x8100, 0xB780]
BWD_GROUP_BYTES = [3488, 2336]
BWD_GROUP_COUNTS = [4, 3]
BWD_END = 0xD2E0
LCAM_TABLE_OFFSET = 0x8000
# (conv_index trong lcam_params.json, ten file trong so)  -> lop 6, lop 7
BWD_STEPS = [(5, "lcam_w5_bwd_int8.mem"), (4, "lcam_w4_bwd_int8.mem")]

# Ten lop theo thu tu: conv1..conv5, fc
VGG_ORDER = ["vgg_features_0", "vgg_features_4", "vgg_features_8",
             "vgg_features_12", "vgg_features_15", "vgg_classifier_2"]


def read_mem_tokens(path):
    """Doc file .mem kieu $readmemh: bo comment //, dong @addr; tra ve list so nguyen (hex)."""
    values = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.split("//")[0].strip()
            if not line or line.startswith("@"):
                continue
            for token in line.split():
                values.append(int(token, 16))
    return values


def to_signed(value, bits):
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value


def find_layer_files(memory_dir, suffix):
    paths = []
    for name in VGG_ORDER:
        path = os.path.join(memory_dir, name + suffix)
        if not os.path.isfile(path):
            raise SystemExit("Khong thay file %s" % path)
        paths.append(path)
    return paths


def load_layer_params(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    params = []
    for index, name in enumerate(VGG_ORDER):
        key = name.replace("_", ".")          # vgg_features_0 -> vgg.features.0
        if key not in data:
            raise SystemExit("quant_params.json: khong co muc '%s'. Cac khoa: %s" % (key, list(data.keys())))
        entry = data[key]
        try:
            m0 = int(entry["requant_M0"])
            shift = int(entry["requant_shift"])
            zp_in = int(entry["input_zero_point"])
            zp_out = int(entry["output_zero_point"])
            relu = int(bool(entry["fused_relu"]))
        except KeyError as exc:
            raise SystemExit("quant_params.json muc '%s' thieu truong %s" % (key, exc))
        if not (0 <= m0 < (1 << 31)):
            raise SystemExit("Lop %d: M0=%d ngoai [0, 2^31)" % (index, m0))
        if not (1 <= shift <= 63):
            raise SystemExit("Lop %d: SHIFT=%d ngoai [1, 63]" % (index, shift))
        for label, value in (("zp_in", zp_in), ("zp_out", zp_out)):
            if not (0 <= value <= 255):
                raise SystemExit("Lop %d: %s=%d ngoai [0, 255]" % (index, label, value))
        params.append((m0, shift, zp_in, zp_out, relu))
    return params


def find_classes(hospital_dir):
    """Tim file json co khoa 'classes' trong thu muc benh vien (uu tien vgg9_int8_config.json)."""
    candidates = []
    preferred = os.path.join(hospital_dir, "INT8", "vgg9_int8_config.json")
    if os.path.isfile(preferred):
        candidates.append(preferred)
    for folder, _dirs, files in os.walk(hospital_dir):
        for name in sorted(files):
            if name.lower().endswith(".json") and name != "quant_params.json":
                full = os.path.join(folder, name)
                if full not in candidates:
                    candidates.append(full)
    for full in candidates:
        try:
            with open(full, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(config, dict) and "classes" in config:
            classes = config["classes"]
            if isinstance(classes, dict):
                classes = [k for k, _ in sorted(classes.items(), key=lambda kv: kv[1])]
            return full, [str(c) for c in classes]
    return None, None


def get_unknown_index(hospital_dir, override):
    path, classes = find_classes(hospital_dir)
    if classes is None:
        if override is None:
            raise SystemExit("%s: khong tim thay file json nao co khoa 'classes'. "
                             "Hay dung --unknown-index N de chi dinh thu cong." % hospital_dir)
        print("  [canh bao] %s: khong co file classes, dung --unknown-index=%d" % (hospital_dir, override))
        return override
    lowered = [c.lower() for c in classes]
    if "unknown" not in lowered:
        if override is None:
            raise SystemExit("%s: 'classes' khong co 'Unknown': %s (dung --unknown-index N)" % (path, classes))
        return override
    index = lowered.index("unknown")
    print("  classes (%s): %s -> Unknown = %d" % (os.path.basename(path), classes, index))
    if override is not None and override != index:
        print("  [canh bao] --unknown-index=%d bi bo qua, dung %d tu file" % (override, index))
    return index


def get_normal_index(hospital_dir, override):
    """Chi so lop 'Normal' (khong phan biet hoa thuong); khong co Normal thi 0xFF."""
    path, classes = find_classes(hospital_dir)
    if classes is None:
        if override is None:
            raise SystemExit("%s: khong tim thay file 'classes' de lay Normal. "
                             "Hay dung --normal-index N (255 neu khong co Normal)." % hospital_dir)
        print("  [canh bao] %s: khong co file classes, dung --normal-index=%d" % (hospital_dir, override))
        return override
    lowered = [c.lower() for c in classes]
    if "normal" not in lowered:
        print("  khong co lop Normal -> NORMAL_CLASS_INDEX = 0xFF")
        return 0xFF
    index = lowered.index("normal")
    print("  Normal = %d" % index)
    if override is not None and override != index:
        print("  [canh bao] --normal-index=%d bi bo qua, dung %d tu file" % (override, index))
    return index


def write_layer(block, offset, groups, in_ch, kernel_area, weights, biases):
    """Ghi 'groups' khoi nhom (8 kenh ra) bat dau tai offset. Tra ve vi tri ngay sau khoi cuoi.

    weights: list int co do dai out*in*kernel_area, layout [o][i][kr][kc]; biases: list int do dai out.
    Neu out < groups*8 thi dem 0 cho du nhom.
    """
    pe_bytes = in_ch * kernel_area
    padded_out = groups * 8
    weights = list(weights)
    biases = list(biases)
    out_ch = len(biases)
    if out_ch > padded_out or len(weights) != out_ch * pe_bytes:
        raise SystemExit("write_layer: so trong so/bias khong khop (out=%d, in=%d, area=%d, so w=%d)"
                         % (out_ch, in_ch, kernel_area, len(weights)))
    weights += [0] * ((padded_out - out_ch) * pe_bytes)
    biases += [0] * (padded_out - out_ch)

    position = offset
    for group in range(groups):
        for pe in range(8):
            struct.pack_into("<i", block, position + 4 * pe, biases[group * 8 + pe])
        position += 32
        for pe in range(8):
            start = (group * 8 + pe) * pe_bytes
            chunk = weights[start:start + pe_bytes]
            block[position:position + pe_bytes] = bytes(w & 0xFF for w in chunk)
            position += pe_bytes
    return position


def _pick(entry, names, label):
    for name in names:
        if name in entry:
            return entry[name]
    raise SystemExit("lcam_params.json: %s thieu truong %s (cac khoa co: %s)"
                     % (label, "/".join(names), list(entry.keys())))


def _check_m0_shift(label, m0, shift):
    if not (0 <= m0 < (1 << 31)):
        raise SystemExit("LCAM %s: M0=%d ngoai [0, 2^31)" % (label, m0))
    if not (1 <= shift <= 63):
        raise SystemExit("LCAM %s: SHIFT=%d ngoai [1, 63]" % (label, shift))


def load_lcam_params(json_path):
    """Tra ve ([(m0, shift) lop 6, (m0, shift) lop 7], (cam_m0, cam_shift))."""
    if not os.path.isfile(json_path):
        raise SystemExit("Khong thay %s (dung --a-lcam-json / --b-lcam-json neu file da doi ten)" % json_path)
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "steps" not in data or "cam" not in data:
        raise SystemExit("%s: can khoa 'steps' va 'cam'. Cac khoa: %s"
                         % (json_path, list(data.keys()) if isinstance(data, dict) else type(data)))

    steps = data["steps"]
    if isinstance(steps, dict):
        steps = list(steps.values())
    by_conv = {}
    for step in steps:
        by_conv[int(_pick(step, ["conv_index"], "steps[]"))] = step

    bwd = []
    for layer, ((conv_index, _name), (in_ch, out_ch)) in enumerate(zip(BWD_STEPS, BWD_LAYERS)):
        label = "steps[conv_index=%d]" % conv_index
        if conv_index not in by_conv:
            raise SystemExit("%s: khong co %s (co: %s)" % (json_path, label, sorted(by_conv)))
        step = by_conv[conv_index]
        m0 = int(_pick(step, ["requant_M0", "M0", "m0"], label))
        shift = int(_pick(step, ["requant_shift", "shift", "SHIFT"], label))
        _check_m0_shift(label, m0, shift)
        if "weight_shape_out_in_kh_kw" in step:
            shape = [int(v) for v in step["weight_shape_out_in_kh_kw"]]
            if shape != [out_ch, in_ch, 3, 3]:
                raise SystemExit("%s: weight_shape %s, can %s" % (label, shape, [out_ch, in_ch, 3, 3]))
        else:
            print("  [canh bao] %s khong co weight_shape_out_in_kh_kw, bo qua kiem hinh dang" % label)
        bwd.append((m0, shift))

    fc = data.get("fc")
    if isinstance(fc, dict) and "shape" in fc:
        if [int(v) for v in fc["shape"]] != [5, 48]:
            raise SystemExit("%s: fc.shape=%s, can [5, 48]" % (json_path, fc["shape"]))
    else:
        print("  [canh bao] %s khong co fc.shape, bo qua kiem tra" % json_path)

    cam = data["cam"]
    cam_m0 = int(_pick(cam, ["requant_M0", "M0", "m0"], "cam"))
    cam_shift = int(_pick(cam, ["requant_shift", "shift", "SHIFT"], "cam"))
    _check_m0_shift("cam", cam_m0, cam_shift)
    return bwd, (cam_m0, cam_shift)


def pack_cnn(hospital_dir, cnn_id, unknown_override, lcam_dir, normal_override=None, lcam_json=None):
    print("Dong goi CNN %d tu %s (LCAM: %s)" % (cnn_id, hospital_dir, lcam_json or lcam_dir))
    int8_dir = os.path.join(hospital_dir, "INT8")
    memory_dir = os.path.join(int8_dir, "Memory")
    preprocess_dir = os.path.join(int8_dir, "Preprocess")

    params = load_layer_params(os.path.join(memory_dir, "quant_params.json"))
    unknown_index = get_unknown_index(hospital_dir, unknown_override)
    normal_index = get_normal_index(hospital_dir, normal_override)
    num_classes = LAYERS[-1][1]

    weight_files = find_layer_files(memory_dir, "_weight_int8.mem")
    bias_files = find_layer_files(memory_dir, "_bias_int32.mem")

    block = bytearray(CNN_BLOCK_BYTES)

    # header
    block[0:4] = b"FCNN"
    block[4] = cnn_id
    block[5] = unknown_index
    block[6] = num_classes
    block[7] = normal_index
    for layer, (m0, shift, zp_in, zp_out, relu) in enumerate(params):
        struct.pack_into("<IBBBB", block, 8 + 8 * layer, m0, shift, zp_in, zp_out, relu)

    # LUT R, G, B
    for channel, name in enumerate("RGB"):
        lut = read_mem_tokens(os.path.join(preprocess_dir, "input_lut_%s.mem" % name))
        if len(lut) != 256 or any(not (0 <= v <= 255) for v in lut):
            raise SystemExit("input_lut_%s.mem: can 256 gia tri 0..255, thay %d" % (name, len(lut)))
        block[0x200 + 256 * channel: 0x200 + 256 * (channel + 1)] = bytes(lut)

    # du lieu cac lop forward
    for layer, (in_ch, out_ch) in enumerate(LAYERS):
        weights = [to_signed(v, 8) for v in read_mem_tokens(weight_files[layer])]
        biases = [to_signed(v, 32) for v in read_mem_tokens(bias_files[layer])]
        kernel_area = 1 if layer == 5 else 9          # FC: moi PE chi co 48 trong so
        if len(weights) != out_ch * in_ch * kernel_area:
            raise SystemExit("%s: can %d trong so, thay %d"
                             % (weight_files[layer], out_ch * in_ch * kernel_area, len(weights)))
        if len(biases) != out_ch:
            raise SystemExit("%s: can %d bias, thay %d" % (bias_files[layer], out_ch, len(biases)))
        if 32 + 8 * in_ch * kernel_area != GROUP_BYTES[layer]:
            raise SystemExit("Noi bo: GROUP_BYTES lop %d khong khop" % layer)

        position = write_layer(block, LAYER_OFFSETS[layer], GROUP_COUNTS[layer], in_ch, kernel_area,
                               weights, biases)
        expected_end = LAYER_OFFSETS[layer] + GROUP_COUNTS[layer] * GROUP_BYTES[layer]
        if position != expected_end:
            raise SystemExit("Noi bo: lop %d ket thuc tai %#x, mong %#x" % (layer, position, expected_end))

    # ---- LayerCAM ----
    # 1. Lop conv3, conv4, conv5 phai co ZP_OUT = 0 (gradient/mask LCAM dua vao gia tri > 0)
    for layer in (2, 3, 4):
        if params[layer][3] != 0:
            raise SystemExit("Lop %d (conv%d): zp_out=%d, LayerCAM can zp_out == 0"
                             % (layer, layer + 1, params[layer][3]))

    # 2. Tham so LCAM
    json_path = lcam_json or os.path.join(lcam_dir, "lcam_params.json")
    bwd_params, (cam_m0, cam_shift) = load_lcam_params(json_path)

    # 3. Bang 0x8000 (24 byte)
    for k, (m0, shift) in enumerate(bwd_params):
        struct.pack_into("<IBBBB", block, LCAM_TABLE_OFFSET + 8 * k, m0, shift, 128, 128, 0)
    struct.pack_into("<IB", block, LCAM_TABLE_OFFSET + 0x10, cam_m0, cam_shift)

    # 4. Trong so BWD (bias = 0)
    for k, ((conv_index, file_name), (in_ch, out_ch)) in enumerate(zip(BWD_STEPS, BWD_LAYERS)):
        path = os.path.join(lcam_dir, file_name)
        if not os.path.isfile(path):
            raise SystemExit("Khong thay file %s" % path)
        weights = [to_signed(v, 8) for v in read_mem_tokens(path)]
        if len(weights) != out_ch * in_ch * 9:
            raise SystemExit("%s: can %d trong so, thay %d" % (path, out_ch * in_ch * 9, len(weights)))
        if 32 + 8 * in_ch * 9 != BWD_GROUP_BYTES[k]:
            raise SystemExit("Noi bo: BWD_GROUP_BYTES %d khong khop" % k)
        position = write_layer(block, BWD_OFFSETS[k], BWD_GROUP_COUNTS[k], in_ch, 9,
                               weights, [0] * out_ch)
        expected_end = BWD_OFFSETS[k] + BWD_GROUP_COUNTS[k] * BWD_GROUP_BYTES[k]
        if position != expected_end:
            raise SystemExit("Noi bo: BWD%d ket thuc tai %#x, mong %#x" % (k + 1, position, expected_end))
    if BWD_OFFSETS[1] + BWD_GROUP_COUNTS[1] * BWD_GROUP_BYTES[1] != BWD_END:
        raise SystemExit("Noi bo: diem cuoi BWD2 khac %#x" % BWD_END)

    print("  LCAM: lop6 M0=%d SHIFT=%d | lop7 M0=%d SHIFT=%d | CAM M0=%d SHIFT=%d | Normal=%#04x Unknown=%d"
          % (bwd_params[0][0], bwd_params[0][1], bwd_params[1][0], bwd_params[1][1],
             cam_m0, cam_shift, normal_index, unknown_index))
    return bytes(block)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="thu muc chua Hospital_A va Hospital_B (vd ./BEST)")
    parser.add_argument("--a-dir", help="thu muc Hospital_A (chua INT8/)")
    parser.add_argument("--b-dir", help="thu muc Hospital_B (chua INT8/)")
    parser.add_argument("--lcam-root", default="cam_out",
                        help="goc LayerCAM: <root>/A/hw_lcam/Hospital_A/LayerCAM_B3 va <root>/B/hw_lcam/Hospital_B/LayerCAM_B3")
    parser.add_argument("--a-lcam-dir", help="thu muc LayerCAM_B3 cua Hospital_A (ghi de --lcam-root)")
    parser.add_argument("--b-lcam-dir", help="thu muc LayerCAM_B3 cua Hospital_B (ghi de --lcam-root)")
    parser.add_argument("--a-lcam-json", help="duong dan JSON LCAM Hospital_A (neu khong ten lcam_params.json)")
    parser.add_argument("--b-lcam-json", help="duong dan JSON LCAM Hospital_B (neu khong ten lcam_params.json)")
    parser.add_argument("--unknown-index", type=int, default=None,
                        help="chi so lop Unknown neu khong tim thay file classes")
    parser.add_argument("--normal-index", type=int, default=None,
                        help="chi so lop Normal neu khong tim thay file classes (255 = khong co)")
    parser.add_argument("--out-bin", default="sd_image.bin")
    parser.add_argument("--out-hex", default="sd_image.hex")
    args = parser.parse_args()

    dir_a = args.a_dir or os.path.join(args.root, "Hospital_A")
    dir_b = args.b_dir or os.path.join(args.root, "Hospital_B")
    lcam_a = args.a_lcam_dir or os.path.join(args.lcam_root, "A", "hw_lcam", "Hospital_A", "LayerCAM_B3")
    lcam_b = args.b_lcam_dir or os.path.join(args.lcam_root, "B", "hw_lcam", "Hospital_B", "LayerCAM_B3")

    image = (pack_cnn(dir_a, 1, args.unknown_index, lcam_a, args.normal_index, args.a_lcam_json)
             + pack_cnn(dir_b, 2, args.unknown_index, lcam_b, args.normal_index, args.b_lcam_json))
    assert len(image) == 2 * CNN_BLOCK_BYTES

    with open(args.out_bin, "wb") as handle:
        handle.write(image)
    with open(args.out_hex, "w", encoding="utf-8") as handle:
        for byte in image:
            handle.write("%02x\n" % byte)
    print("Da ghi %s (%d byte) va %s (%d dong)" % (args.out_bin, len(image), args.out_hex, len(image)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
