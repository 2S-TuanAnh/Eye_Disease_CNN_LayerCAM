#!/usr/bin/env python3
"""PC phia host: gui anh 64x64 RGB qua UART cho FPGA, so ket qua (va LayerCAM) voi golden.

Giao thuc (SEPC.md muc 5 + LayerCAM), 8N1, mac dinh 1 Mbaud:
  FPGA -> PC sau khoi dong: 0x55 (san sang)  hoac  0xEE, ma_loi (1 = SD init, 2 = doc sector, 3 = sai magic)
  PC -> FPGA: 0xA5 + 12288 byte (R,G,B tung pixel, theo hang)
  FPGA -> PC: khung 14 byte (hoac 270 byte neu CAM_FLAG = 1):
      0xAA, FINAL_SOURCE, FINAL_CLASS, 5 logit CNN 1, 5 logit CNN 2, CAM_FLAG (0 hoac 1),
      [neu CAM_FLAG = 1: 256 byte cam_u8, luoi 16x16 theo hang]

Dung:
  python host_cascade.py --port COM5 --expected test_images/expected_cascade.json
  python host_cascade.py --port COM5 --expected test_images/expected_cascade.json --show --save-dir cam_png
  python host_cascade.py --port /dev/ttyUSB0 --baud 115200 --mem image0_raw64.mem
Neu FPGA da bao 0x55 tu truoc luc mo cong thi dung --skip-ready.
--show / --save-dir ve 1 hinh 2 panel (trai: anh goc, phai: LayerCAM, khong khung) tu cam_u8 (can torch, matplotlib, cam_best_cnn_v3.py; chi anh co
source_path trong expected_cascade.json).
"""
import argparse
import json
import os
import sys
import time

import serial  # pyserial
import matplotlib
matplotlib.use("Qt5Agg", force=True)

IMAGE_BYTES = 64 * 64 * 3
START_BYTE = 0xA5
READY_BYTE = 0x55
ERROR_BYTE = 0xEE
RESULT_HEADER = 0xAA
RESULT_LENGTH = 14
CAM_BYTES = 256
HEAT_ALPHA = 0.45      # do dam heatmap, giong code goc
BOOT_ERRORS = {1: "SD init loi", 2: "doc sector loi", 3: "sai magic"}
COMPARE_KEYS = ("final_source", "final_class", "logits_cnn1", "logits_cnn2", "cam_flag", "cam_u8")


def read_mem(path):
    data = bytearray()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.split("//")[0].strip()
            if line:
                data.append(int(line, 16))
    if len(data) != IMAGE_BYTES:
        raise SystemExit("%s: can %d byte, thay %d" % (path, IMAGE_BYTES, len(data)))
    return bytes(data)


def read_exact(port, count, timeout_seconds):
    deadline = time.time() + timeout_seconds
    data = bytearray()
    while len(data) < count and time.time() < deadline:
        chunk = port.read(count - len(data))
        data.extend(chunk)
    return bytes(data)


def wait_ready(port, timeout_seconds):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        first = port.read(1)
        if not first:
            continue
        if first[0] == READY_BYTE:
            return
        if first[0] == ERROR_BYTE:
            code = read_exact(port, 1, 1.0)
            number = code[0] if code else -1
            raise SystemExit("FPGA bao loi khoi dong: ma %d (%s)" % (number, BOOT_ERRORS.get(number, "khong ro")))
    raise SystemExit("Qua thoi gian cho 0x55 (%.0f s). Kiem tra the SD, SDRAM, cong UART, toc do baud." % timeout_seconds)


def run_image(port, image_bytes, result_timeout):
    port.reset_input_buffer()
    port.write(bytes([START_BYTE]) + image_bytes)
    port.flush()
    result = read_exact(port, RESULT_LENGTH, result_timeout)
    if len(result) != RESULT_LENGTH:
        raise SystemExit("Chi nhan %d/%d byte ket qua (FPGA khong tra loi kip hoac bo khung do im lang > 100 ms)"
                         % (len(result), RESULT_LENGTH))
    if result[0] != RESULT_HEADER:
        raise SystemExit("Byte dau khung ket qua = 0x%02X, mong 0xAA. Du lieu tho: %s" % (result[0], result.hex(" ")))
    cam_flag = result[13]
    if cam_flag not in (0, 1):
        raise SystemExit("CAM_FLAG = 0x%02X, chi nhan 0 hoac 1. Du lieu tho: %s" % (cam_flag, result.hex(" ")))
    cam_u8 = []
    if cam_flag == 1:
        cam = read_exact(port, CAM_BYTES, result_timeout)
        if len(cam) != CAM_BYTES:
            raise SystemExit("Chi nhan %d/%d byte CAM" % (len(cam), CAM_BYTES))
        cam_u8 = list(cam)
    return {
        "final_source": result[1],
        "final_class": result[2],
        "logits_cnn1": list(result[3:8]),
        "logits_cnn2": list(result[8:13]),
        "cam_flag": cam_flag,
        "cam_u8": cam_u8,
    }


def describe_difference(key, got, golden):
    """Chuoi mo ta lech; cam_u8 chi in so phan tu khac thay vi ca 256 so."""
    if key == "cam_u8":
        if len(got) != len(golden):
            return "so byte CAM: FPGA=%d golden=%d" % (len(got), len(golden))
        differing = sum(1 for a, b in zip(got, golden) if a != b)
        return "%d/%d phan tu khac" % (differing, len(golden))
    return "FPGA=%s golden=%s" % (got, golden)


def full_res_fov(crop, thr=12):
    """Mat na vung mat o do phan giai anh goc (khong erode) -> vien muot, phu sat mep."""
    import numpy as np
    from scipy import ndimage
    mask = ndimage.binary_fill_holes(crop.max(axis=2) > thr)
    lab, count = ndimage.label(mask)
    if count > 1:
        mask = lab == (1 + int(np.argmax(ndimage.sum(mask, lab, range(1, count + 1)))))
    return mask if mask.any() else np.ones(mask.shape, bool)


def show_cam(source_path, cam_u8, title, save_dir, figures, module_dir, cam_low=0.25, cam_high=0.60):
    """Ve 1 hinh gom 2 panel: trai = anh goc, phai = LayerCAM phu toan vung mat (khong khung). Tra ve True neu ve duoc."""
    if module_dir and module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib
    if not figures and save_dir and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wanted_backend = matplotlib.get_backend()
    from cam_best_cnn_v3 import load_raw, fov_mask
    # cam_best_cnn_v3.py goi matplotlib.use("Agg") khi import -> tra lai backend co giao dien (vd Qt5Agg)
    if not figures and matplotlib.get_backend().lower() != wanted_backend.lower():
        try:
            plt.switch_backend(wanted_backend)
        except Exception as exc:
            print("    [canh bao] khong doi lai duoc backend %s: %s" % (wanted_backend, exc))

    raw = load_raw(source_path)
    height0, width0 = raw.shape[:2]
    crop = raw[:(height0 // 64) * 64, :(width0 // 64) * 64]   # phan du duoi/phai bi bo, giong luc resize
    height, width = crop.shape[:2]

    # Gia tri heat GIONG HET code goc: hw_heat voi mask erode mac dinh (cung chuan hoa, cung mau)
    from scipy import ndimage
    from cam_best_cnn_v3 import hw_heat
    mask_in = fov_mask(raw)                                     # erode=2 nhu code goc
    heat64 = hw_heat(torch.from_numpy(np.asarray(cam_u8, dtype=np.uint8).reshape(16, 16)),
                     torch.from_numpy(mask_in)).numpy()
    # Chi de phu sat vien: lap day phan bi erode bang gia tri gan nhat ben trong (khong doi gia tri/mau ben trong)
    idx = ndimage.distance_transform_edt(~mask_in, return_distances=False, return_indices=True)
    heat64 = heat64[idx[0], idx[1]]
    heat = F.interpolate(torch.from_numpy(heat64)[None, None].float(), size=(height, width),
                         mode="bilinear", align_corners=False)[0, 0].numpy()

    # Loc nhieu: chi cac cum (heat >= cam_low) co loi cam-do (heat >= cam_high) moi giu gia tri CAM.
    # Moi noi khac bi dua ve MOT muc nen phang duy nhat (tim dong deu), chuyen tiep mem quanh cum giu lai
    # -> khong con mang sang/toi khac mau nen, va khong co duong vien.
    fov = full_res_fov(crop)
    lab, count = ndimage.label(heat >= cam_low)
    strong = np.zeros(count + 1, bool)
    if count:
        peaks = np.asarray(ndimage.maximum(heat, lab, range(1, count + 1)))
        strong[1:] = peaks >= cam_high
    strong_px = strong[lab]
    low_vals = heat[fov & ~strong_px & (heat < cam_low)]
    bg_level = float(np.percentile(low_vals, 30)) if low_vals.size else 0.08   # muc nen tim
    if strong_px.any():
        dist = ndimage.distance_transform_edt(~strong_px)
        reach = 0.04 * min(height, width)                               # do rong vung chuyen tiep
        keep = np.clip(1.0 - dist / reach, 0.0, 1.0)
        keep = keep * keep * (3.0 - 2.0 * keep)                        # smoothstep
        keep = ndimage.gaussian_filter(keep, 0.01 * min(height, width))
    else:
        keep = np.zeros_like(heat)
    heat = bg_level + (heat - bg_level) * keep
    rgba = plt.get_cmap("jet")(np.clip(heat, 0.0, 1.0))                # mau jet nhu code goc
    rgba[..., 3] = HEAT_ALPHA * fov

    fig, (ax_raw, ax_cam) = plt.subplots(1, 2, figsize=(11, 5.6))
    ax_raw.imshow(crop)
    ax_raw.set_title("Anh goc", fontsize=10)
    ax_cam.imshow(crop)
    ax_cam.imshow(rgba, interpolation="bilinear")
    ax_cam.set_title("LayerCAM", fontsize=10)
    for ax in (ax_raw, ax_cam):
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(source_path))[0]
        fig.savefig(os.path.join(save_dir, "%s_cam.png" % stem), dpi=150, bbox_inches="tight")
    figures.append(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--expected", help="expected_cascade.json do make_test_images.py tao")
    parser.add_argument("--mem", nargs="*", default=[], help="them file imageN_raw64.mem (khong co golden)")
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--result-timeout", type=float, default=5.0)
    parser.add_argument("--skip-ready", action="store_true")
    parser.add_argument("--show", action="store_true", help="hien heatmap + khung cho khung co cam_flag = 1")
    parser.add_argument("--save-dir", help="luu anh heatmap (png) vao thu muc nay")
    parser.add_argument("--cam-low", type=float, default=0.25, help="nguong vien cum CAM (0-1)")
    parser.add_argument("--cam-high", type=float, default=0.60, help="cum phai co dinh >= nguong nay moi giu nguyen, con lai bi ha xuong nen")
    parser.add_argument("--module-dir", default=".", help="thu muc chua cam_best_cnn_v3.py (cho --show/--save-dir)")
    args = parser.parse_args()

    jobs = []
    if args.expected:
        base = os.path.dirname(os.path.abspath(args.expected))
        with open(args.expected, "r", encoding="utf-8") as handle:
            for entry in json.load(handle):
                jobs.append((os.path.join(base, entry["raw_file"]), entry))
    for path in args.mem:
        jobs.append((path, None))
    if not jobs:
        raise SystemExit("Can --expected hoac --mem")

    draw = args.show or bool(args.save_dir)
    figures = []
    mismatches = 0
    with serial.Serial(args.port, args.baud, bytesize=8, parity=serial.PARITY_NONE,
                       stopbits=1, timeout=0.2) as port:
        if not args.skip_ready:
            print("Cho 0x55 tu FPGA ...")
            wait_ready(port, args.ready_timeout)
        print("FPGA san sang.")

        for path, golden in jobs:
            image_bytes = read_mem(path)
            start = time.time()
            got = run_image(port, image_bytes, args.result_timeout)
            elapsed_ms = (time.time() - start) * 1000.0
            line = "%s: source=%d class=%d cam_flag=%d logits1=%s logits2=%s (%.0f ms)" % (
                os.path.basename(path), got["final_source"], got["final_class"], got["cam_flag"],
                got["logits_cnn1"], got["logits_cnn2"], elapsed_ms)
            if golden is None:
                print(line)
            else:
                keys = [key for key in COMPARE_KEYS if key in golden]
                same = all(got[key] == golden[key] for key in keys)
                print(line + ("  OK" if same else "  KHAC GOLDEN"))
                if not same:
                    mismatches += 1
                    for key in keys:
                        if got[key] != golden[key]:
                            print("    %s: %s" % (key, describe_difference(key, got[key], golden[key])))

            source_path = golden.get("source_path") if golden else None
            if draw and got["cam_flag"] == 1 and source_path:
                title = "%s: class %d (CNN %d)" % (os.path.basename(source_path), got["final_class"],
                                                    got["final_source"])
                try:
                    show_cam(source_path, got["cam_u8"], title, args.save_dir, figures, args.module_dir,
                             args.cam_low, args.cam_high)
                except ImportError as exc:
                    print("    [canh bao] khong ve duoc CAM (thieu module: %s)" % exc)
                    draw = False
                except OSError as exc:
                    print("    [canh bao] khong doc duoc anh goc %s: %s" % (source_path, exc))

    print("Xong: %d anh, %d lech." % (len(jobs), mismatches))
    if args.show and figures:
        import matplotlib.pyplot as plt
        plt.show()
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
