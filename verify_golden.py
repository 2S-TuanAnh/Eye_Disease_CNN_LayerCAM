#!/usr/bin/env python3
"""Doi chieu sd_image.bin + mo hinh numpy (make_test_images.py) voi golden do CNN_Search_BN.py xuat.

Doc <Hospital>/INT8/TestVectors/:
  test_raw_images.mem   anh goc (src_w x src_h), moi dong 1 pixel = BBGGRR (hex)
  testvec_info.json     num_images, src_w, src_h
  golden_logits.txt     logit u8 cua golden cho TOAN BO anh (N x 5)
  dbg_img{0,1}_*.mem    gia tri tung lop cua anh 0 va 1

Cach chay (tu thu muc chua make_test_images.py, CNN_Search_BN.py, sd_image.bin):
  python verify_golden.py --hospital-dir BEST/Hospital_A --cnn 1
  python verify_golden.py --hospital-dir BEST/Hospital_B --cnn 2
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import CNN_Search_BN as search                      # noqa: E402
from make_test_images import CnnBlock, run_cnn, CNN_BLOCK_BYTES   # noqa: E402


def read_hex_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        return np.array([int(line, 16) for line in handle.read().split()], dtype=np.int64)


def load_raw_images(tv_dir):
    with open(os.path.join(tv_dir, "testvec_info.json"), "r", encoding="utf-8") as handle:
        info = json.load(handle)
    n, w, h = info["num_images"], info["src_w"], info["src_h"]
    with open(os.path.join(tv_dir, "test_raw_images.mem"), "r", encoding="utf-8") as handle:
        text = handle.read().replace("\n", "").replace("\r", "")
    data = np.frombuffer(bytes.fromhex(text), dtype=np.uint8)
    if data.size != n * h * w * 3:
        raise SystemExit("test_raw_images.mem: co %d byte, mong %d (%d anh %dx%d)" % (data.size, n * h * w * 3, n, w, h))
    bgr = data.reshape(n, h, w, 3)
    return bgr[..., ::-1].copy(), info        # dong pixel la B,G,R -> doi sang R,G,B


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hospital-dir", required=True, help="vd BEST/Hospital_A")
    parser.add_argument("--cnn", type=int, choices=(1, 2), required=True, help="1 = Hospital_A, 2 = Hospital_B")
    parser.add_argument("--sd-image", default="sd_image.bin")
    args = parser.parse_args()

    tv_dir = os.path.join(args.hospital_dir, "INT8", "TestVectors")
    with open(args.sd_image, "rb") as handle:
        sd = handle.read()
    block = sd[:CNN_BLOCK_BYTES] if args.cnn == 1 else sd[CNN_BLOCK_BYTES:]
    cnn = CnnBlock(block)

    raw, info = load_raw_images(tv_dir)
    golden_logits = np.loadtxt(os.path.join(tv_dir, "golden_logits.txt"), dtype=np.int64)
    golden_logits = golden_logits.reshape(len(raw), -1)
    print("Hospital: %s | %d anh %dx%d | CNN %d (unknown index %d)" %
          (args.hospital_dir, len(raw), info["src_w"], info["src_h"], args.cnn, cnn.unknown_class_index))

    logit_ok = 0
    all_layers = []
    for i in range(len(raw)):
        small = search.enhance_int(search.block_average_resize(raw[i], search.IMG_SIZE), search.ENHANCE_MODE)
        logits, _cls, layers = run_cnn(np.asarray(small, dtype=np.uint8), cnn)
        all_layers.append(layers)
        same = np.array_equal(np.asarray(logits, dtype=np.int64), golden_logits[i])
        logit_ok += int(same)
        if not same:
            print("  anh %2d: LECH logit  ours=%s  golden=%s" % (i, list(map(int, logits)), list(golden_logits[i])))
    print("Logit trung golden: %d/%d anh" % (logit_ok, len(raw)))

    print("\nSo tung lop (dbg_img*):")
    layer_ok = True
    for i in range(2):
        for name in ("input", "pool1", "pool2", "pool3", "conv4", "conv5", "gap", "fc"):
            path = os.path.join(tv_dir, "dbg_img%d_%s.mem" % (i, name))
            if not os.path.isfile(path):
                print("  img%d %-6s: (khong co file dbg)" % (i, name))
                continue
            gold = read_hex_file(path)
            ours = np.asarray(all_layers[i][name], dtype=np.int64).reshape(-1)
            if gold.size != ours.size:
                print("  img%d %-6s: KHAC KICH THUOC  golden=%d ours=%d" % (i, name, gold.size, ours.size))
                layer_ok = False
            elif np.array_equal(gold, ours):
                print("  img%d %-6s: TRUNG (%d gia tri)" % (i, name, ours.size))
            else:
                bad = np.flatnonzero(gold != ours)
                print("  img%d %-6s: LECH %d/%d gia tri, dau tien tai vi tri %d (golden=%d ours=%d)" %
                      (i, name, bad.size, ours.size, bad[0], gold[bad[0]], ours[bad[0]]))
                layer_ok = False

    ok = (logit_ok == len(raw)) and layer_ok
    print("\nKET LUAN:", "BIT-EXACT, dong goi + mo hinh numpy dung." if ok else "CO LECH, xem dong LECH dau tien o tren.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
