#!/usr/bin/env python3
"""So sanh tung lop giua file .mem xuat tu RTL (TB_CNN_RUNNER) va file tham chieu (Python).

Mac dinh so rtl_img0_{ten}_{tag}.mem voi test_images/ref_img0_{ten}_{tag}.mem (do make_test_images.py sinh),
cho cac lop: input, pool1, pool2, pool3, conv4, conv5, gap, fc va tag cnn1 (cnn2 neu co file).

Vi du:
  # RTL vs mo hinh numpy
  python compare_layers.py --rtl-pattern "rtl_img0_{name}_{tag}.mem" --ref-pattern "test_images/ref_img0_{name}_{tag}.mem"

  # RTL vs golden goc (TestVectors), chi CNN 1, file golden khong co hau to tag
  python compare_layers.py --rtl-pattern "rtl_img0_{name}_cnn1.mem" --ref-pattern "TestVectors/dbg_img0_{name}.mem" --tags cnn1

  # Kiem tra mo hinh numpy co khop golden goc khong (nen lam truoc)
  python compare_layers.py --rtl-pattern "test_images/ref_img0_{name}_cnn1.mem" --ref-pattern "TestVectors/dbg_img0_{name}.mem" --tags cnn1

  # Them kiem tra ket qua cascade
  python compare_layers.py ... --rtl-result rtl_img0_result.txt --expected test_images/expected_cascade.json --index 0

Thoat 0 neu moi thu khop, 1 neu co lech. {name} va {tag} la cho trong pattern.
"""
import argparse
import json
import os
import sys

import numpy as np

# (ten lop, shape [row, col, channel] hoac [n])
LAYERS = [
    ("input", (64, 64, 3)),
    ("pool1", (32, 32, 16)),
    ("pool2", (16, 16, 24)),
    ("pool3", (8, 8, 24)),
    ("conv4", (8, 8, 32)),
    ("conv5", (8, 8, 48)),
    ("gap",   (48,)),
    ("fc",    (5,)),
]
INVALID = 999   # gia tri danh dau token khong phai hex (vi du xx tu RTL)


def read_mem(path):
    values = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.split("//")[0].strip()
            if not line:
                continue
            for token in line.split():
                try:
                    values.append(int(token, 16))
                except ValueError:
                    values.append(INVALID)
    return np.array(values, dtype=np.int64)


def describe_index(flat_index, shape):
    if len(shape) == 3:
        rows, cols, channels = shape
        row, rest = divmod(flat_index, cols * channels)
        col, channel = divmod(rest, channels)
        return "[row=%d col=%d ch=%d]" % (row, col, channel)
    return "[%d]" % flat_index


def compare_layer(name, shape, rtl_path, ref_path, max_show):
    """Tra ve (trang_thai, so_lech). trang_thai: OK, LECH, THIEU."""
    if not os.path.exists(rtl_path):
        print("  %-6s THIEU file RTL: %s" % (name, rtl_path))
        return "THIEU", 0
    if not os.path.exists(ref_path):
        print("  %-6s THIEU file tham chieu: %s" % (name, ref_path))
        return "THIEU", 0

    rtl = read_mem(rtl_path)
    ref = read_mem(ref_path)
    expected_count = int(np.prod(shape))
    note = ""
    if len(rtl) != len(ref):
        note = " (RTL %d gia tri, ref %d gia tri; chi so sanh %d dau)" % (len(rtl), len(ref), min(len(rtl), len(ref)))
    count = min(len(rtl), len(ref))
    if count != expected_count:
        note += " [mong %d]" % expected_count
    rtl, ref = rtl[:count], ref[:count]

    different = np.nonzero(rtl != ref)[0]
    if len(different) == 0 and not note.strip().startswith("(RTL"):
        print("  %-6s OK    %d/%d khop%s" % (name, count, count, note))
        return "OK", 0
    if len(different) == 0:
        print("  %-6s LECH  do dai khac nhau%s" % (name, note))
        return "LECH", 1

    valid = (rtl[different] != INVALID) & (ref[different] != INVALID)
    max_diff = int(np.max(np.abs(rtl[different][valid] - ref[different][valid]))) if valid.any() else -1
    print("  %-6s LECH  %d/%d gia tri sai, |diff| lon nhat = %d%s" % (name, len(different), count, max_diff, note))
    for flat_index in different[:max_show]:
        print("           %s RTL=%s ref=%s" % (
            describe_index(int(flat_index), shape),
            "xx" if rtl[flat_index] == INVALID else "%02x" % rtl[flat_index],
            "xx" if ref[flat_index] == INVALID else "%02x" % ref[flat_index]))
    if len(different) > max_show:
        print("           ... con %d gia tri sai nua" % (len(different) - max_show))

    if len(shape) == 3:
        channels = shape[2]
        per_channel = np.bincount(different % channels, minlength=channels)
        bad_channels = [(c, int(per_channel[c])) for c in range(channels) if per_channel[c]]
        print("           kenh co loi (kenh:so_lan): %s" % ", ".join("%d:%d" % item for item in bad_channels[:24]))
        first = int(different[0])
        print("           vi tri sai dau tien: %s" % describe_index(first, shape))
    return "LECH", len(different)


def compare_cascade(result_path, expected_path, index):
    with open(expected_path, "r", encoding="utf-8") as handle:
        expected = json.load(handle)
    golden = next((e for e in expected if e["index"] == index), None)
    if golden is None:
        print("  expected_cascade.json khong co anh index=%d" % index)
        return False
    with open(result_path, "r", encoding="utf-8") as handle:
        numbers = [int(t) for t in handle.read().split()]
    if len(numbers) != 12:
        print("  %s: mong 12 so, thay %d" % (result_path, len(numbers)))
        return False
    got = {"final_source": numbers[0], "final_class": numbers[1],
           "logits_cnn1": numbers[2:7], "logits_cnn2": numbers[7:12]}
    ok = True
    for key in ("final_source", "final_class", "logits_cnn1", "logits_cnn2"):
        same = got[key] == golden[key]
        ok = ok and same
        print("  %-13s %s   RTL=%s golden=%s" % (key, "OK  " if same else "LECH", got[key], golden[key]))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rtl-pattern", default="rtl_img0_{name}_{tag}.mem")
    parser.add_argument("--ref-pattern", default="test_images/ref_img0_{name}_{tag}.mem")
    parser.add_argument("--tags", nargs="+", default=["cnn1", "cnn2"])
    parser.add_argument("--layers", nargs="+", help="chi so sanh cac lop nay (mac dinh: tat ca)")
    parser.add_argument("--max-show", type=int, default=8, help="so vi tri sai in ra moi lop")
    parser.add_argument("--rtl-result", help="rtl_img0_result.txt")
    parser.add_argument("--expected", help="expected_cascade.json")
    parser.add_argument("--index", type=int, default=0, help="chi so anh trong expected_cascade.json")
    args = parser.parse_args()

    selected = [(n, s) for n, s in LAYERS if not args.layers or n in args.layers]
    all_ok = True

    for tag in args.tags:
        rtl_first = args.rtl_pattern.format(name=selected[0][0], tag=tag)
        if not os.path.exists(rtl_first) and tag != args.tags[0]:
            print("[%s] khong co file RTL (CNN nay khong chay) - bo qua" % tag)
            continue
        print("[%s]" % tag)
        first_failure = None
        for name, shape in selected:
            status, _ = compare_layer(name, shape,
                                      args.rtl_pattern.format(name=name, tag=tag),
                                      args.ref_pattern.format(name=name, tag=tag),
                                      args.max_show)
            if status != "OK":
                all_ok = False
                if first_failure is None:
                    first_failure = name
        if first_failure:
            print("  => Lop dau tien lech: %s (loi o day thuong keo theo cac lop sau; debug lop nay truoc)" % first_failure)

    if args.rtl_result and args.expected:
        print("[cascade]")
        if not compare_cascade(args.rtl_result, args.expected, args.index):
            all_ok = False

    print("KET QUA: %s" % ("TAT CA KHOP" if all_ok else "CO LECH"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
