"""Sinh anh SD tong hop (131072 byte) + ket qua ky vong cho 2 CNN, theo bo cuc cua RTL / huong dan LCAM."""
import math, struct, sys
import numpy as np

rng = np.random.default_rng(12345)

LAYERS = [  # in_size, in_ch, out_ch, pool, offset, group_bytes, group_count
    (64, 3, 16, True, 1536, 248, 2),
    (32, 16, 24, True, 2032, 1184, 3),
    (16, 24, 24, True, 5584, 1760, 3),
    (8, 24, 32, False, 10864, 1760, 4),
    (8, 32, 48, False, 17904, 2336, 6),
]
FC_OFFSET, FC_GROUP_BYTES = 31920, 416
BWD = [(48, 32, 33024, 3488, 4), (32, 24, 46976, 2336, 3)]
BLOCK = 0x10000


def calib(vals, target_std):
    std = max(float(np.std(vals)), 1.0)
    scale = target_std / std
    S = int(math.floor(math.log2((2**31 - 1) / scale)))
    S = max(1, min(63, S))
    M0 = int(round(scale * 2**S))
    while M0 >= 2**31:
        S -= 1
        M0 = int(round(scale * 2**S))
    assert 0 < M0 < 2**31 and 1 <= S <= 63
    return M0, S


def requant(acc, M0, S, zp, lower):
    acc = acc.astype(np.int64)
    assert np.abs(acc).max() < 2**31
    v = (acc * M0 + (1 << (S - 1))) >> S
    return np.clip(v + zp, lower, 255)


def conv3x3(xc, w, b):
    """xc: (H,W,Cin) da tru zp (int64); w: (Cout,Cin,3,3); b: (Cout,) -> acc (H,W,Cout)"""
    H, W, Ci = xc.shape
    Co = w.shape[0]
    pad = np.zeros((H + 2, W + 2, Ci), dtype=np.int64)
    pad[1:-1, 1:-1] = xc
    acc = np.zeros((H, W, Co), dtype=np.int64) + b.astype(np.int64)
    for ky in range(3):
        for kx in range(3):
            patch = pad[ky:ky + H, kx:kx + W, :]
            acc += np.einsum('hwi,oi->hwo', patch, w[:, :, ky, kx].astype(np.int64))
    return acc


def pack_groups(block, offset, group_bytes, groups, in_ch, k_area, w, bias):
    """w: (Cout_padded, Cin, k_area) int8 ; bias (Cout_padded,) int32"""
    wpp = in_ch * k_area
    assert group_bytes == 32 + 8 * wpp, (group_bytes, wpp)
    pos = offset
    for g in range(groups):
        for p in range(8):
            struct.pack_into("<i", block, pos + 4 * p, int(bias[g * 8 + p]))
        pos += 32
        for p in range(8):
            ws = w[g * 8 + p].reshape(-1).astype(np.int8)
            assert len(ws) == wpp
            block[pos:pos + wpp] = ws.tobytes()
            pos += wpp
    assert pos == offset + groups * group_bytes
    return pos


def build_cnn(unknown_idx, normal_idx, image):
    block = bytearray(BLOCK)
    block[0:4] = b"FCNN"
    block[5] = unknown_idx
    block[7] = normal_idx
    lut = np.zeros((3, 256), dtype=np.int64)
    for c in range(3):
        lut[c] = np.clip(np.arange(256) + rng.integers(-10, 10), 0, 255)
    block[0x200:0x200 + 768] = lut.astype(np.uint8).reshape(-1).tobytes()

    hdr = []   # (M0, S, zp_in, zp_out, relu)
    x = lut[np.arange(3)[None, None, :], image].astype(np.int64)      # (64,64,3) uint8 sau LUT
    zp_in = 128
    feats = {}
    for li, (sz, ci, co, pool, off, gb, gc) in enumerate(LAYERS):
        w = np.clip(np.round(rng.normal(0, 18, (co, ci, 3, 3))), -127, 127).astype(np.int8)
        b = rng.integers(-300, 300, co).astype(np.int32)
        acc = conv3x3(x - zp_in, w, b)
        if pool:
            Hh = sz // 2
            a4 = acc.reshape(Hh, 2, Hh, 2, co).transpose(0, 2, 4, 1, 3).reshape(Hh, Hh, co, 4)
            pidx = a4.argmax(-1)
            acc_p = a4.max(-1)
        else:
            pidx, acc_p = None, acc
        M0, S = calib(acc_p, 35)
        y = requant(acc_p, M0, S, 0, 0)
        zp_out = 0
        hdr.append((M0, S, zp_in, zp_out, 1))
        wp = w.reshape(co, ci, 9)
        pack_groups(block, off, gb, gc, ci, 9, wp, b)
        if li == 2:
            feats["pool3"], feats["pidx3"] = y.copy(), pidx.copy()
        if li == 3:
            feats["A4"] = y.copy()
        if li == 4:
            feats["A5"] = y.copy()
        x, zp_in = y, 0
    # GAP + FC
    gap = (feats["A5"].reshape(64, 48).sum(0) + 32) >> 6
    assert gap.max() <= 255
    wfc = np.zeros((8, 48), dtype=np.int8)
    wfc[:5] = np.clip(np.round(rng.normal(0, 25, (5, 48))), -127, 127).astype(np.int8)
    bfc = np.zeros(8, dtype=np.int32)
    bfc[:5] = rng.integers(-200, 200, 5)
    acc_fc = bfc.astype(np.int64) + (gap - 0)[None, :] @ wfc.astype(np.int64).T
    acc_fc = acc_fc.reshape(-1)
    M0, S = calib(acc_fc[:5], 30)
    logits = requant(acc_fc, M0, S, 128, 0)[:5]
    hdr.append((M0, S, 0, 128, 0))
    pack_groups(block, FC_OFFSET, FC_GROUP_BYTES, 1, 48, 1, wfc.reshape(8, 48, 1), bfc)
    pred = int(np.argmax(logits))   # primer maximum

    # header
    for li, (M0, S, zi, zo, rl) in enumerate(hdr):
        struct.pack_into("<IBBBB", block, 8 + 8 * li, M0, S, zi, zo, rl)

    # ---- LayerCAM ----
    cls = pred
    wb = []
    for (ci, co, off, gb, gc) in BWD:
        wb.append(np.clip(np.round(rng.normal(0, 20, (co, ci, 3, 3))), -127, 127).astype(np.int8))
    A5, A4 = feats["A5"], feats["A4"]
    dz5 = (A5 > 0) * wfc[cls].astype(np.int64)
    acc6 = conv3x3(dz5, wb[0], np.zeros(32))
    M6, S6 = calib(acc6, 25)
    g5 = np.clip((acc6 * M6 + (1 << (S6 - 1))) >> S6, -127, 127)
    dz4 = (A4 > 0) * g5
    acc7 = conv3x3(dz4, wb[1], np.zeros(24))
    M7, S7 = calib(acc7, 25)
    g4 = np.clip((acc7 * M7 + (1 << (S7 - 1))) >> S7, -127, 127)
    gm = np.maximum(g4, 0) * feats["pool3"]
    gm_up = np.repeat(np.repeat(gm, 2, 0), 2, 1)
    code_up = np.repeat(np.repeat(feats["pidx3"], 2, 0), 2, 1)
    rr, cc = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    expect = ((rr % 2) * 2 + (cc % 2))[..., None]
    cam_acc = ((code_up == expect) * gm_up).sum(-1)
    pos = cam_acc[cam_acc > 0]
    Mc, Sc = calib(cam_acc, 70) if pos.size else (1 << 30, 40)
    cam_u8 = np.clip((cam_acc * Mc + (1 << (Sc - 1))) >> Sc, 0, 255)

    struct.pack_into("<IBBBB", block, 0x8000, M6, S6, 128, 128, 0)
    struct.pack_into("<IBBBB", block, 0x8008, M7, S7, 128, 128, 0)
    struct.pack_into("<IB", block, 0x8010, Mc, Sc)
    for (ci, co, off, gb, gc), w in zip(BWD, wb):
        pack_groups(block, off, gb, gc, ci, 9, w.reshape(co, ci, 9), np.zeros(co, dtype=np.int32))
    assert BWD[1][2] + BWD[1][4] * BWD[1][3] == 0xD2E0

    info = dict(logits=[int(v) for v in logits], pred=pred, cam=cam_u8.reshape(-1).astype(int).tolist(),
                nonzero_cam=int((cam_u8 > 0).sum()), g4_pos=int((g4 > 0).sum()), g5_nz=int((g5 != 0).sum()),
                A4_nz=int((A4 > 0).sum()), M=(M6, S6, M7, S7, Mc, Sc), cam_max=int(cam_u8.max()))
    return bytes(block), info


def main():
    img = rng.integers(0, 256, (64, 64, 3))
    b1, i1 = build_cnn(unknown_idx=4, normal_idx=1, image=img)
    b2, i2 = build_cnn(unknown_idx=4, normal_idx=0xFF, image=img)
    image = b1 + b2
    assert len(image) == 131072
    with open("sd_image.hex", "w") as f:
        f.write("\n".join(f"{v:02x}" for v in image) + "\n")
    with open("img.hex", "w") as f:
        f.write("\n".join(f"{v:02x}" for v in img.reshape(-1).astype(np.uint8)) + "\n")
    with open("expected.txt", "w") as f:
        for i in (i1, i2):
            f.write(" ".join(map(str, i["logits"])) + "\n")
            f.write(f"{i['pred']}\n")
            f.write(" ".join(map(str, i["cam"])) + "\n")
    for n, i in (("CNN1", i1), ("CNN2", i2)):
        print(n, {k: v for k, v in i.items() if k != "cam"})
    with open("cls.hex", "w") as f:
        f.write(f"{i1['pred']:02x}\n{i2['pred']:02x}\n")


main()
