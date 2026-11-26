# Đặc tả triển khai: cascade 2 CNN + LayerCAM trên AX309 (Spartan-6 XC6SLX9)

Tài liệu này dành cho AI/kỹ sư viết RTL. Đọc hết trước khi code. Nguồn sự thật về phép tính:
- Forward CNN: hàm `golden_run` trong `CNN_Search_BN.py`.
- LayerCAM: `int_forward`, `_chain`, `hw_infer` trong `cam_best_cnn_v3.py` (đã sửa theo mục 13.1).

RTL phải **khớp từng bit** với hai nguồn trên. Phần thêm/sửa cho LayerCAM được đánh dấu **(LCAM)**. Phần tối ưu tốc độ (không đổi kết quả) đánh dấu **(OPT)**, đặc tả đầy đủ ở mục 14.

## 0. Mục tiêu và ràng buộc

- PC gửi ảnh **đã resize 64x64 RGB** qua UART. FPGA chạy CNN 1 (Hospital_A). Nếu ra `Unknown` thì chạy CNN 2 (Hospital_B) trên **cùng ảnh**.
- **(LCAM)** Khi đã có kết quả cuối của cascade, nếu lớp cuối **không phải Unknown và không phải Normal** thì FPGA chạy tiếp **LayerCAM khối 3** cho CNN đã quyết định. Xong LayerCAM mới gửi kết quả về PC (phân loại + bản đồ CAM 16x16). Nếu là Unknown hoặc Normal thì bỏ qua LayerCAM, chỉ gửi phân loại.
- Hai CNN **cùng kiến trúc**, chỉ khác trọng số, hệ số requant, zero-point, bảng LUT đầu vào và tham số LayerCAM. Tham số của cả hai nằm trên thẻ SD, được chép vào SDRAM lúc khởi động.
- Kiến trúc **tuần tự, dùng lại khối**: một `COMPUTE_CORE` chạy lần lượt mọi lớp, kể cả FC và **2 lớp conv ngược của LayerCAM (LCAM)**.
- Công cụ: **ISE 14.7 (XST), Verilog-2001**. Không dùng SystemVerilog. Cổng dạng mảng thì dùng vector phẳng, ví dụ `[255:0]` cho 8 giá trị 32-bit, kênh k là `[32*k +: 32]`.
- Một miền clock duy nhất: 50 MHz. Reset đồng bộ.
- **(OPT)** Giữ 50 MHz, không đổi UART. Tối ưu tốc độ gồm 4 việc ở mục 14, kết quả phải **giống hệt** bản chưa tối ưu.

## 1. Quy tắc đặt tên (bắt buộc)

Đọc tên phải biết nó làm gì. Không viết tắt khó hiểu.

| Loại | Quy tắc | Ví dụ |
|---|---|---|
| Cổng input/output | VIẾT HOA, snake_case | `CLK`, `START_CNN`, `IMAGE_WRITE_ADDR` |
| Wire/reg bên trong | chữ thường snake_case | `output_row`, `in_channel_index` |
| Tham số | VIẾT HOA | `CLOCKS_PER_BIT` |
| Module | VIẾT HOA | `COMPUTE_CORE` |
| Trạng thái FSM | `STATE_` + việc đang làm | `STATE_WAIT_IMAGE` |
| Xung 1 chu kỳ bắt đầu | `START_` + việc | `START_LOAD_WEIGHTS` |
| Xung 1 chu kỳ xong | `DONE_` + việc | `DONE_LAYER` |
| Mức báo đang bận | `BUSY_` + việc | `BUSY_UART_TX` |
| Handshake | `_VALID`, `_READY` | `RX_BYTE_VALID` |
| Địa chỉ/dữ liệu/cho phép | `_ADDR`, `_DATA`, `_ENABLE` | `FEATURE_WRITE_ENABLE` |
| Chỉ số / đếm | `_INDEX`, `_COUNT` | `group_index` |
| Tích cực mức thấp | hậu tố `_N` | `RESET_N`, `SD_CS_N` |

Từ vựng cố định: `row`/`col`, `in_channel`/`out_channel`, `kernel_row`/`kernel_col`, `group` (nhóm 8 kênh ra), `pixel`. Cấm: `cnt`, `tmp`, `ptr`, `d1`, `flag`, `sig1`. Độ trễ pipeline đặt tên rõ: `accumulate_enable_delayed`.

## 2. Chân (UCF), tất cả LVCMOS33

**(LCAM) Không thêm chân mới.**

| Cổng top | Chân | Ghi chú |
|---|---|---|
| `CLK` | T8 | 50 MHz |
| `RESET_N` | L3 | nút RESET, nhấn = 0 |
| `UART_RX` | C11 | FPGA nhận từ PC |
| `UART_TX` | D12 | FPGA gửi lên PC |
| `LED[3:0]` | P4, N5, P5, M6 | mức 1 = sáng |
| `SD_CS_N` | N3 | |
| `SD_CLK` | M3 | |
| `SD_MOSI` | L5 | manual đặt tên `SD_DIN` |
| `SD_MISO` | L4 | manual đặt tên `SD_DOUT` |
| `SDRAM_CLK` | H4 | |
| `SDRAM_CKE` | H2 | luôn 1 sau reset |
| `SDRAM_CS_N` | G1 | |
| `SDRAM_RAS_N` | F1 | |
| `SDRAM_CAS_N` | F2 | |
| `SDRAM_WE_N` | E1 | |
| `SDRAM_DQM[1:0]` | H1 (bit1), E2 (bit0) | |
| `SDRAM_BANK[1:0]` | J6 (bit1), G6 (bit0) | |
| `SDRAM_ADDR[12:0]` | J1,K1,K6,K2,L1,M1,M2,N1,P1,K5,K3,J4,J3 | từ bit 12 xuống bit 0 |
| `SDRAM_DATA[15:0]` | F4,F3,F5,G3,F6,H3,G5,H5,D1,C1,C2,B1,B2,A2,B3,A3 | từ bit 15 xuống bit 0 |

Thêm `NET CLK PERIOD = 20 ns;`. SDRAM HY57V2562GTR: 4 bank, 13 bit hàng, 9 bit cột, rộng 16 bit. Địa chỉ từ 24 bit = `{bank[1:0], row[12:0], col[8:0]}`.

## 3. Kiến trúc CNN (`16p-24p-24p-32-48`)

Mọi conv là 3x3, pad=1, stride=1. BN đã fold vào weight/bias. 8 PE song song, mỗi lớp tính theo **nhóm 8 kênh ra**.

| Lớp | `LAYER_INDEX` | Kích thước vào | Kênh vào | Kênh ra | MaxPool 2x2 | Số nhóm | Nguồn đọc | Đích ghi |
|---|---|---|---|---|---|---|---|---|
| Conv1 | 0 | 64 | 3 | 16 | có | 2 | `IMAGE_RAM` (+ LUT) | `FEATURE_RAM_A` |
| Conv2 | 1 | 32 | 16 | 24 | có | 3 | A | `FEATURE_RAM_B` |
| Conv3 | 2 | 16 | 24 | 24 | có | 3 | B | A (**LCAM: ghi thêm, xem 3.1**) |
| Conv4 | 3 | 8 | 24 | 32 | không | 4 | A | B |
| Conv5 | 4 | 8 | 32 | 48 | không | 6 | B | A |
| GAP | | 8x8, 48 kênh | | | | | A | `GAP_RAM` |
| FC | 5 | 1 | 48 | 5 (đệm lên 8) | không | 1 | `GAP_RAM` | `LOGIT_RAM` |

- Ping-pong: mỗi lớp đọc RAM này, ghi RAM kia. `IMAGE_RAM` **không được ghi đè** cho tới khi cả cascade xong.
- Xếp feature map: `addr = (row*IN_SIZE + col)*CHANNELS + channel`. Ảnh: `addr = (row*64 + col)*3 + channel`, channel 0,1,2 = R,G,B.
- FC dùng lại khối conv ở chế độ `IS_FULLY_CONNECTED`: 1 vị trí, không padding, địa chỉ đọc và địa chỉ trọng số đều bằng `in_channel_index` (0..47), không ReLU. Kênh 5..7 đệm 0, ghi vào `LOGIT_RAM` (sâu 8) rồi bỏ qua.

### 3.1 (LCAM) Hai lớp conv ngược, chạy sau FC

LayerCAM lấy gradient từ logit lớp đã chọn, đi ngược qua conv5 và conv4 để tới đầu ra ReLU của conv3 (khối k=3). Dùng lại đúng `COMPUTE_CORE` và vòng lặp mục 9, chỉ đổi tham số:

| Lớp ngược | `LAYER_INDEX` | Kích thước vào | Kênh vào | Kênh ra | Pool | Số nhóm | Nguồn đọc | Đích ghi |
|---|---|---|---|---|---|---|---|---|
| BWD1 (qua conv5) | 6 | 8 | 48 | 32 | không | 4 | `GRADIENT_RAM_A` | `GRADIENT_RAM_B` (có mask ReLU conv4) |
| BWD2 (qua conv4) | 7 | 8 | 32 | 24 | không | 3 | `GRADIENT_RAM_B` | `GRADIENT_RAM_A` |

## 4. Phép tính chính xác (bit-exact)

Ký hiệu: `x` = pixel uint8 sau LUT (lớp 1) hoặc đọc từ feature RAM, `w` = int8, `zp_in`, `zp_out` lấy từ header theo lớp.

1. **Tích lũy (mỗi kênh ra):** `acc = bias_int32 + Σ (x - zp_in) * w` (int32 có dấu, |acc| < 2^30).
   - `x - zp_in` có dấu, rộng 9 bit.
   - Ô padding **bỏ qua hẳn vị trí kernel** (không cộng gì, không tốn chu kỳ).
2. **MaxPool (lớp có pool):** max của 4 acc (có dấu) **trước requant**. Lớp không pool chỉ có 1 vị trí.
   - **(LCAM)** Khi pool, phải ghi nhận **vị trí thắng**: `argmax_code = pool_position (0..3)`, so sánh **`>` nghiêm ngặt** (hòa thì giữ vị trí đầu). `pool_position = (row%2)*2 + (col%2)`.
3. **Requant:** `product = acc * M0` (64 bit có dấu), `value = (product + (1 << (SHIFT-1))) >>> SHIFT` (**dịch số học**), `y = value + zp_out`. Kẹp `y` vào `[lower, 255]`.
   - Forward: `lower = zp_out` nếu `RELU_ENABLE`, ngược lại 0.
   - **(LCAM)** Lớp ngược 6, 7: `lower = 1` (hằng `GRADIENT_CLAMP_LOWER`), xem mục 10.2.
   - Kẹp trên giá trị rộng (64 bit). **(OPT)** `REQUANTIZER` dùng pipeline DSP48A1, 1 kênh/chu kỳ (mục 14.1). Chỉ CAM giữ `REQUANTIZER_SERIAL` (shift-add nhiều chu kỳ, không dùng DSP).
4. **GAP:** `gap[c] = (Σ_64pixel x[c] + 32) >> 6`, c = 0..47.
5. **FC:** như bước 1 và 3 với đầu vào `gap`, `zp_out = 127`, không ReLU (kẹp [0,255]). 5 logit uint8.
6. **Argmax:** trên 5 logit uint8, hòa thì lấy chỉ số nhỏ nhất.

## 5. Giao thức UART (8N1, 1 Mbaud = 50 MHz / 50)

`CLOCKS_PER_BIT = 50` (115200 baud thì 434).

- **PC -> FPGA:** `0xA5` rồi 12.288 byte ảnh theo thứ tự hàng, mỗi pixel `R,G,B`. Im lặng quá ~100 ms giữa hai byte thì bỏ khung.
- **FPGA -> PC, sau khởi động:** `0x55` (sẵn sàng). Lỗi khởi động: `0xEE, mã_lỗi` (1 = SD init lỗi, 2 = đọc sector lỗi, 3 = sai magic).
- **FPGA -> PC, khung kết quả (LCAM: 14 byte, hoặc 270 byte nếu có CAM):**

| Byte | Nội dung |
|---|---|
| 0 | `0xAA` |
| 1 | `FINAL_SOURCE`: `0x01` = CNN 1 quyết định, `0x02` = CNN 2 quyết định, `0x00` = Unknown cuối cùng |
| 2 | `FINAL_CLASS`: chỉ số lớp trong CNN đã quyết định, hoặc `0xFF` nếu Unknown cuối |
| 3..7 | 5 logit CNN 1 |
| 8..12 | 5 logit CNN 2 (0 nếu CNN 2 không chạy) |
| 13 | **(LCAM)** `CAM_FLAG`: `0x01` = có CAM theo sau, `0x00` = không |
| 14..269 | **(LCAM)** chỉ khi `CAM_FLAG=1`: 256 byte `cam_u8`, ma trận 16x16, thứ tự hàng (`index = row*16 + col`) |

Logic cascade: CNN 1 ra lớp khác `UNKNOWN_CLASS_INDEX` thì dừng (`SOURCE=1`). Ngược lại chạy CNN 2: khác Unknown thì `SOURCE=2`, còn Unknown thì `SOURCE=0`, `CLASS=0xFF`. `UNKNOWN_CLASS_INDEX` lấy từ header từng CNN, **không hard-code**.

**(LCAM)** `CAM_FLAG=1` khi và chỉ khi `FINAL_SOURCE != 0` **và** `FINAL_CLASS != NORMAL_CLASS_INDEX` (của CNN quyết định). Khung chỉ được gửi **sau khi LayerCAM xong**.

PC nhận `cam_u8` rồi tự phóng 16x16 -> 64x64 (bilinear), chuẩn hóa và phủ lên ảnh, y như hàm `hw_heat` trong Python. FPGA **không** làm bước này.

**Hiển thị theo kích thước ảnh gốc (khớp `cam_best_cnn_v3.py`):** PC giữ ảnh gốc `raw` (H0 x W0). Resize 64x64 là block-average trên `crop = raw[:(H0//64)*64, :(W0//64)*64]`, **phần dư bên dưới và bên phải bị bỏ**, nên "ảnh gốc" để hiển thị là `crop`. PC tính `mask` FOV từ `raw` (`fov_mask`), tạo `heat64 = hw_heat(cam_u8, mask)` (bilinear 16 -> 64, nhân `mask`, chuẩn hóa từ min đến quantile 0,99 trong `mask`), rồi vẽ `heat64` lên `crop` với extent bằng kích thước `crop` (matplotlib tự kéo giãn), cmap `jet`, alpha 0,45, vùng ngoài `mask` trong suốt. FPGA không làm bước nào trong số này. CAM chỉ có độ phân giải thật 16x16 nên ảnh càng lớn heatmap càng mờ.

LED: `LED[0]` = đã khởi động, `LED[1]` = đang chạy CNN hoặc LayerCAM, `LED[2]` = lỗi, `LED[3]` = đảo mỗi lần có kết quả.

## 6. Định dạng thẻ SD và SDRAM

Ghi **sector thô** (không FAT), thẻ SDHC. Cảnh báo: ghi thô xóa dữ liệu thẻ.

**(LCAM) Mỗi CNN chiếm 64 KB** (128 sector): sector 0..127 là CNN 1, sector 128..255 là CNN 2. Tổng `sd_image.bin` = 131.072 byte. `SD_LOADER` đọc **256 sector**. Chép nguyên khối vào SDRAM từ byte 0. Byte chẵn nằm ở `[7:0]` của từ 16 bit, byte lẻ ở `[15:8]`. Địa chỉ nền CNN k (k = 0,1): `k * 0x10000` (byte).

Bố cục một khối CNN (65.536 byte, little-endian):

| Offset | Nội dung |
|---|---|
| 0x000 | magic `"FCNN"`, `CNN_ID` (1 byte), `UNKNOWN_CLASS_INDEX` (1 byte), `NUM_CLASSES` (1 byte, =5), **(LCAM) `NORMAL_CLASS_INDEX` (1 byte, 0xFF nếu không có lớp Normal)** |
| 0x008 | bảng 6 lớp, mỗi lớp 8 byte: `M0` (u32), `SHIFT` (u8), `ZP_IN` (u8), `ZP_OUT` (u8), `RELU_ENABLE` (u8). Lớp 0..4 conv, lớp 5 FC |
| 0x200 | LUT đầu vào 3x256 byte: R, G, B (0x200..0x4FF) |
| 0x600 | dữ liệu các lớp forward, xếp liền, không đệm (hết ở 0x7E50, phần sau đến 0x7FFF là 0) |
| **0x8000** | **(LCAM) bảng lớp ngược, 3 mục x 8 byte (xem dưới)** |
| **0x8100** | **(LCAM) trọng số BWD1 (lớp 6)** |
| **0xB780** | **(LCAM) trọng số BWD2 (lớp 7)** |

**(LCAM) Bảng ở 0x8000** (cùng khuôn 8 byte như bảng lớp forward):

| Offset | Mục | Giá trị |
|---|---|---|
| 0x8000 | lớp 6 (BWD1) | `M0`, `SHIFT` lấy từ `lcam_params.json` bước conv 5; `ZP_IN=128`, `ZP_OUT=128`, `RELU_ENABLE=0` |
| 0x8008 | lớp 7 (BWD2) | như trên, bước conv 4 |
| 0x8010 | CAM | `CAM_M0` (u32), `CAM_SHIFT` (u8), 3 byte đệm 0 |

Mỗi **khối nhóm** forward (8 kênh ra): `bias` 8 x int32 (32 byte), rồi trọng số PE0..PE7, mỗi PE là `IN_CHANNELS*9` byte int8 theo `[in_channel][kernel_row][kernel_col]`. FC: `IN_CHANNELS=48`, PE5..7 đệm 0.

**(LCAM)** Khối nhóm lớp ngược **cùng khuôn**: 32 byte bias **toàn 0**, rồi 8 PE, mỗi PE `IN_CHANNELS*9` byte theo `[in_channel][kernel_row][kernel_col]`. Dữ liệu trọng số lấy từ `lcam_w5_bwd_int8.mem` và `lcam_w4_bwd_int8.mem` (layout `((o*C_in + i)*3 + ky)*3 + kx`), cắt từng 8 kênh ra `o` thành một nhóm.

Bảng `LAYER_CONFIG_ROM` (hằng số cố định):

| `LAYER_INDEX` | Offset từ nền CNN | Byte mỗi khối nhóm | Số nhóm |
|---|---|---|---|
| 0 | 0x0600 (1536) | 248 | 2 |
| 1 | 0x07F0 (2032) | 1184 | 3 |
| 2 | 0x15D0 (5584) | 1760 | 3 |
| 3 | 0x2A70 (10864) | 1760 | 4 |
| 4 | 0x45F0 (17904) | 2336 | 6 |
| 5 (FC) | 0x7CB0 (31920) | 416 | 1 |
| **6 (BWD1, LCAM)** | **0x8100 (33024)** | **3488** | **4** |
| **7 (BWD2, LCAM)** | **0xB780 (46976)** | **2336** | **3** |

Địa chỉ khối nhóm g của lớp L: `CNN_BASE + offset[L] + g*bytes_per_group[L]`.

**(LCAM) Hàng FC cho LayerCAM:** không lưu thêm. Hàng của lớp `cls` nằm trong khối FC: `CNN_BASE + 0x7CB0 + 32 + cls*48` (48 byte int8, đúng `Wfc[cls, :]`).

## 7. Bộ nhớ trong chip

Dùng chung module `SIMPLE_DUAL_PORT_RAM` (8 bit, 1 cổng ghi, 1 cổng đọc, **đọc đồng bộ trễ 1 chu kỳ**):

| RAM | Độ sâu (byte) | Ghi chú |
|---|---|---|
| `IMAGE_RAM` | 12288 | ảnh thô |
| `FEATURE_RAM_A` | 16384 | |
| `FEATURE_RAM_B` | 6144 | |
| `INPUT_LUT_RAM` | 768 | địa chỉ `in_channel*256 + pixel_thô` |
| `WEIGHT_RAM` x8 | **1024** mỗi cái (**OPT**: 2 bank x 512, bit địa chỉ 9 = bank) | mỗi bank 512 (**LCAM:** BWD1 cần 48*9 = 432 byte mỗi PE). Bank thứ hai để nạp trước nhóm kế (mục 14.4) |
| `GAP_RAM` | 64 | dùng 48. **(LCAM)** Sau khi có logit thì tái dùng giữ hàng FC của LayerCAM |
| `LOGIT_RAM` | 8 | dùng 5 |
| `GRADIENT_RAM_A` **(LCAM)** | 3072 | dz5 (8x8x48), sau đó g4 (8x8x24) |
| `GRADIENT_RAM_B` **(LCAM)** | 2048 | dz4 (8x8x32) |
| `POOL3_COPY_RAM` **(LCAM)** | 1536 | bản sao đầu ra pool của Conv3 (8x8x24) |
| `POOL3_ARGMAX_RAM` **(LCAM)** | 1536 | mã vị trí thắng của pool Conv3, giá trị 0..3, cùng địa chỉ với `POOL3_COPY_RAM` |
| `CAM_RAM` **(LCAM)** | 256 | `cam_u8` 16x16 |

Bias: 8 thanh ghi 32 bit trong `COMPUTE_CORE`. Ước tính BRAM: ~25/32 khối 18Kb, còn đủ. **(OPT)** Nếu mỗi `WEIGHT_RAM` đang chiếm 1 BRAM18 thì 1024x8 vẫn vừa 1 BRAM18 (tối đa 2048x9), nên không tăng BRAM. Kiểm báo cáo XST.

## 8. Danh sách module (12 module)

Mọi module có `CLK`, `RST` (nội bộ, tích cực mức cao, tạo từ `RESET_N` qua 2 FF đồng bộ).

```
TOP_CNN
├─ UART_RECEIVER
├─ UART_TRANSMITTER
├─ HOST_INTERFACE            (nhận ảnh, gửi kết quả + CAM)
├─ SD_LOADER                 (thẻ SD -> SDRAM, 256 sector, chạy 1 lần lúc khởi động)
├─ SDRAM_CONTROLLER
├─ SIMPLE_DUAL_PORT_RAM      (IMAGE_RAM)
├─ CNN_RUNNER                (chạy 1 CNN; (LCAM) thêm chế độ chạy LayerCAM)
│   ├─ PARAMETER_FETCHER
│   ├─ CONV_CONTROLLER       (vòng lặp một lớp, thêm lớp 6, 7)
│   ├─ COMPUTE_CORE          (PE_ARRAY + max-pool + REQUANTIZER)
│   ├─ LAYERCAM_CONTROLLER   (LCAM, mới)
│   ├─ SIMPLE_DUAL_PORT_RAM  (FEATURE A/B, INPUT_LUT, GAP, LOGIT, + 5 RAM LCAM)
│   └─ phần GAP + argmax
└─ FSM cascade               (viết thẳng trong TOP_CNN)
```

| # | Module | Việc làm | Thay đổi **(LCAM)** |
|---|---|---|---|
| 1 | `UART_RECEIVER` | Nhận byte | không đổi |
| 2 | `UART_TRANSMITTER` | Gửi byte | không đổi |
| 3 | `HOST_INTERFACE` | Nhận ảnh, gửi `0x55`, `0xEE`, khung kết quả | Khung dài 14 byte, thêm `CAM_FLAG`. Thêm cổng `FINAL_CAM_VALID`, `CAM_READ_ADDR[7:0]`, `CAM_READ_DATA[7:0]` (đọc trễ 1 chu kỳ). Nếu `FINAL_CAM_VALID=1` thì gửi tiếp 256 byte đọc từ `CAM_RAM` |
| 4 | `SD_LOADER` | SPI, init SD, đọc sector, ghi SDRAM | Đọc **256 sector** (thay vì 128) |
| 5 | `SDRAM_CONTROLLER` | Init, refresh, truy cập từ đơn | **(OPT)** Thêm đọc burst 8 từ, mode register BL=8 và A9=1 (mục 14.3) |
| 6 | `SIMPLE_DUAL_PORT_RAM` | RAM 8 bit | không đổi |
| 7 | `PARAMETER_FETCHER` | Nạp header, LUT, khối nhóm | (a) Khi nạp tham số CNN, đọc thêm 24 byte ở `0x8000` vào bảng header mục 6, 7 và `CAM_M0/CAM_SHIFT`. (b) Xuất thêm `NORMAL_CLASS_INDEX[7:0]` (byte 0x007). (c) Thêm việc nạp hàng FC: `START_FETCH_CLASS_ROW`, `LAYERCAM_CLASS_INDEX[2:0]`, `DONE_FETCH_CLASS_ROW`, ghi 48 byte vào `GAP_RAM` qua `CLASS_ROW_WRITE_ENABLE`, `CLASS_ROW_WRITE_ADDR[5:0]`, `CLASS_ROW_WRITE_DATA[7:0]`. **(OPT)** (d) mọi lần đọc SDRAM dùng burst căn 8 từ với 2 bộ đệm burst (14.3). (e) đệm đôi `WEIGHT_RAM`: cổng `FETCH_BANK`, `bias_shadow`, `APPLY_BIAS` (14.4) |
| 8 | `COMPUTE_CORE` | 8 PE, max-pool, `REQUANTIZER` | Thêm vào: `POOL_POSITION_INDEX[1:0]` và đầu ra `POOL_ARGMAX_CODE[15:0]` (kênh k ở `[2*k +: 2]`, cập nhật khi `POOL_NEXT_POSITION` với so sánh `>` nghiêm ngặt). Thêm đầu vào `CLAMP_LOWER[7:0]` cho requant (forward: như mục 4.3; ngược: 1). Pixel vào có dấu nhờ mã hóa +128 nên PE giữ nguyên. **(OPT)** `REQUANTIZER` thành pipeline DSP48A1 có bộ đệm phụ cho 8 acc, cổng `REQUANT_READY`, `OUTPUT_VALID`, `OUTPUT_CHANNEL_INDEX`, `OUTPUT_DATA` (14.1). Thêm cổng `COMPUTE_BANK` cho 8 PE |
| 9 | `CONV_CONTROLLER` | Vòng lặp một lớp + `LAYER_CONFIG_ROM` | Thêm hàng 6, 7 (mục 3.1, 6). Thêm cờ `IS_BACKWARD` (đặt `CLAMP_LOWER=1`), `APPLY_RELU4_MASK` (chỉ lớp 6), chọn nguồn/đích RAM gradient. Lớp 2: khi ghi đầu ra thì **ghi thêm** `POOL3_COPY_RAM` (cùng địa chỉ, cùng dữ liệu) và `POOL3_ARGMAX_RAM` (dữ liệu `POOL_ARGMAX_CODE` của kênh đang ghi). **(OPT)** Không chờ `DONE_REQUANT` giữa các vị trí: dùng `pending_write_base_addr`, `pending_argmax_code` (14.2). Điều khiển `FETCH_BANK`, `COMPUTE_BANK`, `APPLY_BIAS` theo lịch 14.4 |
| 10 | `CNN_RUNNER` | Chạy trọn một CNN | Thêm cổng `START_LAYERCAM`, `LAYERCAM_CLASS_INDEX[2:0]`, `DONE_LAYERCAM`, `NORMAL_CLASS_INDEX[7:0]`, `CAM_READ_ADDR`, `CAM_READ_DATA`. Chứa 5 RAM LCAM mới |
| 11 | `LAYERCAM_CONTROLLER` **(mới)** | Điều khiển 4 bước LayerCAM (mục 10.4), gồm 1 `REQUANTIZER_SERIAL` riêng cho CAM (**OPT**: giữ bản shift-add, không dùng DSP) và 1 bộ nhân-cộng cho CAM | Cổng: `START_LAYERCAM`, `DONE_LAYERCAM`, `LAYERCAM_CLASS_INDEX`, các cổng điều khiển `PARAMETER_FETCHER` và `CONV_CONTROLLER`, các cổng RAM gradient/pool3/CAM |
| 12 | `TOP_CNN` | Nối tất cả, mux SDRAM, FSM cascade | FSM thêm trạng thái ở mục 10.5 |

Cổng chủ SDRAM (`SD_LOADER` và `CNN_RUNNER` dùng chung, qua mux của `TOP_CNN`): `SDRAM_REQUEST_VALID`, `SDRAM_REQUEST_WRITE`, `SDRAM_REQUEST_WORD_ADDR[23:0]`, `SDRAM_REQUEST_WRITE_DATA[15:0]`, `SDRAM_REQUEST_READY`, `SDRAM_RESPONSE_VALID`, `SDRAM_RESPONSE_READ_DATA[15:0]`. **(OPT)** Thêm `SDRAM_REQUEST_BURST` (đọc 8 từ liên tiếp, địa chỉ căn 8 từ) và `SDRAM_RESPONSE_LAST` (xung cùng từ thứ 8), xem 14.3.

Cổng `PARAMETER_FETCHER` (nạp CNN và nạp khối nhóm) và `COMPUTE_CORE` giữ như bản gốc, cộng thêm các cổng ghi ở bảng trên. Ở lớp 1, đường đọc đầu vào là `IMAGE_RAM -> INPUT_LUT_RAM -> INPUT_PIXEL`.

## 9. Vòng lặp của `CONV_CONTROLLER` (một lớp)

```
for group_index in 0 .. GROUP_COUNT-1:
    PARAMETER_FETCHER nạp khối nhóm (bias + 8 PE), chờ DONE_FETCH_WEIGHTS   // (OPT) nhóm 0 nạp chặn; nhóm sau đã nạp trước, chỉ đổi bank (14.4)
    for output_row, output_col:
        for pool_position in 0 .. (POOL_ENABLE ? 3 : 0):
            conv_row = output_row*(POOL_ENABLE?2:1) + pool_position/2
            conv_col = output_col*(POOL_ENABLE?2:1) + pool_position%2
            LOAD_BIAS_INTO_ACC
            for kernel_row 0..2, kernel_col 0..2:
                input_row = conv_row + kernel_row - 1
                input_col = conv_col + kernel_col - 1
                if input_row/col ngoài [0, IN_SIZE-1]: continue   // padding: bỏ qua
                for in_channel_index 0 .. IN_CHANNELS-1:
                    input_addr  = (input_row*IN_SIZE + input_col)*IN_CHANNELS + in_channel_index
                    WEIGHT_READ_ADDR = in_channel_index*9 + kernel_row*3 + kernel_col
                    ACCUMULATE_ENABLE
            pool: pool_position==0 ? POOL_FIRST_POSITION : POOL_NEXT_POSITION
        START_REQUANT; ghi 8 byte ra RAM đích khi OUTPUT_VALID
        // (OPT) không chờ DONE_REQUANT, sang vị trí kế ngay; chỉ chờ khi REQUANT_READY=0 (14.2)
DONE_LAYER   // (OPT) chỉ phát khi REQUANT_READY=1 và byte cuối đã ghi
```

Địa chỉ ghi: `output_write_addr = (output_row*OUT_SIZE + output_col)*OUT_CHANNELS + group_index*8 + REQUANT_CHANNEL_IN_GROUP`. FC: 1 vị trí, không vòng kernel, `input_addr = WEIGHT_READ_ADDR = in_channel_index`.

**(LCAM)** Lớp 6, 7 chạy **đúng vòng lặp này** (không pool, `OUT_SIZE = IN_SIZE = 8`). Khác nhau chỉ ở nguồn/đích RAM, `ZP_IN=128`, `CLAMP_LOWER=1`, và lớp 6 có mask khi ghi (mục 10.3).

Pipeline: mọi RAM đọc trễ 1 chu kỳ, DSP trễ thêm vài chu kỳ. Trễ các cờ cho khớp dữ liệu. **Làm bản chạy đúng trước**, tối ưu sau. Ước lượng bản chưa tối ưu: ~2,2M chu kỳ mỗi CNN (~44 ms; MAC ~1,0M, requant ~1,0M, nạp trọng số ~0,2M). **(OPT)** Sau mục 14: ~1,0M (~20 ms). **(LCAM)** LayerCAM ~0,4M chưa tối ưu, ~0,2M sau mục 14. Đều là ước tính từ code, chưa đo trên board.

## 10. (LCAM) LayerCAM khối 3

### 10.1 Điều kiện chạy

Sau `STATE_CHECK_*`, nếu `FINAL_SOURCE != 0` và `FINAL_CLASS != NORMAL_CLASS_INDEX` thì chạy LayerCAM với `cls = FINAL_CLASS`. Ngược lại `CAM_FLAG=0`, gửi kết quả luôn.

CNN chạy **sau cùng** luôn là CNN quyết định, nên `FEATURE_RAM_A/B`, `GAP_RAM`, `LOGIT_RAM` và header đang giữ đúng dữ liệu của nó. **Không chạy lại forward.**

### 10.2 Công thức (n=5, k=3; `g` lưu dạng `g + 128`)

```
dz5   = (A5 > 0) ? Wfc[cls, c] : 0                           // 8x8x48, A5 = FEATURE_RAM_A
g5    = clamp(((convT(dz5, W5)*M0_6 + 2^(S_6-1)) >> S_6), -127, 127)   // 8x8x32
dz4   = (A4 > 0) ? g5 : 0                                    // A4 = FEATURE_RAM_B
g4    = clamp(((convT(dz4, W4)*M0_7 + 2^(S_7-1)) >> S_7), -127, 127)   // 8x8x24, không mask
cam_acc[r][c] = Σ_ch  (code3[r/2][c/2][ch] == (r%2)*2 + c%2)
                       ? max(g4[r/2][c/2][ch], 0) * pool3[r/2][c/2][ch] : 0   // 16x16, int32
cam_u8 = clamp(((cam_acc*CAM_M0 + 2^(CAM_S-1)) >> CAM_S), 0, 255)
```

- `convT` = conv 3x3 pad 1 với trọng số đã lật/hoán vị trong file SD, tức **conv xuôi thường** của PE. Padding bỏ qua như mục 4.1.
- `A > 0` so sánh trực tiếp giá trị uint8 (vì `zp_out = 0` cho conv3..5).
- Không cần buffer unpool hay `A3` trước pool: gradient chỉ khác 0 ở vị trí thắng của pool, và ở đó `A3` đúng bằng giá trị đầu ra pool (`pool3`). Công thức trên **bằng đúng** `Σ_c ReLU(G)·A3` của Python (đã kiểm theo quy tắc argmax ở 13.1).

**Mã hóa có dấu cho PE:** gradient lưu dạng `g + 128` (uint8). Dùng `ZP_IN = 128` để `(x - zp_in) = g` (PE không đổi). Đầu ra lớp 6, 7: `ZP_OUT = 128`, kẹp `[1, 255]`, tương đương clamp `[-127, 127]` của golden.

### 10.3 Bốn bước do `LAYERCAM_CONTROLLER` điều khiển

1. **Nạp hàng FC** (`START_FETCH_CLASS_ROW`) vào `GAP_RAM`. Rồi tạo `dz5`: duyệt `i = 0..3071`, `byte = (FEATURE_RAM_A[i] > 0) ? (GAP_RAM[i % 48] + 128) : 128`, ghi `GRADIENT_RAM_A[i]`. Hàng FC là int8 có dấu, nên cộng 128 để đúng mã hóa.
2. **BWD1** (lớp 6): đọc `GRADIENT_RAM_A`, ghi `GRADIENT_RAM_B`. Khi ghi byte tại địa chỉ `a`, đọc `FEATURE_RAM_B[a]` (cổng đọc B rảnh): nếu `FEATURE_RAM_B[a] == 0` thì ghi `128`, ngược lại ghi giá trị requant.
3. **BWD2** (lớp 7): đọc `GRADIENT_RAM_B`, ghi `GRADIENT_RAM_A[0..1535]` (đã hết cần `dz5`). Không mask.
4. **CAM:** với mỗi `r, c` trong 16x16, duyệt `ch = 0..23`, địa chỉ chung `(((r>>1)*8 + (c>>1))*24 + ch)` cho cả 3 RAM `GRADIENT_RAM_A`, `POOL3_COPY_RAM`, `POOL3_ARGMAX_RAM`. Cộng dồn theo công thức 10.2 vào int32 (tối đa ~7,8e5, an toàn). Requant bằng `REQUANTIZER` riêng, ghi `CAM_RAM[r*16 + c]`. Xong thì `DONE_LAYERCAM`.

### 10.4 Forward cần ghi thêm (không tốn chu kỳ)

Ở lớp 2 (Conv3, có pool), mỗi lần ghi đầu ra thì **ghi song song** `POOL3_COPY_RAM` (cùng dữ liệu) và `POOL3_ARGMAX_RAM` (`POOL_ARGMAX_CODE` của kênh đó). Chạy cho cả hai CNN, mọi ảnh (vì chưa biết có cần CAM hay không).

### 10.5 FSM cascade (thêm trạng thái)

```
... -> STATE_CHECK_CNN1 -> (STATE_RUN_CNN2 -> STATE_CHECK_CNN2) -> STATE_CHECK_LAYERCAM
STATE_CHECK_LAYERCAM: cần CAM ? STATE_RUN_LAYERCAM : STATE_SEND_RESULT (CAM_FLAG=0)
STATE_RUN_LAYERCAM  : START_LAYERCAM, chờ DONE_LAYERCAM -> STATE_SEND_RESULT (CAM_FLAG=1)
STATE_SEND_RESULT   -> STATE_WAIT_IMAGE
```

`NORMAL_CLASS_INDEX` lấy từ header của CNN quyết định.

## 11. Phần Python cần viết thêm / sửa

| Script | Việc làm |
|---|---|
| `pack_sd_image.py` | Như bản gốc, **sửa/thêm (LCAM)**: (1) mỗi CNN 65.536 byte, ra `sd_image.bin` 131.072 byte và `sd_image.hex`; (2) header byte 0x007 = chỉ số `Normal` trong `classes` (0xFF nếu không có); (3) đọc `cam_out/<A|B>/hw_lcam/Hospital_X/LayerCAM_B3/`: `lcam_params.json` (lấy `requant_M0/shift` của bước `conv_index` 5 -> lớp 6, 4 -> lớp 7, và `cam.requant_M0/shift`), `lcam_w5_bwd_int8.mem`, `lcam_w4_bwd_int8.mem`; (4) ghi bảng 0x8000 và trọng số ở 0x8100, 0xB780 theo mục 6 (mỗi nhóm thêm 32 byte bias 0, cắt 8 kênh ra mỗi nhóm); (5) **assert** `ZP_OUT` của conv3, conv4, conv5 bằng 0 |
| `make_test_images.py` | Như bản gốc |
| `host_cascade.py` | Đọc 14 byte; nếu byte 13 = 1 thì đọc thêm 256 byte, reshape 16x16, phóng 64x64 bilinear, so với `*_layercam_int8_u8_16x16.npy` của Python. **Hiển thị trên ảnh gốc:** `from cam_best_cnn_v3 import fov_mask, hw_heat, draw_panel, heat_to_boxes` (module có guard `__main__`, import được) để kết quả giống hệt Python. Với mỗi khung có CAM: `mask = fov_mask(raw)`; `heat = hw_heat(torch.from_numpy(cam_u8), torch.from_numpy(mask))`; `crop = raw[:(H0//64)*64, :(W0//64)*64]`; vẽ bằng `draw_panel(ax, crop, heat.numpy(), boxes, title, mask)` (`boxes` từ `heat_to_boxes(heat.numpy(), mask, 0.15, 1, W, H)`). Cần `scipy`, `torch`, `matplotlib` trên PC |

Cách tạo tham số LayerCAM (mỗi CNN một lần, calib trên **train**):

```
python cam_best_cnn_v3.py --images <Hospital_X>/test --calib-images <Hospital_X>/train --calib-n 128 \
    --mode <A|B> --out cam_out/<A|B> --hw --export-hw --per-folder 5
```

### 13.1 Sửa bắt buộc trong `cam_best_cnn_v3.py` (hàm `int_forward`)

Vị trí thắng của pool phải tính theo **acc** (khớp phần cứng), không theo giá trị sau requant. Thay đoạn pool:

```python
if hw["pools"][i - 1]:
    xp, _ = F.max_pool2d(x.double(), 2, return_indices=True)       # giá trị pool (không đổi)
    _, idx = F.max_pool2d(acc.double(), 2, return_indices=True)    # LCAM: argmax theo acc, hòa -> vị trí đầu
    tr[f"pidx{i}"] = idx
    x = xp.long()
    tr[f"pool{i}"] = x
```

(Mục đích: requant có kẹp nên hai acc khác nhau có thể cho cùng giá trị; phần cứng chọn theo acc.)

## 12. Thứ tự làm và kiểm thử

1. `UART_RECEIVER`, `UART_TRANSMITTER`, `HOST_INTERFACE`, echo qua UART.
2. `REQUANTIZER` và `REQUANTIZER_SERIAL` (test riêng, vector biên ở 14.1), `COMPUTE_CORE`, `CONV_CONTROLLER`, `PARAMETER_FETCHER`, `CNN_RUNNER` trong **mô phỏng**, thay SDRAM bằng `SDRAM_SIM_MODEL` nạp từ `sd_image.hex`. So từng lớp với `TestVectors/dbg_img{i}_{tên}.mem`: `input`, `pool1`, `pool2`, `pool3`, `conv4`, `conv5`, `gap`, `fc` (`conv1..3` là trước pool, nên với lớp có pool hãy so `pool*`).
3. **(LCAM)** Mô phỏng LayerCAM, so với `hw_lcam/.../vectors/`. Ghi chú: file `.mem` có dấu dạng bù 2, còn RAM gradient của FPGA lưu `+128`, nên so `(giá_trị_RAM - 128) & 0xFF`.
   - `dz5` <-> `GRADIENT_RAM_A` bước 1.
   - `dz4` <-> `GRADIENT_RAM_B` sau BWD1 (đã mask).
   - `g4` <-> `GRADIENT_RAM_A` sau BWD2.
   - `pidx3` <-> `POOL3_ARGMAX_RAM`.
   - `cam_acc` <-> tích lũy trước requant.
   - `cam_u8` <-> `CAM_RAM`.
   - Pass khi khớp 100%.
4. Mô phỏng cả cascade (kể cả nhánh Normal/Unknown không chạy CAM), so logit và `CAM_FLAG`.
5. `SDRAM_CONTROLLER` trên board.
6. `SD_LOADER` (init, đọc sector), rồi chép vào SDRAM (đọc lại so checksum, **256 sector**).
7. Chạy trọn trên board với `host_cascade.py`.
8. **(OPT)** Tối ưu theo mục 14, từng việc một (14.1 -> 14.4). Sau mỗi việc chạy lại bước 2-4: kết quả giống hệt, chỉ chu kỳ giảm. Đo số chu kỳ mỗi lớp trong mô phỏng để xác nhận.

## 13. Những chỗ dễ sai

- Dịch số học (`>>>`) trên số có dấu khi requant.
- Padding = bỏ qua vị trí kernel (không nạp 0 cho pixel). Lớp ngược cũng vậy.
- ReLU: cận dưới kẹp là `zp_out`. FC cận dưới 0. **Lớp ngược cận dưới = 1** (không phải 0, không phải `zp_out`).
- Max-pool trên acc có dấu, trước requant, mỗi vị trí pool nạp lại bias. **Hòa thì giữ vị trí đầu (`>` nghiêm ngặt).**
- Thứ tự trọng số `[in_channel][kernel_row][kernel_col]`, thứ tự LUT R, G, B.
- `IMAGE_RAM` giữ đến hết cascade. Mỗi CNN nạp lại LUT, requant, zero-point riêng.
- **(LCAM)** `WEIGHT_RAM` phải sâu 512 (BWD1 cần 432 byte mỗi PE).
- **(LCAM)** Gradient lưu dạng `g + 128`, `ZP_IN = 128`. Quên cộng 128 khi tạo `dz5` hoặc khi ghi giá trị bị mask (phải ghi 128, không phải 0).
- **(LCAM)** Mask ReLU chỉ ở BWD1 (theo `A4`). Ở BWD2 **không** mask theo `A3`.
- **(LCAM)** Hàng FC lấy theo `FINAL_CLASS` của CNN quyết định, không phải CNN 1 mặc định.
- **(LCAM)** Kênh nhanh nhất trong địa chỉ RAM, nên `dz5[i]` dùng kênh `i % 48`.
- **(LCAM)** Kích thước block SD là 64 KB, địa chỉ nền CNN 2 là `0x10000`. Quên sửa `SD_LOADER` (256 sector) là CNN 2 đọc ra rác.
- Nhớ trễ cờ điều khiển cho khớp độ trễ RAM/DSP. Spartan-6 dùng ISE, chỉ Verilog-2001.
- **(OPT)** Requant DSP: chỉ `a_hi` có dấu; `a_lo`, `m_hi`, `m_lo` **không dấu** (mở rộng 0 lên 18 bit). Nhầm dấu là lệch bit với số âm.
- **(OPT)** Chốt `pending_write_base_addr` và `pending_argmax_code` cùng chu kỳ `START_REQUANT`. Chốt muộn thì ghi sai địa chỉ khi vị trí kế đã đổi.
- **(OPT)** `DONE_LAYER` phải chờ requant xả hết, nếu không lớp sau đọc dữ liệu chưa ghi (đặc biệt lớp 2 -> POOL3 RAM, và FC -> argmax).
- **(OPT)** SDRAM burst: địa chỉ căn 8 từ, A9=1 để ghi vẫn là từ đơn, refresh chỉ chèn giữa hai burst. Mode register phải khớp datasheet.
- **(OPT)** Đệm đôi: `APPLY_BIAS` trước `START_FETCH_WEIGHTS` của nhóm kế (nó ghi đè `bias_shadow`). Không ghi vào bank đang được PE đọc.
- **(OPT)** DSP: đúng 12 DSP48A1 (8 PE + 4 requant). Vượt thì kiểm lại suy diễn của XST.

## 14. (OPT) Tối ưu tốc độ (giữ 50 MHz, không đổi UART)

Mục tiêu: giảm số chu kỳ, **kết quả giữ nguyên từng bit**. Không nâng clock, không tối ưu UART, không tăng số PE.

Nút cổ chai bản gốc (ước tính từ code, chưa đo trên board), mỗi CNN: MAC ~1,0M chu kỳ; requant shift-add ~34 chu kỳ x 8 kênh x 3.649 nhóm-vị-trí ≈ 1,0M; nạp trọng số từng từ đơn ~15,4k từ x ~12 chu kỳ ≈ 0,2M. Tổng ~2,2M (~44 ms).

| Việc | Mục | Tiết kiệm (ước tính) | Phụ thuộc |
|---|---|---|---|
| `REQUANTIZER` pipeline DSP48A1, 1 kênh/chu kỳ | 14.1 | ~0,9M | không |
| Chồng requant với MAC vị trí kế | 14.2 | giữ phần còn lại của requant (~16 chu kỳ/vị trí) nằm dưới MAC | 14.1 |
| SDRAM đọc burst 8 từ | 14.3 | ~0,15M | không |
| Nạp trước nhóm kế (đệm đôi `WEIGHT_RAM`) | 14.4 | che nốt phần nạp còn lại | 14.3 |

Kết quả mong đợi: ~1,0M chu kỳ mỗi CNN (~20 ms), LayerCAM ~0,2M (~4 ms).

**Cách làm:** chỉ làm sau khi bản gốc đã khớp golden trong mô phỏng (mục 12, bước 2-4). Làm từng việc theo thứ tự 14.1 -> 14.4. Sau mỗi việc chạy lại bước 2-4: **kết quả phải giống hệt**, chỉ số chu kỳ giảm. Không sửa Python, không đổi định dạng SD.

### 14.1 `REQUANTIZER` pipeline bằng DSP48A1

Thay bản shift-add trong `COMPUTE_CORE`. Bản shift-add cũ đổi tên `REQUANTIZER_SERIAL`, **chỉ giữ cho CAM** (chỉ chạy 256 lần, không cần DSP).

| Cổng | Hướng | Ý nghĩa |
|---|---|---|
| `START_REQUANT` | vào | Xung 1 chu kỳ: chốt `ACC_VECTOR[255:0]` (8 acc int32, kênh k ở `[32*k +: 32]`) vào bộ đệm phụ **bên trong module**. Chỉ hợp lệ khi `REQUANT_READY=1` |
| `REQUANT_READY` | ra | 1 = rảnh. Lên 1 sau `DONE_REQUANT` |
| `M0[31:0]`, `SHIFT[5:0]`, `ZP_OUT[7:0]`, `CLAMP_LOWER[7:0]` | vào | Giữ nguyên suốt một lớp. `SHIFT >= 1` như golden |
| `OUTPUT_VALID`, `OUTPUT_CHANNEL_INDEX[2:0]`, `OUTPUT_DATA[7:0]` | ra | Mỗi chu kỳ 1 kênh, kênh 0..7 liên tiếp, độ trễ cố định `REQUANT_LATENCY` (<= 10) |
| `DONE_REQUANT` | ra | Xung 1 chu kỳ, cùng chu kỳ với `OUTPUT_VALID` của kênh 7 |

Tính cho từng kênh (một kênh vào pipeline mỗi chu kỳ), **bit-exact** với mục 4.3:

```
acc  = ACC_VECTOR[32*k +: 32]                 // có dấu
a_hi = acc[31:17]        (có dấu, 15 bit)     a_lo = {1'b0, acc[16:0]}   (không dấu, mở rộng 18 bit)
m_hi = {3'b0, M0[31:17]} (không dấu)          m_lo = {1'b0, M0[16:0]}    (không dấu)
hh = a_hi*m_hi   hl = a_hi*m_lo   lh = a_lo*m_hi   ll = a_lo*m_lo        // 4 DSP48A1, 18x18 có dấu
product64 = (hh <<< 34) + ((hl + lh) <<< 17) + ll                          // = acc*M0 chính xác, 64 bit có dấu
value = (product64 + (1 << (SHIFT-1))) >>> SHIFT                           // dịch số học
y     = value + ZP_OUT;  kẹp y vào [CLAMP_LOWER, 255] trên giá trị rộng
```

- Cộng các tích riêng trong fabric hoặc dùng cascade `PCIN >> 17`, cách nào cũng được miễn bit-exact. Chia tầng thanh ghi vừa đủ cho 50 MHz.
- Ngân sách DSP: 8 (PE) + 4 (requant) = **12 / 16**. Đặt `(* USE_DSP48 = "yes" *)` cho 4 phép nhân và kiểm báo cáo XST: đúng 12 DSP48A1.
- Nếu thiếu DSP, phương án lùi: 1 DSP chạy 4 chu kỳ mỗi kênh (vẫn che được bằng 14.2 vì MAC mỗi vị trí >= ~48 chu kỳ). Mặc định dùng 4 DSP.
- Test riêng bằng testbench so với mô hình số nguyên Python: >= 100.000 vector ngẫu nhiên và vector biên: `acc` = 0, ±1, ±(2^30-1), -2^30; `M0` = 1, 2^31-1, 2^32-1; `SHIFT` = 1, giữa, lớn nhất trong header; trường hợp hòa khi làm tròn với số âm; `CLAMP_LOWER` = 0, `zp_out`, 1.

### 14.2 Chồng requant với MAC vị trí kế

Requant chỉ cần 8 acc đã max-pool, đã được chốt trong `REQUANTIZER` (14.1). Nên `CONV_CONTROLLER` **không chờ** `DONE_REQUANT`.

Sửa vòng lặp mục 9, tại điểm cuối mỗi vị trí đầu ra (sau pool position cuối, pipeline PE đã xả):

```
nếu REQUANT_READY == 0: chờ                       // hiếm, MAC mỗi vị trí >> 8 + REQUANT_LATENCY chu kỳ
pending_write_base_addr = (output_row*OUT_SIZE + output_col)*OUT_CHANNELS + group_index*8
pending_argmax_code     = POOL_ARGMAX_CODE         // chỉ lớp 2
START_REQUANT                                      // ACC_VECTOR = giá trị pool-max cuối
sang vị trí kế NGAY (LOAD_BIAS_INTO_ACC, POOL_FIRST_POSITION, ...) không chờ DONE_REQUANT

song song, mỗi chu kỳ có OUTPUT_VALID:
    RAM_đích[pending_write_base_addr + OUTPUT_CHANNEL_INDEX] = OUTPUT_DATA
    lớp 2: POOL3_COPY_RAM cùng địa chỉ/dữ liệu; POOL3_ARGMAX_RAM = pending_argmax_code[2*OUTPUT_CHANNEL_INDEX +: 2]
    lớp 6: mask đọc FEATURE_RAM_B tại cùng địa chỉ (phát địa chỉ sớm 1 chu kỳ, trễ dữ liệu cho khớp OUTPUT_DATA); FEATURE_RAM_B[a]==0 thì ghi 128
```

- `pending_write_base_addr`, `pending_argmax_code` là thanh ghi trong `CONV_CONTROLLER`, chốt **cùng chu kỳ** với `START_REQUANT`. Chỉ một requant chạy tại một thời điểm (bảo đảm bởi `REQUANT_READY`) nên một thanh ghi là đủ.
- Không xung đột RAM: RAM nguồn và RAM đích khác nhau (ping-pong); cổng đọc `FEATURE_RAM_B` cho mask lớp 6 vẫn rảnh.
- Hết nhóm: không cần chờ. **Hết lớp:** `DONE_LAYER` chỉ phát khi `REQUANT_READY=1` **và** byte cuối đã ghi vào RAM đích, nếu không lớp sau đọc dữ liệu chưa ghi.
- Lớp FC (1 vị trí) không có gì để chồng: chờ `DONE_REQUANT` rồi mới argmax.

### 14.3 SDRAM đọc burst 8 từ

Nạp trọng số là đọc liên tiếp, nên đọc burst thay vì từng từ đơn (~12 -> ~2 chu kỳ/từ).

`SDRAM_CONTROLLER`:
- Mode register lúc init: burst length 8, kiểu tuần tự, **A9 = 1** (ghi vẫn là từ đơn, nên `SD_LOADER` giữ nguyên), CAS latency giữ như bản hiện tại (CL=2 ở 50 MHz). Gợi ý `A[12:0] = 13'h0223` (A9=1, CL=2, BL=8). **Đối chiếu datasheet HY57V2562GTR trước khi dùng.**
- Đọc burst: ACTIVATE, READ có auto-precharge (`A10=1`), chờ CL, nhận 8 từ. Mỗi từ có 1 xung `SDRAM_RESPONSE_VALID`, từ thứ 8 thêm `SDRAM_RESPONSE_LAST`. Khoảng 14-16 chu kỳ mỗi burst.
- Cổng yêu cầu thêm `SDRAM_REQUEST_BURST` (đọc 8 từ). **Địa chỉ phải căn 8 từ** (`SDRAM_REQUEST_WORD_ADDR[2:0]=0`): burst tuần tự quay vòng trong khối 8 từ, và cột 9 bit nên burst căn 8 không bao giờ vượt hàng.
- Refresh chỉ chèn **giữa** hai burst, không ngắt giữa chừng. Chu kỳ refresh ~390 chu kỳ lớn hơn nhiều một burst.
- Nếu còn chỗ nào đọc từ đơn (debug, checksum của `SD_LOADER`): dùng burst, lấy từ đầu, bỏ 7 từ sau.

`PARAMETER_FETCHER` (mọi lần đọc SDRAM: header, LUT, khối nhóm, hàng FC):
- Địa chỉ từ bắt đầu `word_addr = byte_addr/2` (mọi offset và kích thước khối ở mục 6 đều chẵn). Đọc từ `word_addr & ~7` đến hết khối theo từng burst, **bỏ các từ thừa** ở đầu và cuối. Đọc dư tối đa 7 từ sau khối vẫn nằm trong dữ liệu SDRAM nên vô hại.
- Dùng **2 bộ đệm burst luân phiên** (mỗi bộ 8x16 bit): ghi bộ này ra `WEIGHT_RAM` (RAM 8 bit nên 1 byte/chu kỳ) trong khi đọc burst kế vào bộ kia. Tốc độ ~2 chu kỳ/từ.

### 14.4 Nạp trước nhóm kế (đệm đôi `WEIGHT_RAM`)

- `WEIGHT_RAM` sâu 1024 (mục 7): địa chỉ = `{bank, addr[8:0]}`. PE đọc bank `COMPUTE_BANK`, `PARAMETER_FETCHER` ghi bank `FETCH_BANK` (khác nhau khi nạp trước). Hai cổng của `SIMPLE_DUAL_PORT_RAM` độc lập nên đọc và ghi cùng lúc được.
- Bias nạp vào `bias_shadow[255:0]` trong `PARAMETER_FETCHER`. Xung `APPLY_BIAS` chép sang 8 thanh ghi bias của `COMPUTE_CORE`.
- Cổng mới: `FETCH_BANK` (vào `PARAMETER_FETCHER`), `COMPUTE_BANK` (từ `CONV_CONTROLLER` tới 8 PE), `APPLY_BIAS`. `START_FETCH_WEIGHTS` / `DONE_FETCH_WEIGHTS` giữ nguyên.

Lịch của `CONV_CONTROLLER` cho một lớp:

```
FETCH_BANK = 0; START_FETCH_WEIGHTS(group 0); chờ DONE_FETCH_WEIGHTS       // nhóm đầu: nạp chặn
for group_index in 0 .. GROUP_COUNT-1:
    COMPUTE_BANK = group_index[0]; APPLY_BIAS
    if group_index+1 < GROUP_COUNT:
        FETCH_BANK = ~group_index[0]; START_FETCH_WEIGHTS(group_index+1)   // chạy ngầm trong lúc MAC
    ... vòng MAC của nhóm group_index (mục 9) ...
    if group_index+1 < GROUP_COUNT: chờ DONE_FETCH_WEIGHTS                 // thường đã xong
```

- `APPLY_BIAS` chỉ phát khi không còn phép cộng dồn nào đang chạy (pipeline PE đã xả). `START_FETCH_WEIGHTS` của nhóm kế phát **sau** `APPLY_BIAS`, vì nó ghi đè `bias_shadow`.
- Không nạp trước xuyên lớp (nhóm 0 của lớp sau vẫn nạp chặn). Nhóm đầu mỗi lớp lộ ra khoảng ~14k chu kỳ tổng cộng, chấp nhận được.
- `START_FETCH_CLASS_ROW` (LayerCAM) dùng `GAP_RAM`, không đụng `WEIGHT_RAM`, và chạy khi không có nạp trước nào đang hoạt động.
- Fetch một nhóm (<= 3.488 byte, ~2 chu kỳ/từ) luôn ngắn hơn MAC của nhóm đó (tối thiểu ~18k chu kỳ ở conv5/BWD), nên che hoàn toàn.
