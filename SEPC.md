# Đặc tả triển khai: cascade 2 CNN trên AX309 (Spartan-6 XC6SLX9)

Tài liệu này dành cho AI/kỹ sư viết RTL. Đọc hết trước khi code. Nguồn sự thật về phép tính là hàm `golden_run` trong `CNN_Search_BN.py`: RTL phải **khớp từng bit** với nó.

## 0. Mục tiêu và ràng buộc

- PC gửi ảnh **đã resize 64x64 RGB** qua UART. FPGA chạy CNN 1 (Hospital_A). Nếu ra `Unknown` thì chạy CNN 2 (Hospital_B) trên **cùng ảnh**. FPGA gửi kết quả về PC, kèm thông tin để PC biết CNN nào quyết định.
- Hai CNN **cùng kiến trúc**, chỉ khác trọng số, hệ số requant, zero-point và bảng LUT đầu vào. Tham số của cả hai nằm trên thẻ SD, được chép vào SDRAM lúc khởi động. Mỗi lần chạy một CNN thì nạp tham số của CNN đó từ SDRAM.
- Kiến trúc **tuần tự, dùng lại khối**: một bộ tính conv duy nhất chạy lần lượt mọi lớp (kể cả FC) và cả hai CNN.
- Công cụ: **ISE 14.7 (XST), Verilog-2001**. Không dùng SystemVerilog (`logic`, `always_ff`, `int`, mảng packed 2 chiều ở cổng). Cổng dạng mảng thì dùng vector phẳng, ví dụ `[255:0]` cho 8 giá trị 32-bit, kênh k là `[32*k +: 32]`.
- Một miền clock duy nhất: 50 MHz. Reset đồng bộ.

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

Từ vựng cố định: `row`/`col` (không dùng y/x), `in_channel`/`out_channel` (không dùng cin/cout), `kernel_row`/`kernel_col`, `group` (nhóm 8 kênh ra), `pixel`. Cấm: `cnt`, `tmp`, `ptr`, `d1`, `flag`, `sig1`.

Độ trễ pipeline thì đặt tên rõ: `accumulate_enable_delayed`, không dùng `ae_d2`.

## 2. Chân (UCF), tất cả LVCMOS33

| Cổng top | Chân | Ghi chú |
|---|---|---|
| `CLK` | T8 | 50 MHz |
| `RESET_N` | L3 | nút RESET, nhấn = 0 |
| `UART_RX` | C11 | FPGA nhận từ PC |
| `UART_TX` | D12 | FPGA gửi lên PC |
| `LED[3:0]` | P4, N5, P5, M6 | mức 1 = sáng |
| `SD_CS_N` | N3 | |
| `SD_CLK` | M3 | |
| `SD_MOSI` | L5 | manual đặt tên `SD_DIN` (FPGA -> thẻ) |
| `SD_MISO` | L4 | manual đặt tên `SD_DOUT` (thẻ -> FPGA) |
| `SDRAM_CLK` | H4 | |
| `SDRAM_CKE` | H2 | luôn 1 sau reset |
| `SDRAM_CS_N` | G1 | |
| `SDRAM_RAS_N` | F1 | |
| `SDRAM_CAS_N` | F2 | |
| `SDRAM_WE_N` | E1 | |
| `SDRAM_DQM[1:0]` | H1 (bit1), E2 (bit0) | |
| `SDRAM_BANK[1:0]` | J6 (bit1), G6 (bit0) | |
| `SDRAM_ADDR[12:0]` | J1,K1,K6,K2,L1,M1,M2,N1,P1,K5,K3,J4,J3 | liệt kê từ bit 12 xuống bit 0 |
| `SDRAM_DATA[15:0]` | F4,F3,F5,G3,F6,H3,G5,H5,D1,C1,C2,B1,B2,A2,B3,A3 | liệt kê từ bit 15 xuống bit 0 |

Thêm `NET CLK PERIOD = 20 ns;`. SDRAM là HY57V2562GTR: 4 bank, 13 bit hàng, 9 bit cột, rộng 16 bit, tổng 16M từ. Địa chỉ từ 24 bit = `{bank[1:0], row[12:0], col[8:0]}`.

## 3. Kiến trúc CNN (`16p-24p-24p-32-48`)

Mọi conv là 3x3, pad=1, stride=1. BN đã được fold vào weight/bias. Có 8 PE chạy song song, mỗi PE tính một kênh ra, nên một lớp được tính theo từng **nhóm 8 kênh ra**.

| Lớp | `LAYER_INDEX` | Kích thước vào | Kênh vào | Kênh ra | MaxPool 2x2 | Số nhóm | Nguồn đọc | Đích ghi |
|---|---|---|---|---|---|---|---|---|
| Conv1 | 0 | 64 | 3 | 16 | có | 2 | `IMAGE_RAM` (+ LUT) | `FEATURE_RAM_A` |
| Conv2 | 1 | 32 | 16 | 24 | có | 3 | A | `FEATURE_RAM_B` |
| Conv3 | 2 | 16 | 24 | 24 | có | 3 | B | A |
| Conv4 | 3 | 8 | 24 | 32 | không | 4 | A | B |
| Conv5 | 4 | 8 | 32 | 48 | không | 6 | B | A |
| GAP | | 8x8, 48 kênh | | | | | A | `GAP_RAM` |
| FC | 5 | 1 | 48 | 5 (đệm lên 8) | không | 1 | `GAP_RAM` | `LOGIT_RAM` |

- Bộ nhớ feature map luân phiên (ping-pong): mỗi lớp đọc từ RAM này và ghi sang RAM kia. `IMAGE_RAM` **không được ghi đè** cho tới khi cả cascade xong (CNN 2 cần dùng lại).
- Cách xếp feature map: `addr = (row*IN_SIZE + col)*CHANNELS + channel`. Ảnh: `addr = (row*64 + col)*3 + channel`, với channel 0,1,2 = R,G,B.
- FC dùng lại đúng khối conv ở chế độ `IS_FULLY_CONNECTED`: chỉ có một vị trí, không padding, địa chỉ đọc vào và địa chỉ trọng số đều bằng `in_channel_index` (0..47), không ReLU. 3 kênh ra thừa (kênh 5..7) được đệm bằng 0 trong file SD, cứ ghi vào `LOGIT_RAM` (độ sâu 8) rồi bỏ qua khi đọc.

## 4. Phép tính chính xác (bit-exact)

Ký hiệu: `x` = pixel uint8 sau LUT (lớp 1) hoặc đọc từ feature RAM, `w` = int8, `zp_in`, `zp_out` lấy từ header theo lớp.

1. **Tích lũy (mỗi kênh ra):** `acc = bias_int32 + Σ (x - zp_in) * w` (acc int32 có dấu, |acc| < 2^30).
   - `x - zp_in` có dấu, rộng 9 bit (-127..255).
   - Ô nằm ngoài ảnh (padding) có giá trị = `zp_in`, nghĩa là đóng góp bằng 0. **Bỏ qua hẳn vị trí kernel đó** (không cộng gì, không tốn chu kỳ).
2. **MaxPool (lớp có pool):** với khối 2x2 (4 vị trí conv `(2r,2c),(2r,2c+1),(2r+1,2c),(2r+1,2c+1)`), lấy **max của 4 acc (có dấu) trước khi requant**. Kết quả vẫn khớp golden vì requant, ReLU và clamp đều không giảm. Lớp không pool thì chỉ có 1 vị trí.
3. **Requant:** `product = acc * M0` (acc int32 có dấu, M0 uint31 nên product có dấu 64 bit), `value = (product + (1 << (SHIFT-1))) >>> SHIFT` (**dịch số học**, 64 bit), `y = value + zp_out`.
   Kẹp `y` vào `[lower, 255]` với `lower = zp_out` nếu `RELU_ENABLE` bằng 1, ngược lại `lower = 0`. Phải kẹp trên giá trị rộng (64 bit), không cắt bit trước khi kẹp.
   Phiên bản đầu được phép nhân bằng shift-add nhiều chu kỳ (đơn giản). Tối ưu bằng DSP48A1 để sau.
4. **GAP:** `gap[c] = (Σ_64pixel x[c] + 32) >> 6`, với c = 0..47.
5. **FC:** như bước 1 và 3 với đầu vào là `gap`, `zp_in` lấy từ header (lớp 5), `zp_out = 127`, không ReLU (kẹp [0,255]). 5 logit là uint8.
6. **Argmax:** trên 5 logit uint8 (khớp `lg_rq.argmax`). Nếu hòa thì lấy chỉ số nhỏ nhất.

## 5. Giao thức UART (8N1, 1 Mbaud = 50 MHz / 50)

Tham số `CLOCKS_PER_BIT = 50`. Có thể đổi về 115200 baud (434) để thử.

- **PC -> FPGA:** `0xA5` rồi 12.288 byte ảnh theo thứ tự hàng, mỗi pixel `R,G,B`. Nếu giữa hai byte im lặng quá ~100 ms thì bỏ khung, quay lại chờ `0xA5`.
- **FPGA -> PC, sau khi khởi động xong:** 1 byte `0x55` (sẵn sàng). Lỗi khởi động thì gửi `0xEE, mã_lỗi` (1 = SD init lỗi, 2 = đọc sector lỗi, 3 = sai magic).
- **FPGA -> PC, khung kết quả 13 byte:**

| Byte | Nội dung |
|---|---|
| 0 | `0xAA` |
| 1 | `FINAL_SOURCE`: `0x01` = CNN 1 quyết định, `0x02` = CNN 2 quyết định, `0x00` = Unknown cuối cùng |
| 2 | `FINAL_CLASS`: chỉ số lớp trong CNN đã quyết định, hoặc `0xFF` nếu Unknown cuối |
| 3..7 | 5 logit CNN 1 |
| 8..12 | 5 logit CNN 2 (bằng 0 nếu CNN 2 không chạy) |

Logic cascade: CNN 1 ra lớp khác `UNKNOWN_CLASS_INDEX` thì dừng (`SOURCE=1`). Ngược lại chạy CNN 2: ra lớp khác Unknown thì `SOURCE=2`, còn Unknown thì `SOURCE=0`, `CLASS=0xFF`. `UNKNOWN_CLASS_INDEX` lấy từ header của từng CNN, **không hard-code 4**.

LED: `LED[0]` = đã khởi động xong, `LED[1]` = đang chạy CNN, `LED[2]` = lỗi, `LED[3]` = đảo mỗi lần có kết quả.

## 6. Định dạng thẻ SD và SDRAM

Ghi **sector thô** (không FAT), 128 sector = 65.536 byte. Cảnh báo: ghi thô sẽ xóa dữ liệu thẻ. Dùng thẻ SDHC. Sector 0..63 là CNN 1, sector 64..127 là CNN 2. Chép nguyên khối vào SDRAM từ byte 0. Byte chẵn nằm ở `[7:0]` của từ 16 bit, byte lẻ ở `[15:8]`. Địa chỉ nền CNN k: `k * 0x8000` (byte).

Bố cục một khối CNN (32.768 byte, little-endian):

| Offset | Nội dung |
|---|---|
| 0x000 | magic `"FCNN"` (46 43 4E 4E), `CNN_ID` (1 byte), `UNKNOWN_CLASS_INDEX` (1 byte), `NUM_CLASSES` (1 byte, =5), 1 byte dự trữ |
| 0x008 | bảng 6 lớp, mỗi lớp 8 byte: `M0` (u32), `SHIFT` (u8), `ZP_IN` (u8), `ZP_OUT` (u8), `RELU_ENABLE` (u8). Lớp 0..4 là conv, lớp 5 là FC |
| 0x200 | LUT đầu vào 3x256 byte: R rồi G rồi B, chỉ số = pixel thô 0..255 (0x200..0x4FF) |
| 0x600 | dữ liệu các lớp, xếp liền, không đệm |

Mỗi **khối nhóm** (8 kênh ra): `bias` 8 x int32 (32 byte), rồi trọng số của PE0, PE1, ..., PE7, mỗi PE là `IN_CHANNELS*9` byte int8 theo thứ tự `[in_channel][kernel_row][kernel_col]`. Thứ tự này trùng với dữ liệu `.mem` của Python (flatten theo `[out_channel][in_channel][kh][kw]`), nên mỗi nhóm là một đoạn liên tục. FC: `IN_CHANNELS=48`, PE5..7 đệm bằng 0 (cả bias lẫn trọng số).

Bảng `LAYER_CONFIG_ROM` (hằng số cố định):

| `LAYER_INDEX` | Offset từ nền CNN | Byte mỗi khối nhóm | Số nhóm |
|---|---|---|---|
| 0 | 0x0600 (1536) | 248 | 2 |
| 1 | 0x07F0 (2032) | 1184 | 3 |
| 2 | 0x15D0 (5584) | 1760 | 3 |
| 3 | 0x2A70 (10864) | 1760 | 4 |
| 4 | 0x45F0 (17904) | 2336 | 6 |
| 5 (FC) | 0x7CB0 (31920) | 416 | 1 |

Địa chỉ khối nhóm g của lớp L: `CNN_BASE + offset[L] + g*bytes_per_group[L]`. Hết dữ liệu tại 0x7E50, phần còn lại của 32.768 byte là 0.

## 7. Bộ nhớ trong chip

Dùng một module chung `SIMPLE_DUAL_PORT_RAM` (rộng 8 bit, 1 cổng ghi, 1 cổng đọc, **đọc đồng bộ trễ 1 chu kỳ**) cho tất cả:

| RAM | Độ sâu (byte) | Ghi chú |
|---|---|---|
| `IMAGE_RAM` | 12288 | ảnh thô, ghi bởi `HOST_INTERFACE` |
| `FEATURE_RAM_A` | 16384 | |
| `FEATURE_RAM_B` | 6144 | |
| `INPUT_LUT_RAM` | 768 | địa chỉ `in_channel*256 + pixel_thô` |
| `WEIGHT_RAM` x8 | 288 mỗi cái | một cái cho mỗi PE; nên là distributed RAM, dùng BRAM cũng được |
| `GAP_RAM` | 64 | dùng 48 |
| `LOGIT_RAM` | 8 | dùng 5 |

Bias: 8 thanh ghi 32 bit nằm trong `COMPUTE_CORE`.

## 8. Danh sách module (11 module)

Mọi module có `CLK`, `RST` (RST nội bộ tích cực mức cao, tạo từ `RESET_N` qua 2 FF đồng bộ). Chỉ liệt kê cổng quan trọng.

Cây phân cấp (module nào nằm trong module nào):

```
TOP_CNN
├─ UART_RECEIVER
├─ UART_TRANSMITTER
├─ HOST_INTERFACE            (nhận ảnh, gửi kết quả)
├─ SD_LOADER                 (thẻ SD -> SDRAM, chạy 1 lần lúc khởi động)
├─ SDRAM_CONTROLLER
├─ SIMPLE_DUAL_PORT_RAM      (IMAGE_RAM, dùng chung 1 module RAM cho mọi bộ nhớ)
├─ CNN_RUNNER                (chạy trọn 1 CNN, gọi 2 lần với CNN_SELECT khác nhau)
│   ├─ PARAMETER_FETCHER     (SDRAM -> header, LUT, bias, trọng số)
│   ├─ CONV_CONTROLLER       (vòng lặp một lớp + LAYER_CONFIG_ROM bên trong)
│   ├─ COMPUTE_CORE          (PE_ARRAY + max-pool + REQUANTIZER)
│   ├─ SIMPLE_DUAL_PORT_RAM  (FEATURE_RAM_A/B, INPUT_LUT_RAM, GAP_RAM, LOGIT_RAM)
│   └─ phần GAP + argmax     (hai vòng lặp ngắn, viết thẳng trong CNN_RUNNER)
└─ FSM cascade               (viết thẳng trong TOP_CNN)
```

| # | Module | Việc làm | Cổng chính |
|---|---|---|---|
| 1 | `UART_RECEIVER` | Nhận byte, đồng bộ 2 FF, lấy mẫu giữa bit | `UART_RX` -> `RX_BYTE[7:0]`, `RX_BYTE_VALID` |
| 2 | `UART_TRANSMITTER` | Gửi byte | `TX_BYTE[7:0]`, `TX_BYTE_VALID` -> `UART_TX`, `BUSY_UART_TX` |
| 3 | `HOST_INTERFACE` | Nhận: chờ `0xA5`, ghi 12288 byte vào `IMAGE_RAM` (có timeout). Gửi: `0x55` (sẵn sàng), `0xEE`+mã lỗi, khung kết quả 13 byte, qua `UART_TRANSMITTER` | `RX_BYTE`, `RX_BYTE_VALID`, `TX_BYTE`, `TX_BYTE_VALID`, `BUSY_UART_TX`, `IMAGE_WRITE_ENABLE`, `IMAGE_WRITE_ADDR[13:0]`, `IMAGE_WRITE_DATA[7:0]`, `DONE_IMAGE_RECEIVE`, `START_SEND_READY`, `START_SEND_ERROR`, `ERROR_CODE[7:0]`, `START_SEND_RESULT`, `FINAL_SOURCE[7:0]`, `FINAL_CLASS[7:0]`, `LOGITS_CNN1[39:0]`, `LOGITS_CNN2[39:0]`, `DONE_SEND` |
| 4 | `SD_LOADER` | Một FSM làm 3 việc nối tiếp: (a) SPI mode 0, truyền 1 byte, chia clock được; (b) init SD: 80 xung clock chậm ~400 kHz với CS=1, CMD0 (CRC 0x95), CMD8 arg 0x1AA (CRC 0x87), lặp CMD55+ACMD41 arg 0x40000000, CMD58 kiểm tra CCS, rồi đổi sang clock nhanh (~6 MHz); (c) đọc 128 sector bằng CMD17 (chờ token 0xFE, 512 byte, bỏ 2 byte CRC), ghép 2 byte thành 1 từ (byte đầu ở `[7:0]`) và ghi vào SDRAM từ địa chỉ 0 | `SD_CS_N`, `SD_CLK`, `SD_MOSI`, `SD_MISO`, `START_BOOT_LOAD`, `DONE_BOOT_LOAD`, `ERROR_CODE[7:0]`, và cổng chủ SDRAM `SDRAM_REQUEST_*` (xem bên dưới bảng) |
| 5 | `SDRAM_CONTROLLER` | Init (chờ 200 µs, PRECHARGE ALL, 2 AUTO REFRESH, LOAD MODE = BL1, CL3, tuần tự, `13'h030`), refresh định kỳ (~370 chu kỳ), mỗi truy cập một từ: ACTIVE -> READ/WRITE với auto-precharge. Không cần burst (hiệu năng không quan trọng). Xuất `SDRAM_CLK` ngược pha `CLK` bằng ODDR2, `SDRAM_DATA` dùng IOBUF tri-state. **Khó nhất, test riêng.** | `REQUEST_VALID`, `REQUEST_WRITE`, `REQUEST_WORD_ADDR[23:0]`, `REQUEST_WRITE_DATA[15:0]`, `REQUEST_READY`, `RESPONSE_VALID`, `RESPONSE_READ_DATA[15:0]`, và các chân `SDRAM_*` |
| 6 | `SIMPLE_DUAL_PORT_RAM` | RAM 8 bit, 1 cổng ghi, 1 cổng đọc, đọc đồng bộ trễ 1 chu kỳ. Tham số `DEPTH` | `WRITE_ENABLE`, `WRITE_ADDR`, `WRITE_DATA[7:0]`, `READ_ADDR`, `READ_DATA[7:0]` |
| 7 | `PARAMETER_FETCHER` | Đọc byte liên tiếp từ SDRAM (tách mỗi từ thành 2 byte) rồi chia theo mục đích. **Nạp tham số CNN** (header + LUT) và **nạp khối nhóm** (bias + 8 `WEIGHT_RAM`) | xem bên dưới bảng |
| 8 | `COMPUTE_CORE` | Gồm: `PE_ARRAY` (8 PE, mỗi PE `acc <= acc + (pixel - zero_point) * weight`, có dấu, DSP48A1), bộ max-pool (8 thanh ghi max có dấu 32 bit), `REQUANTIZER` (module con, có testbench riêng, công thức mục 4.3, xử lý lần lượt 8 kênh) | xem bên dưới bảng |
| 9 | `CONV_CONTROLLER` | Vòng lặp một lớp (mục 9) và hằng số `LAYER_CONFIG_ROM` (bảng mục 3 và 6). Sinh địa chỉ đọc đầu vào, địa chỉ trọng số, các cờ điều khiển `COMPUTE_CORE`, địa chỉ ghi đầu ra `output_write_addr = (output_row*OUT_SIZE + output_col)*OUT_CHANNELS + group_index*8 + REQUANT_CHANNEL_IN_GROUP` | `START_LAYER`, `LAYER_INDEX[2:0]`, `DONE_LAYER`, các hằng số lớp (`IN_SIZE`, `IN_CHANNELS`, `OUT_CHANNELS`, `POOL_ENABLE`, `GROUP_COUNT`, `GROUP_BYTES`, `LAYER_OFFSET`, `IS_FULLY_CONNECTED`), địa chỉ RAM nguồn/đích |
| 10 | `CNN_RUNNER` | **Chạy trọn một CNN**: `PARAMETER_FETCHER` nạp tham số -> lớp 0..4 -> GAP (mục 4.4) -> FC -> argmax (mục 4.6). Chứa các RAM nội bộ, giữ chỉ số ping-pong, nối `CONV_CONTROLLER`, `COMPUTE_CORE`, `PARAMETER_FETCHER` | `START_CNN`, `CNN_SELECT`, `DONE_CNN`, `PREDICTED_CLASS_INDEX[2:0]`, `LOGITS_OUTPUT[39:0]`, `UNKNOWN_CLASS_INDEX[7:0]`, cổng đọc `IMAGE_RAM` (`IMAGE_READ_ADDR[13:0]`, `IMAGE_READ_DATA[7:0]`), cổng chủ SDRAM `SDRAM_REQUEST_*`, `ERROR_BAD_MAGIC` |
| 11 | `TOP_CNN` | Nối tất cả theo mục 2. Chứa `IMAGE_RAM`, mux cổng SDRAM giữa `SD_LOADER` (lúc khởi động) và `CNN_RUNNER` (lúc chạy) theo cờ `boot_finished`, và **FSM cascade**: `STATE_INIT_SDRAM` -> `STATE_COPY_SD_TO_SDRAM` -> `STATE_SEND_READY` -> `STATE_WAIT_IMAGE` -> `STATE_RUN_CNN1` -> `STATE_CHECK_CNN1` -> (`STATE_RUN_CNN2` -> `STATE_CHECK_CNN2`) -> `STATE_SEND_RESULT` -> `STATE_WAIT_IMAGE`. Quyết định `FINAL_SOURCE`/`FINAL_CLASS` theo mục 5 | `CLK`, `RESET_N`, `UART_RX`, `UART_TX`, `LED[3:0]`, chân SD, chân SDRAM |

Cổng chủ SDRAM (cùng tên ở `SD_LOADER` và `CNN_RUNNER`, nối vào mux của `TOP_CNN`):
`SDRAM_REQUEST_VALID`, `SDRAM_REQUEST_WRITE`, `SDRAM_REQUEST_WORD_ADDR[23:0]`, `SDRAM_REQUEST_WRITE_DATA[15:0]`, `SDRAM_REQUEST_READY`, `SDRAM_RESPONSE_VALID`, `SDRAM_RESPONSE_READ_DATA[15:0]`. Hai bên không bao giờ chạy cùng lúc.

Cổng của `PARAMETER_FETCHER`:
- Nạp tham số CNN: `START_FETCH_PARAMETERS`, `CNN_SELECT`, `DONE_FETCH_PARAMETERS`, `ERROR_BAD_MAGIC` (header sai magic), `UNKNOWN_CLASS_INDEX[7:0]`. Header 56 byte được giữ trong thanh ghi và trả ra theo `LAYER_INDEX[2:0]`: `LAYER_MULTIPLIER[30:0]`, `LAYER_SHIFT[5:0]`, `LAYER_INPUT_ZERO_POINT[7:0]`, `LAYER_OUTPUT_ZERO_POINT[7:0]`, `LAYER_RELU_ENABLE`. 768 byte LUT ghi vào `INPUT_LUT_RAM` qua `INPUT_LUT_WRITE_ENABLE`, `INPUT_LUT_WRITE_ADDR[9:0]`, `INPUT_LUT_WRITE_DATA[7:0]`.
- Nạp khối nhóm: `START_FETCH_WEIGHTS`, `GROUP_INDEX[2:0]`, `DONE_FETCH_WEIGHTS`. Nhận từ `CONV_CONTROLLER` các hằng số `LAYER_OFFSET`, `GROUP_BYTES`, `IN_CHANNELS`. Ghi ra `BIAS_WRITE_ENABLE`, `BIAS_WRITE_INDEX[2:0]`, `BIAS_WRITE_DATA[31:0]`, `WEIGHT_WRITE_ENABLE`, `WEIGHT_WRITE_PE_INDEX[2:0]`, `WEIGHT_WRITE_ADDR[8:0]`, `WEIGHT_WRITE_DATA[7:0]`. 32 byte đầu của khối vào bias, phần còn lại vào `WEIGHT_RAM` của PE0, rồi PE1, ..., PE7.

Cổng của `COMPUTE_CORE`:
- Đầu vào: `INPUT_PIXEL[7:0]`, `INPUT_ZERO_POINT[7:0]`, `WEIGHT_READ_ADDR[8:0]` (chung cho 8 PE), `LOAD_BIAS_INTO_ACC`, `ACCUMULATE_ENABLE`, `POOL_FIRST_POSITION` (nạp thẳng acc vào max), `POOL_NEXT_POSITION` (lấy max), `START_REQUANT`, `LAYER_MULTIPLIER`, `LAYER_SHIFT`, `LAYER_OUTPUT_ZERO_POINT`, `LAYER_RELU_ENABLE`, và các cổng ghi bias/trọng số của `PARAMETER_FETCHER`.
- Đầu ra: `REQUANT_BYTE[7:0]`, `REQUANT_BYTE_VALID`, `REQUANT_CHANNEL_IN_GROUP[2:0]`, `DONE_REQUANT`.
- Lớp không pool thì chỉ có 1 vị trí nên max = acc, không cần mạch bypass.

Ở lớp 1, đường đọc đầu vào là `IMAGE_RAM[addr] -> INPUT_LUT_RAM[in_channel*256 + pixel_thô] -> INPUT_PIXEL` (thêm 1 chu kỳ trễ). Các lớp khác đọc thẳng từ feature RAM.

## 9. Vòng lặp của `CONV_CONTROLLER` (một lớp)

```
for group_index in 0 .. GROUP_COUNT-1:
    PARAMETER_FETCHER nạp khối nhóm (START_FETCH_WEIGHTS: bias + 8 PE), chờ DONE_FETCH_WEIGHTS
    for output_row, output_col:                        // vị trí ra (đã pool nếu có pool)
        for pool_position in 0 .. (POOL_ENABLE ? 3 : 0):
            conv_row = output_row*(POOL_ENABLE?2:1) + pool_position/2   // pool_position = 0..3: hàng = /2, cột = %2
            conv_col = output_col*(POOL_ENABLE?2:1) + pool_position%2
            LOAD_BIAS_INTO_ACC
            for kernel_row 0..2, kernel_col 0..2:
                input_row = conv_row + kernel_row - 1
                input_col = conv_col + kernel_col - 1
                if input_row/col nằm ngoài [0, IN_SIZE-1]: continue   // padding: bỏ qua
                for in_channel_index 0 .. IN_CHANNELS-1:
                    input_addr  = (input_row*IN_SIZE + input_col)*IN_CHANNELS + in_channel_index
                    WEIGHT_READ_ADDR = in_channel_index*9 + kernel_row*3 + kernel_col
                    ACCUMULATE_ENABLE
            pool: pool_position==0 ? POOL_FIRST_POSITION : POOL_NEXT_POSITION
        START_REQUANT; ghi 8 byte ra RAM đích; chờ DONE_REQUANT
DONE_LAYER
```

Với FC (`IS_FULLY_CONNECTED`): 1 vị trí, không vòng kernel, `input_addr = WEIGHT_READ_ADDR = in_channel_index`, ghi 8 byte vào `LOGIT_RAM`.

Pipeline: mọi RAM đọc trễ 1 chu kỳ, DSP trễ thêm vài chu kỳ. Phải trễ các cờ (`ACCUMULATE_ENABLE`, `LOAD_BIAS_INTO_ACC`, ...) cho khớp dữ liệu, đặt tên rõ như `accumulate_enable_delayed`. **Làm bản chạy đúng trước** (thêm chu kỳ chờ nếu cần), tối ưu sau. Ước lượng: ~1M chu kỳ cho mỗi CNN (~20..30 ms ở 50 MHz).

## 10. Phần Python cần viết thêm

| Script | Việc làm |
|---|---|
| `pack_sd_image.py` | Đọc thư mục `.../BEST/Hospital_A` và `Hospital_B`: `INT8/Memory/quant_params.json` (M0, shift, zp, relu, theo thứ tự lớp), `*_weight_int8.mem`, `*_bias_int32.mem`, `INT8/Preprocess/input_lut_{R,G,B}.mem`, `INT8/vgg9_int8_config.json` (chỉ số `Unknown` trong `classes`). Ra `sd_image.bin` (65.536 byte) và `sd_image.hex` (1 byte/dòng, cho `$readmemh`). Ghi thô lên thẻ bằng `dd` hoặc Win32DiskImager |
| `make_test_images.py` | Dùng lại `load_resize_enhance` trong `CNN_Search_BN.py` để tạo ảnh 64x64 giống hệt PC sẽ gửi. Xuất `image{i}_raw64.mem` (12288 dòng hex, thứ tự R,G,B từng pixel). Xuất kết quả mong đợi (logit CNN 1, CNN 2, kết quả cascade) bằng `golden_u8_predict` |
| `host_cascade.py` | pyserial 1 Mbaud: chờ `0x55`, gửi `0xA5` + 12288 byte, đọc 13 byte, so với golden |

PC **bắt buộc** resize bằng đúng `block_average_resize` (ENHANCE_MODE = "none"). Sai khác một bit là kết quả lệch.

## 11. Thứ tự làm và kiểm thử

1. `UART_RECEIVER`, `UART_TRANSMITTER`, `HOST_INTERFACE`, echo qua UART. Pass khi nhận đủ ảnh và gửi lại đúng checksum.
2. `REQUANTIZER` (test riêng trước), `COMPUTE_CORE`, `CONV_CONTROLLER`, `PARAMETER_FETCHER`, `CNN_RUNNER` trong **mô phỏng**, thay `SDRAM_CONTROLLER` bằng mô hình hành vi `SDRAM_SIM_MODEL` (cùng giao diện, nạp từ `sd_image.hex`).
   So từng lớp với file golden `TestVectors/dbg_img{i}_{tên}.mem` (hex, 1 byte/dòng, thứ tự `[row][col][channel]`): `input`, `pool1`, `pool2`, `pool3`, `conv4`, `conv5`, `gap`, `fc`. Lưu ý `conv1`..`conv3` trong file golden là **trước pool**, nên với lớp có pool hãy so với file `pool*`. Pass khi khớp 100%.
3. Mô phỏng cả cascade, so logit với `golden_logits.txt` và kết quả `make_test_images.py`.
4. `SDRAM_CONTROLLER` trên board: ghi mẫu, đọc lại, báo qua UART.
5. `SD_LOADER` (phần init và đọc sector): đọc sector 0, gửi magic qua UART.
6. `SD_LOADER` (phần chép vào SDRAM): chép xong thì đọc lại, so checksum.
7. Chạy trọn trên board với `host_cascade.py`.

## 12. Những chỗ dễ sai

- Dịch số học (`>>>`) trên số có dấu khi requant. Không dùng `>>` cho số có dấu.
- Padding = bỏ qua vị trí kernel (đóng góp 0), **không** nạp giá trị 0 cho pixel (vì `zp_in` của lớp 1 là 127, không phải 0).
- ReLU: cận dưới của kẹp là `zp_out`, không phải 0. FC không ReLU nên cận dưới là 0.
- Max-pool trên acc có dấu, trước requant. Mỗi vị trí pool phải **nạp lại bias** vào acc.
- Thứ tự trọng số `[in_channel][kernel_row][kernel_col]` và thứ tự kênh LUT R, G, B.
- Khối nhóm FC phải đệm đủ 8 PE (đã xử lý trong `pack_sd_image.py`).
- `IMAGE_RAM` giữ nguyên đến hết cascade. Mỗi CNN nạp lại LUT, hệ số requant, zero-point riêng.
- Nhớ trễ cờ điều khiển cho khớp độ trễ RAM/DSP.
- Spartan-6 dùng ISE. Chỉ Verilog-2001.
