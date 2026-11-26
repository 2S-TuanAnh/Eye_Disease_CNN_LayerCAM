# Hướng dẫn sửa/bổ sung RTL theo `SEPC_LayerCAM.md` (chỉ phần LCAM)

Đối tượng đọc: AI/kỹ sư sửa code. Đọc `SEPC_LayerCAM.md` mục 3.1, 6, 7, 8, 10 trước. Verilog-2001, ISE 14.7. Giữ quy tắc đặt tên mục 1 của spec.

## 0. Hiện trạng

Code Verilog đang có là **bản gốc, chưa LCAM, chưa OPT**: SD 128 sector (CNN2 ở `0x8000`), 6 lớp (0..5), `WEIGHT_RAM` sâu 288, `REQUANTIZER` shift-add 8 kênh, `PARAMETER_FETCHER` đọc từng từ, HOST gửi khung 13 byte.

**Làm LCAM trước. KHÔNG làm mục 14 (OPT)** cho tới khi LCAM khớp golden.

### File còn thiếu (theo `TOP_CNN.v`)

Mọi module `TOP_CNN` gọi đều đã có đủ file. Cần **tạo mới** 2 file Verilog:

| File mới | Việc |
|---|---|
| `LAYERCAM_CONTROLLER.v` | Điều khiển 4 bước LayerCAM (mục 3 dưới đây) |
| `REQUANTIZER_SERIAL.v` | Bản **1 kênh** của `REQUANTIZER` dùng cho CAM (mục 2.2) |

Python đã có đủ 4 file (`cam_best_cnn_v3.py`, `pack_sd_image.py`, `make_test_images.py`, `host_cascade.py`) trong đó `cam_best_cnn_v3.py` đã sửa xong và kiểm đạt (6.1); **3 file còn lại** (`pack_sd_image.py`, `make_test_images.py`, `host_cascade.py`) chưa theo LCAM, cần sửa ở mục 6. Còn thiếu (không cần gửi): UCF (không đổi chân), testbench; `sd_image.hex` và các `.mem` do script sinh ra. `lcam_params*.vh` **không cần** đưa vào RTL (tham số nằm trên thẻ SD); 2 file trùng tên `localparam` nên không `include` cùng lúc.

### Hằng số kiểm chứng (từ JSON mới đã sửa 13.1, `n_images = 128`; khớp 100% với 2 file `.vh` mới)

| | Hospital_A (CNN1) | Hospital_B (CNN2) |
|---|---|---|
| `NORMAL_CLASS_INDEX` / `UNKNOWN_CLASS_INDEX` | 1 / 4 | `0xFF` (không có Normal) / 4 |
| Lớp 6: `M0`, `SHIFT` | 1928382021, 42 | 1283153167, 42 |
| Lớp 7: `M0`, `SHIFT` | 1277726974, 39 | 1831985940, 40 |
| CAM: `M0`, `SHIFT` | 1235282224, 35 | 1595037662, 35 |

Dùng để kiểm tra đầu ra của `pack_sd_image.py` (đọc lại bảng `0x8000` trong `sd_image.bin`). Lưu ý tên file: `lcam_params_e59f3b.json` / `lcam_params_83c94c.vh` là Hospital_A, `lcam_params_a6b4f9.json` / `lcam_params_5c197b.vh` là Hospital_B. `pack_sd_image.py` đọc đúng tên **`lcam_params.json`** trong thư mục `LayerCAM_B3`, nên nếu dùng file đã đổi tên thì đặt lại tên, hoặc thêm tham số chỉ đường dẫn JSON.

Mọi `M0` < 2^31 nên vừa cổng 31 bit hiện có.

## 1. Thứ tự làm

1. Sửa các file "đổi nhỏ" (mục 2) → mô phỏng lại forward, kết quả **phải giống cũ** (chưa bật LCAM).
2. Viết `REQUANTIZER_SERIAL`, `LAYERCAM_CONTROLLER`.
3. Nối vào `CNN_RUNNER`, rồi `TOP_CNN`, `HOST_INTERFACE`.
4. Mô phỏng theo mục 12 bước 3-4 của spec.

## 2. Sửa từng file

### 2.1 Đổi nhỏ

**`SD_LOADER.v`**
- Tham số `TOTAL_SECTOR_COUNT` 128 → **256**.
- `sector_has_magic`: `8'd64` → **`8'd128`** (magic CNN2 nằm ở sector 128). Sửa comment đầu file.
- Các phần khác (`sector_index` 8 bit, địa chỉ từ `{8'd0, sector_index, byte[8:1]}`) đã đủ cho 256 sector.

**`SDRAM_SIM_MODEL.v`** (chỉ mô phỏng)
- `image_bytes [0:65535]` → `[0:131071]`, vòng init 131072.
- `memory_words [0:32767]` → `[0:65535]`, vòng init 65536; `pending_word_addr` 15 → **16 bit**, mọi chỗ cắt `REQUEST_WORD_ADDR[14:0]` → `[15:0]`.

**`LAYER_CONFIG_ROM.v`**
- `LAYER_OFFSET` `[14:0]` → **`[15:0]`** (offset 46976 > 32767), đổi literal `15'd` → `16'd`.
- Thêm 2 case (cùng khuôn case 3, `POOL_ENABLE=0`, `IS_FULLY_CONNECTED=0`, `IN_SIZE=8`, `IN_SIZE_LOG2=3`, `OUT_SIZE=8`, `OUT_SIZE_LOG2=3`):

| case | `IN_CHANNELS` | `OUT_CHANNELS` | `GROUP_COUNT` | `GROUP_BYTES` | `LAYER_OFFSET` |
|---|---|---|---|---|---|
| `3'd6` (BWD1) | 48 | 32 | 4 | 3488 | 33024 |
| `3'd7` (BWD2) | 32 | 24 | 3 | 2336 | 46976 |

**`REQUANTIZER.v`**
- Thay cổng `LAYER_RELU_ENABLE` bằng **`CLAMP_LOWER[7:0]`**.
- `lower_bound = $signed({56'd0, CLAMP_LOWER})` (bỏ biểu thức `LAYER_RELU_ENABLE ? ... : 0`). Phần còn lại giữ nguyên.

**`COMPUTE_CORE.v`**
- `weight_ram` `.DEPTH(288)` → **`512`** (BWD1 cần 432 byte/PE).
- Cổng vào: bỏ `LAYER_RELU_ENABLE`, thêm `CLAMP_LOWER[7:0]` (nối thẳng vào `REQUANTIZER`); thêm `POOL_POSITION_INDEX[1:0]`.
- Cổng ra mới `POOL_ARGMAX_CODE[15:0]` (kênh k ở `[2*k +: 2]`).
- Thêm `pool_position_delayed <= POOL_POSITION_INDEX` cạnh các `*_delayed` có sẵn (cùng 1 nhịp trễ như cờ pool).
- Trong vòng `for pe_loop`, cạnh khối max-pool:
  - `pool_first_position_delayed` → `code[k] <= pool_position_delayed`.
  - `pool_next_position_delayed` và `accumulator > max_pool_register` (đúng điều kiện `>` nghiêm ngặt đang có) → `code[k] <= pool_position_delayed`.
- Quy ước mã: `pool_position = (row%2)*2 + (col%2)`, đúng như `pool_position` trong `CONV_CONTROLLER`.

**`CONV_CONTROLLER.v`**
- `LAYER_OFFSET` `[14:0]` → `[15:0]`.
- Thêm cổng ra `POOL_POSITION_INDEX[1:0]`: thêm `issue_pool_position <= pool_position` trong `STATE_POOL_ISSUE`, đi qua chuỗi `stage1_`, `stage2_` như `issue_pool_first`, chọn bằng `use_lut_pipeline` y hệt `POOL_FIRST_POSITION`. Chỉ cần đúng ở chu kỳ có cờ pool.
- Thêm cổng ra `MASK_READ_ADDR[13:0] = output_base_addr + REQUANT_CHANNEL_IN_GROUP` (cùng biểu thức đang gán vào `OUTPUT_WRITE_ADDR`, nhưng là wire, chưa qua thanh ghi). Dùng để đọc `FEATURE_RAM_B` làm mask lớp 6: RAM trễ 1 chu kỳ nên dữ liệu trùng đúng chu kỳ `OUTPUT_WRITE_DATA`.
- **Không đổi vòng lặp**: lớp 6, 7 chạy y hệt (không pool, 8x8). Không cần thêm cờ `IS_BACKWARD`/`APPLY_RELU4_MASK` vào controller, `CNN_RUNNER` tự suy ra từ `active_layer_index` (mục 2.3).

**`PARAMETER_FETCHER.v`**

(a) **Mở rộng địa chỉ** (nền CNN2 giờ là `0x10000`):
- `cnn_base_byte_addr` → `[16:0]`: `CNN_SELECT ? 17'h10000 : 17'h0`.
- `stream_word_addr` → `[15:0]`: `CNN_SELECT ? 16'h8000 : 16'h0`. `LUT_WORD_OFFSET` thành `[15:0]`.
- Cổng `LAYER_OFFSET` → `[15:0]`; `group_start_byte_addr` → `[16:0]`; chỗ gán `stream_word_addr <= group_start_byte_addr[16:1]`; chỗ LUT dùng `cnn_base_byte_addr[16:1] + LUT_WORD_OFFSET`.
- `SDRAM_REQUEST_WORD_ADDR <= {8'd0, stream_word_addr}`. Sửa comment `CNN_SELECT`.

(b) **`NORMAL_CLASS_INDEX`**: thêm cổng ra `[7:0]`; trong `MODE_HEADER` thêm `6'd7: NORMAL_CLASS_INDEX <= delivered_byte;` (cạnh `6'd5` của Unknown).

(c) **Đọc bảng LCAM (24 byte ở `+0x8000`)**: thêm `MODE_LCAM_TABLE`. `fetch_mode` rộng ra `[2:0]` (sẽ có 5 chế độ).
- Trong `STATE_FINISH_STREAM`: `MODE_LUT` không kết thúc nữa mà chuyển sang `MODE_LCAM_TABLE`: `stream_word_addr <= cnn_base_byte_addr[16:1] + 16'h4000`, `stream_total_bytes <= 24`, rồi `STATE_REQUEST_WORD`. `MODE_LCAM_TABLE` → `STATE_FINISH_PARAMETERS`.
- Phân tích byte (trong `STATE_DELIVER_BYTE`): `stream_byte_index < 16` thì dùng **đúng khối `case (field_index)` của header** nhưng slot = `{2'b11, stream_byte_index[3]}` (byte 0..7 → slot 6, 8..15 → slot 7). Mảng `layer_*_register[0:7]` đã có đủ 8 slot nên không phải mở rộng.
- `stream_byte_index` 16..23 (mục CAM): field 0..2 → `m0_assemble`; field 3 → `CAM_M0 <= {delivered_byte[6:0], m0_assemble}`; field 4 → `CAM_SHIFT <= delivered_byte[5:0]`. Thêm 2 cổng ra `CAM_M0[30:0]`, `CAM_SHIFT[5:0]`.

(d) **Nạp hàng FC**: thêm cổng vào `START_FETCH_CLASS_ROW`, `LAYERCAM_CLASS_INDEX[2:0]`; cổng ra `DONE_FETCH_CLASS_ROW`, `CLASS_ROW_WRITE_ENABLE`, `CLASS_ROW_WRITE_ADDR[5:0]`, `CLASS_ROW_WRITE_DATA[7:0]`.
- `MODE_CLASS_ROW` mới. Trong `STATE_IDLE` thêm nhánh `else if (START_FETCH_CLASS_ROW)`: byte địa chỉ = `cnn_base_byte_addr + 17'h7CD0 + cls*48` (`0x7CB0 + 32`; `cls*48 = (cls<<5) + (cls<<4)`), `stream_word_addr` = byte/2, `stream_total_bytes = 48`.
- `STATE_DELIVER_BYTE`: `CLASS_ROW_WRITE_ENABLE <= 1`, `ADDR <= stream_byte_index[5:0]`, `DATA <= delivered_byte`.
- `STATE_FINISH_STREAM`: `MODE_CLASS_ROW` → trạng thái mới `STATE_FINISH_CLASS_ROW = 3'd7` → xung `DONE_FETCH_CLASS_ROW` (giống `STATE_FINISH_WEIGHTS`). Nhớ mặc định hạ `CLASS_ROW_WRITE_ENABLE` và `DONE_FETCH_CLASS_ROW` mỗi chu kỳ.
- `cnn_base_byte_addr` giữ nguyên từ lần `START_FETCH_PARAMETERS` cuối (chính là CNN quyết định) nên **không cần** cổng `CNN_SELECT` mới.

**`HOST_INTERFACE.v`**
- Thêm cổng: vào `FINAL_CAM_VALID`, `CAM_READ_DATA[7:0]`; ra `CAM_READ_ADDR[7:0]`.
- `tx_byte_index`, `tx_byte_count` `[3:0]` → **`[8:0]`**. Thêm `saved_final_cam_valid` chốt cùng chỗ với `saved_final_source` khi `START_SEND_RESULT`.
- Khung kết quả: `tx_byte_count <= saved cam ? 270 : 14` (tính từ `FINAL_CAM_VALID` tại thời điểm chốt). Thêm `4'd13: tx_byte_combinational = {7'd0, saved_final_cam_valid}`; `default` với `tx_byte_index >= 14` → `CAM_READ_DATA`. Sửa comment 13 → 14 byte.
- `CAM_READ_ADDR = tx_byte_index[7:0] - 8'd14`.
- **Độ trễ RAM**: `CAM_RAM` đọc trễ 1 chu kỳ, còn `SEND_IDLE` lấy `TX_BYTE` ngay sau khi `tx_byte_index` đổi. Thêm trạng thái `SEND_WAIT_RAM` (1 chu kỳ) giữa `SEND_BUSY_LOW` và `SEND_IDLE`, nếu không byte CAM bị lệch 1 vị trí.

### 2.2 `REQUANTIZER_SERIAL.v` (mới)

Sao `REQUANTIZER.v`, bỏ vòng kênh:
- Cổng: `CLK`, `RST`, `START_REQUANT`, `ACCUMULATOR_INPUT[31:0]` (signed), `LAYER_MULTIPLIER[30:0]`, `LAYER_SHIFT[5:0]`, `REQUANT_BYTE[7:0]`, `DONE_REQUANT` (xung 1 chu kỳ, cùng lúc `REQUANT_BYTE` hợp lệ).
- Cố định `zp_out = 0`, `lower = 0` (kẹp `[0,255]`). Bỏ `requant_channel`; `STATE_CLAMP` đi thẳng `STATE_DONE`. Công thức (nhân shift-add 31 bước, cộng làm tròn `1 <<< (SHIFT-1)`, dịch số học) giữ nguyên.
- Giữ `ACCUMULATOR_INPUT`, `M0`, `SHIFT` ổn định từ `START_REQUANT` đến `DONE_REQUANT`.

### 2.3 `CNN_RUNNER.v` (sửa nhiều nhất)

**Cổng mới**: vào `START_LAYERCAM`, `LAYERCAM_CLASS_INDEX[2:0]`, `CAM_READ_ADDR[7:0]`; ra `DONE_LAYERCAM`, `NORMAL_CLASS_INDEX[7:0]` (lấy từ fetcher), `CAM_READ_DATA[7:0]`.

**A. Chọn lớp hiệu dụng.** `wire [2:0] active_layer_index = lcam_active ? lcam_layer_index : layer_index;`
Dùng `active_layer_index` thay `layer_index` ở: `.LAYER_INDEX` của `PARAMETER_FETCHER` và `CONV_CONTROLLER`, các `layer_dest_*`, `case` của `input_pixel_to_core`. FSM forward của runner giữ `layer_index` và đứng yên ở `STATE_IDLE` khi LayerCAM chạy (nó chỉ nhìn `done_layer` ở `STATE_LAYER_WAIT` nên không bị ảnh hưởng).
`start_layer` vào `CONV_CONTROLLER` = `start_layer | lcam_start_layer`. `done_layer` nối thêm vào `LAYERCAM_CONTROLLER`.

**B. Cận dưới requant.** `wire backward = (active_layer_index == 3'd6) || (active_layer_index == 3'd7);`
`clamp_lower = backward ? 8'd1 : (layer_relu_enable ? layer_output_zero_point : 8'd0);` nối vào `COMPUTE_CORE.CLAMP_LOWER`. Nối `POOL_POSITION_INDEX` từ `CONV_CONTROLLER` sang `COMPUTE_CORE`, lấy `POOL_ARGMAX_CODE`.

**C. 5 RAM mới** (`SIMPLE_DUAL_PORT_RAM`):

| RAM | DEPTH / ADDR | Ghi | Đọc |
|---|---|---|---|
| `gradient_ram_a` | 3072 / 12 | bước 1 của LCAM (dz5), hoặc `conv_write_enable && active==7` (addr `conv_write_addr[11:0]`) | bước CAM: addr từ controller; lớp 6: `conv_input_read_addr[11:0]` |
| `gradient_ram_b` | 2048 / 11 | `conv_write_enable && active==6`, addr `conv_write_addr[10:0]`, **data = `(feature_b_read_data == 8'd0) ? 8'd128 : conv_write_data`** | lớp 7: `conv_input_read_addr[10:0]` |
| `pool3_copy_ram` | 1536 / 11 | `conv_write_enable && active==2`, `conv_write_data` | addr từ controller |
| `pool3_argmax_ram` | 1536 / 11 | cùng điều kiện, data = `{6'd0, POOL_ARGMAX_CODE[2*conv_write_addr[2:0] +: 2]}` | addr từ controller (chung với `pool3_copy_ram`) |
| `cam_ram` | 256 / 8 | từ controller | `CAM_READ_ADDR` |

Kênh ghi = `conv_write_addr[2:0]` vì mọi `base` chia hết cho 8. Với lớp 2, `POOL_ARGMAX_CODE` vẫn ổn định lúc ghi vì bản gốc không chồng requant với MAC kế.

**D. Sửa RAM có sẵn**
- `feature_ram_b.READ_ADDR`: `(active_layer_index == 3'd6) ? mask_read_addr[12:0] : conv_input_read_addr[12:0]` (`mask_read_addr` từ `CONV_CONTROLLER.MASK_READ_ADDR`). Lớp 6 đọc nguồn từ `gradient_ram_a` nên cổng đọc B rảnh.
- `feature_ram_a.READ_ADDR`: thêm nhánh `lcam_dz5_read_addr` (khi controller ở bước 1), ưu tiên cùng chỗ với `gap_active`.
- `gap_ram`: ghi = `CLASS_ROW_WRITE_*` của fetcher **hoặc** `gap_write_*` hiện có (không trùng thời điểm). Đọc: bước 1 dùng `lcam_gap_read_addr`, còn lại giữ `conv_input_read_addr[5:0]`.
- `input_pixel_to_core`: thêm `3'd6: gradient_ram_a read data`, `3'd7: gradient_ram_b read data`.
- `layer_dest_*`: lớp 6, 7 không ghi vào feature A/B/logit (đã ngầm đúng vì `layer_dest_is_*` chỉ khớp 0..5; giữ như vậy).

**E.** Khởi tạo `LAYERCAM_CONTROLLER` và nối `START_FETCH_CLASS_ROW` / `DONE_FETCH_CLASS_ROW`, `CAM_M0`, `CAM_SHIFT`.

## 3. `LAYERCAM_CONTROLLER.v` (mới)

Ý tưởng: 1 FSM, dùng lại `CONV_CONTROLLER` cho BWD1/BWD2 bằng cách đổi `LCAM_LAYER_INDEX`.

**Cổng** (tên gợi ý): `START_LAYERCAM`, `DONE_LAYERCAM`; tới fetcher `START_FETCH_CLASS_ROW`, `DONE_FETCH_CLASS_ROW`; tới conv `LCAM_ACTIVE`, `LCAM_LAYER_INDEX[2:0]`, `LCAM_START_LAYER`, `DONE_LAYER`; bước 1 `DZ5_READ_ADDR[11:0]` (→ `feature_ram_a`), `FEATURE_A_READ_DATA`, `GAP_READ_ADDR[5:0]`, `GAP_READ_DATA`, `GRADIENT_WRITE_ENABLE/ADDR[11:0]/DATA`; bước CAM `CAM_PHASE_ACTIVE`, `CAM_SOURCE_READ_ADDR[10:0]` (cho `gradient_ram_a`, `pool3_copy_ram`, `pool3_argmax_ram`), 3 cổng data tương ứng, `CAM_WRITE_ENABLE/ADDR[7:0]/DATA`; `CAM_M0[30:0]`, `CAM_SHIFT[5:0]`.

**FSM** (`LCAM_ACTIVE = 1` từ `START_LAYERCAM` đến `DONE_LAYERCAM`):

1. `STATE_FETCH_CLASS_ROW`: xung `START_FETCH_CLASS_ROW`, chờ `DONE_FETCH_CLASS_ROW` (hàng FC vào `GAP_RAM`).
2. `STATE_BUILD_DZ5`: `i = 0..3071`. Phát `DZ5_READ_ADDR = i`, `GAP_READ_ADDR = i % 48` (dùng bộ đếm kênh 0..47, không dùng chia). Sau 1 chu kỳ (trễ RAM, trễ `i` cho khớp): `byte = (FEATURE_A_READ_DATA > 0) ? (GAP_READ_DATA ^ 8'h80) : 8'h80` rồi ghi `gradient_ram_a[i]`. (`w ^ 0x80` = `w + 128` cho int8.) Xả pipeline 1 chu kỳ cuối.
3. `STATE_RUN_BWD1`: `LCAM_LAYER_INDEX = 6`, xung `LCAM_START_LAYER`, chờ `DONE_LAYER`.
4. `STATE_RUN_BWD2`: như trên với `7`. Sau bước này `gradient_ram_a[0..1535]` = g4 (dạng `g+128`).
5. `STATE_CAM`: lặp `r` 0..15, `c` 0..15, `ch` 0..23.
   - Địa chỉ đọc chung cho 3 RAM: `(((r>>1)*8 + (c>>1))*24 + ch)`.
   - Sau 1 chu kỳ trễ: nếu `argmax_data == {r[0], c[0]}` và `g_byte > 8'd128` thì `cam_acc += (g_byte - 128) * pool3_data`; ngược lại +0. (ReLU của `g` với `g` lưu `+128`.) `cam_acc` 32 bit, tối đa ~7,8e5.
   - Sau `ch = 23` và xả 1 chu kỳ: nạp `cam_acc` vào `REQUANTIZER_SERIAL` (`CAM_M0`, `CAM_SHIFT`), chờ `DONE_REQUANT`, ghi `cam_ram[r*16 + c] = REQUANT_BYTE`, xoá `cam_acc`.
6. Hết 256 điểm: xung `DONE_LAYERCAM`.

`REQUANTIZER_SERIAL` đặt **bên trong** `LAYERCAM_CONTROLLER`.

## 4. `TOP_CNN.v`

- Thêm `STATE_CHECK_LAYERCAM = 4'd11`, `STATE_RUN_LAYERCAM = 4'd12` (vẫn vừa 4 bit).
- Chốt thêm `normal_class_index` của từng CNN cạnh `cnn?_unknown_class_index` (tại `done_cnn`): `cnn1_normal_class_index`, `cnn2_normal_class_index` (8 bit, nối từ `CNN_RUNNER.NORMAL_CLASS_INDEX`).
- **Đổi đích nhảy**: ở `STATE_CHECK_CNN1` (nhánh không Unknown) và `STATE_CHECK_CNN2` (cả 2 nhánh): **bỏ** `start_send_result <= 1` + `state <= STATE_SEND_RESULT`, thay bằng `state <= STATE_CHECK_LAYERCAM`.
- `STATE_CHECK_LAYERCAM`:
  - `decided_normal = (final_source == 8'h01) ? cnn1_normal_class_index : cnn2_normal_class_index`.
  - Nếu `final_source != 8'h00 && final_class != decided_normal`: `start_layercam <= 1`, `layercam_class_index <= final_class[2:0]`, `state <= STATE_RUN_LAYERCAM`.
  - Ngược lại: `final_cam_valid <= 0`, `start_send_result <= 1`, `state <= STATE_SEND_RESULT`.
- `STATE_RUN_LAYERCAM`: khi `done_layercam`: `final_cam_valid <= 1`, `start_send_result <= 1`, `state <= STATE_SEND_RESULT`. (`final_cam_valid` phải gán **cùng cạnh** với `start_send_result` vì `HOST_INTERFACE` chốt ngay.)
- **Không đổi `cnn_select`** sau lần chạy cuối (nó chọn nền SD của CNN quyết định cho LayerCAM). Xoá `final_cam_valid` khi `done_image_receive`.
- `start_layercam` là xung 1 chu kỳ, mặc định hạ xuống 0 mỗi chu kỳ như `start_cnn`.
- Nối `CNN_RUNNER` (cổng mới mục 2.3) và `HOST_INTERFACE` (`FINAL_CAM_VALID`, `CAM_READ_ADDR`, `CAM_READ_DATA`, dây `cam_read_addr/data`).
- `cnn_running` thêm 2 trạng thái mới (LED[1] sáng khi chạy CAM). Sửa comment đầu file.

## 5. Bẫy cần nhớ

- Gradient trong RAM là `g + 128`; ô bị mask phải ghi **128** (không phải 0). `ZP_IN = 128`, cận dưới requant lớp ngược = **1**.
- Mask ReLU chỉ ở BWD1, theo `FEATURE_RAM_B` (A4). BWD2 **không** mask.
- `FEATURE_RAM_A` lúc LayerCAM giữ đầu ra conv5 (A5), `FEATURE_RAM_B` giữ conv4 (A4); đầu ra conv3 đã bị ghi đè nên mới cần `pool3_*`.
- `POOL3_*` được ghi ở **mọi ảnh, cả 2 CNN**; CNN chạy sau cùng luôn là CNN quyết định, nên **không chạy lại forward**.
- Hàng FC lấy theo `FINAL_CLASS` của CNN quyết định, không mặc định CNN1.
- `LAYER_OFFSET`, nền SD và `stream_word_addr` phải đủ bit (mục 2.1), quên là CNN2 đọc rác.
- Pool khi hòa giữ vị trí đầu (`>` nghiêm ngặt); mã vị trí so theo **acc**, trước requant.
- Sau mỗi bước sửa phải chạy lại mô phỏng forward 5 lớp: kết quả forward **không được đổi**.
- Pass LCAM khi khớp 100% với `hw_lcam/.../vectors/` (so `(giá_trị_RAM - 128) & 0xFF` với file `.mem`).

## 6. Python (sửa trước khi làm RTL, vì sinh golden và ảnh SD)

### 6.1 `cam_best_cnn_v3.py` (đã sửa 13.1, đã kiểm: ĐẠT)

Bản `cam_best_cnn_v3_1.py` đúng yêu cầu: `idx` của pool lấy từ `acc` (`F.max_pool2d(acc.double(), ...)`), giá trị pool = requant của max acc, `relu{i}` vẫn là đầu ra trước pool, `pool{i}` và `pidx{i}` được giữ trong `trim_trace` (đủ cho `_chain` và `export_hw_vector`). `--calib-n` mặc định 128, thêm `assert |acc| < 2^30` (khớp giả thiết RTL). Cú pháp chạy được, JSON/`.vh` mới nhất quán với nhau và `M0 < 2^31`, `SHIFT` trong `[1, 63]`.

Việc còn lại:
- Đổi tên file thành `cam_best_cnn_v3.py` (các script khác `import cam_best_cnn_v3`).
- `F.max_pool2d` trên GPU không bảo đảm chọn phần tử đầu khi hòa. Sinh tham số/vector bằng `--device cpu`, hoặc kiểm bằng `argmax` numpy trong `make_test_images.py` (6.3): hai bên phải cho cùng `pidx3`.
- Hàm `make_input_from_small` chưa được gọi ở đâu (vô hại), và dòng tính `A_n` trong `_chain` hơi rối nhưng cho kết quả đúng (`relu5` vì conv5 không pool).

### 6.2 `pack_sd_image.py`

- `CNN_BLOCK_BYTES = 0x8000` → **`0x10000`**; `assert len(image) == 2 * CNN_BLOCK_BYTES` giữ nguyên (ra 131.072 byte). Sửa docstring (2 khối 64 KB, thêm vùng LCAM).
- Hằng mới: `BWD_LAYERS = [(48, 32), (32, 24)]` (vào, ra), `BWD_OFFSETS = [0x8100, 0xB780]`, `BWD_GROUP_BYTES = [3488, 2336]`, `BWD_GROUP_COUNTS = [4, 3]`, `LCAM_TABLE_OFFSET = 0x8000`.
- Thêm tham số `--lcam-root` (mặc định `cam_out`). Thư mục LCAM: `<lcam-root>/A/hw_lcam/Hospital_A/LayerCAM_B3` và `.../B/hw_lcam/Hospital_B/LayerCAM_B3`. Truyền vào `pack_cnn(hospital_dir, cnn_id, unknown_override, lcam_dir)`.
- `get_unknown_index`: tách thêm hàm lấy `normal_index` từ `classes` (không phân biệt hoa thường; không có `Normal` thì `0xFF`). Ghi `block[7] = normal_index` (hiện đang `0`).
- Trong `pack_cnn`, sau vòng lớp forward:
  1. `assert` `zp_out == 0` cho `params[2]`, `params[3]`, `params[4]` (conv3, conv4, conv5).
  2. Đọc `lcam_params.json`; lấy `steps` theo `conv_index` (5 → lớp 6, 4 → lớp 7) và `cam`. Kiểm `0 <= M0 < 2**31`, `1 <= shift <= 63`, và `weight_shape_out_in_kh_kw` khớp `BWD_LAYERS` (`[32,48,3,3]`, `[24,32,3,3]`); `fc.shape == [5,48]`.
  3. Bảng `0x8000`: `struct.pack_into("<IBBBB", block, 0x8000, m0_6, sh_6, 128, 128, 0)`, tại `0x8008` tương tự cho lớp 7, tại `0x8010` `struct.pack_into("<IB", block, 0x8010, cam_m0, cam_shift)` (3 byte sau để 0).
  4. Trọng số BWD: đọc `lcam_w5_bwd_int8.mem`, `lcam_w4_bwd_int8.mem` bằng `read_mem_tokens` + `to_signed(v, 8)`; số phần tử phải bằng `out*in*9`. Đóng gói **đúng khuôn vòng forward** nhưng bias toàn 0 (32 byte 0 mỗi nhóm), PE `p` của nhóm `g` lấy `weights[(g*8+p)*in*9 : +in*9]` (layout file đã là `((o*Cin+i)*3+ky)*3+kx` nên không hoán vị). Tái dùng bằng cách tách phần ghi nhóm của vòng forward thành hàm `write_layer(block, offset, groups, in_ch, kernel_area, weights, biases)`. `assert` điểm cuối = offset + groups*group_bytes (`0xD2E0` cho lớp 7).
- Hàng FC cho LayerCAM **không ghi thêm** (nằm sẵn trong khối FC `0x7CB0 + 32 + cls*48`).

### 6.3 `make_test_images.py` (golden của cascade và CAM)

- `CNN_BLOCK_BYTES = 0x10000`.
- `CnnBlock.__init__`: thêm `self.normal_class_index = block[7]`; đọc bảng `0x8000` (lớp 6, 7: `M0, SHIFT`; CAM: `<IB`), và `self.bwd_weights[k]` shape `(out, in, 3, 3)` từ `BWD_OFFSETS` theo đúng cách đọc khối nhóm forward (nên tái dùng một hàm `read_layer_groups`).
- `run_cnn`: với lớp pool, ngoài `max_pool_2x2` thêm mã vị trí thắng, tính **trên acc**: `acc.reshape(H//2,2,W//2,2,C).transpose(0,2,4,1,3).reshape(H//2,W//2,C,4).argmax(-1)` (`argmax` numpy lấy phần tử đầu khi hòa; thứ tự 4 phần tử là `r*2+c`). Lưu `layers["pidx3"]` cho lớp 3 (conv3), giữ `layers["pool3"]`, `conv4`, `conv5` (đã có).
- Hàm mới `run_layercam(layers, cnn, cls)`:
  ```
  dz5 = (A5 > 0) * Wfc[cls]                         # (8,8,48) int
  g5  = clip(requant_signed(conv3x3_accumulate(dz5 + 128, bwd_w[0], 0, 128), M0_6, S_6), -127, 127)
  dz4 = (A4 > 0) * g5
  g4  = clip(requant_signed(conv3x3_accumulate(dz4 + 128, bwd_w[1], 0, 128), M0_7, S_7), -127, 127)
  code_up = repeat(repeat(pidx3, 2, 0), 2, 1); expect = (r%2)*2 + (c%2) lưới 16x16
  cam_acc = sum_ch( (code_up == expect[..., None]) * repeat2(max(g4,0) * pool3) )   # int64
  cam_u8  = clip((cam_acc*CAM_M0 + (1 << (CAM_S-1))) >> CAM_S, 0, 255)
  ```
  `requant_signed` = `(acc*M0 + (1<<(S-1))) >> S` (không cộng zp, vì `+128` rồi kẹp `[1,255]` tương đương kẹp `[-127,127]`). Dùng bias 0, `zp_in = 128` để `x - zp_in = g`.
- Vòng ảnh: sau khi có `source, final_class`, chọn CNN quyết định (`cnn1` hoặc `cnn2`, **và layers tương ứng**). `cam_flag = source != 0 and final_class != cnn_quyet_dinh.normal_class_index`. Nếu `cam_flag`: chạy `run_layercam`, ghi `ref_img{i}_cam_u8_{cnn}.mem` (và tùy chọn `dz5`, `dz4`, `g4` dạng `v & 0xFF`, `pidx3`).
- `expected_cascade.json` thêm: `cam_flag`, `cam_u8` (256 số, thứ tự hàng), `source_path` (đường dẫn ảnh gốc, để `host_cascade.py` vẽ).
- Đối chiếu: `cam_u8` của script này phải bằng `*_layercam_int8_u8_16x16.npy` / `hw_lcam/.../vectors/*_cam_u8.mem` của `cam_best_cnn_v3.py`. Lệch là sai ở 6.1 hoặc ở hàm trên.

### 6.4 `host_cascade.py`

- `RESULT_LENGTH = 13` → `14`; thêm `CAM_BYTES = 256`.
- `run_image`: đọc 14 byte; `cam_flag = result[13]` (chỉ nhận 0 hoặc 1, khác thì báo lỗi). Nếu `cam_flag == 1` đọc thêm `read_exact(port, 256, result_timeout)`. Trả thêm `"cam_flag"` và `"cam_u8"` (list 256).
- So golden: thêm `cam_flag` và `cam_u8` vào tuple khóa so sánh; in lệch bằng số phần tử khác nhau thay vì in cả 256 số.
- Thêm cờ `--show` (và `--save-dir`). Với mỗi khung `cam_flag == 1` và có `source_path`:
  ```python
  from cam_best_cnn_v3 import load_raw, fov_mask, hw_heat, draw_panel, heat_to_boxes
  raw  = load_raw(path)
  mask = fov_mask(raw)                                   # 64x64 bool
  heat = hw_heat(torch.from_numpy(cam_u8).reshape(16,16), torch.from_numpy(mask)).numpy()
  H0, W0 = raw.shape[:2]
  crop = raw[:(H0//64)*64, :(W0//64)*64]                 # phần dư dưới/phải bị bỏ, giống lúc resize
  boxes = heat_to_boxes(heat, mask, 0.15, 1, crop.shape[1], crop.shape[0])
  fig, ax = plt.subplots(); draw_panel(ax, crop, heat, boxes, title, mask)
  ```
  Import `torch`, `matplotlib` bên trong nhánh `--show` để chạy không cần chúng khi chỉ kiểm khung. Sửa docstring (khung 14 hoặc 270 byte).
- Cập nhật `--mem` (không có golden) cũng in `cam_flag`.

### 6.5 Thứ tự chạy Python

1. (Xong) 6.1 và JSON mới. Nếu chạy lại `--export-hw`, dùng `--device cpu`.
2. `pack_sd_image.py --root ... --lcam-root cam_out` ra `sd_image.bin/.hex` (131.072 byte).
3. `make_test_images.py` ra golden (logit + CAM) → dùng cho mô phỏng và `host_cascade.py`.
4. Đối chiếu bảng `0x8000` trong `sd_image.bin` với bảng hằng số ở mục 0.
