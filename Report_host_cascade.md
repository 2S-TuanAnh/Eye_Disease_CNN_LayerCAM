# host_cascade.py

## Mục đích
Chương trình phía PC để kiểm tra FPGA chạy cascade 2 CNN phân loại bệnh mắt. Chương trình gửi ảnh xuống board, nhận kết quả, đối chiếu với golden và vẽ bản đồ LayerCAM để xem mô hình chú ý vào đâu.

## Làm gì
1. **Gửi ảnh qua UART** (mặc định 1 Mbaud, 8N1): chờ byte sẵn sàng `0x55`, gửi `0xA5` rồi 12288 byte ảnh 64×64 RGB.
2. **Nhận khung kết quả** 14 byte (270 byte nếu `CAM_FLAG = 1`): CNN quyết định, lớp dự đoán, 5 logit của mỗi CNN và 256 byte `cam_u8` (lưới 16×16).
3. **So với golden** trong `expected_cascade.json`: in `OK` hoặc `KHAC GOLDEN`, kèm chi tiết phần bị lệch.
4. **Vẽ LayerCAM** (khi dùng `--show` hoặc `--save-dir`) thành một hình gồm hai panel: trái là ảnh gốc, phải là ảnh gốc phủ heatmap, không có khung vuông.

## Thuật toán vẽ LayerCAM
1. Cắt ảnh gốc cho chia hết 64, giống bước resize trên phần cứng.
2. Tạo heat 64×64 bằng `hw_heat` của `cam_best_cnn_v3.py` (mask vùng mắt erode 2). Cách chuẩn hóa và màu `jet` (alpha 0,45) giống code gốc.
3. **Phủ sát viền:** phần rìa bị erode được lấp bằng giá trị gần nhất bên trong, rồi phóng to bilinear lên độ phân giải ảnh gốc.
4. **Lọc nhiễu:**
   - Gom các cụm heat liền nhau từ `cam_low` trở lên.
   - Cụm có đỉnh từ `cam_high` trở lên (vùng vàng–cam–đỏ) được giữ nguyên.
   - Mọi vùng còn lại được đưa về một mức nền tím phẳng (phân vị 30% của các giá trị thấp).
   - Quanh cụm giữ lại, màu chuyển dần sang nền (smoothstep theo khoảng cách, rộng 4% kích thước ảnh, rồi làm mờ nhẹ) nên không có đường viền.
5. Che phần nền đen ngoài vùng mắt bằng mask tính ở độ phân giải gốc, nên mép mượt.

## Tham số chính
| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--port` | bắt buộc | Cổng COM/tty của FPGA |
| `--baud` | 1000000 | Tốc độ UART |
| `--expected` | | File golden `expected_cascade.json` |
| `--mem` | | Thêm file `imageN_raw64.mem` (không có golden) |
| `--skip-ready` | tắt | Bỏ qua bước chờ `0x55` |
| `--show` | tắt | Hiện cửa sổ hình LayerCAM |
| `--save-dir` | | Lưu `<tên ảnh>_cam.png` vào thư mục này |
| `--cam-low` | 0,25 | Ngưỡng viền cụm CAM |
| `--cam-high` | 0,60 | Cụm phải có đỉnh ≥ ngưỡng này mới được giữ |
| `--module-dir` | `.` | Thư mục chứa `cam_best_cnn_v3.py` |

Chỉnh nhanh khi cần:
- Tăng `--cam-high` thì chỉ còn lõi mạnh hơn.
- Tăng `--cam-low` thì halo quanh lõi mỏng hơn.

## Ví dụ
```
python host_cascade.py --port COM4 --expected test_images/expected_cascade.json --show --save-dir cam_png --skip-ready
```

## Yêu cầu
`pyserial`, `numpy`, `scipy`, `torch`, `matplotlib` (Qt5Agg cần `PyQt5` để hiện cửa sổ), và `cam_best_cnn_v3.py` cùng thư mục. Hình LayerCAM chỉ vẽ cho ảnh có `source_path` trong golden và `CAM_FLAG = 1`.

## Ghi chú
- Phần so khớp golden và giao thức UART không đổi. Chỉ phần hiển thị LayerCAM được sửa.
- `cam_best_cnn_v3.py` gọi `matplotlib.use("Agg")` khi import, nên `show_cam` đặt lại backend ban đầu (Qt5Agg) để cửa sổ hiện được.
