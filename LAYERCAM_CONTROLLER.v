// Dieu khien 4 buoc LayerCAM (chay sau khi CNN quyet dinh da xong forward; muc 3 huong dan).
//
//   1. STATE_FETCH_*   : nap hang FC cua lop du doan (48 byte int8) vao GAP_RAM (qua PARAMETER_FETCHER).
//   2. STATE_BUILD_DZ5 : dz5[i] = (A5[i] > 0) ? w[i % 48] : 0, luu dang (g + 128) = w ^ 0x80; o bi mask = 128.
//                        Ghi vao gradient_ram_a[0..3071].
//   3. STATE_BWD1/BWD2 : dung lai CONV_CONTROLLER voi LCAM_LAYER_INDEX = 6 roi 7 (khong pool, 8x8).
//                        Sau BWD2, gradient_ram_a[0..1535] = g4 (dang g + 128).
//   4. STATE_CAM_*     : voi moi diem (r, c) cua luoi 16x16:
//                          acc = sum_ch [ argmax3(r>>1,c>>1,ch) == {r[0],c[0]} && g4 > 128 ] * (g4 - 128) * pool3
//                        roi requant 1 kenh (REQUANTIZER_SERIAL, CAM_M0 / CAM_SHIFT) -> cam_ram[r*16 + c].
//
// LCAM_ACTIVE = 1 tu luc nhan START_LAYERCAM den luc xung DONE_LAYERCAM.
// DZ5_PHASE_ACTIVE = 1 trong buoc 2 (CNN_RUNNER dua DZ5_READ_ADDR vao feature_ram_a, GAP_READ_ADDR vao gap_ram).
// CAM_PHASE_ACTIVE = 1 trong buoc 4 (CNN_RUNNER dua CAM_SOURCE_READ_ADDR vao gradient_ram_a, pool3_copy_ram,
//                    pool3_argmax_ram; ca 3 RAM tre 1 chu ky).
module LAYERCAM_CONTROLLER (
    input  wire        CLK,
    input  wire        RST,

    input  wire        START_LAYERCAM,
    output reg         DONE_LAYERCAM,

    // ---- PARAMETER_FETCHER: nap hang FC ----
    output reg         START_FETCH_CLASS_ROW,
    input  wire        DONE_FETCH_CLASS_ROW,

    // ---- CONV_CONTROLLER (qua CNN_RUNNER) ----
    output reg         LCAM_ACTIVE,
    output reg  [2:0]  LCAM_LAYER_INDEX,
    output reg         LCAM_START_LAYER,
    input  wire        DONE_LAYER,

    // ---- Buoc 1-2: tao dz5 ----
    output reg         DZ5_PHASE_ACTIVE,
    output reg  [11:0] DZ5_READ_ADDR,                 // -> feature_ram_a (A5)
    input  wire [7:0]  FEATURE_A_READ_DATA,
    output reg  [5:0]  GAP_READ_ADDR,                 // -> gap_ram (hang FC)
    input  wire [7:0]  GAP_READ_DATA,
    output reg         GRADIENT_WRITE_ENABLE,         // -> gradient_ram_a
    output reg  [11:0] GRADIENT_WRITE_ADDR,
    output reg  [7:0]  GRADIENT_WRITE_DATA,

    // ---- Buoc 4: CAM ----
    output reg         CAM_PHASE_ACTIVE,
    output reg  [10:0] CAM_SOURCE_READ_ADDR,          // chung cho gradient_ram_a, pool3_copy_ram, pool3_argmax_ram
    input  wire [7:0]  GRADIENT_READ_DATA,            // g4 (dang g + 128)
    input  wire [7:0]  POOL3_READ_DATA,
    input  wire [7:0]  POOL3_ARGMAX_READ_DATA,        // [1:0] = ma vi tri thang
    output reg         CAM_WRITE_ENABLE,
    output reg  [7:0]  CAM_WRITE_ADDR,
    output reg  [7:0]  CAM_WRITE_DATA,
    input  wire [30:0] CAM_M0,
    input  wire [5:0]  CAM_SHIFT
);
    // ------------------------------------------------------------
    // Trang thai
    // ------------------------------------------------------------
    localparam STATE_IDLE               = 4'd0;
    localparam STATE_FETCH_ROW_START    = 4'd1;
    localparam STATE_FETCH_ROW_WAIT     = 4'd2;
    localparam STATE_BUILD_DZ5          = 4'd3;
    localparam STATE_BUILD_DRAIN        = 4'd4;
    localparam STATE_BWD1_START         = 4'd5;
    localparam STATE_BWD1_WAIT          = 4'd6;
    localparam STATE_BWD2_START         = 4'd7;
    localparam STATE_BWD2_WAIT          = 4'd8;
    localparam STATE_CAM_ISSUE          = 4'd9;
    localparam STATE_CAM_FLUSH          = 4'd10;
    localparam STATE_CAM_REQUANT_START  = 4'd11;
    localparam STATE_CAM_REQUANT_WAIT   = 4'd12;
    localparam STATE_CAM_NEXT           = 4'd13;
    localparam STATE_FINISH             = 4'd14;

    reg [3:0]  state;

    // Buoc 2
    reg [11:0] build_index;
    reg [5:0]  build_channel;           // build_index % 48 (bo dem, khong dung phep chia)
    reg        build_valid_1;
    reg        build_valid_2;
    reg [11:0] build_index_1;
    reg [11:0] build_index_2;
    reg [2:0]  drain_counter;

    // Buoc 4
    reg [3:0]  cam_row;
    reg [3:0]  cam_col;
    reg [4:0]  cam_channel;
    reg        cam_valid_1;
    reg        cam_valid_2;
    reg [1:0]  cam_expect_1;
    reg [1:0]  cam_expect_2;
    reg [2:0]  flush_counter;
    reg [31:0] cam_acc;
    reg        start_requant;

    wire [7:0] requant_byte;
    wire       done_requant;

    // dia chi doc chung = (((r>>1)*8 + (c>>1)) * 24 + ch), pixel*24 = pixel*16 + pixel*8
    wire [5:0]  cam_pooled_pixel = {cam_row[3:1], cam_col[3:1]};
    wire [10:0] cam_pixel_base   = ({5'd0, cam_pooled_pixel} << 4) + ({5'd0, cam_pooled_pixel} << 3);
    wire [10:0] cam_read_addr    = cam_pixel_base + {6'd0, cam_channel};

    // g luu (g + 128): ReLU(g) = g_byte - 128 khi g_byte > 128. Tich <= 127 * 255.
    wire [7:0]  gradient_positive = GRADIENT_READ_DATA - 8'd128;
    wire [15:0] cam_product       = gradient_positive * POOL3_READ_DATA;
    wire        cam_hit           = cam_valid_2 &&
                                    (POOL3_ARGMAX_READ_DATA[1:0] == cam_expect_2) &&
                                    (GRADIENT_READ_DATA > 8'd128);

    REQUANTIZER_SERIAL requantizer_serial_inst (
        .CLK               (CLK),
        .RST               (RST),
        .START_REQUANT     (start_requant),
        .ACCUMULATOR_INPUT (cam_acc),
        .LAYER_MULTIPLIER  (CAM_M0),
        .LAYER_SHIFT       (CAM_SHIFT),
        .REQUANT_BYTE      (requant_byte),
        .DONE_REQUANT      (done_requant)
    );

    always @(posedge CLK) begin
        if (RST) begin
            state                 <= STATE_IDLE;
            DONE_LAYERCAM         <= 1'b0;
            START_FETCH_CLASS_ROW <= 1'b0;
            LCAM_ACTIVE           <= 1'b0;
            LCAM_LAYER_INDEX      <= 3'd6;
            LCAM_START_LAYER      <= 1'b0;
            DZ5_PHASE_ACTIVE      <= 1'b0;
            DZ5_READ_ADDR         <= 12'd0;
            GAP_READ_ADDR         <= 6'd0;
            GRADIENT_WRITE_ENABLE <= 1'b0;
            GRADIENT_WRITE_ADDR   <= 12'd0;
            GRADIENT_WRITE_DATA   <= 8'd0;
            CAM_PHASE_ACTIVE      <= 1'b0;
            CAM_SOURCE_READ_ADDR  <= 11'd0;
            CAM_WRITE_ENABLE      <= 1'b0;
            CAM_WRITE_ADDR        <= 8'd0;
            CAM_WRITE_DATA        <= 8'd0;
            build_index           <= 12'd0;
            build_channel         <= 6'd0;
            build_valid_1         <= 1'b0;
            build_valid_2         <= 1'b0;
            build_index_1         <= 12'd0;
            build_index_2         <= 12'd0;
            drain_counter         <= 3'd0;
            cam_row               <= 4'd0;
            cam_col               <= 4'd0;
            cam_channel           <= 5'd0;
            cam_valid_1           <= 1'b0;
            cam_valid_2           <= 1'b0;
            cam_expect_1          <= 2'd0;
            cam_expect_2          <= 2'd0;
            flush_counter         <= 3'd0;
            cam_acc               <= 32'd0;
            start_requant         <= 1'b0;
        end else begin
            // ---- xung mac dinh ----
            DONE_LAYERCAM         <= 1'b0;
            START_FETCH_CLASS_ROW <= 1'b0;
            LCAM_START_LAYER      <= 1'b0;
            CAM_WRITE_ENABLE      <= 1'b0;
            start_requant         <= 1'b0;
            cam_valid_1           <= 1'b0;

            // ---- duong ong buoc 2: dia chi (T) -> RAM tra du lieu (T+1) -> ghi thanh ghi (T+2) ----
            build_valid_2         <= build_valid_1;
            build_index_2         <= build_index_1;
            GRADIENT_WRITE_ENABLE <= build_valid_2;
            GRADIENT_WRITE_ADDR   <= build_index_2;
            GRADIENT_WRITE_DATA   <= (FEATURE_A_READ_DATA != 8'd0) ? (GAP_READ_DATA ^ 8'h80) : 8'h80;
            build_valid_1         <= 1'b0;

            // ---- duong ong buoc 4: dia chi (T) -> RAM tra du lieu (T+1) -> cong don (T+2) ----
            cam_valid_2           <= cam_valid_1;
            cam_expect_2          <= cam_expect_1;
            if (cam_hit)
                cam_acc <= cam_acc + {16'd0, cam_product};

            case (state)
                STATE_IDLE: begin
                    if (START_LAYERCAM) begin
                        LCAM_ACTIVE <= 1'b1;
                        state       <= STATE_FETCH_ROW_START;
                    end
                end

                // ---- 1. nap hang FC vao GAP_RAM ----
                STATE_FETCH_ROW_START: begin
                    START_FETCH_CLASS_ROW <= 1'b1;
                    state                 <= STATE_FETCH_ROW_WAIT;
                end

                STATE_FETCH_ROW_WAIT: begin
                    if (DONE_FETCH_CLASS_ROW) begin
                        DZ5_PHASE_ACTIVE <= 1'b1;
                        build_index      <= 12'd0;
                        build_channel    <= 6'd0;
                        state            <= STATE_BUILD_DZ5;
                    end
                end

                // ---- 2. dz5 -> gradient_ram_a ----
                STATE_BUILD_DZ5: begin
                    DZ5_READ_ADDR <= build_index;
                    GAP_READ_ADDR <= build_channel;
                    build_valid_1 <= 1'b1;
                    build_index_1 <= build_index;
                    build_channel <= (build_channel == 6'd47) ? 6'd0 : (build_channel + 1'b1);
                    if (build_index == 12'd3071) begin
                        drain_counter <= 3'd5;
                        state         <= STATE_BUILD_DRAIN;
                    end else begin
                        build_index <= build_index + 1'b1;
                    end
                end

                // xa duong ong de lan ghi cuoi vao gradient_ram_a xong truoc khi BWD1 doc
                STATE_BUILD_DRAIN: begin
                    if (drain_counter == 3'd0) begin
                        DZ5_PHASE_ACTIVE <= 1'b0;
                        LCAM_LAYER_INDEX <= 3'd6;
                        state            <= STATE_BWD1_START;
                    end else begin
                        drain_counter <= drain_counter - 1'b1;
                    end
                end

                // ---- 3. hai lop nguoc ----
                STATE_BWD1_START: begin
                    LCAM_START_LAYER <= 1'b1;
                    state            <= STATE_BWD1_WAIT;
                end

                STATE_BWD1_WAIT: begin
                    if (DONE_LAYER) begin
                        LCAM_LAYER_INDEX <= 3'd7;
                        state            <= STATE_BWD2_START;
                    end
                end

                STATE_BWD2_START: begin
                    LCAM_START_LAYER <= 1'b1;
                    state            <= STATE_BWD2_WAIT;
                end

                STATE_BWD2_WAIT: begin
                    if (DONE_LAYER) begin
                        CAM_PHASE_ACTIVE <= 1'b1;
                        cam_row          <= 4'd0;
                        cam_col          <= 4'd0;
                        cam_channel      <= 5'd0;
                        cam_acc          <= 32'd0;
                        state            <= STATE_CAM_ISSUE;
                    end
                end

                // ---- 4. CAM: 24 kenh moi diem ----
                STATE_CAM_ISSUE: begin
                    CAM_SOURCE_READ_ADDR <= cam_read_addr;
                    cam_valid_1          <= 1'b1;
                    cam_expect_1         <= {cam_row[0], cam_col[0]};
                    if (cam_channel == 5'd23) begin
                        cam_channel   <= 5'd0;
                        flush_counter <= 3'd3;
                        state         <= STATE_CAM_FLUSH;
                    end else begin
                        cam_channel <= cam_channel + 1'b1;
                    end
                end

                // cho nhip cuoi qua duong ong de cam_acc on dinh
                STATE_CAM_FLUSH: begin
                    if (flush_counter == 3'd0)
                        state <= STATE_CAM_REQUANT_START;
                    else
                        flush_counter <= flush_counter - 1'b1;
                end

                STATE_CAM_REQUANT_START: begin
                    start_requant <= 1'b1;
                    state         <= STATE_CAM_REQUANT_WAIT;
                end

                STATE_CAM_REQUANT_WAIT: begin
                    if (done_requant) begin
                        CAM_WRITE_ENABLE <= 1'b1;
                        CAM_WRITE_ADDR   <= {cam_row, cam_col};
                        CAM_WRITE_DATA   <= requant_byte;
                        cam_acc          <= 32'd0;
                        state            <= STATE_CAM_NEXT;
                    end
                end

                STATE_CAM_NEXT: begin
                    if (cam_col == 4'd15) begin
                        cam_col <= 4'd0;
                        if (cam_row == 4'd15) begin
                            state <= STATE_FINISH;
                        end else begin
                            cam_row <= cam_row + 1'b1;
                            state   <= STATE_CAM_ISSUE;
                        end
                    end else begin
                        cam_col <= cam_col + 1'b1;
                        state   <= STATE_CAM_ISSUE;
                    end
                end

                STATE_FINISH: begin
                    DONE_LAYERCAM    <= 1'b1;
                    LCAM_ACTIVE      <= 1'b0;
                    CAM_PHASE_ACTIVE <= 1'b0;
                    state            <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
