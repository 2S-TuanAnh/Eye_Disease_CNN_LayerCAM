// Vong lap mot lop (muc 9 dac ta). Chua LAYER_CONFIG_ROM.
//
// Pipeline (da tinh ky):
//  - Chu ky T0: INPUT_READ_ADDR (thanh ghi) + bo co "issue_*" cung hop le.
//  - RAM nguon tre 1 -> du lieu pixel o T0+1.
//      Lop 1..5: INPUT_PIXEL vao COMPUTE_CORE o T0+1  -> cac cong dieu khien duoc tre 1 nhip (stage1).
//      Lop 0   : them INPUT_LUT_RAM (tre 1)  -> INPUT_PIXEL o T0+2 -> cac cong dieu khien tre 2 nhip (stage2).
//  - POOL_FIRST/NEXT di cung chuoi tre voi ACCUMULATE_ENABLE va duoc phat o chu ky ngay sau
//    ACCUMULATE cuoi (trang thai POOL_ISSUE), dung yeu cau cua COMPUTE_CORE.
//  - Moi vi tri conv: LOAD_BIAS_INTO_ACC cung chu ky voi ACCUMULATE_ENABLE dau tien (COMPUTE_CORE cong bias + tich dau).
//  - Padding: bo han tap kernel nam ngoai anh (khong ton chu ky accumulate).
//  - Lop khong pool van phat POOL_FIRST_POSITION cho vi tri duy nhat.
//  - LayerCAM: LAYER_INDEX 6, 7 (BWD1/BWD2) chay y het lop khong pool 8x8. POOL_POSITION_INDEX = (row%2)*2 + (col%2)
//    di cung chuoi tre voi co pool (chi hop le o chu ky co co pool). MASK_READ_ADDR la dia chi (wire) cua byte
//    dang requant, dung de doc FEATURE_RAM_B lam mask ReLU cho lop 6 (RAM tre 1 chu ky = trung OUTPUT_WRITE_DATA).
module CONV_CONTROLLER (
    input  wire        CLK,
    input  wire        RST,

    input  wire        START_LAYER,
    input  wire [2:0]  LAYER_INDEX,
    output reg         DONE_LAYER,

    // ---- Nap khoi nhom (PARAMETER_FETCHER) ----
    output reg         START_FETCH_WEIGHTS,
    output wire [2:0]  GROUP_INDEX,
    input  wire        DONE_FETCH_WEIGHTS,
    output wire [15:0] LAYER_OFFSET,
    output wire [11:0] GROUP_BYTES,

    // ---- Doc RAM nguon (CNN_RUNNER chon RAM theo LAYER_INDEX) ----
    output reg  [13:0] INPUT_READ_ADDR,
    output reg  [5:0]  INPUT_CHANNEL_INDEX,      // thang hang voi INPUT_READ_ADDR (dung cho dia chi LUT lop 0)

    // ---- Dieu khien COMPUTE_CORE ----
    output wire [8:0]  WEIGHT_READ_ADDR,
    output wire        LOAD_BIAS_INTO_ACC,
    output wire        ACCUMULATE_ENABLE,
    output wire        POOL_FIRST_POSITION,
    output wire        POOL_NEXT_POSITION,
    output wire [1:0]  POOL_POSITION_INDEX,
    output reg         START_REQUANT,
    input  wire [7:0]  REQUANT_BYTE,
    input  wire        REQUANT_BYTE_VALID,
    input  wire [2:0]  REQUANT_CHANNEL_IN_GROUP,
    input  wire        DONE_REQUANT,

    // ---- Ghi RAM dich ----
    output reg         OUTPUT_WRITE_ENABLE,
    output reg  [13:0] OUTPUT_WRITE_ADDR,
    output reg  [7:0]  OUTPUT_WRITE_DATA,
    output wire [13:0] MASK_READ_ADDR            // = output_base_addr + kenh dang requant (chua qua thanh ghi)
);
    // ------------------------------------------------------------
    // Hang so lop
    // ------------------------------------------------------------
    wire [6:0] in_size;
    wire [2:0] in_size_log2;
    wire [5:0] in_channels;
    wire [5:0] out_size;
    wire [2:0] out_size_log2;
    wire [5:0] out_channels;
    wire       pool_enable;
    wire [2:0] group_count;
    wire       is_fully_connected;

    LAYER_CONFIG_ROM layer_config_rom_inst (
        .LAYER_INDEX(LAYER_INDEX),
        .IN_SIZE(in_size),
        .IN_SIZE_LOG2(in_size_log2),
        .IN_CHANNELS(in_channels),
        .OUT_SIZE(out_size),
        .OUT_SIZE_LOG2(out_size_log2),
        .OUT_CHANNELS(out_channels),
        .POOL_ENABLE(pool_enable),
        .GROUP_COUNT(group_count),
        .GROUP_BYTES(GROUP_BYTES),
        .LAYER_OFFSET(LAYER_OFFSET),
        .IS_FULLY_CONNECTED(is_fully_connected)
    );

    // ------------------------------------------------------------
    // Trang thai
    // ------------------------------------------------------------
    localparam STATE_IDLE                = 4'd0;
    localparam STATE_FETCH_WEIGHTS_START = 4'd1;
    localparam STATE_FETCH_WEIGHTS_WAIT  = 4'd2;
    localparam STATE_POSITION_BEGIN      = 4'd3;
    localparam STATE_TAP_CHECK           = 4'd4;
    localparam STATE_TAP_MULTIPLY        = 4'd5;
    localparam STATE_ACCUMULATE          = 4'd6;
    localparam STATE_POOL_ISSUE          = 4'd7;
    localparam STATE_SETTLE              = 4'd8;
    localparam STATE_REQUANT_START       = 4'd9;
    localparam STATE_REQUANT_WAIT        = 4'd10;
    localparam STATE_ADVANCE_POSITION    = 4'd11;
    localparam STATE_LAYER_FINISH        = 4'd12;

    reg [3:0]  state;
    reg [2:0]  group_index;
    reg [5:0]  output_row;
    reg [5:0]  output_col;
    reg [1:0]  pool_position;          // 0..3 (lop khong pool: luon 0)
    reg [1:0]  kernel_row;
    reg [1:0]  kernel_col;
    reg [5:0]  in_channel_index;
    reg [8:0]  weight_channel_offset;  // in_channel_index * 9 (conv) hoac * 1 (FC)
    reg [11:0] pixel_index_register;   // input_row * IN_SIZE + input_col
    reg [13:0] pixel_base_addr;        // pixel_index * IN_CHANNELS
    reg        first_issue_pending;    // accumulate dau tien cua vi tri conv -> nap bias
    reg [2:0]  settle_counter;

    assign GROUP_INDEX = group_index;

    // ------------------------------------------------------------
    // Vi tri conv va kiem tra padding
    // ------------------------------------------------------------
    wire [7:0] conv_row = pool_enable ? ({1'b0, output_row, 1'b0} + {7'd0, pool_position[1]})
                                      : {2'b00, output_row};
    wire [7:0] conv_col = pool_enable ? ({1'b0, output_col, 1'b0} + {7'd0, pool_position[0]})
                                      : {2'b00, output_col};

    // input_row = conv_row + kernel_row - 1  ->  dung row_with_offset = input_row + 1 de tranh so am
    wire [7:0] row_with_offset = conv_row + {6'd0, kernel_row};
    wire [7:0] col_with_offset = conv_col + {6'd0, kernel_col};
    wire       tap_row_valid   = (row_with_offset != 8'd0) && (row_with_offset <= {1'b0, in_size});
    wire       tap_col_valid   = (col_with_offset != 8'd0) && (col_with_offset <= {1'b0, in_size});
    wire       tap_valid       = is_fully_connected || (tap_row_valid && tap_col_valid);

    wire [6:0]  input_row = row_with_offset[6:0] - 7'd1;
    wire [6:0]  input_col = col_with_offset[6:0] - 7'd1;
    wire [11:0] pixel_index_comb = ({5'd0, input_row} << in_size_log2) + {5'd0, input_col};

    wire [3:0] tap_index = is_fully_connected
                           ? 4'd0
                           : (({2'b00, kernel_row} << 1) + {2'b00, kernel_row} + {2'b00, kernel_col});
    wire       last_tap  = is_fully_connected || ((kernel_row == 2'd2) && (kernel_col == 2'd2));
    wire [8:0] weight_channel_step = is_fully_connected ? 9'd1 : 9'd9;

    // ------------------------------------------------------------
    // Dia chi ghi dau ra: chay tu do, vi tri on dinh nhieu chu ky truoc REQUANT
    //   output_write_addr = (output_row*OUT_SIZE + output_col)*OUT_CHANNELS + group_index*8 + kenh
    // ------------------------------------------------------------
    reg [10:0] output_pixel_index;
    reg [13:0] output_base_addr;
    always @(posedge CLK) begin
        output_pixel_index <= ({5'd0, output_row} << out_size_log2) + {5'd0, output_col};
        output_base_addr   <= output_pixel_index * out_channels + {8'd0, group_index, 3'b000};
    end

    wire [13:0] mask_read_addr_comb = output_base_addr + {11'd0, REQUANT_CHANNEL_IN_GROUP};
    assign MASK_READ_ADDR = mask_read_addr_comb;

    // ------------------------------------------------------------
    // Bo co "issue" va chuoi tre
    // ------------------------------------------------------------
    reg [8:0] issue_weight_addr;
    reg       issue_load_bias;
    reg       issue_accumulate;
    reg       issue_pool_first;
    reg       issue_pool_next;
    reg [1:0] issue_pool_position;

    reg [8:0] stage1_weight_addr;
    reg       stage1_load_bias;
    reg       stage1_accumulate;
    reg       stage1_pool_first;
    reg       stage1_pool_next;
    reg [1:0] stage1_pool_position;

    reg [8:0] stage2_weight_addr;
    reg       stage2_load_bias;
    reg       stage2_accumulate;
    reg       stage2_pool_first;
    reg       stage2_pool_next;
    reg [1:0] stage2_pool_position;

    always @(posedge CLK) begin
        if (RST) begin
            stage1_weight_addr <= 9'd0; stage1_load_bias <= 1'b0; stage1_accumulate <= 1'b0;
            stage1_pool_first  <= 1'b0; stage1_pool_next <= 1'b0; stage1_pool_position <= 2'd0;
            stage2_weight_addr <= 9'd0; stage2_load_bias <= 1'b0; stage2_accumulate <= 1'b0;
            stage2_pool_first  <= 1'b0; stage2_pool_next <= 1'b0; stage2_pool_position <= 2'd0;
        end else begin
            stage1_weight_addr <= issue_weight_addr;
            stage1_load_bias   <= issue_load_bias;
            stage1_accumulate  <= issue_accumulate;
            stage1_pool_first  <= issue_pool_first;
            stage1_pool_next   <= issue_pool_next;
            stage1_pool_position <= issue_pool_position;

            stage2_weight_addr <= stage1_weight_addr;
            stage2_load_bias   <= stage1_load_bias;
            stage2_accumulate  <= stage1_accumulate;
            stage2_pool_first  <= stage1_pool_first;
            stage2_pool_next   <= stage1_pool_next;
            stage2_pool_position <= stage1_pool_position;
        end
    end

    wire use_lut_pipeline = (LAYER_INDEX == 3'd0);

    assign WEIGHT_READ_ADDR    = use_lut_pipeline ? stage2_weight_addr : stage1_weight_addr;
    assign LOAD_BIAS_INTO_ACC  = use_lut_pipeline ? stage2_load_bias   : stage1_load_bias;
    assign ACCUMULATE_ENABLE   = use_lut_pipeline ? stage2_accumulate  : stage1_accumulate;
    assign POOL_FIRST_POSITION = use_lut_pipeline ? stage2_pool_first  : stage1_pool_first;
    assign POOL_NEXT_POSITION  = use_lut_pipeline ? stage2_pool_next   : stage1_pool_next;
    assign POOL_POSITION_INDEX = use_lut_pipeline ? stage2_pool_position : stage1_pool_position;

    // ------------------------------------------------------------
    // FSM chinh
    // ------------------------------------------------------------
    always @(posedge CLK) begin
        if (RST) begin
            state                 <= STATE_IDLE;
            group_index           <= 3'd0;
            output_row            <= 6'd0;
            output_col            <= 6'd0;
            pool_position         <= 2'd0;
            kernel_row            <= 2'd0;
            kernel_col            <= 2'd0;
            in_channel_index      <= 6'd0;
            weight_channel_offset <= 9'd0;
            pixel_index_register  <= 12'd0;
            pixel_base_addr       <= 14'd0;
            first_issue_pending   <= 1'b0;
            settle_counter        <= 3'd0;
            DONE_LAYER            <= 1'b0;
            START_FETCH_WEIGHTS   <= 1'b0;
            START_REQUANT         <= 1'b0;
            INPUT_READ_ADDR       <= 14'd0;
            INPUT_CHANNEL_INDEX   <= 6'd0;
            issue_weight_addr     <= 9'd0;
            issue_load_bias       <= 1'b0;
            issue_accumulate      <= 1'b0;
            issue_pool_first      <= 1'b0;
            issue_pool_next       <= 1'b0;
            issue_pool_position   <= 2'd0;
            OUTPUT_WRITE_ENABLE   <= 1'b0;
            OUTPUT_WRITE_ADDR     <= 14'd0;
            OUTPUT_WRITE_DATA     <= 8'd0;
        end else begin
            DONE_LAYER          <= 1'b0;
            START_FETCH_WEIGHTS <= 1'b0;
            START_REQUANT       <= 1'b0;
            issue_load_bias     <= 1'b0;
            issue_accumulate    <= 1'b0;
            issue_pool_first    <= 1'b0;
            issue_pool_next     <= 1'b0;

            // Ghi byte requant ra RAM dich (VALID chi xuat hien trong STATE_REQUANT_WAIT)
            OUTPUT_WRITE_ENABLE <= REQUANT_BYTE_VALID;
            OUTPUT_WRITE_ADDR   <= mask_read_addr_comb;
            OUTPUT_WRITE_DATA   <= REQUANT_BYTE;

            case (state)
                STATE_IDLE: begin
                    if (START_LAYER) begin
                        group_index   <= 3'd0;
                        output_row    <= 6'd0;
                        output_col    <= 6'd0;
                        pool_position <= 2'd0;
                        state         <= STATE_FETCH_WEIGHTS_START;
                    end
                end

                STATE_FETCH_WEIGHTS_START: begin
                    START_FETCH_WEIGHTS <= 1'b1;
                    state               <= STATE_FETCH_WEIGHTS_WAIT;
                end

                STATE_FETCH_WEIGHTS_WAIT: begin
                    if (DONE_FETCH_WEIGHTS) begin
                        output_row    <= 6'd0;
                        output_col    <= 6'd0;
                        pool_position <= 2'd0;
                        state         <= STATE_POSITION_BEGIN;
                    end
                end

                // Bat dau 1 vi tri conv (1 trong toi da 4 vi tri cua khoi pool)
                STATE_POSITION_BEGIN: begin
                    kernel_row          <= 2'd0;
                    kernel_col          <= 2'd0;
                    first_issue_pending <= 1'b1;
                    state               <= STATE_TAP_CHECK;
                end

                // Kiem tra tap kernel (kernel_row, kernel_col) co nam trong anh khong
                STATE_TAP_CHECK: begin
                    if (tap_valid) begin
                        pixel_index_register <= pixel_index_comb;
                        state                <= STATE_TAP_MULTIPLY;
                    end else if (last_tap) begin
                        state <= STATE_POOL_ISSUE;
                    end else begin
                        if (kernel_col == 2'd2) begin
                            kernel_col <= 2'd0;
                            kernel_row <= kernel_row + 1'b1;
                        end else begin
                            kernel_col <= kernel_col + 1'b1;
                        end
                    end
                end

                STATE_TAP_MULTIPLY: begin
                    pixel_base_addr       <= is_fully_connected ? 14'd0
                                                                : (pixel_index_register * in_channels);
                    in_channel_index      <= 6'd0;
                    weight_channel_offset <= 9'd0;
                    state                 <= STATE_ACCUMULATE;
                end

                // Moi chu ky phat 1 accumulate cho 1 kenh vao
                STATE_ACCUMULATE: begin
                    INPUT_READ_ADDR       <= pixel_base_addr + {8'd0, in_channel_index};
                    INPUT_CHANNEL_INDEX   <= in_channel_index;
                    issue_weight_addr     <= weight_channel_offset + {5'd0, tap_index};
                    issue_accumulate      <= 1'b1;
                    issue_load_bias       <= first_issue_pending;
                    first_issue_pending   <= 1'b0;
                    weight_channel_offset <= weight_channel_offset + weight_channel_step;
                    in_channel_index      <= in_channel_index + 1'b1;

                    if (in_channel_index == (in_channels - 6'd1)) begin
                        if (last_tap) begin
                            state <= STATE_POOL_ISSUE;
                        end else begin
                            if (kernel_col == 2'd2) begin
                                kernel_col <= 2'd0;
                                kernel_row <= kernel_row + 1'b1;
                            end else begin
                                kernel_col <= kernel_col + 1'b1;
                            end
                            state <= STATE_TAP_CHECK;
                        end
                    end
                end

                // Chu ky ngay sau ACCUMULATE cuoi: phat co pool (se tre cung chuoi voi accumulate)
                STATE_POOL_ISSUE: begin
                    issue_pool_first <= (pool_position == 2'd0);
                    issue_pool_next  <= (pool_position != 2'd0);
                    issue_pool_position <= pool_position;
                    if (pool_enable && (pool_position != 2'd3)) begin
                        pool_position <= pool_position + 1'b1;
                        state         <= STATE_POSITION_BEGIN;
                    end else begin
                        settle_counter <= 3'd6;
                        state          <= STATE_SETTLE;
                    end
                end

                // Cho co pool lan truyen qua chuoi tre + COMPUTE_CORE (toi da ~5 chu ky)
                STATE_SETTLE: begin
                    if (settle_counter == 3'd0)
                        state <= STATE_REQUANT_START;
                    else
                        settle_counter <= settle_counter - 1'b1;
                end

                STATE_REQUANT_START: begin
                    START_REQUANT <= 1'b1;
                    state         <= STATE_REQUANT_WAIT;
                end

                STATE_REQUANT_WAIT: begin
                    if (DONE_REQUANT)
                        state <= STATE_ADVANCE_POSITION;
                end

                STATE_ADVANCE_POSITION: begin
                    pool_position <= 2'd0;
                    if (output_col == (out_size - 6'd1)) begin
                        output_col <= 6'd0;
                        if (output_row == (out_size - 6'd1)) begin
                            output_row <= 6'd0;
                            if (group_index == (group_count - 3'd1)) begin
                                state <= STATE_LAYER_FINISH;
                            end else begin
                                group_index <= group_index + 1'b1;
                                state       <= STATE_FETCH_WEIGHTS_START;
                            end
                        end else begin
                            output_row <= output_row + 1'b1;
                            state      <= STATE_POSITION_BEGIN;
                        end
                    end else begin
                        output_col <= output_col + 1'b1;
                        state      <= STATE_POSITION_BEGIN;
                    end
                end

                STATE_LAYER_FINISH: begin
                    DONE_LAYER <= 1'b1;
                    state      <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
