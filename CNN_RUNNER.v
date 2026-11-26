// Chay tron 1 CNN: nap tham so -> conv0..4 -> GAP -> FC (lop 5) -> argmax.
// Chua: PARAMETER_FETCHER, CONV_CONTROLLER, COMPUTE_CORE, FEATURE_RAM_A/B, INPUT_LUT_RAM, GAP_RAM, LOGIT_RAM.
// IMAGE_RAM nam o TOP_CNN (khong bi ghi de); runner chi doc qua IMAGE_READ_ADDR/DATA.
//
// Chon RAM theo active_layer_index:
//   nguon : 0 = IMAGE_RAM + LUT, 1,3 = A, 2,4 = B, 5 = GAP_RAM, 6 = gradient_ram_a, 7 = gradient_ram_b
//   dich  : 0,2,4 = A, 1,3 = B, 5 = LOGIT_RAM, 6 = gradient_ram_b (co mask ReLU theo B), 7 = gradient_ram_a
//   (lop 2 con ghi them pool3_copy_ram / pool3_argmax_ram cho LayerCAM)
//
// LayerCAM (START_LAYERCAM): chay LAYERCAM_CONTROLLER tren CNN vua chay forward xong (CNN quyet dinh).
// Khi do active_layer_index = lcam_layer_index (6 hoac 7); FSM forward dung yen o STATE_IDLE va giu layer_index.
// FEATURE_RAM_A giu A5 (conv5), FEATURE_RAM_B giu A4 (conv4); GAP_RAM bi ghi de bang hang FC (nap lai o anh sau).
module CNN_RUNNER (
    input  wire        CLK,
    input  wire        RST,

    input  wire        START_CNN,
    input  wire        CNN_SELECT,
    output reg         DONE_CNN,
    output reg  [2:0]  PREDICTED_CLASS_INDEX,
    output reg  [39:0] LOGITS_OUTPUT,              // logit k = [8*k +: 8]
    output wire [7:0]  UNKNOWN_CLASS_INDEX,
    output wire        ERROR_BAD_MAGIC,
    output wire [7:0]  NORMAL_CLASS_INDEX,         // 0xFF = CNN nay khong co lop Normal

    // ---- LayerCAM ----
    input  wire        START_LAYERCAM,
    input  wire [2:0]  LAYERCAM_CLASS_INDEX,
    output wire        DONE_LAYERCAM,
    input  wire [7:0]  CAM_READ_ADDR,
    output wire [7:0]  CAM_READ_DATA,               // cam_ram tre 1 chu ky

    output wire [13:0] IMAGE_READ_ADDR,
    input  wire [7:0]  IMAGE_READ_DATA,

    output wire        SDRAM_REQUEST_VALID,
    output wire        SDRAM_REQUEST_WRITE,
    output wire [23:0] SDRAM_REQUEST_WORD_ADDR,
    output wire [15:0] SDRAM_REQUEST_WRITE_DATA,
    input  wire        SDRAM_REQUEST_READY,
    input  wire        SDRAM_RESPONSE_VALID,
    input  wire [15:0] SDRAM_RESPONSE_READ_DATA
);
    // ------------------------------------------------------------
    // FSM cua runner
    // ------------------------------------------------------------
    localparam STATE_IDLE               = 4'd0;
    localparam STATE_FETCH_PARAMETERS   = 4'd1;
    localparam STATE_LAYER_START        = 4'd2;
    localparam STATE_LAYER_WAIT         = 4'd3;
    localparam STATE_GAP_CHANNEL_INIT   = 4'd4;
    localparam STATE_GAP_READ_ISSUE     = 4'd5;
    localparam STATE_GAP_READ_WAIT      = 4'd6;
    localparam STATE_GAP_ACCUMULATE     = 4'd7;
    localparam STATE_GAP_WRITE          = 4'd8;
    localparam STATE_ARGMAX_ISSUE       = 4'd9;
    localparam STATE_ARGMAX_WAIT        = 4'd10;
    localparam STATE_ARGMAX_CHECK       = 4'd11;
    localparam STATE_FINISH             = 4'd12;

    reg [3:0] state;
    reg [2:0] layer_index;
    reg       start_fetch_parameters;
    reg       start_layer;

    // GAP
    reg [5:0]  gap_channel;
    reg [6:0]  gap_pixel_index;
    reg [13:0] gap_pixel_address;
    reg [13:0] gap_read_addr;
    reg [15:0] gap_sum;
    reg        gap_write_enable;
    reg [5:0]  gap_write_addr;
    reg [7:0]  gap_write_data;
    wire [15:0] gap_sum_rounded = gap_sum + 16'd32;

    // Argmax
    reg [2:0]  argmax_index;
    reg [2:0]  logit_read_addr;
    reg [7:0]  best_value;
    reg [2:0]  best_index;
    reg [39:0] logits_shift;

    wire gap_active = (state >= STATE_GAP_CHANNEL_INIT) && (state <= STATE_GAP_WRITE);

    // ------------------------------------------------------------
    // LayerCAM: lop hieu dung
    // ------------------------------------------------------------
    wire        lcam_active;
    wire [2:0]  lcam_layer_index;
    wire        lcam_start_layer;
    wire        lcam_dz5_phase_active;
    wire [11:0] lcam_dz5_read_addr;
    wire [5:0]  lcam_gap_read_addr;
    wire        lcam_gradient_write_enable;
    wire [11:0] lcam_gradient_write_addr;
    wire [7:0]  lcam_gradient_write_data;
    wire        lcam_cam_phase_active;
    wire [10:0] lcam_cam_source_read_addr;
    wire        lcam_cam_write_enable;
    wire [7:0]  lcam_cam_write_addr;
    wire [7:0]  lcam_cam_write_data;
    wire        start_fetch_class_row;
    wire        done_fetch_class_row;
    wire        class_row_write_enable;
    wire [5:0]  class_row_write_addr;
    wire [7:0]  class_row_write_data;
    wire [30:0] cam_m0;
    wire [5:0]  cam_shift;

    wire [2:0] active_layer_index = lcam_active ? lcam_layer_index : layer_index;
    wire       backward           = (active_layer_index == 3'd6) || (active_layer_index == 3'd7);

    // ------------------------------------------------------------
    // Day noi
    // ------------------------------------------------------------
    wire        done_fetch_parameters;
    wire [30:0] layer_multiplier;
    wire [5:0]  layer_shift;
    wire [7:0]  layer_input_zero_point;
    wire [7:0]  layer_output_zero_point;
    wire        layer_relu_enable;

    wire        input_lut_write_enable;
    wire [9:0]  input_lut_write_addr;
    wire [7:0]  input_lut_write_data;

    wire        start_fetch_weights;
    wire [2:0]  group_index;
    wire        done_fetch_weights;
    wire [15:0] layer_offset;
    wire [11:0] group_bytes;

    wire        bias_write_enable;
    wire [2:0]  bias_write_index;
    wire [31:0] bias_write_data;
    wire        weight_write_enable;
    wire [2:0]  weight_write_pe_index;
    wire [8:0]  weight_write_addr;
    wire [7:0]  weight_write_data;

    wire        done_layer;
    wire [13:0] conv_input_read_addr;
    wire [5:0]  conv_input_channel_index;
    wire [8:0]  weight_read_addr;
    wire        load_bias_into_acc;
    wire        accumulate_enable;
    wire        pool_first_position;
    wire        pool_next_position;
    wire [1:0]  pool_position_index;
    wire [15:0] pool_argmax_code;
    wire [13:0] mask_read_addr;
    wire        start_requant;
    wire [7:0]  requant_byte;
    wire        requant_byte_valid;
    wire [2:0]  requant_channel_in_group;
    wire        done_requant;
    wire        conv_write_enable;
    wire [13:0] conv_write_addr;
    wire [7:0]  conv_write_data;

    // ------------------------------------------------------------
    // PARAMETER_FETCHER
    // ------------------------------------------------------------
    PARAMETER_FETCHER parameter_fetcher_inst (
        .CLK(CLK), .RST(RST),
        .START_FETCH_PARAMETERS(start_fetch_parameters),
        .CNN_SELECT(CNN_SELECT),
        .DONE_FETCH_PARAMETERS(done_fetch_parameters),
        .ERROR_BAD_MAGIC(ERROR_BAD_MAGIC),
        .UNKNOWN_CLASS_INDEX(UNKNOWN_CLASS_INDEX),
        .NORMAL_CLASS_INDEX(NORMAL_CLASS_INDEX),
        .CAM_M0(cam_m0),
        .CAM_SHIFT(cam_shift),
        .LAYER_INDEX(active_layer_index),
        .LAYER_MULTIPLIER(layer_multiplier),
        .LAYER_SHIFT(layer_shift),
        .LAYER_INPUT_ZERO_POINT(layer_input_zero_point),
        .LAYER_OUTPUT_ZERO_POINT(layer_output_zero_point),
        .LAYER_RELU_ENABLE(layer_relu_enable),
        .INPUT_LUT_WRITE_ENABLE(input_lut_write_enable),
        .INPUT_LUT_WRITE_ADDR(input_lut_write_addr),
        .INPUT_LUT_WRITE_DATA(input_lut_write_data),
        .START_FETCH_WEIGHTS(start_fetch_weights),
        .GROUP_INDEX(group_index),
        .LAYER_OFFSET(layer_offset),
        .GROUP_BYTES(group_bytes),
        .DONE_FETCH_WEIGHTS(done_fetch_weights),
        .START_FETCH_CLASS_ROW(start_fetch_class_row),
        .LAYERCAM_CLASS_INDEX(LAYERCAM_CLASS_INDEX),
        .DONE_FETCH_CLASS_ROW(done_fetch_class_row),
        .CLASS_ROW_WRITE_ENABLE(class_row_write_enable),
        .CLASS_ROW_WRITE_ADDR(class_row_write_addr),
        .CLASS_ROW_WRITE_DATA(class_row_write_data),
        .BIAS_WRITE_ENABLE(bias_write_enable),
        .BIAS_WRITE_INDEX(bias_write_index),
        .BIAS_WRITE_DATA(bias_write_data),
        .WEIGHT_WRITE_ENABLE(weight_write_enable),
        .WEIGHT_WRITE_PE_INDEX(weight_write_pe_index),
        .WEIGHT_WRITE_ADDR(weight_write_addr),
        .WEIGHT_WRITE_DATA(weight_write_data),
        .SDRAM_REQUEST_VALID(SDRAM_REQUEST_VALID),
        .SDRAM_REQUEST_WRITE(SDRAM_REQUEST_WRITE),
        .SDRAM_REQUEST_WORD_ADDR(SDRAM_REQUEST_WORD_ADDR),
        .SDRAM_REQUEST_WRITE_DATA(SDRAM_REQUEST_WRITE_DATA),
        .SDRAM_REQUEST_READY(SDRAM_REQUEST_READY),
        .SDRAM_RESPONSE_VALID(SDRAM_RESPONSE_VALID),
        .SDRAM_RESPONSE_READ_DATA(SDRAM_RESPONSE_READ_DATA)
    );

    // ------------------------------------------------------------
    // CONV_CONTROLLER
    // ------------------------------------------------------------
    CONV_CONTROLLER conv_controller_inst (
        .CLK(CLK), .RST(RST),
        .START_LAYER(start_layer | lcam_start_layer),
        .LAYER_INDEX(active_layer_index),
        .DONE_LAYER(done_layer),
        .START_FETCH_WEIGHTS(start_fetch_weights),
        .GROUP_INDEX(group_index),
        .DONE_FETCH_WEIGHTS(done_fetch_weights),
        .LAYER_OFFSET(layer_offset),
        .GROUP_BYTES(group_bytes),
        .INPUT_READ_ADDR(conv_input_read_addr),
        .INPUT_CHANNEL_INDEX(conv_input_channel_index),
        .WEIGHT_READ_ADDR(weight_read_addr),
        .LOAD_BIAS_INTO_ACC(load_bias_into_acc),
        .ACCUMULATE_ENABLE(accumulate_enable),
        .POOL_FIRST_POSITION(pool_first_position),
        .POOL_NEXT_POSITION(pool_next_position),
        .POOL_POSITION_INDEX(pool_position_index),
        .START_REQUANT(start_requant),
        .REQUANT_BYTE(requant_byte),
        .REQUANT_BYTE_VALID(requant_byte_valid),
        .REQUANT_CHANNEL_IN_GROUP(requant_channel_in_group),
        .DONE_REQUANT(done_requant),
        .OUTPUT_WRITE_ENABLE(conv_write_enable),
        .OUTPUT_WRITE_ADDR(conv_write_addr),
        .OUTPUT_WRITE_DATA(conv_write_data),
        .MASK_READ_ADDR(mask_read_addr)
    );

    // ------------------------------------------------------------
    // RAM
    // ------------------------------------------------------------
    wire [7:0] feature_a_read_data;
    wire [7:0] feature_b_read_data;
    wire [7:0] lut_read_data;
    wire [7:0] gap_read_data;
    wire [7:0] logit_read_data;

    wire [7:0] gradient_a_read_data;
    wire [7:0] gradient_b_read_data;
    wire [7:0] pool3_copy_read_data;
    wire [7:0] pool3_argmax_read_data;

    // Lop 6, 7 khong khop dich nao trong 3 dong nay (ghi vao gradient_ram_a/b rieng)
    wire layer_dest_is_a     = (active_layer_index == 3'd0) || (active_layer_index == 3'd2) || (active_layer_index == 3'd4);
    wire layer_dest_is_b     = (active_layer_index == 3'd1) || (active_layer_index == 3'd3);
    wire layer_dest_is_logit = (active_layer_index == 3'd5);

    // Kenh vao cua pixel dang doc, tre 1 nhip de khop du lieu IMAGE_RAM (dia chi LUT)
    reg [1:0] input_channel_for_lut_delayed;
    always @(posedge CLK) begin
        input_channel_for_lut_delayed <= conv_input_channel_index[1:0];
    end

    assign IMAGE_READ_ADDR = conv_input_read_addr;

    SIMPLE_DUAL_PORT_RAM #(.DEPTH(16384), .ADDR_WIDTH(14)) feature_ram_a (
        .CLK(CLK),
        .WRITE_ENABLE(conv_write_enable && layer_dest_is_a),
        .WRITE_ADDR(conv_write_addr),
        .WRITE_DATA(conv_write_data),
        .READ_ADDR(gap_active ? gap_read_addr :
                   (lcam_dz5_phase_active ? {2'b00, lcam_dz5_read_addr} : conv_input_read_addr)),
        .READ_DATA(feature_a_read_data)
    );

    SIMPLE_DUAL_PORT_RAM #(.DEPTH(6144), .ADDR_WIDTH(13)) feature_ram_b (
        .CLK(CLK),
        .WRITE_ENABLE(conv_write_enable && layer_dest_is_b),
        .WRITE_ADDR(conv_write_addr[12:0]),
        .WRITE_DATA(conv_write_data),
        // Lop 6 (BWD1) doc nguon tu gradient_ram_a nen cong doc B ranh -> doc A4 lam mask ReLU
        .READ_ADDR((active_layer_index == 3'd6) ? mask_read_addr[12:0] : conv_input_read_addr[12:0]),
        .READ_DATA(feature_b_read_data)
    );

    SIMPLE_DUAL_PORT_RAM #(.DEPTH(768), .ADDR_WIDTH(10)) input_lut_ram (
        .CLK(CLK),
        .WRITE_ENABLE(input_lut_write_enable),
        .WRITE_ADDR(input_lut_write_addr),
        .WRITE_DATA(input_lut_write_data),
        .READ_ADDR({input_channel_for_lut_delayed, IMAGE_READ_DATA}),   // in_channel*256 + pixel tho
        .READ_DATA(lut_read_data)
    );

    // gap_ram: GAP (forward) hoac hang FC 48 byte (LayerCAM); hai nguon ghi khong trung thoi diem
    SIMPLE_DUAL_PORT_RAM #(.DEPTH(64), .ADDR_WIDTH(6)) gap_ram (
        .CLK(CLK),
        .WRITE_ENABLE(gap_write_enable | class_row_write_enable),
        .WRITE_ADDR(class_row_write_enable ? class_row_write_addr : gap_write_addr),
        .WRITE_DATA(class_row_write_enable ? class_row_write_data : gap_write_data),
        .READ_ADDR(lcam_dz5_phase_active ? lcam_gap_read_addr : conv_input_read_addr[5:0]),
        .READ_DATA(gap_read_data)
    );

    SIMPLE_DUAL_PORT_RAM #(.DEPTH(8), .ADDR_WIDTH(3)) logit_ram (
        .CLK(CLK),
        .WRITE_ENABLE(conv_write_enable && layer_dest_is_logit),
        .WRITE_ADDR(conv_write_addr[2:0]),
        .WRITE_DATA(conv_write_data),
        .READ_ADDR(logit_read_addr),
        .READ_DATA(logit_read_data)
    );

    // ------------------------------------------------------------
    // RAM cua LayerCAM
    // ------------------------------------------------------------
    // gradient_ram_a: dz5 (buoc 1, 3072) -> g4 (lop 7, 1536). Doc: BWD1 (lop 6) hoac buoc CAM.
    wire gradient_a_conv_write = conv_write_enable && (active_layer_index == 3'd7);
    SIMPLE_DUAL_PORT_RAM #(.DEPTH(3072), .ADDR_WIDTH(12)) gradient_ram_a (
        .CLK(CLK),
        .WRITE_ENABLE(lcam_gradient_write_enable | gradient_a_conv_write),
        .WRITE_ADDR(lcam_gradient_write_enable ? lcam_gradient_write_addr : conv_write_addr[11:0]),
        .WRITE_DATA(lcam_gradient_write_enable ? lcam_gradient_write_data : conv_write_data),
        .READ_ADDR(lcam_cam_phase_active ? {1'b0, lcam_cam_source_read_addr} : conv_input_read_addr[11:0]),
        .READ_DATA(gradient_a_read_data)
    );

    // gradient_ram_b: g5 (dau ra lop 6, 2048). O bi mask ReLU cua A4 (A4 == 0) ghi 128 (= gradient 0).
    SIMPLE_DUAL_PORT_RAM #(.DEPTH(2048), .ADDR_WIDTH(11)) gradient_ram_b (
        .CLK(CLK),
        .WRITE_ENABLE(conv_write_enable && (active_layer_index == 3'd6)),
        .WRITE_ADDR(conv_write_addr[10:0]),
        .WRITE_DATA((feature_b_read_data == 8'd0) ? 8'd128 : conv_write_data),
        .READ_ADDR(conv_input_read_addr[10:0]),
        .READ_DATA(gradient_b_read_data)
    );

    // pool3: dau ra conv3 sau pool (8x8x24) va ma vi tri thang cua pool; ghi o MOI anh (lop 2 forward)
    wire pool3_write_enable = conv_write_enable && (active_layer_index == 3'd2);
    wire [1:0] pool3_argmax_code = pool_argmax_code[2*conv_write_addr[2:0] +: 2];   // kenh = addr[2:0] (moi base chia het cho 8)

    SIMPLE_DUAL_PORT_RAM #(.DEPTH(1536), .ADDR_WIDTH(11)) pool3_copy_ram (
        .CLK(CLK),
        .WRITE_ENABLE(pool3_write_enable),
        .WRITE_ADDR(conv_write_addr[10:0]),
        .WRITE_DATA(conv_write_data),
        .READ_ADDR(lcam_cam_source_read_addr),
        .READ_DATA(pool3_copy_read_data)
    );

    SIMPLE_DUAL_PORT_RAM #(.DEPTH(1536), .ADDR_WIDTH(11)) pool3_argmax_ram (
        .CLK(CLK),
        .WRITE_ENABLE(pool3_write_enable),
        .WRITE_ADDR(conv_write_addr[10:0]),
        .WRITE_DATA({6'd0, pool3_argmax_code}),
        .READ_ADDR(lcam_cam_source_read_addr),
        .READ_DATA(pool3_argmax_read_data)
    );

    SIMPLE_DUAL_PORT_RAM #(.DEPTH(256), .ADDR_WIDTH(8)) cam_ram (
        .CLK(CLK),
        .WRITE_ENABLE(lcam_cam_write_enable),
        .WRITE_ADDR(lcam_cam_write_addr),
        .WRITE_DATA(lcam_cam_write_data),
        .READ_ADDR(CAM_READ_ADDR),
        .READ_DATA(CAM_READ_DATA)
    );

    // Pixel vao COMPUTE_CORE
    reg [7:0] input_pixel_to_core;
    always @(*) begin
        case (active_layer_index)
            3'd0:         input_pixel_to_core = lut_read_data;
            3'd1, 3'd3:   input_pixel_to_core = feature_a_read_data;
            3'd2, 3'd4:   input_pixel_to_core = feature_b_read_data;
            3'd5:         input_pixel_to_core = gap_read_data;
            3'd6:         input_pixel_to_core = gradient_a_read_data;
            3'd7:         input_pixel_to_core = gradient_b_read_data;
            default:      input_pixel_to_core = 8'd0;
        endcase
    end

    // ------------------------------------------------------------
    // COMPUTE_CORE
    // ------------------------------------------------------------
    // Can duoi requant: lop nguoc kep o 1 (gradient 128 +- 127, 0 khong dung); lop khac: ZP_OUT neu ReLU, nguoc lai 0
    wire [7:0] clamp_lower = backward ? 8'd1 : (layer_relu_enable ? layer_output_zero_point : 8'd0);

    COMPUTE_CORE compute_core_inst (
        .CLK(CLK), .RST(RST),
        .INPUT_PIXEL(input_pixel_to_core),
        .INPUT_ZERO_POINT(layer_input_zero_point),
        .WEIGHT_READ_ADDR(weight_read_addr),
        .LOAD_BIAS_INTO_ACC(load_bias_into_acc),
        .ACCUMULATE_ENABLE(accumulate_enable),
        .POOL_FIRST_POSITION(pool_first_position),
        .POOL_NEXT_POSITION(pool_next_position),
        .POOL_POSITION_INDEX(pool_position_index),
        .START_REQUANT(start_requant),
        .LAYER_MULTIPLIER(layer_multiplier),
        .LAYER_SHIFT(layer_shift),
        .LAYER_OUTPUT_ZERO_POINT(layer_output_zero_point),
        .CLAMP_LOWER(clamp_lower),
        .BIAS_WRITE_ENABLE(bias_write_enable),
        .BIAS_WRITE_INDEX(bias_write_index),
        .BIAS_WRITE_DATA(bias_write_data),
        .WEIGHT_WRITE_ENABLE(weight_write_enable),
        .WEIGHT_WRITE_PE_INDEX(weight_write_pe_index),
        .WEIGHT_WRITE_ADDR(weight_write_addr),
        .WEIGHT_WRITE_DATA(weight_write_data),
        .REQUANT_BYTE(requant_byte),
        .REQUANT_BYTE_VALID(requant_byte_valid),
        .REQUANT_CHANNEL_IN_GROUP(requant_channel_in_group),
        .DONE_REQUANT(done_requant),
        .POOL_ARGMAX_CODE(pool_argmax_code)
    );

    // ------------------------------------------------------------
    // LAYERCAM_CONTROLLER
    // ------------------------------------------------------------
    LAYERCAM_CONTROLLER layercam_controller_inst (
        .CLK(CLK), .RST(RST),
        .START_LAYERCAM(START_LAYERCAM),
        .DONE_LAYERCAM(DONE_LAYERCAM),
        .START_FETCH_CLASS_ROW(start_fetch_class_row),
        .DONE_FETCH_CLASS_ROW(done_fetch_class_row),
        .LCAM_ACTIVE(lcam_active),
        .LCAM_LAYER_INDEX(lcam_layer_index),
        .LCAM_START_LAYER(lcam_start_layer),
        .DONE_LAYER(done_layer),
        .DZ5_PHASE_ACTIVE(lcam_dz5_phase_active),
        .DZ5_READ_ADDR(lcam_dz5_read_addr),
        .FEATURE_A_READ_DATA(feature_a_read_data),
        .GAP_READ_ADDR(lcam_gap_read_addr),
        .GAP_READ_DATA(gap_read_data),
        .GRADIENT_WRITE_ENABLE(lcam_gradient_write_enable),
        .GRADIENT_WRITE_ADDR(lcam_gradient_write_addr),
        .GRADIENT_WRITE_DATA(lcam_gradient_write_data),
        .CAM_PHASE_ACTIVE(lcam_cam_phase_active),
        .CAM_SOURCE_READ_ADDR(lcam_cam_source_read_addr),
        .GRADIENT_READ_DATA(gradient_a_read_data),
        .POOL3_READ_DATA(pool3_copy_read_data),
        .POOL3_ARGMAX_READ_DATA(pool3_argmax_read_data),
        .CAM_WRITE_ENABLE(lcam_cam_write_enable),
        .CAM_WRITE_ADDR(lcam_cam_write_addr),
        .CAM_WRITE_DATA(lcam_cam_write_data),
        .CAM_M0(cam_m0),
        .CAM_SHIFT(cam_shift)
    );

    // ------------------------------------------------------------
    // FSM: tham so -> lop 0..4 -> GAP -> lop 5 (FC) -> argmax
    // ------------------------------------------------------------
    always @(posedge CLK) begin
        if (RST) begin
            state                  <= STATE_IDLE;
            layer_index            <= 3'd0;
            start_fetch_parameters <= 1'b0;
            start_layer            <= 1'b0;
            DONE_CNN               <= 1'b0;
            PREDICTED_CLASS_INDEX  <= 3'd0;
            LOGITS_OUTPUT          <= 40'd0;
            gap_channel            <= 6'd0;
            gap_pixel_index        <= 7'd0;
            gap_pixel_address      <= 14'd0;
            gap_read_addr          <= 14'd0;
            gap_sum                <= 16'd0;
            gap_write_enable       <= 1'b0;
            gap_write_addr         <= 6'd0;
            gap_write_data         <= 8'd0;
            argmax_index           <= 3'd0;
            logit_read_addr        <= 3'd0;
            best_value             <= 8'd0;
            best_index             <= 3'd0;
            logits_shift           <= 40'd0;
        end else begin
            start_fetch_parameters <= 1'b0;
            start_layer            <= 1'b0;
            DONE_CNN               <= 1'b0;
            gap_write_enable       <= 1'b0;

            case (state)
                STATE_IDLE: begin
                    if (START_CNN) begin
                        layer_index            <= 3'd0;
                        start_fetch_parameters <= 1'b1;
                        state                  <= STATE_FETCH_PARAMETERS;
                    end
                end

                STATE_FETCH_PARAMETERS: begin
                    if (done_fetch_parameters) begin
                        if (ERROR_BAD_MAGIC)
                            state <= STATE_FINISH;
                        else
                            state <= STATE_LAYER_START;
                    end
                end

                STATE_LAYER_START: begin
                    start_layer <= 1'b1;
                    state       <= STATE_LAYER_WAIT;
                end

                STATE_LAYER_WAIT: begin
                    if (done_layer) begin
                        if (layer_index == 3'd4) begin
                            gap_channel <= 6'd0;
                            state       <= STATE_GAP_CHANNEL_INIT;
                        end else if (layer_index == 3'd5) begin
                            argmax_index <= 3'd0;
                            best_value   <= 8'd0;
                            best_index   <= 3'd0;
                            state        <= STATE_ARGMAX_ISSUE;
                        end else begin
                            layer_index <= layer_index + 1'b1;
                            state       <= STATE_LAYER_START;
                        end
                    end
                end

                // ---- GAP: gap[c] = (sum_64pixel x[c] + 32) >> 6 ; dia chi = pixel*48 + c ----
                STATE_GAP_CHANNEL_INIT: begin
                    gap_sum           <= 16'd0;
                    gap_pixel_index   <= 7'd0;
                    gap_pixel_address <= {8'd0, gap_channel};
                    state             <= STATE_GAP_READ_ISSUE;
                end

                STATE_GAP_READ_ISSUE: begin
                    gap_read_addr <= gap_pixel_address;
                    state         <= STATE_GAP_READ_WAIT;
                end

                STATE_GAP_READ_WAIT: begin
                    state <= STATE_GAP_ACCUMULATE;      // RAM tre 1 chu ky
                end

                STATE_GAP_ACCUMULATE: begin
                    gap_sum           <= gap_sum + {8'd0, feature_a_read_data};
                    gap_pixel_address <= gap_pixel_address + 14'd48;
                    if (gap_pixel_index == 7'd63) begin
                        state <= STATE_GAP_WRITE;
                    end else begin
                        gap_pixel_index <= gap_pixel_index + 1'b1;
                        state           <= STATE_GAP_READ_ISSUE;
                    end
                end

                STATE_GAP_WRITE: begin
                    gap_write_enable <= 1'b1;
                    gap_write_addr   <= gap_channel;
                    gap_write_data   <= gap_sum_rounded[13:6];     // (sum + 32) >> 6, toi da 255
                    if (gap_channel == 6'd47) begin
                        layer_index <= 3'd5;
                        state       <= STATE_LAYER_START;
                    end else begin
                        gap_channel <= gap_channel + 1'b1;
                        state       <= STATE_GAP_CHANNEL_INIT;
                    end
                end

                // ---- Argmax 5 logit uint8, hoa -> chi so nho nhat (so sanh '>' nghiem ngat) ----
                STATE_ARGMAX_ISSUE: begin
                    logit_read_addr <= argmax_index;
                    state           <= STATE_ARGMAX_WAIT;
                end

                STATE_ARGMAX_WAIT: begin
                    state <= STATE_ARGMAX_CHECK;
                end

                STATE_ARGMAX_CHECK: begin
                    logits_shift <= {logit_read_data, logits_shift[39:8]};
                    if (logit_read_data > best_value) begin
                        best_value <= logit_read_data;
                        best_index <= argmax_index;
                    end
                    if (argmax_index == 3'd4) begin
                        state <= STATE_FINISH;
                    end else begin
                        argmax_index <= argmax_index + 1'b1;
                        state        <= STATE_ARGMAX_ISSUE;
                    end
                end

                STATE_FINISH: begin
                    DONE_CNN              <= 1'b1;
                    PREDICTED_CLASS_INDEX <= best_index;
                    LOGITS_OUTPUT         <= logits_shift;
                    state                 <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
