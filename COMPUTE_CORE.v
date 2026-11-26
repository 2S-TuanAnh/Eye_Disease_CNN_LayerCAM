// 8 PE song song + max-pool + REQUANTIZER.
// SUA so voi ban cu:
//  1. "pe_index[2:0]" (part-select tren genvar) khong hop le trong Verilog-2001/XST -> so sanh truc tiep voi pe_index.
//  2. weight_read_data [0:7] (mang wire noi vao cong module) -> doi thanh vector phang [63:0].
//  3. LayerCAM: WEIGHT_RAM sau 512 (lop BWD1 can 432 byte/PE); LAYER_RELU_ENABLE -> CLAMP_LOWER[7:0];
//     them POOL_POSITION_INDEX (vao) va POOL_ARGMAX_CODE (ra): ma vi tri thang cua max-pool theo tung kenh
//     (kenh k o [2*k +: 2]), tinh tren acc truoc requant, khi hoa giu vi tri dau (so sanh > nghiem ngat).
//
// Pipeline (khong doi):
//   Cong vao o chu ky T (WEIGHT_READ_ADDR, ACCUMULATE_ENABLE, LOAD_BIAS_INTO_ACC, INPUT_PIXEL).
//   WEIGHT_RAM tre 1 -> trong so o T+1. Cac co/pixel duoc tre 1 -> *_delayed o T+1. Acc cap nhat cuoi T+1.
//   POOL_FIRST/NEXT phai keo o chu ky NGAY SAU ACCUMULATE_ENABLE cuoi (T+1) -> *_delayed o T+2,
//   luc do accumulator da on dinh. CONV_CONTROLLER tuan thu dieu nay.
module COMPUTE_CORE (
    input  wire        CLK,
    input  wire        RST,

    input  wire [7:0]  INPUT_PIXEL,
    input  wire [7:0]  INPUT_ZERO_POINT,
    input  wire [8:0]  WEIGHT_READ_ADDR,
    input  wire        LOAD_BIAS_INTO_ACC,
    input  wire        ACCUMULATE_ENABLE,
    input  wire        POOL_FIRST_POSITION,
    input  wire        POOL_NEXT_POSITION,
    input  wire [1:0]  POOL_POSITION_INDEX,      // (row%2)*2 + (col%2), cung nhip voi cac co POOL_*

    input  wire        START_REQUANT,
    input  wire [30:0] LAYER_MULTIPLIER,
    input  wire [5:0]  LAYER_SHIFT,
    input  wire [7:0]  LAYER_OUTPUT_ZERO_POINT,
    input  wire [7:0]  CLAMP_LOWER,

    input  wire        BIAS_WRITE_ENABLE,
    input  wire [2:0]  BIAS_WRITE_INDEX,
    input  wire [31:0] BIAS_WRITE_DATA,

    input  wire        WEIGHT_WRITE_ENABLE,
    input  wire [2:0]  WEIGHT_WRITE_PE_INDEX,
    input  wire [8:0]  WEIGHT_WRITE_ADDR,
    input  wire [7:0]  WEIGHT_WRITE_DATA,

    output wire [7:0]  REQUANT_BYTE,
    output wire        REQUANT_BYTE_VALID,
    output wire [2:0]  REQUANT_CHANNEL_IN_GROUP,
    output wire        DONE_REQUANT,
    output wire [15:0] POOL_ARGMAX_CODE          // kenh k = [2*k +: 2]
);
    reg [7:0] input_pixel_delayed;
    reg [7:0] input_zero_point_delayed;
    reg       load_bias_delayed;
    reg       accumulate_enable_delayed;
    reg       pool_first_position_delayed;
    reg       pool_next_position_delayed;
    reg [1:0] pool_position_delayed;

    always @(posedge CLK) begin
        if (RST) begin
            input_pixel_delayed         <= 8'd0;
            input_zero_point_delayed    <= 8'd0;
            load_bias_delayed           <= 1'b0;
            accumulate_enable_delayed   <= 1'b0;
            pool_first_position_delayed <= 1'b0;
            pool_next_position_delayed  <= 1'b0;
            pool_position_delayed       <= 2'd0;
        end else begin
            input_pixel_delayed         <= INPUT_PIXEL;
            input_zero_point_delayed    <= INPUT_ZERO_POINT;
            load_bias_delayed           <= LOAD_BIAS_INTO_ACC;
            accumulate_enable_delayed   <= ACCUMULATE_ENABLE;
            pool_first_position_delayed <= POOL_FIRST_POSITION;
            pool_next_position_delayed  <= POOL_NEXT_POSITION;
            pool_position_delayed       <= POOL_POSITION_INDEX;
        end
    end

    wire [63:0]       weight_read_data_packed;     // PE k = [8*k +: 8]
    reg signed [31:0] bias_register        [0:7];
    reg signed [31:0] accumulator_register [0:7];
    reg signed [31:0] max_pool_register    [0:7];
    reg [1:0]         pool_argmax_register [0:7];

    // pixel - zero_point : 9 bit co dau (-255..255)
    wire signed [8:0] pixel_minus_zero_point =
        $signed({1'b0, input_pixel_delayed}) -
        $signed({1'b0, input_zero_point_delayed});

    genvar pe_index;
    generate
        for (pe_index = 0; pe_index < 8; pe_index = pe_index + 1) begin : PE_WEIGHT_RAM
            SIMPLE_DUAL_PORT_RAM #(
                .DEPTH(512),
                .ADDR_WIDTH(9)
            ) weight_ram (
                .CLK(CLK),
                .WRITE_ENABLE(WEIGHT_WRITE_ENABLE && (WEIGHT_WRITE_PE_INDEX == pe_index)),
                .WRITE_ADDR(WEIGHT_WRITE_ADDR),
                .WRITE_DATA(WEIGHT_WRITE_DATA),
                .READ_ADDR(WEIGHT_READ_ADDR),
                .READ_DATA(weight_read_data_packed[8*pe_index +: 8])
            );
        end
    endgenerate

    integer pe_loop;
    always @(posedge CLK) begin
        if (RST) begin
            for (pe_loop = 0; pe_loop < 8; pe_loop = pe_loop + 1) begin
                bias_register[pe_loop]        <= 32'sd0;
                accumulator_register[pe_loop] <= 32'sd0;
                max_pool_register[pe_loop]    <= 32'sd0;
                pool_argmax_register[pe_loop] <= 2'd0;
            end
        end else begin
            if (BIAS_WRITE_ENABLE)
                bias_register[BIAS_WRITE_INDEX] <= $signed(BIAS_WRITE_DATA);

            for (pe_loop = 0; pe_loop < 8; pe_loop = pe_loop + 1) begin
                if (load_bias_delayed) begin
                    if (accumulate_enable_delayed)
                        accumulator_register[pe_loop] <=
                            bias_register[pe_loop] +
                            (pixel_minus_zero_point *
                             $signed(weight_read_data_packed[8*pe_loop +: 8]));
                    else
                        accumulator_register[pe_loop] <= bias_register[pe_loop];
                end else if (accumulate_enable_delayed) begin
                    accumulator_register[pe_loop] <=
                        accumulator_register[pe_loop] +
                        (pixel_minus_zero_point *
                         $signed(weight_read_data_packed[8*pe_loop +: 8]));
                end

                // Max-pool tren acc co dau, TRUOC requant
                if (pool_first_position_delayed) begin
                    max_pool_register[pe_loop]    <= accumulator_register[pe_loop];
                    pool_argmax_register[pe_loop] <= pool_position_delayed;
                end else if (pool_next_position_delayed) begin
                    if (accumulator_register[pe_loop] > max_pool_register[pe_loop]) begin
                        max_pool_register[pe_loop]    <= accumulator_register[pe_loop];
                        pool_argmax_register[pe_loop] <= pool_position_delayed;
                    end
                end
            end
        end
    end

    wire [255:0] accumulator_packed = {
        max_pool_register[7], max_pool_register[6], max_pool_register[5], max_pool_register[4],
        max_pool_register[3], max_pool_register[2], max_pool_register[1], max_pool_register[0]
    };

    assign POOL_ARGMAX_CODE = {
        pool_argmax_register[7], pool_argmax_register[6], pool_argmax_register[5], pool_argmax_register[4],
        pool_argmax_register[3], pool_argmax_register[2], pool_argmax_register[1], pool_argmax_register[0]
    };

    REQUANTIZER requantizer_inst (
        .CLK(CLK),
        .RST(RST),
        .ACCUMULATOR_INPUTS(accumulator_packed),
        .LAYER_MULTIPLIER(LAYER_MULTIPLIER),
        .LAYER_SHIFT(LAYER_SHIFT),
        .LAYER_OUTPUT_ZERO_POINT(LAYER_OUTPUT_ZERO_POINT),
        .CLAMP_LOWER(CLAMP_LOWER),
        .START_REQUANT(START_REQUANT),
        .REQUANT_BYTE(REQUANT_BYTE),
        .REQUANT_BYTE_VALID(REQUANT_BYTE_VALID),
        .REQUANT_CHANNEL_IN_GROUP(REQUANT_CHANNEL_IN_GROUP),
        .DONE_REQUANT(DONE_REQUANT)
    );
endmodule
