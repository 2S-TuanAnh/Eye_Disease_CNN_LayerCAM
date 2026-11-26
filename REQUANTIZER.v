// Requant 8 kenh, xu ly lan luot tung kenh, moi kenh ~35 chu ky.
// SUA so voi ban cu: ban cu nhan 64x64 bit to hop trong 1 chu ky (khong dat 50 MHz, ngon DSP48A1
// cua XC6SLX9). Ban nay dung nhan shift-add 31 chu ky + chia nho phep tinh thanh nhieu tang thanh ghi.
// Giao dien (cong) giu nguyen nen TB_REQUANTIZER.v dung duoc khong can sua.
//
// Cong thuc (muc 4.3 dac ta):
//   product = acc * M0                        (signed 64-bit)
//   value   = (product + (1 << (SHIFT-1))) >>> SHIFT      (SHIFT = 0: khong cong lam tron)
//   y       = value + ZP_OUT
//   ket qua = clamp(y, CLAMP_LOWER, 255)   (CLAMP_LOWER do CNN_RUNNER chon: ZP_OUT neu ReLU, 0 neu khong, 1 cho lop nguoc LayerCAM)
//
// Yeu cau: ACCUMULATOR_INPUTS va cac LAYER_* phai giu on dinh tu START_REQUANT den DONE_REQUANT.
// REQUANT_BYTE_VALID len 1 chu ky cho moi kenh (kenh 0..7, theo REQUANT_CHANNEL_IN_GROUP).
// DONE_REQUANT len 1 chu ky, sau byte hop le cuoi cung 1 chu ky.
module REQUANTIZER (
    input  wire         CLK,
    input  wire         RST,
    input  wire [255:0] ACCUMULATOR_INPUTS,        // 8 x 32-bit signed, kenh k = [32*k +: 32]
    input  wire [30:0]  LAYER_MULTIPLIER,
    input  wire [5:0]   LAYER_SHIFT,
    input  wire [7:0]   LAYER_OUTPUT_ZERO_POINT,
    input  wire [7:0]   CLAMP_LOWER,
    input  wire         START_REQUANT,
    output reg  [7:0]   REQUANT_BYTE,
    output reg          REQUANT_BYTE_VALID,
    output reg  [2:0]   REQUANT_CHANNEL_IN_GROUP,
    output reg          DONE_REQUANT
);
    localparam STATE_IDLE     = 3'd0;
    localparam STATE_SELECT   = 3'd1;
    localparam STATE_MULTIPLY = 3'd2;
    localparam STATE_ROUND    = 3'd3;
    localparam STATE_SHIFT    = 3'd4;
    localparam STATE_CLAMP    = 3'd5;
    localparam STATE_DONE     = 3'd6;

    reg [2:0]         requant_state;
    reg [2:0]         requant_channel;
    reg [4:0]         multiply_step_index;
    reg signed [63:0] addend_register;
    reg [30:0]        multiplier_remaining;
    reg signed [63:0] product_register;
    reg signed [63:0] rounded_register;
    reg signed [63:0] shifted_register;

    wire signed [31:0] accumulator_selected =
        ACCUMULATOR_INPUTS[32*requant_channel +: 32];

    // Tang cuoi: cong zero-point + kep
    wire signed [63:0] zero_point_wide  = $signed({56'd0, LAYER_OUTPUT_ZERO_POINT});
    wire signed [63:0] with_zero_point  = shifted_register + zero_point_wide;
    wire signed [63:0] lower_bound      = $signed({56'd0, CLAMP_LOWER});

    always @(posedge CLK) begin
        if (RST) begin
            requant_state            <= STATE_IDLE;
            requant_channel          <= 3'd0;
            multiply_step_index      <= 5'd0;
            addend_register          <= 64'sd0;
            multiplier_remaining     <= 31'd0;
            product_register         <= 64'sd0;
            rounded_register         <= 64'sd0;
            shifted_register         <= 64'sd0;
            REQUANT_BYTE             <= 8'd0;
            REQUANT_BYTE_VALID       <= 1'b0;
            REQUANT_CHANNEL_IN_GROUP <= 3'd0;
            DONE_REQUANT             <= 1'b0;
        end else begin
            REQUANT_BYTE_VALID <= 1'b0;
            DONE_REQUANT       <= 1'b0;

            case (requant_state)
                STATE_IDLE: begin
                    if (START_REQUANT) begin
                        requant_channel <= 3'd0;
                        requant_state   <= STATE_SELECT;
                    end
                end

                STATE_SELECT: begin
                    addend_register      <= {{32{accumulator_selected[31]}}, accumulator_selected};
                    multiplier_remaining <= LAYER_MULTIPLIER;
                    product_register     <= 64'sd0;
                    multiply_step_index  <= 5'd0;
                    requant_state        <= STATE_MULTIPLY;
                end

                // product += acc << i  neu bit i cua M0 = 1. |acc| < 2^31 nen khong tran 64 bit.
                STATE_MULTIPLY: begin
                    if (multiplier_remaining[0])
                        product_register <= product_register + addend_register;
                    addend_register      <= addend_register <<< 1;
                    multiplier_remaining <= {1'b0, multiplier_remaining[30:1]};
                    if (multiply_step_index == 5'd30)
                        requant_state <= STATE_ROUND;
                    else
                        multiply_step_index <= multiply_step_index + 1'b1;
                end

                STATE_ROUND: begin
                    if (LAYER_SHIFT == 6'd0)
                        rounded_register <= product_register;
                    else
                        rounded_register <= product_register +
                                            (64'sd1 <<< (LAYER_SHIFT - 6'd1));
                    requant_state <= STATE_SHIFT;
                end

                STATE_SHIFT: begin
                    shifted_register <= rounded_register >>> LAYER_SHIFT;   // dich so hoc
                    requant_state    <= STATE_CLAMP;
                end

                STATE_CLAMP: begin
                    if (with_zero_point < lower_bound)
                        REQUANT_BYTE <= lower_bound[7:0];
                    else if (with_zero_point > 64'sd255)
                        REQUANT_BYTE <= 8'd255;
                    else
                        REQUANT_BYTE <= with_zero_point[7:0];
                    REQUANT_BYTE_VALID       <= 1'b1;
                    REQUANT_CHANNEL_IN_GROUP <= requant_channel;
                    if (requant_channel == 3'd7) begin
                        requant_state <= STATE_DONE;
                    end else begin
                        requant_channel <= requant_channel + 1'b1;
                        requant_state   <= STATE_SELECT;
                    end
                end

                STATE_DONE: begin
                    DONE_REQUANT  <= 1'b1;
                    requant_state <= STATE_IDLE;
                end

                default: requant_state <= STATE_IDLE;
            endcase
        end
    end
endmodule
