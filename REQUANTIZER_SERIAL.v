// Requant 1 kenh (ban rut gon cua REQUANTIZER.v) dung cho buoc CAM cua LayerCAM.
// Khac REQUANTIZER 8 kenh: bo vong kenh, zero-point dau ra co dinh = 0, can duoi co dinh = 0 (kep [0, 255]).
//
// Cong thuc (giong REQUANTIZER, muc 4.3 dac ta):
//   product = acc * M0                        (signed 64-bit, nhan shift-add 31 chu ky)
//   value   = (product + (1 << (SHIFT-1))) >>> SHIFT      (SHIFT = 0: khong cong lam tron)
//   y       = clamp(value, 0, 255)
//
// Giao dien:
//   START_REQUANT          : xung 1 chu ky khi dang IDLE.
//   ACCUMULATOR_INPUT      : signed 32 bit; |acc| < 2^31 nen tich khong tran 64 bit.
//   LAYER_MULTIPLIER/SHIFT : phai giu on dinh, cung ACCUMULATOR_INPUT, tu START_REQUANT den DONE_REQUANT.
//   DONE_REQUANT           : xung 1 chu ky; REQUANT_BYTE hop le (va duoc giu) tai chu ky nay.
// Thoi gian: ~36 chu ky tu START_REQUANT den DONE_REQUANT.
module REQUANTIZER_SERIAL (
    input  wire        CLK,
    input  wire        RST,
    input  wire        START_REQUANT,
    input  wire [31:0] ACCUMULATOR_INPUT,          // signed
    input  wire [30:0] LAYER_MULTIPLIER,
    input  wire [5:0]  LAYER_SHIFT,
    output reg  [7:0]  REQUANT_BYTE,
    output reg         DONE_REQUANT
);
    localparam STATE_IDLE     = 3'd0;
    localparam STATE_LOAD     = 3'd1;
    localparam STATE_MULTIPLY = 3'd2;
    localparam STATE_ROUND    = 3'd3;
    localparam STATE_SHIFT    = 3'd4;
    localparam STATE_CLAMP    = 3'd5;
    localparam STATE_DONE     = 3'd6;

    reg [2:0]         requant_state;
    reg [4:0]         multiply_step_index;
    reg signed [63:0] addend_register;
    reg [30:0]        multiplier_remaining;
    reg signed [63:0] product_register;
    reg signed [63:0] rounded_register;
    reg signed [63:0] shifted_register;

    wire signed [31:0] accumulator_signed = ACCUMULATOR_INPUT;

    always @(posedge CLK) begin
        if (RST) begin
            requant_state        <= STATE_IDLE;
            multiply_step_index  <= 5'd0;
            addend_register      <= 64'sd0;
            multiplier_remaining <= 31'd0;
            product_register     <= 64'sd0;
            rounded_register     <= 64'sd0;
            shifted_register     <= 64'sd0;
            REQUANT_BYTE         <= 8'd0;
            DONE_REQUANT         <= 1'b0;
        end else begin
            DONE_REQUANT <= 1'b0;

            case (requant_state)
                STATE_IDLE: begin
                    if (START_REQUANT)
                        requant_state <= STATE_LOAD;
                end

                STATE_LOAD: begin
                    addend_register      <= {{32{accumulator_signed[31]}}, accumulator_signed};
                    multiplier_remaining <= LAYER_MULTIPLIER;
                    product_register     <= 64'sd0;
                    multiply_step_index  <= 5'd0;
                    requant_state        <= STATE_MULTIPLY;
                end

                // product += acc << i  neu bit i cua M0 = 1
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

                // zp_out = 0, can duoi = 0: kep [0, 255]
                STATE_CLAMP: begin
                    if (shifted_register < 64'sd0)
                        REQUANT_BYTE <= 8'd0;
                    else if (shifted_register > 64'sd255)
                        REQUANT_BYTE <= 8'd255;
                    else
                        REQUANT_BYTE <= shifted_register[7:0];
                    requant_state <= STATE_DONE;
                end

                // REQUANT_BYTE da on dinh tu STATE_CLAMP, giu nguyen den lan START ke tiep
                STATE_DONE: begin
                    DONE_REQUANT  <= 1'b1;
                    requant_state <= STATE_IDLE;
                end

                default: requant_state <= STATE_IDLE;
            endcase
        end
    end
endmodule
