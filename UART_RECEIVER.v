// 8N1, lay mau giua bit. 2 FF dong bo tin hieu vao.
module UART_RECEIVER #(
    parameter CLOCKS_PER_BIT = 50
) (
    input  wire       CLK,
    input  wire       RST,
    input  wire       UART_RX,
    output reg  [7:0] RX_BYTE,
    output reg        RX_BYTE_VALID
);
    localparam STATE_IDLE  = 2'd0;
    localparam STATE_START = 2'd1;
    localparam STATE_DATA  = 2'd2;
    localparam STATE_STOP  = 2'd3;

    // Thay the $clog2 (ISE/XST khong ho tro) bang constant function
    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction

    localparam COUNTER_WIDTH = (clog2(CLOCKS_PER_BIT) < 1) ? 1 : clog2(CLOCKS_PER_BIT);

    reg rx_meta, rx_sync;
    always @(posedge CLK) begin
        rx_meta <= UART_RX;
        rx_sync <= rx_meta;
    end

    reg [1:0]                state;
    reg [COUNTER_WIDTH-1:0]  clock_counter;
    reg [2:0]                bit_index;
    reg [7:0]                shift_register;

    always @(posedge CLK) begin
        if (RST) begin
            state            <= STATE_IDLE;
            clock_counter    <= 0;
            bit_index        <= 0;
            shift_register   <= 0;
            RX_BYTE          <= 0;
            RX_BYTE_VALID    <= 1'b0;
        end else begin
            RX_BYTE_VALID <= 1'b0;
            case (state)
                STATE_IDLE: begin
                    if (rx_sync == 1'b0) begin
                        clock_counter <= 0;
                        state         <= STATE_START;
                    end
                end
                STATE_START: begin
                    // Cho nua bit de lay mau giua start bit
                    if (clock_counter == (CLOCKS_PER_BIT/2) - 1) begin
                        clock_counter <= 0;
                        if (rx_sync == 1'b0) begin
                            bit_index <= 0;
                            state     <= STATE_DATA;
                        end else begin
                            state     <= STATE_IDLE;  // nhieu / false start
                        end
                    end else begin
                        clock_counter <= clock_counter + 1'b1;
                    end
                end
                STATE_DATA: begin
                    if (clock_counter == CLOCKS_PER_BIT - 1) begin
                        clock_counter        <= 0;
                        shift_register[bit_index] <= rx_sync;
                        if (bit_index == 3'd7)
                            state <= STATE_STOP;
                        else
                            bit_index <= bit_index + 1'b1;
                    end else begin
                        clock_counter <= clock_counter + 1'b1;
                    end
                end
                STATE_STOP: begin
                    if (clock_counter == CLOCKS_PER_BIT - 1) begin
                        clock_counter <= 0;
                        // Chi phat byte khi stop bit = 1
                        if (rx_sync == 1'b1) begin
                            RX_BYTE       <= shift_register;
                            RX_BYTE_VALID <= 1'b1;
                        end
                        state <= STATE_IDLE;
                    end else begin
                        clock_counter <= clock_counter + 1'b1;
                    end
                end
            endcase
        end
    end
endmodule
