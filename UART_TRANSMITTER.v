// 8N1, BUSY_UART_TX len 1 ngay khi bat dau truyen va xuong khi xong STOP.
module UART_TRANSMITTER #(
    parameter CLOCKS_PER_BIT = 50
) (
    input  wire       CLK,
    input  wire       RST,
    input  wire [7:0] TX_BYTE,
    input  wire       TX_BYTE_VALID,
    output reg        UART_TX,
    output reg        BUSY_UART_TX
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

    reg [1:0]                state;
    reg [COUNTER_WIDTH-1:0]  clock_counter;
    reg [2:0]                bit_index;
    reg [7:0]                shift_register;

    always @(posedge CLK) begin
        if (RST) begin
            state         <= STATE_IDLE;
            clock_counter <= 0;
            bit_index     <= 0;
            shift_register<= 0;
            UART_TX       <= 1'b1;
            BUSY_UART_TX  <= 1'b0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    UART_TX      <= 1'b1;
                    BUSY_UART_TX <= 1'b0;
                    if (TX_BYTE_VALID) begin
                        shift_register<= TX_BYTE;
                        clock_counter <= 0;
                        UART_TX       <= 1'b0;   // start bit
                        BUSY_UART_TX  <= 1'b1;
                        state         <= STATE_START;
                    end
                end
                STATE_START: begin
                    if (clock_counter == CLOCKS_PER_BIT - 1) begin
                        clock_counter <= 0;
                        bit_index     <= 0;
                        UART_TX       <= shift_register[0];
                        state         <= STATE_DATA;
                    end else begin
                        clock_counter <= clock_counter + 1'b1;
                    end
                end
                STATE_DATA: begin
                    if (clock_counter == CLOCKS_PER_BIT - 1) begin
                        clock_counter <= 0;
                        if (bit_index == 3'd7) begin
                            UART_TX <= 1'b1;     // stop bit
                            state   <= STATE_STOP;
                        end else begin
                            bit_index <= bit_index + 1'b1;
                            UART_TX   <= shift_register[bit_index + 1'b1];
                        end
                    end else begin
                        clock_counter <= clock_counter + 1'b1;
                    end
                end
                STATE_STOP: begin
                    if (clock_counter == CLOCKS_PER_BIT - 1) begin
                        clock_counter <= 0;
                        UART_TX       <= 1'b1;
                        BUSY_UART_TX  <= 1'b0;
                        state         <= STATE_IDLE;
                    end else begin
                        clock_counter <= clock_counter + 1'b1;
                    end
                end
            endcase
        end
    end
endmodule
