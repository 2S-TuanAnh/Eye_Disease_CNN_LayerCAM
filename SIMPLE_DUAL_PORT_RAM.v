// RAM 8-bit, 1 cong ghi + 1 cong doc, doc dong bo tre 1 chu ky.
// XST se infer BRAM (depth lon) hoac distributed RAM (depth nho).
module SIMPLE_DUAL_PORT_RAM #(
    parameter DEPTH      = 12288,
    parameter ADDR_WIDTH = 14
) (
    input  wire                  CLK,
    input  wire                  WRITE_ENABLE,
    input  wire [ADDR_WIDTH-1:0] WRITE_ADDR,
    input  wire [7:0]            WRITE_DATA,
    input  wire [ADDR_WIDTH-1:0] READ_ADDR,
    output reg  [7:0]            READ_DATA
);
    reg [7:0] memory_array [0:DEPTH-1];

    always @(posedge CLK) begin
        if (WRITE_ENABLE)
            memory_array[WRITE_ADDR] <= WRITE_DATA;
        READ_DATA <= memory_array[READ_ADDR];
    end
endmodule
