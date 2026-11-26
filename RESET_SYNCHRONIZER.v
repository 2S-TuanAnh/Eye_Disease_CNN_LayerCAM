// Tao RST dong bo, tich cuc muc cao tu nut RESET_N (tich cuc muc thap).
// Nhan nut: RST len 1 ngay (bat dong bo). Nha nut: sau 2 chu ky CLK RST xuong 0.
// SUA: RST phai la "wire" vi duoc gan bang assign (ban cu khai bao "output reg" -> loi).
module RESET_SYNCHRONIZER (
    input  wire CLK,
    input  wire RESET_N,
    output wire RST
);
    reg [1:0] reset_sync_shift;

    always @(posedge CLK or negedge RESET_N) begin
        if (!RESET_N)
            reset_sync_shift <= 2'b11;
        else
            reset_sync_shift <= {reset_sync_shift[0], 1'b0};
    end

    assign RST = reset_sync_shift[1];
endmodule
