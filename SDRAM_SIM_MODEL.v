`timescale 1ns/1ps
// CHI DUNG CHO MO PHONG (khong tong hop). Thay the SDRAM_CONTROLLER, cung giao dien nguoi dung.
// Nap noi dung tu sd_image.hex (1 byte / dong): byte chan -> [7:0], byte le -> [15:8].
// Giao thuc: xem PARAMETER_FETCHER.v. REQUEST_READY = 1 khi ranh; sau khi nhan, READY xuong 0 den khi xong.
// Doc: RESPONSE_VALID xung 1 chu ky sau READ_LATENCY_CYCLES chu ky; READY len lai cung luc.
//
// So voi ban cu:
//  - Dung luong theo tham so DEPTH_BYTES (mac dinh 131072, dung voi file hien tai).
//  - Dia chi vuot dung luong KHONG bi cat/quay vong nua: doc ra 0000 va in CANH BAO.
//  - In so byte khac 0 sau khi nap, de biet file da doc duoc hay chua.
// Muon dung luong lon hon: sua DEPTH_BYTES (so byte cua sd_image.hex), ADDR_BITS = log2(DEPTH_BYTES/2).
module SDRAM_SIM_MODEL #(
    parameter HEX_FILE             = "/home/ise/Xilinx-Project/CNN_Eye_Disease/CNN_Eye_Disease/sd_image.hex",   // nen dung duong dan tuyet doi neu ISim khong tim thay
    parameter DEPTH_BYTES          = 131072,           // so byte (so dong) cua file hex
    parameter ADDR_BITS            = 16,               // log2(DEPTH_BYTES / 2)
    parameter READ_LATENCY_CYCLES  = 4,
    parameter WRITE_LATENCY_CYCLES = 3
) (
    input  wire        CLK,
    input  wire        RST,

    input  wire        REQUEST_VALID,
    input  wire        REQUEST_WRITE,
    input  wire [23:0] REQUEST_WORD_ADDR,
    input  wire [15:0] REQUEST_WRITE_DATA,
    output reg         REQUEST_READY,
    output reg         RESPONSE_VALID,
    output reg  [15:0] RESPONSE_READ_DATA,
    output wire        INIT_DONE
);
    localparam DEPTH_WORDS = DEPTH_BYTES / 2;

    assign INIT_DONE = 1'b1;

    reg [7:0]  image_bytes  [0:DEPTH_BYTES-1];
    reg [15:0] memory_words [0:DEPTH_WORDS-1];
    integer    word_loop;
    integer    nonzero_bytes;

    // Nap sd_image.hex -> memory_words
    initial begin
        for (word_loop = 0; word_loop < DEPTH_BYTES; word_loop = word_loop + 1)
            image_bytes[word_loop] = 8'h00;
        $readmemh(HEX_FILE, image_bytes);

        nonzero_bytes = 0;
        for (word_loop = 0; word_loop < DEPTH_BYTES; word_loop = word_loop + 1)
            if (image_bytes[word_loop] != 8'h00)
                nonzero_bytes = nonzero_bytes + 1;
        $display("SDRAM_SIM_MODEL: da nap %0s, %0d/%0d byte khac 0 (byte0=%h byte1=%h)",
                 HEX_FILE, nonzero_bytes, DEPTH_BYTES, image_bytes[0], image_bytes[1]);
        if (nonzero_bytes == 0)
            $display("SDRAM_SIM_MODEL: CANH BAO - file hex rong hoac khong doc duoc, kiem tra duong dan");

        for (word_loop = 0; word_loop < DEPTH_WORDS; word_loop = word_loop + 1)
            memory_words[word_loop] = {image_bytes[2*word_loop + 1], image_bytes[2*word_loop]};
    end

    reg                  busy;
    reg                  pending_read;
    reg                  pending_out_of_range;
    reg [ADDR_BITS-1:0]  pending_word_addr;
    reg [3:0]            latency_counter;

    // Dia chi yeu cau co nam trong bo nho khong
    wire address_in_range = (REQUEST_WORD_ADDR < DEPTH_WORDS);

    always @(posedge CLK) begin
        if (RST) begin
            REQUEST_READY        <= 1'b1;
            RESPONSE_VALID       <= 1'b0;
            RESPONSE_READ_DATA   <= 16'd0;
            busy                 <= 1'b0;
            pending_read         <= 1'b0;
            pending_out_of_range <= 1'b0;
            pending_word_addr    <= {ADDR_BITS{1'b0}};
            latency_counter      <= 4'd0;
        end else begin
            RESPONSE_VALID <= 1'b0;

            if (REQUEST_READY && REQUEST_VALID) begin
                REQUEST_READY        <= 1'b0;
                busy                 <= 1'b1;
                pending_word_addr    <= REQUEST_WORD_ADDR[ADDR_BITS-1:0];
                pending_out_of_range <= ~address_in_range;

                if (!address_in_range)
                    $display("SDRAM_SIM_MODEL: CANH BAO - dia chi %h (word) vuot dung luong %0d word, %0s tra ve 0000 / bo qua",
                             REQUEST_WORD_ADDR, DEPTH_WORDS, REQUEST_WRITE ? "ghi" : "doc");

                if (REQUEST_WRITE) begin
                    if (address_in_range)
                        memory_words[REQUEST_WORD_ADDR[ADDR_BITS-1:0]] <= REQUEST_WRITE_DATA;
                    pending_read    <= 1'b0;
                    latency_counter <= WRITE_LATENCY_CYCLES;
                end else begin
                    pending_read    <= 1'b1;
                    latency_counter <= READ_LATENCY_CYCLES;
                end
            end else if (busy) begin
                if (latency_counter == 4'd0) begin
                    if (pending_read) begin
                        RESPONSE_VALID     <= 1'b1;
                        RESPONSE_READ_DATA <= pending_out_of_range ? 16'h0000
                                                                   : memory_words[pending_word_addr];
                    end
                    busy          <= 1'b0;
                    REQUEST_READY <= 1'b1;
                end else begin
                    latency_counter <= latency_counter - 1'b1;
                end
            end
        end
    end
endmodule
