`timescale 1ns/1ps
// Testbench cho CNN_RUNNER (chay rieng, khong qua TOP_CNN).
// Cach dung: sua 4 tham so o dau file (thu muc, ten file anh, lop CAM cho CNN 1 / CNN 2).
// Khong can file cls.hex nua: lop CAM khai bao truc tiep bang tham so CLS_CNN1, CLS_CNN2.
module tb_runner;

    // ------------------------------------------------------------------
    // Tham so nguoi dung chinh
    // ------------------------------------------------------------------
    // Duong dan tuyet doi (dung dau /, khong co dau cach)
    parameter IMG_FILE     = "/home/ise/Xilinx-Project/CNN_Eye_Disease/CNN_Eye_Disease/Test_images/image7_raw64.mem";
    parameter RESULT_FILE  = "/home/ise/Xilinx-Project/CNN_Eye_Disease/CNN_Eye_Disease/Output_RTL/results7.txt";
    // Lop ve CAM: lay theo final_class trong expected_cascade.json
    // Anh 0: CNN1 = 0 (CNN 1 quyet dinh). Anh 5: CNN1 = 4, CNN2 = 1 (CNN 2 quyet dinh)
    parameter [2:0] CLS_CNN1 = 3'd4;
    parameter [2:0] CLS_CNN2 = 3'd3;

    // ------------------------------------------------------------------
    // Clock / reset (50 MHz)
    // ------------------------------------------------------------------
    reg CLK = 0, RST = 1;
    always #10 CLK = ~CLK;

    // ------------------------------------------------------------------
    // Tin hieu noi voi CNN_RUNNER
    // ------------------------------------------------------------------
    reg         START_CNN = 0, CNN_SELECT = 0;
    wire        DONE_CNN;
    wire [2:0]  PREDICTED;
    wire [39:0] LOGITS;
    wire [7:0]  UNKNOWN_IDX, NORMAL_IDX;
    wire        BAD_MAGIC;
    reg         START_LCAM = 0;
    reg  [2:0]  LCAM_CLS = 0;
    wire        DONE_LCAM;
    reg  [7:0]  CAM_ADDR = 0;
    wire [7:0]  CAM_DATA;

    // Anh dau vao: 64x64x3 = 12288 byte, doc dong bo 1 chu ky
    wire [13:0] IMAGE_ADDR;
    reg  [7:0]  IMAGE_DATA;
    reg  [7:0]  img_mem [0:12287];
    always @(posedge CLK) IMAGE_DATA <= img_mem[IMAGE_ADDR];

    // Giao tiep SDRAM (model mo phong, da nap san trong so)
    wire        rq_valid, rq_write, rq_ready, rs_valid;
    wire [23:0] rq_addr;
    wire [15:0] rq_wdata, rs_data;
    wire        init_done;

    SDRAM_SIM_MODEL sdram (
        .CLK(CLK), .RST(RST),
        .REQUEST_VALID(rq_valid), .REQUEST_WRITE(rq_write),
        .REQUEST_WORD_ADDR(rq_addr), .REQUEST_WRITE_DATA(rq_wdata),
        .REQUEST_READY(rq_ready),
        .RESPONSE_VALID(rs_valid), .RESPONSE_READ_DATA(rs_data),
        .INIT_DONE(init_done)
    );

    CNN_RUNNER dut (
        .CLK(CLK), .RST(RST),
        .START_CNN(START_CNN), .CNN_SELECT(CNN_SELECT), .DONE_CNN(DONE_CNN),
        .PREDICTED_CLASS_INDEX(PREDICTED), .LOGITS_OUTPUT(LOGITS),
        .UNKNOWN_CLASS_INDEX(UNKNOWN_IDX), .NORMAL_CLASS_INDEX(NORMAL_IDX),
        .ERROR_BAD_MAGIC(BAD_MAGIC),
        .START_LAYERCAM(START_LCAM), .LAYERCAM_CLASS_INDEX(LCAM_CLS), .DONE_LAYERCAM(DONE_LCAM),
        .CAM_READ_ADDR(CAM_ADDR), .CAM_READ_DATA(CAM_DATA),
        .IMAGE_READ_ADDR(IMAGE_ADDR), .IMAGE_READ_DATA(IMAGE_DATA),
        .SDRAM_REQUEST_VALID(rq_valid), .SDRAM_REQUEST_WRITE(rq_write),
        .SDRAM_REQUEST_WORD_ADDR(rq_addr), .SDRAM_REQUEST_WRITE_DATA(rq_wdata),
        .SDRAM_REQUEST_READY(rq_ready),
        .SDRAM_RESPONSE_VALID(rs_valid), .SDRAM_RESPONSE_READ_DATA(rs_data)
    );

    // ------------------------------------------------------------------
    // Dem chu ky + timeout (40 trieu chu ky ~ 0.8 s o 50 MHz)
    // ------------------------------------------------------------------
    integer cycles = 0, t0, i, sel, f;
    always @(posedge CLK) cycles <= cycles + 1;
    always @(posedge CLK) if (cycles > 40000000) begin
        $display("TIMEOUT");
        $finish;
    end

    // ------------------------------------------------------------------
    // Kich ban: voi moi CNN (sel = 0, 1) chay forward roi LayerCAM
    // ------------------------------------------------------------------
    initial begin
        $readmemh(IMG_FILE, img_mem);
        // Kiem tra da doc duoc file: neu in ra xx thi duong dan sai
        $display("img_mem[0]=%h img_mem[12287]=%h", img_mem[0], img_mem[12287]);

        f = $fopen(RESULT_FILE, "w");
        if (f == 0) begin
            $display("LOI: khong mo duoc %0s de ghi", RESULT_FILE);
            $finish;
        end

        repeat (5) @(posedge CLK);
        RST = 0;
        repeat (5) @(posedge CLK);

        for (sel = 0; sel < 2; sel = sel + 1) begin
            // ---- Forward ----
            CNN_SELECT = sel[0];
            t0 = cycles;
            @(posedge CLK); START_CNN = 1;
            @(posedge CLK); START_CNN = 0;
            while (!DONE_CNN) @(posedge CLK);
            $fdisplay(f, "FWD %0d bad_magic=%0d unknown=%0d normal=%0d pred=%0d logits %0d %0d %0d %0d %0d",
                sel, BAD_MAGIC, UNKNOWN_IDX, NORMAL_IDX, PREDICTED,
                LOGITS[7:0], LOGITS[15:8], LOGITS[23:16], LOGITS[31:24], LOGITS[39:32]);
            $display("sel=%0d forward done in %0d cycles", sel, cycles - t0);
            repeat (3) @(posedge CLK);

            // ---- LayerCAM ----
            LCAM_CLS = (sel == 0) ? CLS_CNN1 : CLS_CNN2;
            t0 = cycles;
            @(posedge CLK); START_LCAM = 1;
            @(posedge CLK); START_LCAM = 0;
            while (!DONE_LCAM) @(posedge CLK);
            $display("sel=%0d layercam done in %0d cycles", sel, cycles - t0);
            repeat (3) @(posedge CLK);

            // ---- Doc 256 gia tri CAM (RAM doc co tre 1-2 chu ky) ----
            $fwrite(f, "CAM %0d", sel);
            for (i = 0; i < 256; i = i + 1) begin
                CAM_ADDR = i[7:0];
                @(posedge CLK); @(posedge CLK); #1;
                $fwrite(f, " %0d", CAM_DATA);
            end
            $fwrite(f, "\n");
        end

        $fclose(f);
        $display("SIM DONE");
        $finish;
    end
endmodule