// Top: cascade 2 CNN tren AX309 (Spartan-6 XC6SLX9). Mot mien clock 50 MHz, reset dong bo.
// Luong: khoi dong SDRAM -> chep the SD vao SDRAM -> gui 0x55 -> cho anh -> CNN 1 -> (CNN 2 neu Unknown)
//        -> (LayerCAM neu co CNN quyet dinh va lop khac Normal) -> gui ket qua (14 byte, hoac 270 byte kem 256 byte CAM).
// Loi khoi dong: gui 0xEE, ma_loi roi dung o STATE_HALT_ERROR (LED[2] sang). Nhan RESET de thu lai.
// LED[0] = da khoi dong xong, LED[1] = dang chay CNN / LayerCAM, LED[2] = loi, LED[3] = dao moi lan co ket qua.
module TOP_CNN #(
    parameter UART_CLOCKS_PER_BIT = 50          // 1 Mbaud; dung 434 de thu 115200
) (
    input  wire        CLK,
    input  wire        RESET_N,

    input  wire        UART_RX,
    output wire        UART_TX,

    output wire [3:0]  LED,

    output wire        SD_CS_N,
    output wire        SD_CLK,
    output wire        SD_MOSI,
    input  wire        SD_MISO,

    output wire        SDRAM_CLK,
    output wire        SDRAM_CKE,
    output wire        SDRAM_CS_N,
    output wire        SDRAM_RAS_N,
    output wire        SDRAM_CAS_N,
    output wire        SDRAM_WE_N,
    output wire [1:0]  SDRAM_DQM,
    output wire [1:0]  SDRAM_BANK,
    output wire [12:0] SDRAM_ADDR,
    inout  wire [15:0] SDRAM_DATA
);
    // ------------------------------------------------------------------
    // Reset
    // ------------------------------------------------------------------
    wire RST;
    RESET_SYNCHRONIZER reset_synchronizer_inst (
        .CLK     (CLK),
        .RESET_N (RESET_N),
        .RST     (RST)
    );

    // ------------------------------------------------------------------
    // UART
    // ------------------------------------------------------------------
    wire [7:0] rx_byte;
    wire       rx_byte_valid;
    wire [7:0] tx_byte;
    wire       tx_byte_valid;
    wire       busy_uart_tx;

    UART_RECEIVER #(.CLOCKS_PER_BIT(UART_CLOCKS_PER_BIT)) uart_receiver_inst (
        .CLK           (CLK),
        .RST           (RST),
        .UART_RX       (UART_RX),
        .RX_BYTE       (rx_byte),
        .RX_BYTE_VALID (rx_byte_valid)
    );

    UART_TRANSMITTER #(.CLOCKS_PER_BIT(UART_CLOCKS_PER_BIT)) uart_transmitter_inst (
        .CLK           (CLK),
        .RST           (RST),
        .TX_BYTE       (tx_byte),
        .TX_BYTE_VALID (tx_byte_valid),
        .UART_TX       (UART_TX),
        .BUSY_UART_TX  (busy_uart_tx)
    );

    // ------------------------------------------------------------------
    // HOST_INTERFACE
    // ------------------------------------------------------------------
    wire        image_write_enable;
    wire [13:0] image_write_addr;
    wire [7:0]  image_write_data;
    wire        done_image_receive;
    wire        done_send;

    reg         start_send_ready;
    reg         start_send_error;
    reg  [7:0]  send_error_code;
    reg         start_send_result;
    reg  [7:0]  final_source;
    reg  [7:0]  final_class;
    reg  [39:0] logits_cnn1;
    reg  [39:0] logits_cnn2;
    reg         final_cam_valid;
    wire [7:0]  cam_read_addr;
    wire [7:0]  cam_read_data;

    HOST_INTERFACE host_interface_inst (
        .CLK                (CLK),
        .RST                (RST),
        .RX_BYTE            (rx_byte),
        .RX_BYTE_VALID      (rx_byte_valid),
        .TX_BYTE            (tx_byte),
        .TX_BYTE_VALID      (tx_byte_valid),
        .BUSY_UART_TX       (busy_uart_tx),
        .IMAGE_WRITE_ENABLE (image_write_enable),
        .IMAGE_WRITE_ADDR   (image_write_addr),
        .IMAGE_WRITE_DATA   (image_write_data),
        .DONE_IMAGE_RECEIVE (done_image_receive),
        .START_SEND_READY   (start_send_ready),
        .START_SEND_ERROR   (start_send_error),
        .ERROR_CODE         (send_error_code),
        .START_SEND_RESULT  (start_send_result),
        .FINAL_SOURCE       (final_source),
        .FINAL_CLASS        (final_class),
        .LOGITS_CNN1        (logits_cnn1),
        .LOGITS_CNN2        (logits_cnn2),
        .FINAL_CAM_VALID    (final_cam_valid),
        .CAM_READ_DATA      (cam_read_data),
        .CAM_READ_ADDR      (cam_read_addr),
        .DONE_SEND          (done_send)
    );

    // ------------------------------------------------------------------
    // IMAGE_RAM (khong bi ghi de cho den het cascade)
    // ------------------------------------------------------------------
    wire [13:0] image_read_addr;
    wire [7:0]  image_read_data;

    SIMPLE_DUAL_PORT_RAM #(.DEPTH(12288), .ADDR_WIDTH(14)) image_ram (
        .CLK          (CLK),
        .WRITE_ENABLE (image_write_enable),
        .WRITE_ADDR   (image_write_addr),
        .WRITE_DATA   (image_write_data),
        .READ_ADDR    (image_read_addr),
        .READ_DATA    (image_read_data)
    );

    // ------------------------------------------------------------------
    // SDRAM_CONTROLLER va mux giua SD_LOADER / CNN_RUNNER
    // ------------------------------------------------------------------
    reg         boot_finished;

    wire        sdram_request_valid;
    wire        sdram_request_write;
    wire [23:0] sdram_request_word_addr;
    wire [15:0] sdram_request_write_data;
    wire        sdram_request_ready;
    wire        sdram_response_valid;
    wire [15:0] sdram_response_read_data;
    wire        sdram_init_done;

    wire        loader_request_valid;
    wire        loader_request_write;
    wire [23:0] loader_request_word_addr;
    wire [15:0] loader_request_write_data;

    wire        runner_request_valid;
    wire        runner_request_write;
    wire [23:0] runner_request_word_addr;
    wire [15:0] runner_request_write_data;

    assign sdram_request_valid      = boot_finished ? runner_request_valid      : loader_request_valid;
    assign sdram_request_write      = boot_finished ? runner_request_write      : loader_request_write;
    assign sdram_request_word_addr  = boot_finished ? runner_request_word_addr  : loader_request_word_addr;
    assign sdram_request_write_data = boot_finished ? runner_request_write_data : loader_request_write_data;

    wire runner_request_ready  = boot_finished  & sdram_request_ready;
    wire loader_request_ready  = ~boot_finished & sdram_request_ready;
    wire runner_response_valid = boot_finished  & sdram_response_valid;
    wire loader_response_valid = ~boot_finished & sdram_response_valid;

    SDRAM_CONTROLLER sdram_controller_inst (
        .CLK                (CLK),
        .RST                (RST),
        .REQUEST_VALID      (sdram_request_valid),
        .REQUEST_WRITE      (sdram_request_write),
        .REQUEST_WORD_ADDR  (sdram_request_word_addr),
        .REQUEST_WRITE_DATA (sdram_request_write_data),
        .REQUEST_READY      (sdram_request_ready),
        .RESPONSE_VALID     (sdram_response_valid),
        .RESPONSE_READ_DATA (sdram_response_read_data),
        .INIT_DONE          (sdram_init_done),
        .SDRAM_CLK          (SDRAM_CLK),
        .SDRAM_CKE          (SDRAM_CKE),
        .SDRAM_CS_N         (SDRAM_CS_N),
        .SDRAM_RAS_N        (SDRAM_RAS_N),
        .SDRAM_CAS_N        (SDRAM_CAS_N),
        .SDRAM_WE_N         (SDRAM_WE_N),
        .SDRAM_DQM          (SDRAM_DQM),
        .SDRAM_BANK         (SDRAM_BANK),
        .SDRAM_ADDR         (SDRAM_ADDR),
        .SDRAM_DATA         (SDRAM_DATA)
    );

    // ------------------------------------------------------------------
    // SD_LOADER
    // ------------------------------------------------------------------
    reg        start_boot_load;
    wire       done_boot_load;
    wire [7:0] boot_error_code;

    SD_LOADER sd_loader_inst (
        .CLK                      (CLK),
        .RST                      (RST),
        .SD_CS_N                  (SD_CS_N),
        .SD_CLK                   (SD_CLK),
        .SD_MOSI                  (SD_MOSI),
        .SD_MISO                  (SD_MISO),
        .START_BOOT_LOAD          (start_boot_load),
        .DONE_BOOT_LOAD           (done_boot_load),
        .ERROR_CODE               (boot_error_code),
        .SDRAM_REQUEST_VALID      (loader_request_valid),
        .SDRAM_REQUEST_WRITE      (loader_request_write),
        .SDRAM_REQUEST_WORD_ADDR  (loader_request_word_addr),
        .SDRAM_REQUEST_WRITE_DATA (loader_request_write_data),
        .SDRAM_REQUEST_READY      (loader_request_ready),
        .SDRAM_RESPONSE_VALID     (loader_response_valid),
        .SDRAM_RESPONSE_READ_DATA (sdram_response_read_data)
    );

    // ------------------------------------------------------------------
    // CNN_RUNNER
    // ------------------------------------------------------------------
    reg        start_cnn;
    reg        cnn_select;
    wire       done_cnn;
    wire [2:0] predicted_class_index;
    wire [39:0] logits_output;
    wire [7:0] unknown_class_index;
    wire [7:0] normal_class_index;
    wire       error_bad_magic;
    reg        start_layercam;
    reg  [2:0] layercam_class_index;
    wire       done_layercam;

    CNN_RUNNER cnn_runner_inst (
        .CLK                      (CLK),
        .RST                      (RST),
        .START_CNN                (start_cnn),
        .CNN_SELECT               (cnn_select),
        .DONE_CNN                 (done_cnn),
        .PREDICTED_CLASS_INDEX    (predicted_class_index),
        .LOGITS_OUTPUT            (logits_output),
        .UNKNOWN_CLASS_INDEX      (unknown_class_index),
        .NORMAL_CLASS_INDEX       (normal_class_index),
        .ERROR_BAD_MAGIC          (error_bad_magic),
        .START_LAYERCAM           (start_layercam),
        .LAYERCAM_CLASS_INDEX     (layercam_class_index),
        .DONE_LAYERCAM            (done_layercam),
        .CAM_READ_ADDR            (cam_read_addr),
        .CAM_READ_DATA            (cam_read_data),
        .IMAGE_READ_ADDR          (image_read_addr),
        .IMAGE_READ_DATA          (image_read_data),
        .SDRAM_REQUEST_VALID      (runner_request_valid),
        .SDRAM_REQUEST_WRITE      (runner_request_write),
        .SDRAM_REQUEST_WORD_ADDR  (runner_request_word_addr),
        .SDRAM_REQUEST_WRITE_DATA (runner_request_write_data),
        .SDRAM_REQUEST_READY      (runner_request_ready),
        .SDRAM_RESPONSE_VALID     (runner_response_valid),
        .SDRAM_RESPONSE_READ_DATA (sdram_response_read_data)
    );

    // ------------------------------------------------------------------
    // FSM cascade
    // ------------------------------------------------------------------
    localparam [3:0] STATE_INIT_SDRAM       = 4'd0;
    localparam [3:0] STATE_COPY_SD_TO_SDRAM = 4'd1;
    localparam [3:0] STATE_SEND_READY       = 4'd2;
    localparam [3:0] STATE_WAIT_IMAGE       = 4'd3;
    localparam [3:0] STATE_RUN_CNN1         = 4'd4;
    localparam [3:0] STATE_CHECK_CNN1       = 4'd5;
    localparam [3:0] STATE_RUN_CNN2         = 4'd6;
    localparam [3:0] STATE_CHECK_CNN2       = 4'd7;
    localparam [3:0] STATE_SEND_RESULT      = 4'd8;
    localparam [3:0] STATE_SEND_ERROR       = 4'd9;
    localparam [3:0] STATE_HALT_ERROR       = 4'd10;
    localparam [3:0] STATE_CHECK_LAYERCAM   = 4'd11;
    localparam [3:0] STATE_RUN_LAYERCAM     = 4'd12;

    reg [3:0]  state;
    reg [7:0]  cnn1_class_index;
    reg [7:0]  cnn1_unknown_class_index;
    reg [7:0]  cnn1_normal_class_index;
    reg [7:0]  cnn2_class_index;
    reg [7:0]  cnn2_unknown_class_index;
    reg [7:0]  cnn2_normal_class_index;
    reg        ready_sent;
    reg        error_latched;
    reg        result_toggle;

    wire cnn1_is_unknown = (cnn1_class_index == cnn1_unknown_class_index);
    wire cnn2_is_unknown = (cnn2_class_index == cnn2_unknown_class_index);

    // CNN quyet dinh (final_source = 1 -> CNN 1, = 2 -> CNN 2); chi co nghia khi final_source != 0
    wire [7:0] decided_normal_class_index = (final_source == 8'h01) ? cnn1_normal_class_index
                                                                     : cnn2_normal_class_index;

    always @(posedge CLK) begin
        if (RST) begin
            state                    <= STATE_INIT_SDRAM;
            boot_finished            <= 1'b0;
            start_boot_load          <= 1'b0;
            start_send_ready         <= 1'b0;
            start_send_error         <= 1'b0;
            send_error_code          <= 8'd0;
            start_send_result        <= 1'b0;
            final_source             <= 8'd0;
            final_class              <= 8'hFF;
            final_cam_valid          <= 1'b0;
            start_layercam           <= 1'b0;
            layercam_class_index     <= 3'd0;
            logits_cnn1              <= 40'd0;
            logits_cnn2              <= 40'd0;
            start_cnn                <= 1'b0;
            cnn_select               <= 1'b0;
            cnn1_class_index         <= 8'd0;
            cnn1_unknown_class_index <= 8'd0;
            cnn1_normal_class_index  <= 8'hFF;
            cnn2_class_index         <= 8'd0;
            cnn2_unknown_class_index <= 8'd0;
            cnn2_normal_class_index  <= 8'hFF;
            ready_sent               <= 1'b0;
            error_latched            <= 1'b0;
            result_toggle            <= 1'b0;
        end else begin
            start_boot_load   <= 1'b0;
            start_send_ready  <= 1'b0;
            start_send_error  <= 1'b0;
            start_send_result <= 1'b0;
            start_cnn         <= 1'b0;
            start_layercam    <= 1'b0;

            case (state)
                STATE_INIT_SDRAM: begin
                    if (sdram_init_done) begin
                        start_boot_load <= 1'b1;
                        state           <= STATE_COPY_SD_TO_SDRAM;
                    end
                end

                STATE_COPY_SD_TO_SDRAM: begin
                    if (done_boot_load) begin
                        if (boot_error_code != 8'd0) begin
                            send_error_code  <= boot_error_code;
                            start_send_error <= 1'b1;
                            error_latched    <= 1'b1;
                            state            <= STATE_SEND_ERROR;
                        end else begin
                            boot_finished    <= 1'b1;
                            start_send_ready <= 1'b1;
                            state            <= STATE_SEND_READY;
                        end
                    end
                end

                STATE_SEND_READY: begin
                    if (done_send) begin
                        ready_sent <= 1'b1;
                        state      <= STATE_WAIT_IMAGE;
                    end
                end

                STATE_SEND_ERROR: begin
                    if (done_send)
                        state <= STATE_HALT_ERROR;
                end

                STATE_HALT_ERROR: begin
                    state <= STATE_HALT_ERROR;
                end

                STATE_WAIT_IMAGE: begin
                    if (done_image_receive) begin
                        final_cam_valid <= 1'b0;
                        logits_cnn1 <= 40'd0;
                        logits_cnn2 <= 40'd0;
                        cnn_select  <= 1'b0;
                        start_cnn   <= 1'b1;
                        state       <= STATE_RUN_CNN1;
                    end
                end

                STATE_RUN_CNN1: begin
                    if (done_cnn) begin
                        logits_cnn1              <= logits_output;
                        cnn1_class_index         <= {5'd0, predicted_class_index};
                        cnn1_unknown_class_index <= unknown_class_index;
                        cnn1_normal_class_index  <= normal_class_index;
                        if (error_bad_magic)
                            error_latched <= 1'b1;
                        state <= STATE_CHECK_CNN1;
                    end
                end

                STATE_CHECK_CNN1: begin
                    if (!cnn1_is_unknown) begin
                        final_source <= 8'h01;
                        final_class  <= cnn1_class_index;
                        state        <= STATE_CHECK_LAYERCAM;
                    end else begin
                        cnn_select <= 1'b1;
                        start_cnn  <= 1'b1;
                        state      <= STATE_RUN_CNN2;
                    end
                end

                STATE_RUN_CNN2: begin
                    if (done_cnn) begin
                        logits_cnn2              <= logits_output;
                        cnn2_class_index         <= {5'd0, predicted_class_index};
                        cnn2_unknown_class_index <= unknown_class_index;
                        cnn2_normal_class_index  <= normal_class_index;
                        if (error_bad_magic)
                            error_latched <= 1'b1;
                        state <= STATE_CHECK_CNN2;
                    end
                end

                STATE_CHECK_CNN2: begin
                    if (!cnn2_is_unknown) begin
                        final_source <= 8'h02;
                        final_class  <= cnn2_class_index;
                    end else begin
                        final_source <= 8'h00;
                        final_class  <= 8'hFF;
                    end
                    state <= STATE_CHECK_LAYERCAM;
                end

                // final_source / final_class da on dinh (gan o buoc truoc). Chay LayerCAM neu co CNN quyet dinh
                // va lop du doan khac Normal cua CNN do. cnn_select giu nguyen (= CNN quyet dinh, chon nen SD).
                STATE_CHECK_LAYERCAM: begin
                    if (final_source != 8'h00 && final_class != decided_normal_class_index) begin
                        start_layercam       <= 1'b1;
                        layercam_class_index <= final_class[2:0];
                        state                <= STATE_RUN_LAYERCAM;
                    end else begin
                        final_cam_valid   <= 1'b0;
                        start_send_result <= 1'b1;
                        state             <= STATE_SEND_RESULT;
                    end
                end

                // final_cam_valid phai gan CUNG canh voi start_send_result (HOST_INTERFACE chot ngay)
                STATE_RUN_LAYERCAM: begin
                    if (done_layercam) begin
                        final_cam_valid   <= 1'b1;
                        start_send_result <= 1'b1;
                        state             <= STATE_SEND_RESULT;
                    end
                end

                STATE_SEND_RESULT: begin
                    if (done_send) begin
                        result_toggle <= ~result_toggle;
                        state         <= STATE_WAIT_IMAGE;
                    end
                end

                default: state <= STATE_INIT_SDRAM;
            endcase
        end
    end

    wire cnn_running = (state == STATE_RUN_CNN1) || (state == STATE_CHECK_CNN1) ||
                       (state == STATE_RUN_CNN2) || (state == STATE_CHECK_CNN2) ||
                       (state == STATE_CHECK_LAYERCAM) || (state == STATE_RUN_LAYERCAM);

    assign LED[0] = ready_sent;
    assign LED[1] = cnn_running;
    assign LED[2] = error_latched;
    assign LED[3] = result_toggle;
endmodule
