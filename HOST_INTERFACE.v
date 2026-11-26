// - Nhan: cho 0xA5 roi 12288 byte, ghi vao IMAGE_RAM (co timeout).
// - Gui: 0x55 (ready), 0xEE+err (loi boot), hoac khung ket qua 14 byte (byte 13 = cam_flag);
//   neu cam_flag = 1 gui them 256 byte CAM (doc tu CAM_RAM qua CAM_READ_ADDR/CAM_READ_DATA) -> khung 270 byte.
// - DONE_SEND xung 1 chu ky khi da day het goi tin da yeu cau.
module HOST_INTERFACE (
    input  wire        CLK,
    input  wire        RST,

    // UART RX
    input  wire [7:0]  RX_BYTE,
    input  wire        RX_BYTE_VALID,

    // UART TX
    output reg  [7:0]  TX_BYTE,
    output reg         TX_BYTE_VALID,
    input  wire        BUSY_UART_TX,

    // Ghi IMAGE_RAM
    output reg         IMAGE_WRITE_ENABLE,
    output reg  [13:0] IMAGE_WRITE_ADDR,
    output reg  [7:0]  IMAGE_WRITE_DATA,
    output reg         DONE_IMAGE_RECEIVE,

    // Lenh gui tu TOP
    input  wire        START_SEND_READY,
    input  wire        START_SEND_ERROR,
    input  wire [7:0]  ERROR_CODE,
    input  wire        START_SEND_RESULT,
    input  wire [7:0]  FINAL_SOURCE,
    input  wire [7:0]  FINAL_CLASS,
    input  wire [39:0] LOGITS_CNN1,
    input  wire [39:0] LOGITS_CNN2,
    input  wire        FINAL_CAM_VALID,       // 1 = co 256 byte CAM di kem (phai on dinh cung chu ky voi START_SEND_RESULT)
    input  wire [7:0]  CAM_READ_DATA,         // CAM_RAM tre 1 chu ky
    output wire [7:0]  CAM_READ_ADDR,
    output reg         DONE_SEND
);
    localparam TIMEOUT_WIDTH    = 23;
    localparam TIMEOUT_MAX      = 23'd4_999_999;  // ~100 ms @ 50 MHz
    localparam IMAGE_BYTE_COUNT = 14'd12288;

    localparam STATE_WAIT_SEND_CMD = 3'd0;
    localparam STATE_SENDING       = 3'd1;
    localparam STATE_WAIT_HEADER   = 3'd2;
    localparam STATE_RECEIVING     = 3'd3;

    localparam SEND_IDLE      = 2'd0;
    localparam SEND_BUSY_HIGH = 2'd1;
    localparam SEND_BUSY_LOW  = 2'd2;
    localparam SEND_WAIT_RAM  = 2'd3;      // 1 chu ky cho CAM_RAM tra du lieu sau khi tx_byte_index doi

    localparam [8:0] RESULT_HEADER_BYTES = 9'd14;
    localparam [8:0] RESULT_FULL_BYTES   = 9'd270;   // 14 + 256

    localparam PKT_NONE   = 2'd0;
    localparam PKT_READY  = 2'd1;
    localparam PKT_ERROR  = 2'd2;
    localparam PKT_RESULT = 2'd3;

    reg [2:0]  state;
    reg [1:0]  send_state;
    reg [1:0]  tx_packet_type;
    reg [8:0]  tx_byte_index;
    reg [8:0]  tx_byte_count;
    reg [13:0] image_byte_index;
    reg [TIMEOUT_WIDTH-1:0] timeout_counter;

    reg [7:0]  saved_error_code;
    reg [7:0]  saved_final_source;
    reg [7:0]  saved_final_class;
    reg        saved_final_cam_valid;
    reg [39:0] saved_logits_cnn1;
    reg [39:0] saved_logits_cnn2;

    // Dia chi CAM_RAM = tx_byte_index - 14 (chi 8 bit thap la du: 0..255)
    assign CAM_READ_ADDR = tx_byte_index[7:0] - 8'd14;

    // Byte tiep theo can gui, phu thuoc tx_packet_type + tx_byte_index
    reg [7:0] tx_byte_combinational;
    always @(*) begin
        case (tx_packet_type)
            PKT_READY:  tx_byte_combinational = 8'h55;
            PKT_ERROR: begin
                if (tx_byte_index == 9'd0)
                    tx_byte_combinational = 8'hEE;
                else
                    tx_byte_combinational = saved_error_code;
            end
            PKT_RESULT: begin
                case (tx_byte_index)
                    9'd0:  tx_byte_combinational = 8'hAA;
                    9'd1:  tx_byte_combinational = saved_final_source;
                    9'd2:  tx_byte_combinational = saved_final_class;
                    9'd3:  tx_byte_combinational = saved_logits_cnn1[7:0];
                    9'd4:  tx_byte_combinational = saved_logits_cnn1[15:8];
                    9'd5:  tx_byte_combinational = saved_logits_cnn1[23:16];
                    9'd6:  tx_byte_combinational = saved_logits_cnn1[31:24];
                    9'd7:  tx_byte_combinational = saved_logits_cnn1[39:32];
                    9'd8:  tx_byte_combinational = saved_logits_cnn2[7:0];
                    9'd9:  tx_byte_combinational = saved_logits_cnn2[15:8];
                    9'd10: tx_byte_combinational = saved_logits_cnn2[23:16];
                    9'd11: tx_byte_combinational = saved_logits_cnn2[31:24];
                    9'd12: tx_byte_combinational = saved_logits_cnn2[39:32];
                    9'd13: tx_byte_combinational = {7'd0, saved_final_cam_valid};   // cam_flag
                    default: tx_byte_combinational = (tx_byte_index >= RESULT_HEADER_BYTES) ? CAM_READ_DATA : 8'h00;
                endcase
            end
            default: tx_byte_combinational = 8'h00;
        endcase
    end

    always @(posedge CLK) begin
        if (RST) begin
            state               <= STATE_WAIT_SEND_CMD;
            send_state          <= SEND_IDLE;
            tx_packet_type      <= PKT_NONE;
            tx_byte_index       <= 0;
            tx_byte_count       <= 0;
            TX_BYTE             <= 0;
            TX_BYTE_VALID       <= 1'b0;
            IMAGE_WRITE_ENABLE  <= 1'b0;
            IMAGE_WRITE_ADDR    <= 0;
            IMAGE_WRITE_DATA    <= 0;
            DONE_IMAGE_RECEIVE  <= 1'b0;
            DONE_SEND           <= 1'b0;
            image_byte_index    <= 0;
            timeout_counter     <= 0;
            saved_error_code    <= 0;
            saved_final_source  <= 0;
            saved_final_class   <= 0;
            saved_final_cam_valid <= 1'b0;
            saved_logits_cnn1   <= 0;
            saved_logits_cnn2   <= 0;
        end else begin
            // Xung mac dinh tat
            DONE_IMAGE_RECEIVE <= 1'b0;
            DONE_SEND          <= 1'b0;
            IMAGE_WRITE_ENABLE <= 1'b0;
            TX_BYTE_VALID      <= 1'b0;

            case (state)
                STATE_WAIT_SEND_CMD: begin
                    if (START_SEND_READY) begin
                        tx_packet_type <= PKT_READY;
                        tx_byte_index  <= 0;
                        tx_byte_count  <= 9'd1;
                        send_state     <= SEND_IDLE;
                        state          <= STATE_SENDING;
                    end else if (START_SEND_ERROR) begin
                        tx_packet_type    <= PKT_ERROR;
                        saved_error_code  <= ERROR_CODE;
                        tx_byte_index     <= 0;
                        tx_byte_count     <= 9'd2;
                        send_state        <= SEND_IDLE;
                        state             <= STATE_SENDING;
                    end else if (START_SEND_RESULT) begin
                        tx_packet_type     <= PKT_RESULT;
                        saved_final_source <= FINAL_SOURCE;
                        saved_final_class  <= FINAL_CLASS;
                        saved_final_cam_valid <= FINAL_CAM_VALID;
                        saved_logits_cnn1  <= LOGITS_CNN1;
                        saved_logits_cnn2  <= LOGITS_CNN2;
                        tx_byte_index      <= 0;
                        tx_byte_count      <= FINAL_CAM_VALID ? RESULT_FULL_BYTES : RESULT_HEADER_BYTES;
                        send_state         <= SEND_IDLE;
                        state              <= STATE_SENDING;
                    end else if (RX_BYTE_VALID && RX_BYTE == 8'hA5) begin
                        image_byte_index <= 0;
                        timeout_counter  <= 0;
                        state            <= STATE_RECEIVING;
                    end
                end

                STATE_RECEIVING: begin
                    if (RX_BYTE_VALID) begin
                        IMAGE_WRITE_ENABLE <= 1'b1;
                        IMAGE_WRITE_ADDR   <= image_byte_index;
                        IMAGE_WRITE_DATA   <= RX_BYTE;
                        image_byte_index   <= image_byte_index + 1'b1;
                        timeout_counter    <= 0;
                        if (image_byte_index == IMAGE_BYTE_COUNT - 1) begin
                            DONE_IMAGE_RECEIVE <= 1'b1;
                            state              <= STATE_WAIT_SEND_CMD;
                        end
                    end else begin
                        if (timeout_counter == TIMEOUT_MAX) begin
                            // Bo khung, quay lai cho 0xA5
                            timeout_counter <= 0;
                            state           <= STATE_WAIT_SEND_CMD;
                        end else begin
                            timeout_counter <= timeout_counter + 1'b1;
                        end
                    end
                end

                STATE_SENDING: begin
                    case (send_state)
                        SEND_IDLE: begin
                            if (!BUSY_UART_TX) begin
                                if (tx_byte_index < tx_byte_count) begin
                                    TX_BYTE        <= tx_byte_combinational;
                                    TX_BYTE_VALID  <= 1'b1;
                                    send_state     <= SEND_BUSY_HIGH;
                                end else begin
                                    DONE_SEND <= 1'b1;
                                    state     <= STATE_WAIT_SEND_CMD;
                                end
                            end
                        end
                        SEND_BUSY_HIGH: begin
                            if (BUSY_UART_TX)
                                send_state <= SEND_BUSY_LOW;
                        end
                        SEND_BUSY_LOW: begin
                            if (!BUSY_UART_TX) begin
                                tx_byte_index <= tx_byte_index + 1'b1;
                                send_state    <= SEND_WAIT_RAM;
                            end
                        end
                        // CAM_READ_ADDR vua doi -> CAM_RAM tra du lieu sau 1 chu ky; cho de khong lech 1 byte
                        SEND_WAIT_RAM: begin
                            send_state <= SEND_IDLE;
                        end
                    endcase
                end
            endcase
        end
    end
endmodule
