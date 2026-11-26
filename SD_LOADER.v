// Chep 256 sector tho (131.072 byte) tu the SDHC qua SPI vao SDRAM, bat dau tu tu 0.
// Chay mot lan luc khoi dong: xung START_BOOT_LOAD -> xung DONE_BOOT_LOAD.
// ERROR_CODE (giu muc, hop le khi DONE_BOOT_LOAD): 0 = OK, 1 = SD init loi, 2 = doc sector loi, 3 = sai magic "FCNN".
// Byte chan nam o [7:0] cua tu 16 bit, byte le o [15:8]. Tu thu k cua sector s nam o dia chi tu s*256 + k.
// Kiem tra magic tai byte 0..3 cua sector 0 (CNN 1) va sector 128 (CNN 2).
// SPI mode 0, MSB truoc. Giai doan init ~400 kHz, giai doan doc ~6 MHz.
module SD_LOADER #(
    parameter SLOW_CLOCK_DIVIDER   = 61,       // nua chu ky SD_CLK = (DIVIDER + 1) chu ky CLK -> ~403 kHz
    parameter FAST_CLOCK_DIVIDER   = 3,        // -> 6.25 MHz
    parameter DUMMY_BYTE_COUNT     = 10,       // 80 xung clock khi CS = 1
    parameter TOTAL_SECTOR_COUNT   = 256,
    parameter ACMD41_RETRY_LIMIT   = 4095,
    parameter CMD0_RETRY_LIMIT     = 10,
    parameter TOKEN_WAIT_LIMIT     = 100000
) (
    input  wire        CLK,
    input  wire        RST,

    output wire        SD_CS_N,
    output wire        SD_CLK,
    output wire        SD_MOSI,
    input  wire        SD_MISO,

    input  wire        START_BOOT_LOAD,
    output reg         DONE_BOOT_LOAD,
    output reg  [7:0]  ERROR_CODE,

    output reg         SDRAM_REQUEST_VALID,
    output wire        SDRAM_REQUEST_WRITE,
    output reg  [23:0] SDRAM_REQUEST_WORD_ADDR,
    output reg  [15:0] SDRAM_REQUEST_WRITE_DATA,
    input  wire        SDRAM_REQUEST_READY,
    input  wire        SDRAM_RESPONSE_VALID,
    input  wire [15:0] SDRAM_RESPONSE_READ_DATA
);
    assign SDRAM_REQUEST_WRITE = 1'b1;

    localparam [7:0] CODE_NONE               = 8'd0;
    localparam [7:0] CODE_SD_INIT_FAILED     = 8'd1;
    localparam [7:0] CODE_SECTOR_READ_FAILED = 8'd2;
    localparam [7:0] CODE_BAD_MAGIC          = 8'd3;

    // ------------------------------------------------------------------
    // Dong bo MISO
    // ------------------------------------------------------------------
    reg miso_meta, miso_sync;
    always @(posedge CLK) begin
        miso_meta <= SD_MISO;
        miso_sync <= miso_meta;
    end

    // ------------------------------------------------------------------
    // Bo truyen 1 byte SPI mode 0
    // ------------------------------------------------------------------
    reg        spi_start;
    reg  [7:0] spi_tx_byte;
    reg  [7:0] spi_rx_byte;
    reg        spi_done;
    reg        spi_busy;
    reg  [7:0] spi_shift_out;
    reg  [7:0] spi_shift_in;
    reg  [3:0] spi_bit_count;
    reg  [7:0] spi_clock_count;
    reg  [7:0] spi_clock_divider;
    reg        spi_clock_level;
    reg        spi_mosi_level;
    reg        spi_cs_n_level;

    assign SD_CLK  = spi_clock_level;
    assign SD_MOSI = spi_mosi_level;
    assign SD_CS_N = spi_cs_n_level;

    always @(posedge CLK) begin
        if (RST) begin
            spi_busy        <= 1'b0;
            spi_done        <= 1'b0;
            spi_rx_byte     <= 8'd0;
            spi_shift_out   <= 8'd0;
            spi_shift_in    <= 8'd0;
            spi_bit_count   <= 4'd0;
            spi_clock_count <= 8'd0;
            spi_clock_level <= 1'b0;
            spi_mosi_level  <= 1'b1;
        end else begin
            spi_done <= 1'b0;
            if (!spi_busy) begin
                if (spi_start) begin
                    spi_busy        <= 1'b1;
                    spi_shift_out   <= spi_tx_byte;
                    spi_mosi_level  <= spi_tx_byte[7];
                    spi_bit_count   <= 4'd0;
                    spi_clock_count <= 8'd0;
                    spi_clock_level <= 1'b0;
                end
            end else begin
                if (spi_clock_count == spi_clock_divider) begin
                    spi_clock_count <= 8'd0;
                    if (!spi_clock_level) begin
                        // suon len: lay mau MISO
                        spi_clock_level <= 1'b1;
                        spi_shift_in    <= {spi_shift_in[6:0], miso_sync};
                    end else begin
                        // suon xuong: doi bit MOSI
                        spi_clock_level <= 1'b0;
                        spi_bit_count   <= spi_bit_count + 1'b1;
                        if (spi_bit_count == 4'd7) begin
                            spi_busy       <= 1'b0;
                            spi_done       <= 1'b1;
                            spi_rx_byte    <= spi_shift_in;
                            spi_mosi_level <= 1'b1;
                        end else begin
                            spi_shift_out  <= {spi_shift_out[6:0], 1'b1};
                            spi_mosi_level <= spi_shift_out[6];
                        end
                    end
                end else begin
                    spi_clock_count <= spi_clock_count + 1'b1;
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // FSM chinh
    // ------------------------------------------------------------------
    localparam [4:0] STATE_IDLE                  = 5'd0;
    localparam [4:0] STATE_SPI_WAIT              = 5'd1;
    localparam [4:0] STATE_DUMMY_CLOCKS          = 5'd2;
    localparam [4:0] STATE_LOAD_COMMAND          = 5'd3;
    localparam [4:0] STATE_COMMAND_BYTE          = 5'd4;
    localparam [4:0] STATE_COMMAND_RESPONSE_POLL = 5'd5;
    localparam [4:0] STATE_COMMAND_RESPONSE_CHECK = 5'd6;
    localparam [4:0] STATE_COMMAND_EXTRA         = 5'd7;
    localparam [4:0] STATE_COMMAND_EXTRA_STORE   = 5'd8;
    localparam [4:0] STATE_COMMAND_FINISH        = 5'd9;
    localparam [4:0] STATE_AFTER_COMMAND         = 5'd10;
    localparam [4:0] STATE_START_SECTOR_READ     = 5'd11;
    localparam [4:0] STATE_TOKEN_POLL            = 5'd12;
    localparam [4:0] STATE_TOKEN_CHECK           = 5'd13;
    localparam [4:0] STATE_DATA_BYTE_SEND        = 5'd14;
    localparam [4:0] STATE_DATA_BYTE_STORE       = 5'd15;
    localparam [4:0] STATE_SDRAM_REQUEST         = 5'd16;
    localparam [4:0] STATE_CRC_BYTE              = 5'd17;
    localparam [4:0] STATE_SECTOR_FINISH         = 5'd18;
    localparam [4:0] STATE_SECTOR_NEXT           = 5'd19;
    localparam [4:0] STATE_FINISH                = 5'd20;
    localparam [4:0] STATE_ERROR                 = 5'd21;

    localparam [2:0] KIND_CMD0   = 3'd0;
    localparam [2:0] KIND_CMD8   = 3'd1;
    localparam [2:0] KIND_CMD55  = 3'd2;
    localparam [2:0] KIND_ACMD41 = 3'd3;
    localparam [2:0] KIND_CMD58  = 3'd4;
    localparam [2:0] KIND_CMD17  = 3'd5;

    localparam PHASE_INIT = 1'b0;
    localparam PHASE_READ = 1'b1;

    reg [4:0]  state;
    reg [4:0]  spi_next_state;
    reg [4:0]  after_sdram_state;

    reg        loader_phase;
    reg [2:0]  init_step;
    reg [2:0]  command_kind;
    reg [5:0]  command_index;
    reg [31:0] command_argument;
    reg [7:0]  command_crc;
    reg [2:0]  command_extra_count;
    reg        command_keep_cs;
    reg [2:0]  command_byte_index;
    reg [4:0]  response_poll_count;
    reg [2:0]  extra_byte_index;
    reg [7:0]  response_r1;
    reg [31:0] response_extra;

    reg [3:0]  dummy_byte_counter;
    reg [3:0]  cmd0_retry_count;
    reg [12:0] acmd41_retry_count;
    reg [16:0] token_poll_count;

    reg [7:0]  sector_index;
    reg [8:0]  data_byte_index;
    reg [7:0]  low_byte_register;
    reg        crc_byte_count;
    reg        magic_mismatch;
    reg [7:0]  pending_error_code;

    wire [7:0] phase_error_code = (loader_phase == PHASE_INIT) ? CODE_SD_INIT_FAILED : CODE_SECTOR_READ_FAILED;

    reg [7:0] command_byte_to_send;
    always @(*) begin
        case (command_byte_index)
            3'd0:    command_byte_to_send = {2'b01, command_index};
            3'd1:    command_byte_to_send = command_argument[31:24];
            3'd2:    command_byte_to_send = command_argument[23:16];
            3'd3:    command_byte_to_send = command_argument[15:8];
            3'd4:    command_byte_to_send = command_argument[7:0];
            default: command_byte_to_send = command_crc;
        endcase
    end

    function [7:0] expected_magic_byte;
        input [1:0] index;
        begin
            case (index)
                2'd0:    expected_magic_byte = 8'h46;   // 'F'
                2'd1:    expected_magic_byte = 8'h43;   // 'C'
                default: expected_magic_byte = 8'h4E;   // 'N'
            endcase
        end
    endfunction

    wire sector_has_magic = (sector_index == 8'd0) || (sector_index == 8'd128);

    always @(posedge CLK) begin
        if (RST) begin
            state                    <= STATE_IDLE;
            spi_next_state           <= STATE_IDLE;
            after_sdram_state        <= STATE_IDLE;
            spi_start                <= 1'b0;
            spi_tx_byte              <= 8'hFF;
            spi_clock_divider        <= SLOW_CLOCK_DIVIDER;
            spi_cs_n_level           <= 1'b1;
            DONE_BOOT_LOAD           <= 1'b0;
            ERROR_CODE               <= CODE_NONE;
            SDRAM_REQUEST_VALID      <= 1'b0;
            SDRAM_REQUEST_WORD_ADDR  <= 24'd0;
            SDRAM_REQUEST_WRITE_DATA <= 16'd0;
            loader_phase             <= PHASE_INIT;
            init_step                <= 3'd0;
            command_kind             <= KIND_CMD0;
            command_index            <= 6'd0;
            command_argument         <= 32'd0;
            command_crc              <= 8'hFF;
            command_extra_count      <= 3'd0;
            command_keep_cs          <= 1'b0;
            command_byte_index       <= 3'd0;
            response_poll_count      <= 5'd0;
            extra_byte_index         <= 3'd0;
            response_r1              <= 8'd0;
            response_extra           <= 32'd0;
            dummy_byte_counter       <= 4'd0;
            cmd0_retry_count         <= 4'd0;
            acmd41_retry_count       <= 13'd0;
            token_poll_count         <= 17'd0;
            sector_index             <= 8'd0;
            data_byte_index          <= 9'd0;
            low_byte_register        <= 8'd0;
            crc_byte_count           <= 1'b0;
            magic_mismatch           <= 1'b0;
            pending_error_code       <= CODE_NONE;
        end else begin
            spi_start      <= 1'b0;
            DONE_BOOT_LOAD <= 1'b0;

            case (state)
                STATE_IDLE: begin
                    if (START_BOOT_LOAD) begin
                        ERROR_CODE         <= CODE_NONE;
                        spi_clock_divider  <= SLOW_CLOCK_DIVIDER;
                        spi_cs_n_level     <= 1'b1;
                        dummy_byte_counter <= 4'd0;
                        loader_phase       <= PHASE_INIT;
                        init_step          <= 3'd0;
                        cmd0_retry_count   <= 4'd0;
                        acmd41_retry_count <= 13'd0;
                        sector_index       <= 8'd0;
                        magic_mismatch     <= 1'b0;
                        state              <= STATE_DUMMY_CLOCKS;
                    end
                end

                // Cho bo truyen SPI xong 1 byte roi nhay sang spi_next_state (spi_rx_byte da hop le)
                STATE_SPI_WAIT: begin
                    if (spi_done)
                        state <= spi_next_state;
                end

                STATE_DUMMY_CLOCKS: begin
                    if (dummy_byte_counter == DUMMY_BYTE_COUNT) begin
                        command_kind <= KIND_CMD0;
                        state        <= STATE_LOAD_COMMAND;
                    end else begin
                        dummy_byte_counter <= dummy_byte_counter + 1'b1;
                        spi_tx_byte        <= 8'hFF;
                        spi_start          <= 1'b1;
                        spi_next_state     <= STATE_DUMMY_CLOCKS;
                        state              <= STATE_SPI_WAIT;
                    end
                end

                // ---------------------------------------------- gui lenh SD
                STATE_LOAD_COMMAND: begin
                    command_keep_cs <= 1'b0;
                    case (command_kind)
                        KIND_CMD0: begin
                            command_index       <= 6'd0;
                            command_argument    <= 32'h0000_0000;
                            command_crc         <= 8'h95;
                            command_extra_count <= 3'd0;
                        end
                        KIND_CMD8: begin
                            command_index       <= 6'd8;
                            command_argument    <= 32'h0000_01AA;
                            command_crc         <= 8'h87;
                            command_extra_count <= 3'd4;
                        end
                        KIND_CMD55: begin
                            command_index       <= 6'd55;
                            command_argument    <= 32'h0000_0000;
                            command_crc         <= 8'hFF;
                            command_extra_count <= 3'd0;
                        end
                        KIND_ACMD41: begin
                            command_index       <= 6'd41;
                            command_argument    <= 32'h4000_0000;
                            command_crc         <= 8'hFF;
                            command_extra_count <= 3'd0;
                        end
                        KIND_CMD58: begin
                            command_index       <= 6'd58;
                            command_argument    <= 32'h0000_0000;
                            command_crc         <= 8'hFF;
                            command_extra_count <= 3'd4;
                        end
                        default: begin   // KIND_CMD17: doc 1 sector (SDHC: dia chi theo sector)
                            command_index       <= 6'd17;
                            command_argument    <= {24'd0, sector_index};
                            command_crc         <= 8'hFF;
                            command_extra_count <= 3'd0;
                            command_keep_cs     <= 1'b1;
                        end
                    endcase
                    spi_cs_n_level      <= 1'b0;
                    command_byte_index  <= 3'd0;
                    response_poll_count <= 5'd0;
                    state               <= STATE_COMMAND_BYTE;
                end

                STATE_COMMAND_BYTE: begin
                    spi_tx_byte        <= command_byte_to_send;
                    spi_start          <= 1'b1;
                    command_byte_index <= command_byte_index + 1'b1;
                    spi_next_state     <= (command_byte_index == 3'd5) ? STATE_COMMAND_RESPONSE_POLL
                                                                        : STATE_COMMAND_BYTE;
                    state              <= STATE_SPI_WAIT;
                end

                STATE_COMMAND_RESPONSE_POLL: begin
                    spi_tx_byte    <= 8'hFF;
                    spi_start      <= 1'b1;
                    spi_next_state <= STATE_COMMAND_RESPONSE_CHECK;
                    state          <= STATE_SPI_WAIT;
                end

                STATE_COMMAND_RESPONSE_CHECK: begin
                    if (!spi_rx_byte[7]) begin
                        response_r1      <= spi_rx_byte;
                        extra_byte_index <= 3'd0;
                        response_extra   <= 32'd0;
                        if (command_extra_count == 3'd0)
                            state <= STATE_COMMAND_FINISH;
                        else
                            state <= STATE_COMMAND_EXTRA;
                    end else if (response_poll_count == 5'd15) begin
                        pending_error_code <= phase_error_code;
                        state              <= STATE_ERROR;
                    end else begin
                        response_poll_count <= response_poll_count + 1'b1;
                        state               <= STATE_COMMAND_RESPONSE_POLL;
                    end
                end

                STATE_COMMAND_EXTRA: begin
                    spi_tx_byte    <= 8'hFF;
                    spi_start      <= 1'b1;
                    spi_next_state <= STATE_COMMAND_EXTRA_STORE;
                    state          <= STATE_SPI_WAIT;
                end

                STATE_COMMAND_EXTRA_STORE: begin
                    response_extra   <= {response_extra[23:0], spi_rx_byte};
                    extra_byte_index <= extra_byte_index + 1'b1;
                    if ((extra_byte_index + 1'b1) == command_extra_count)
                        state <= STATE_COMMAND_FINISH;
                    else
                        state <= STATE_COMMAND_EXTRA;
                end

                STATE_COMMAND_FINISH: begin
                    if (command_keep_cs) begin
                        state <= STATE_AFTER_COMMAND;
                    end else begin
                        spi_cs_n_level <= 1'b1;
                        spi_tx_byte    <= 8'hFF;      // 8 xung clock sau khi nha CS
                        spi_start      <= 1'b1;
                        spi_next_state <= STATE_AFTER_COMMAND;
                        state          <= STATE_SPI_WAIT;
                    end
                end

                STATE_AFTER_COMMAND: begin
                    if (loader_phase == PHASE_INIT) begin
                        case (init_step)
                            3'd0: begin   // CMD0: mong R1 = 0x01 (idle)
                                if (response_r1 == 8'h01) begin
                                    init_step    <= 3'd1;
                                    command_kind <= KIND_CMD8;
                                    state        <= STATE_LOAD_COMMAND;
                                end else if (cmd0_retry_count < CMD0_RETRY_LIMIT) begin
                                    cmd0_retry_count <= cmd0_retry_count + 1'b1;
                                    command_kind     <= KIND_CMD0;
                                    state            <= STATE_LOAD_COMMAND;
                                end else begin
                                    pending_error_code <= CODE_SD_INIT_FAILED;
                                    state              <= STATE_ERROR;
                                end
                            end
                            3'd1: begin   // CMD8: R1 = 0x01 va echo 0x1AA
                                if (response_r1 == 8'h01 && response_extra[15:0] == 16'h01AA) begin
                                    init_step    <= 3'd2;
                                    command_kind <= KIND_CMD55;
                                    state        <= STATE_LOAD_COMMAND;
                                end else begin
                                    pending_error_code <= CODE_SD_INIT_FAILED;
                                    state              <= STATE_ERROR;
                                end
                            end
                            3'd2: begin   // CMD55 xong -> ACMD41
                                if (response_r1 <= 8'h01) begin
                                    init_step    <= 3'd3;
                                    command_kind <= KIND_ACMD41;
                                    state        <= STATE_LOAD_COMMAND;
                                end else begin
                                    pending_error_code <= CODE_SD_INIT_FAILED;
                                    state              <= STATE_ERROR;
                                end
                            end
                            3'd3: begin   // ACMD41: 0x00 = san sang, 0x01 = lap lai
                                if (response_r1 == 8'h00) begin
                                    init_step    <= 3'd4;
                                    command_kind <= KIND_CMD58;
                                    state        <= STATE_LOAD_COMMAND;
                                end else if (response_r1 == 8'h01 && acmd41_retry_count < ACMD41_RETRY_LIMIT) begin
                                    acmd41_retry_count <= acmd41_retry_count + 1'b1;
                                    init_step          <= 3'd2;
                                    command_kind       <= KIND_CMD55;
                                    state              <= STATE_LOAD_COMMAND;
                                end else begin
                                    pending_error_code <= CODE_SD_INIT_FAILED;
                                    state              <= STATE_ERROR;
                                end
                            end
                            default: begin   // CMD58: OCR bit 30 (CCS) phai = 1 (SDHC)
                                if (response_r1 <= 8'h01 && response_extra[30]) begin
                                    spi_clock_divider <= FAST_CLOCK_DIVIDER;
                                    loader_phase      <= PHASE_READ;
                                    sector_index      <= 8'd0;
                                    state             <= STATE_START_SECTOR_READ;
                                end else begin
                                    pending_error_code <= CODE_SD_INIT_FAILED;
                                    state              <= STATE_ERROR;
                                end
                            end
                        endcase
                    end else begin
                        // CMD17: R1 phai = 0x00
                        if (response_r1 == 8'h00) begin
                            token_poll_count <= 17'd0;
                            state            <= STATE_TOKEN_POLL;
                        end else begin
                            pending_error_code <= CODE_SECTOR_READ_FAILED;
                            state              <= STATE_ERROR;
                        end
                    end
                end

                // ---------------------------------------------- doc sector
                STATE_START_SECTOR_READ: begin
                    command_kind <= KIND_CMD17;
                    state        <= STATE_LOAD_COMMAND;
                end

                STATE_TOKEN_POLL: begin
                    spi_tx_byte    <= 8'hFF;
                    spi_start      <= 1'b1;
                    spi_next_state <= STATE_TOKEN_CHECK;
                    state          <= STATE_SPI_WAIT;
                end

                STATE_TOKEN_CHECK: begin
                    if (spi_rx_byte == 8'hFE) begin
                        data_byte_index <= 9'd0;
                        state           <= STATE_DATA_BYTE_SEND;
                    end else if (spi_rx_byte == 8'hFF && token_poll_count < TOKEN_WAIT_LIMIT) begin
                        token_poll_count <= token_poll_count + 1'b1;
                        state            <= STATE_TOKEN_POLL;
                    end else begin
                        pending_error_code <= CODE_SECTOR_READ_FAILED;
                        state              <= STATE_ERROR;
                    end
                end

                STATE_DATA_BYTE_SEND: begin
                    spi_tx_byte    <= 8'hFF;
                    spi_start      <= 1'b1;
                    spi_next_state <= STATE_DATA_BYTE_STORE;
                    state          <= STATE_SPI_WAIT;
                end

                STATE_DATA_BYTE_STORE: begin
                    if (sector_has_magic && data_byte_index < 9'd4) begin
                        if (spi_rx_byte != expected_magic_byte(data_byte_index[1:0]))
                            magic_mismatch <= 1'b1;
                    end

                    data_byte_index <= data_byte_index + 1'b1;   // 511 -> 0 (tran so)

                    if (!data_byte_index[0]) begin
                        // byte chan: giu lai
                        low_byte_register <= spi_rx_byte;
                        state             <= STATE_DATA_BYTE_SEND;
                    end else begin
                        // byte le: ghep thanh tu va ghi SDRAM
                        SDRAM_REQUEST_WORD_ADDR  <= {8'd0, sector_index, data_byte_index[8:1]};
                        SDRAM_REQUEST_WRITE_DATA <= {spi_rx_byte, low_byte_register};
                        after_sdram_state        <= (data_byte_index == 9'd511) ? STATE_CRC_BYTE
                                                                                : STATE_DATA_BYTE_SEND;
                        crc_byte_count           <= 1'b0;
                        state                    <= STATE_SDRAM_REQUEST;
                    end
                end

                STATE_SDRAM_REQUEST: begin
                    SDRAM_REQUEST_VALID <= 1'b1;
                    if (SDRAM_REQUEST_VALID && SDRAM_REQUEST_READY) begin
                        SDRAM_REQUEST_VALID <= 1'b0;
                        state               <= after_sdram_state;
                    end
                end

                STATE_CRC_BYTE: begin     // bo 2 byte CRC
                    spi_tx_byte    <= 8'hFF;
                    spi_start      <= 1'b1;
                    crc_byte_count <= 1'b1;
                    spi_next_state <= crc_byte_count ? STATE_SECTOR_FINISH : STATE_CRC_BYTE;
                    state          <= STATE_SPI_WAIT;
                end

                STATE_SECTOR_FINISH: begin
                    spi_cs_n_level <= 1'b1;
                    spi_tx_byte    <= 8'hFF;
                    spi_start      <= 1'b1;
                    spi_next_state <= STATE_SECTOR_NEXT;
                    state          <= STATE_SPI_WAIT;
                end

                STATE_SECTOR_NEXT: begin
                    if (sector_index == TOTAL_SECTOR_COUNT - 1) begin
                        state <= STATE_FINISH;
                    end else begin
                        sector_index <= sector_index + 1'b1;
                        state        <= STATE_START_SECTOR_READ;
                    end
                end

                STATE_FINISH: begin
                    ERROR_CODE     <= magic_mismatch ? CODE_BAD_MAGIC : CODE_NONE;
                    DONE_BOOT_LOAD <= 1'b1;
                    spi_cs_n_level <= 1'b1;
                    state          <= STATE_IDLE;
                end

                STATE_ERROR: begin
                    ERROR_CODE          <= pending_error_code;
                    DONE_BOOT_LOAD      <= 1'b1;
                    spi_cs_n_level      <= 1'b1;
                    SDRAM_REQUEST_VALID <= 1'b0;
                    state               <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
