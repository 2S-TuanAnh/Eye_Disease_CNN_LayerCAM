// Bo dieu khien SDRAM HY57V2562GTR (4 bank, 13 bit hang, 9 bit cot, 16 bit du lieu) @ 50 MHz.
// - Moi truy cap mot tu 16 bit: ACTIVE -> READ/WRITE co auto-precharge (A10 = 1). Khong burst (BL = 1).
// - Giao dien nguoi dung giong SDRAM_SIM_MODEL:
//     Master giu REQUEST_VALID = 1 (cung dia chi/du lieu) den chu ky co REQUEST_READY = 1 -> da nhan.
//     Doc : sau do cho RESPONSE_VALID (xung 1 chu ky kem RESPONSE_READ_DATA); READY len lai cung luc.
//     Ghi : khong co response; READY ve 1 khi ghi xong.
// - REQUEST_WORD_ADDR[23:0] = {bank[1:0], row[12:0], col[8:0]}.
// - SDRAM_CLK = ~CLK (ODDR2). Lenh duoc phat tai suon len CLK, SDRAM lay mau o suon xuong CLK (cach 10 ns).
// - Khi mo phong, define SIMULATION de thay ODDR2 bang assign SDRAM_CLK = ~CLK.
// - READ_CAPTURE_CYCLES: so chu ky tu luc lenh READ duoc ghi vao thanh ghi den luc lay mau du lieu.
//   Gia tri 4 ung voi CL = 3 va SDRAM_CLK nghich pha. Neu doc sai tren board thi thu 3 hoac 5.
module SDRAM_CONTROLLER #(
    parameter INIT_WAIT_CYCLES         = 10000,   // 200 us @ 50 MHz
    parameter REFRESH_INTERVAL_CYCLES  = 370,     // < 7.8 us
    parameter READ_CAPTURE_CYCLES      = 4,
    parameter WRITE_RECOVERY_CYCLES    = 5
) (
    input  wire        CLK,
    input  wire        RST,

    // Giao dien nguoi dung
    input  wire        REQUEST_VALID,
    input  wire        REQUEST_WRITE,
    input  wire [23:0] REQUEST_WORD_ADDR,
    input  wire [15:0] REQUEST_WRITE_DATA,
    output reg         REQUEST_READY,
    output reg         RESPONSE_VALID,
    output reg  [15:0] RESPONSE_READ_DATA,
    output wire        INIT_DONE,

    // Chan SDRAM
    output wire        SDRAM_CLK,
    output reg         SDRAM_CKE,
    output wire        SDRAM_CS_N,
    output wire        SDRAM_RAS_N,
    output wire        SDRAM_CAS_N,
    output wire        SDRAM_WE_N,
    output reg  [1:0]  SDRAM_DQM,
    output reg  [1:0]  SDRAM_BANK,
    output reg  [12:0] SDRAM_ADDR,
    inout  wire [15:0] SDRAM_DATA
);
    // {CS_N, RAS_N, CAS_N, WE_N}
    localparam [3:0] COMMAND_NOP          = 4'b0111;
    localparam [3:0] COMMAND_ACTIVE       = 4'b0011;
    localparam [3:0] COMMAND_READ         = 4'b0101;
    localparam [3:0] COMMAND_WRITE        = 4'b0100;
    localparam [3:0] COMMAND_PRECHARGE    = 4'b0010;
    localparam [3:0] COMMAND_AUTO_REFRESH = 4'b0001;
    localparam [3:0] COMMAND_LOAD_MODE    = 4'b0000;

    localparam [12:0] MODE_REGISTER_VALUE = 13'h030;   // BL1, tuan tu, CL3

    localparam [4:0] STATE_INIT_WAIT              = 5'd0;
    localparam [4:0] STATE_INIT_PRECHARGE         = 5'd1;
    localparam [4:0] STATE_INIT_WAIT_PRECHARGE    = 5'd2;
    localparam [4:0] STATE_INIT_REFRESH_FIRST     = 5'd3;
    localparam [4:0] STATE_INIT_WAIT_REFRESH_FIRST  = 5'd4;
    localparam [4:0] STATE_INIT_REFRESH_SECOND    = 5'd5;
    localparam [4:0] STATE_INIT_WAIT_REFRESH_SECOND = 5'd6;
    localparam [4:0] STATE_INIT_LOAD_MODE         = 5'd7;
    localparam [4:0] STATE_INIT_WAIT_MODE         = 5'd8;
    localparam [4:0] STATE_IDLE                   = 5'd9;
    localparam [4:0] STATE_ACTIVE_WAIT            = 5'd10;
    localparam [4:0] STATE_READ_WAIT              = 5'd11;
    localparam [4:0] STATE_WRITE_WAIT             = 5'd12;
    localparam [4:0] STATE_REFRESH_WAIT           = 5'd13;

    reg [4:0]  state;
    reg [3:0]  command;
    reg [13:0] wait_count;
    reg        init_done_reg;

    reg        latched_write;
    reg [23:0] latched_word_addr;
    reg [15:0] latched_write_data;

    reg [9:0]  refresh_counter;
    reg        refresh_pending;

    reg        data_output_enable;
    reg [15:0] data_output;

    assign {SDRAM_CS_N, SDRAM_RAS_N, SDRAM_CAS_N, SDRAM_WE_N} = command;
    assign SDRAM_DATA = data_output_enable ? data_output : 16'hZZZZ;
    assign INIT_DONE  = init_done_reg;

`ifdef SIMULATION
    assign SDRAM_CLK = ~CLK;
`else
    ODDR2 #(
        .DDR_ALIGNMENT("NONE"),
        .INIT(1'b0),
        .SRTYPE("SYNC")
    ) sdram_clock_oddr2 (
        .Q (SDRAM_CLK),
        .C0(CLK),
        .C1(~CLK),
        .CE(1'b1),
        .D0(1'b0),
        .D1(1'b1),
        .R (1'b0),
        .S (1'b0)
    );
`endif

    always @(posedge CLK) begin
        if (RST) begin
            state              <= STATE_INIT_WAIT;
            command            <= COMMAND_NOP;
            wait_count         <= INIT_WAIT_CYCLES - 1;
            init_done_reg      <= 1'b0;
            SDRAM_CKE          <= 1'b1;
            SDRAM_DQM          <= 2'b11;
            SDRAM_BANK         <= 2'b00;
            SDRAM_ADDR         <= 13'd0;
            REQUEST_READY      <= 1'b0;
            RESPONSE_VALID     <= 1'b0;
            RESPONSE_READ_DATA <= 16'd0;
            latched_write      <= 1'b0;
            latched_word_addr  <= 24'd0;
            latched_write_data <= 16'd0;
            refresh_counter    <= 10'd0;
            refresh_pending    <= 1'b0;
            data_output_enable <= 1'b0;
            data_output        <= 16'd0;
        end else begin
            command            <= COMMAND_NOP;
            RESPONSE_VALID     <= 1'b0;
            data_output_enable <= 1'b0;

            case (state)
                // ---------------------------------------------------- khoi dong
                STATE_INIT_WAIT: begin
                    if (wait_count == 14'd0)
                        state <= STATE_INIT_PRECHARGE;
                    else
                        wait_count <= wait_count - 1'b1;
                end

                STATE_INIT_PRECHARGE: begin
                    command    <= COMMAND_PRECHARGE;
                    SDRAM_ADDR <= 13'h0400;          // A10 = 1: precharge tat ca bank
                    wait_count <= 14'd2;
                    state      <= STATE_INIT_WAIT_PRECHARGE;
                end

                STATE_INIT_WAIT_PRECHARGE: begin
                    if (wait_count == 14'd0)
                        state <= STATE_INIT_REFRESH_FIRST;
                    else
                        wait_count <= wait_count - 1'b1;
                end

                STATE_INIT_REFRESH_FIRST: begin
                    command    <= COMMAND_AUTO_REFRESH;
                    wait_count <= 14'd7;
                    state      <= STATE_INIT_WAIT_REFRESH_FIRST;
                end

                STATE_INIT_WAIT_REFRESH_FIRST: begin
                    if (wait_count == 14'd0)
                        state <= STATE_INIT_REFRESH_SECOND;
                    else
                        wait_count <= wait_count - 1'b1;
                end

                STATE_INIT_REFRESH_SECOND: begin
                    command    <= COMMAND_AUTO_REFRESH;
                    wait_count <= 14'd7;
                    state      <= STATE_INIT_WAIT_REFRESH_SECOND;
                end

                STATE_INIT_WAIT_REFRESH_SECOND: begin
                    if (wait_count == 14'd0)
                        state <= STATE_INIT_LOAD_MODE;
                    else
                        wait_count <= wait_count - 1'b1;
                end

                STATE_INIT_LOAD_MODE: begin
                    command    <= COMMAND_LOAD_MODE;
                    SDRAM_BANK <= 2'b00;
                    SDRAM_ADDR <= MODE_REGISTER_VALUE;
                    wait_count <= 14'd2;
                    state      <= STATE_INIT_WAIT_MODE;
                end

                STATE_INIT_WAIT_MODE: begin
                    if (wait_count == 14'd0) begin
                        SDRAM_DQM       <= 2'b00;
                        init_done_reg   <= 1'b1;
                        REQUEST_READY   <= 1'b1;
                        refresh_counter <= 10'd0;
                        state           <= STATE_IDLE;
                    end else begin
                        wait_count <= wait_count - 1'b1;
                    end
                end

                // ---------------------------------------------------- hoat dong
                STATE_IDLE: begin
                    if (REQUEST_READY && REQUEST_VALID) begin
                        latched_write      <= REQUEST_WRITE;
                        latched_word_addr  <= REQUEST_WORD_ADDR;
                        latched_write_data <= REQUEST_WRITE_DATA;
                        REQUEST_READY      <= 1'b0;
                        command            <= COMMAND_ACTIVE;
                        SDRAM_BANK         <= REQUEST_WORD_ADDR[23:22];
                        SDRAM_ADDR         <= REQUEST_WORD_ADDR[21:9];
                        wait_count         <= 14'd1;
                        state              <= STATE_ACTIVE_WAIT;
                    end else if (refresh_pending) begin
                        REQUEST_READY   <= 1'b0;
                        refresh_pending <= 1'b0;
                        command         <= COMMAND_AUTO_REFRESH;
                        wait_count      <= 14'd5;
                        state           <= STATE_REFRESH_WAIT;
                    end
                end

                STATE_ACTIVE_WAIT: begin
                    if (wait_count == 14'd0) begin
                        SDRAM_BANK <= latched_word_addr[23:22];
                        // A10 = 1: auto-precharge, A[8:0] = cot
                        SDRAM_ADDR <= {2'b00, 1'b1, 1'b0, latched_word_addr[8:0]};
                        if (latched_write) begin
                            command            <= COMMAND_WRITE;
                            data_output        <= latched_write_data;
                            data_output_enable <= 1'b1;
                            wait_count         <= WRITE_RECOVERY_CYCLES - 1;
                            state              <= STATE_WRITE_WAIT;
                        end else begin
                            command    <= COMMAND_READ;
                            wait_count <= READ_CAPTURE_CYCLES - 1;
                            state      <= STATE_READ_WAIT;
                        end
                    end else begin
                        wait_count <= wait_count - 1'b1;
                    end
                end

                STATE_READ_WAIT: begin
                    if (wait_count == 14'd0) begin
                        RESPONSE_READ_DATA <= SDRAM_DATA;
                        RESPONSE_VALID     <= 1'b1;
                        REQUEST_READY      <= ~refresh_pending;
                        state              <= STATE_IDLE;
                    end else begin
                        wait_count <= wait_count - 1'b1;
                    end
                end

                STATE_WRITE_WAIT: begin
                    if (wait_count == 14'd0) begin
                        REQUEST_READY <= ~refresh_pending;
                        state         <= STATE_IDLE;
                    end else begin
                        wait_count <= wait_count - 1'b1;
                    end
                end

                STATE_REFRESH_WAIT: begin
                    if (wait_count == 14'd0) begin
                        REQUEST_READY <= ~refresh_pending;
                        state         <= STATE_IDLE;
                    end else begin
                        wait_count <= wait_count - 1'b1;
                    end
                end

                default: state <= STATE_INIT_WAIT;
            endcase

            // Bo dem refresh chay tu do sau khi khoi dong xong (dat sau case de co uu tien so voi lenh xoa)
            if (init_done_reg) begin
                if (refresh_counter == REFRESH_INTERVAL_CYCLES - 1) begin
                    refresh_counter <= 10'd0;
                    refresh_pending <= 1'b1;
                end else begin
                    refresh_counter <= refresh_counter + 1'b1;
                end
            end
        end
    end
endmodule
