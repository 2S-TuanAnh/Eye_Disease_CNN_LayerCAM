// Doc tham so CNN tu SDRAM (moi tu 16 bit = 2 byte, byte chan o [7:0]) va chia theo muc dich.
//
// Bon che do hoat dong:
//  (1) START_FETCH_PARAMETERS : doc header 56 byte (offset 0x000..0x037), LUT 768 byte (0x200..0x4FF),
//      roi bang LCAM 24 byte (+0x8000: lop 6, lop 7 moi lop 8 byte nhu header, roi CAM_M0 / CAM_SHIFT).
//      ERROR_BAD_MAGIC la MUC (sticky): len 1 neu magic "FCNN" sai, giu den lan START_FETCH_PARAMETERS ke tiep.
//      DONE_FETCH_PARAMETERS luon xung 1 chu ky khi xong (ca khi bi loi magic; khi do bo qua LUT).
//  (2) START_FETCH_WEIGHTS : doc 1 khoi nhom = bias 8 x int32 LE (32 byte) + trong so PE0..PE7.
//      So trong so moi PE = (GROUP_BYTES - 32) / 8 (suy ra tu GROUP_BYTES, nen FC = 48, BWD1 = 432).
//  (3) START_FETCH_CLASS_ROW  : (LayerCAM) doc hang FC cua lop LAYERCAM_CLASS_INDEX (48 byte tai
//      +0x7CB0 + 32 + cls*48) cua CNN dang chon (cnn_base_byte_addr giu tu lan START_FETCH_PARAMETERS cuoi),
//      day ra CLASS_ROW_WRITE_* (CNN_RUNNER ghi vao GAP_RAM). DONE_FETCH_CLASS_ROW xung 1 chu ky khi xong.
//
// Gia dinh (dung voi bang muc 6): moi dia chi bat dau va do dai moi luong deu CHAN -> moi tu SDRAM cho dung 2 byte.
// (dac ta liet ke IN_CHANNELS la cong vao; ban nay khong can vi da co GROUP_BYTES.)
//
// Giao thuc master SDRAM (chung cho SD_LOADER / CNN_RUNNER / SDRAM_CONTROLLER / SDRAM_SIM_MODEL):
//   - Master giu SDRAM_REQUEST_VALID = 1 (cung dia chi/du lieu) cho den chu ky co REQUEST_READY = 1 -> da nhan.
//   - Doc: sau do cho SDRAM_RESPONSE_VALID (xung 1 chu ky, kem RESPONSE_READ_DATA).
//   - Ghi: khong co response; READY ve 1 khi ghi xong.
module PARAMETER_FETCHER (
    input  wire        CLK,
    input  wire        RST,

    // ---- Nap tham so CNN ----
    input  wire        START_FETCH_PARAMETERS,
    input  wire        CNN_SELECT,                 // 0 = CNN 1 (nen 0x00000), 1 = CNN 2 (nen 0x10000)
    output reg         DONE_FETCH_PARAMETERS,
    output reg         ERROR_BAD_MAGIC,
    output reg  [7:0]  UNKNOWN_CLASS_INDEX,
    output reg  [7:0]  NORMAL_CLASS_INDEX,         // 0xFF = CNN nay khong co lop Normal
    output reg  [30:0] CAM_M0,                     // requant diem CAM (bang LCAM)
    output reg  [5:0]  CAM_SHIFT,

    // ---- Header theo lop ----
    input  wire [2:0]  LAYER_INDEX,
    output wire [30:0] LAYER_MULTIPLIER,
    output wire [5:0]  LAYER_SHIFT,
    output wire [7:0]  LAYER_INPUT_ZERO_POINT,
    output wire [7:0]  LAYER_OUTPUT_ZERO_POINT,
    output wire        LAYER_RELU_ENABLE,

    // ---- Ghi INPUT_LUT_RAM ----
    output reg         INPUT_LUT_WRITE_ENABLE,
    output reg  [9:0]  INPUT_LUT_WRITE_ADDR,
    output reg  [7:0]  INPUT_LUT_WRITE_DATA,

    // ---- Nap khoi nhom ----
    input  wire        START_FETCH_WEIGHTS,
    input  wire [2:0]  GROUP_INDEX,
    input  wire [15:0] LAYER_OFFSET,
    input  wire [11:0] GROUP_BYTES,
    output reg         DONE_FETCH_WEIGHTS,

    output reg         BIAS_WRITE_ENABLE,
    output reg  [2:0]  BIAS_WRITE_INDEX,
    output reg  [31:0] BIAS_WRITE_DATA,
    output reg         WEIGHT_WRITE_ENABLE,
    output reg  [2:0]  WEIGHT_WRITE_PE_INDEX,
    output reg  [8:0]  WEIGHT_WRITE_ADDR,
    output reg  [7:0]  WEIGHT_WRITE_DATA,

    // ---- Nap hang FC cho LayerCAM ----
    input  wire        START_FETCH_CLASS_ROW,
    input  wire [2:0]  LAYERCAM_CLASS_INDEX,
    output reg         DONE_FETCH_CLASS_ROW,
    output reg         CLASS_ROW_WRITE_ENABLE,
    output reg  [5:0]  CLASS_ROW_WRITE_ADDR,
    output reg  [7:0]  CLASS_ROW_WRITE_DATA,

    // ---- Master SDRAM ----
    output reg         SDRAM_REQUEST_VALID,
    output wire        SDRAM_REQUEST_WRITE,
    output reg  [23:0] SDRAM_REQUEST_WORD_ADDR,
    output wire [15:0] SDRAM_REQUEST_WRITE_DATA,
    input  wire        SDRAM_REQUEST_READY,
    input  wire        SDRAM_RESPONSE_VALID,
    input  wire [15:0] SDRAM_RESPONSE_READ_DATA
);
    assign SDRAM_REQUEST_WRITE      = 1'b0;
    assign SDRAM_REQUEST_WRITE_DATA = 16'd0;

    localparam STATE_IDLE              = 3'd0;
    localparam STATE_REQUEST_WORD      = 3'd1;
    localparam STATE_WAIT_RESPONSE     = 3'd2;
    localparam STATE_DELIVER_BYTE      = 3'd3;
    localparam STATE_FINISH_STREAM     = 3'd4;
    localparam STATE_FINISH_PARAMETERS = 3'd5;
    localparam STATE_FINISH_WEIGHTS    = 3'd6;
    localparam STATE_FINISH_CLASS_ROW  = 3'd7;

    localparam MODE_HEADER     = 3'd0;
    localparam MODE_LUT        = 3'd1;
    localparam MODE_GROUP      = 3'd2;
    localparam MODE_LCAM_TABLE = 3'd3;
    localparam MODE_CLASS_ROW  = 3'd4;

    localparam [15:0] HEADER_BYTE_COUNT = 16'd56;
    localparam [15:0] LUT_BYTE_COUNT    = 16'd768;
    localparam [15:0] LUT_WORD_OFFSET   = 16'd256;     // 0x200 byte / 2
    localparam [15:0] LCAM_WORD_OFFSET  = 16'h4000;    // +0x8000 byte / 2
    localparam [15:0] LCAM_BYTE_COUNT   = 16'd24;
    localparam [15:0] CLASS_ROW_BYTES   = 16'd48;

    reg [2:0]  state;
    reg [2:0]  fetch_mode;
    reg [16:0] cnn_base_byte_addr;       // 0x00000 hoac 0x10000
    reg [15:0] stream_word_addr;
    reg [15:0] stream_total_bytes;
    reg [15:0] stream_byte_index;
    reg        byte_phase;               // 0 = byte thap, 1 = byte cao cua tu hien tai
    reg [15:0] word_register;
    reg        magic_mismatch;

    reg [23:0] m0_assemble;              // ghep 3 byte thap cua M0
    reg [31:0] bias_assemble;            // ghep dan bias (dich tu tren xuong)
    reg [8:0]  weights_per_pe;
    reg [2:0]  weight_pe_index;
    reg [8:0]  weight_addr_in_pe;

    // ---- Bang header (chi so 0..5 = lop forward, 6..7 = BWD1/BWD2 tu bang LCAM) ----
    reg [30:0] layer_multiplier_register         [0:7];
    reg [5:0]  layer_shift_register              [0:7];
    reg [7:0]  layer_input_zero_point_register   [0:7];
    reg [7:0]  layer_output_zero_point_register  [0:7];
    reg        layer_relu_enable_register        [0:7];

    assign LAYER_MULTIPLIER        = layer_multiplier_register[LAYER_INDEX];
    assign LAYER_SHIFT             = layer_shift_register[LAYER_INDEX];
    assign LAYER_INPUT_ZERO_POINT  = layer_input_zero_point_register[LAYER_INDEX];
    assign LAYER_OUTPUT_ZERO_POINT = layer_output_zero_point_register[LAYER_INDEX];
    assign LAYER_RELU_ENABLE       = layer_relu_enable_register[LAYER_INDEX];

    // ---- Tinh toan to hop ----
    wire [7:0]  delivered_byte = byte_phase ? word_register[15:8] : word_register[7:0];
    wire        last_byte_of_stream = ((stream_byte_index + 16'd1) == stream_total_bytes);
    wire [2:0]  layer_slot  = stream_byte_index[5:3] - 3'd1;   // byte 8..15 -> lop 0, ..., 48..55 -> lop 5
    wire [2:0]  lcam_slot   = {2'b11, stream_byte_index[3]};   // bang LCAM: byte 0..7 -> lop 6, 8..15 -> lop 7
    wire [2:0]  field_index = stream_byte_index[2:0];

    wire [15:0] group_offset_bytes     = GROUP_INDEX * GROUP_BYTES;
    wire [16:0] group_start_byte_addr  = cnn_base_byte_addr + {1'b0, LAYER_OFFSET} + {1'b0, group_offset_bytes};
    wire [11:0] weight_bytes_in_group  = GROUP_BYTES - 12'd32;

    // Hang FC: byte = nen + 0x7CB0 + 32 + cls*48, cls*48 = (cls<<5) + (cls<<4)
    wire [16:0] class_row_byte_addr    = cnn_base_byte_addr + 17'h07CD0
                                         + {9'd0, LAYERCAM_CLASS_INDEX, 5'd0}
                                         + {10'd0, LAYERCAM_CLASS_INDEX, 4'd0};

    always @(posedge CLK) begin
        if (RST) begin
            state                   <= STATE_IDLE;
            fetch_mode              <= MODE_HEADER;
            cnn_base_byte_addr      <= 17'd0;
            stream_word_addr        <= 16'd0;
            stream_total_bytes      <= 16'd0;
            stream_byte_index       <= 16'd0;
            byte_phase              <= 1'b0;
            word_register           <= 16'd0;
            magic_mismatch          <= 1'b0;
            m0_assemble             <= 24'd0;
            bias_assemble           <= 32'd0;
            weights_per_pe          <= 9'd0;
            weight_pe_index         <= 3'd0;
            weight_addr_in_pe       <= 9'd0;
            DONE_FETCH_PARAMETERS   <= 1'b0;
            DONE_FETCH_WEIGHTS      <= 1'b0;
            ERROR_BAD_MAGIC         <= 1'b0;
            UNKNOWN_CLASS_INDEX     <= 8'd0;
            NORMAL_CLASS_INDEX      <= 8'hFF;
            CAM_M0                  <= 31'd0;
            CAM_SHIFT               <= 6'd0;
            DONE_FETCH_CLASS_ROW    <= 1'b0;
            CLASS_ROW_WRITE_ENABLE  <= 1'b0;
            CLASS_ROW_WRITE_ADDR    <= 6'd0;
            CLASS_ROW_WRITE_DATA    <= 8'd0;
            INPUT_LUT_WRITE_ENABLE  <= 1'b0;
            INPUT_LUT_WRITE_ADDR    <= 10'd0;
            INPUT_LUT_WRITE_DATA    <= 8'd0;
            BIAS_WRITE_ENABLE       <= 1'b0;
            BIAS_WRITE_INDEX        <= 3'd0;
            BIAS_WRITE_DATA         <= 32'd0;
            WEIGHT_WRITE_ENABLE     <= 1'b0;
            WEIGHT_WRITE_PE_INDEX   <= 3'd0;
            WEIGHT_WRITE_ADDR       <= 9'd0;
            WEIGHT_WRITE_DATA       <= 8'd0;
            SDRAM_REQUEST_VALID     <= 1'b0;
            SDRAM_REQUEST_WORD_ADDR <= 24'd0;
        end else begin
            DONE_FETCH_PARAMETERS  <= 1'b0;
            DONE_FETCH_WEIGHTS     <= 1'b0;
            DONE_FETCH_CLASS_ROW   <= 1'b0;
            CLASS_ROW_WRITE_ENABLE <= 1'b0;
            INPUT_LUT_WRITE_ENABLE <= 1'b0;
            BIAS_WRITE_ENABLE      <= 1'b0;
            WEIGHT_WRITE_ENABLE    <= 1'b0;

            case (state)
                STATE_IDLE: begin
                    if (START_FETCH_PARAMETERS) begin
                        cnn_base_byte_addr <= CNN_SELECT ? 17'h10000 : 17'h00000;
                        stream_word_addr   <= CNN_SELECT ? 16'h8000  : 16'h0000;
                        fetch_mode         <= MODE_HEADER;
                        stream_total_bytes <= HEADER_BYTE_COUNT;
                        stream_byte_index  <= 16'd0;
                        byte_phase         <= 1'b0;
                        magic_mismatch     <= 1'b0;
                        ERROR_BAD_MAGIC    <= 1'b0;
                        state              <= STATE_REQUEST_WORD;
                    end else if (START_FETCH_WEIGHTS) begin
                        fetch_mode         <= MODE_GROUP;
                        stream_word_addr   <= group_start_byte_addr[16:1];
                        stream_total_bytes <= {4'd0, GROUP_BYTES};
                        stream_byte_index  <= 16'd0;
                        byte_phase         <= 1'b0;
                        weights_per_pe     <= weight_bytes_in_group[11:3];
                        weight_pe_index    <= 3'd0;
                        weight_addr_in_pe  <= 9'd0;
                        state              <= STATE_REQUEST_WORD;
                    end else if (START_FETCH_CLASS_ROW) begin
                        fetch_mode         <= MODE_CLASS_ROW;
                        stream_word_addr   <= class_row_byte_addr[16:1];
                        stream_total_bytes <= CLASS_ROW_BYTES;
                        stream_byte_index  <= 16'd0;
                        byte_phase         <= 1'b0;
                        state              <= STATE_REQUEST_WORD;
                    end
                end

                STATE_REQUEST_WORD: begin
                    SDRAM_REQUEST_VALID     <= 1'b1;
                    SDRAM_REQUEST_WORD_ADDR <= {8'd0, stream_word_addr};
                    if (SDRAM_REQUEST_VALID && SDRAM_REQUEST_READY) begin
                        SDRAM_REQUEST_VALID <= 1'b0;
                        state               <= STATE_WAIT_RESPONSE;
                    end
                end

                STATE_WAIT_RESPONSE: begin
                    if (SDRAM_RESPONSE_VALID) begin
                        word_register <= SDRAM_RESPONSE_READ_DATA;
                        byte_phase    <= 1'b0;
                        state         <= STATE_DELIVER_BYTE;
                    end
                end

                STATE_DELIVER_BYTE: begin
                    case (fetch_mode)
                        MODE_HEADER: begin
                            case (stream_byte_index[5:0])
                                6'd0: if (delivered_byte != 8'h46) magic_mismatch <= 1'b1;   // 'F'
                                6'd1: if (delivered_byte != 8'h43) magic_mismatch <= 1'b1;   // 'C'
                                6'd2: if (delivered_byte != 8'h4E) magic_mismatch <= 1'b1;   // 'N'
                                6'd3: if (delivered_byte != 8'h4E) magic_mismatch <= 1'b1;   // 'N'
                                6'd5: UNKNOWN_CLASS_INDEX <= delivered_byte;
                                6'd7: NORMAL_CLASS_INDEX  <= delivered_byte;
                                default: ;
                            endcase
                            if (stream_byte_index >= 16'd8) begin
                                case (field_index)
                                    3'd0: m0_assemble[7:0]   <= delivered_byte;
                                    3'd1: m0_assemble[15:8]  <= delivered_byte;
                                    3'd2: m0_assemble[23:16] <= delivered_byte;
                                    3'd3: layer_multiplier_register[layer_slot] <=
                                              {delivered_byte[6:0], m0_assemble};
                                    3'd4: layer_shift_register[layer_slot]             <= delivered_byte[5:0];
                                    3'd5: layer_input_zero_point_register[layer_slot]  <= delivered_byte;
                                    3'd6: layer_output_zero_point_register[layer_slot] <= delivered_byte;
                                    3'd7: layer_relu_enable_register[layer_slot]       <= delivered_byte[0];
                                endcase
                            end
                        end

                        // Bang LCAM (24 byte): byte 0..15 = 2 muc lop (cung khuon header), 16..23 = CAM (<IB)
                        MODE_LCAM_TABLE: begin
                            if (stream_byte_index < 16'd16) begin
                                case (field_index)
                                    3'd0: m0_assemble[7:0]   <= delivered_byte;
                                    3'd1: m0_assemble[15:8]  <= delivered_byte;
                                    3'd2: m0_assemble[23:16] <= delivered_byte;
                                    3'd3: layer_multiplier_register[lcam_slot] <=
                                              {delivered_byte[6:0], m0_assemble};
                                    3'd4: layer_shift_register[lcam_slot]             <= delivered_byte[5:0];
                                    3'd5: layer_input_zero_point_register[lcam_slot]  <= delivered_byte;
                                    3'd6: layer_output_zero_point_register[lcam_slot] <= delivered_byte;
                                    3'd7: layer_relu_enable_register[lcam_slot]       <= delivered_byte[0];
                                endcase
                            end else begin
                                case (field_index)
                                    3'd0: m0_assemble[7:0]   <= delivered_byte;
                                    3'd1: m0_assemble[15:8]  <= delivered_byte;
                                    3'd2: m0_assemble[23:16] <= delivered_byte;
                                    3'd3: CAM_M0             <= {delivered_byte[6:0], m0_assemble};
                                    3'd4: CAM_SHIFT          <= delivered_byte[5:0];
                                    default: ;
                                endcase
                            end
                        end

                        MODE_CLASS_ROW: begin
                            CLASS_ROW_WRITE_ENABLE <= 1'b1;
                            CLASS_ROW_WRITE_ADDR   <= stream_byte_index[5:0];
                            CLASS_ROW_WRITE_DATA   <= delivered_byte;
                        end

                        MODE_LUT: begin
                            INPUT_LUT_WRITE_ENABLE <= 1'b1;
                            INPUT_LUT_WRITE_ADDR   <= stream_byte_index[9:0];
                            INPUT_LUT_WRITE_DATA   <= delivered_byte;
                        end

                        MODE_GROUP: begin
                            if (stream_byte_index < 16'd32) begin
                                // bias int32 little-endian: byte 0 thap nhat
                                bias_assemble <= {delivered_byte, bias_assemble[31:8]};
                                if (stream_byte_index[1:0] == 2'd3) begin
                                    BIAS_WRITE_ENABLE <= 1'b1;
                                    BIAS_WRITE_INDEX  <= stream_byte_index[4:2];
                                    BIAS_WRITE_DATA   <= {delivered_byte, bias_assemble[31:8]};
                                end
                            end else begin
                                WEIGHT_WRITE_ENABLE   <= 1'b1;
                                WEIGHT_WRITE_PE_INDEX <= weight_pe_index;
                                WEIGHT_WRITE_ADDR     <= weight_addr_in_pe;
                                WEIGHT_WRITE_DATA     <= delivered_byte;
                                if ((weight_addr_in_pe + 9'd1) == weights_per_pe) begin
                                    weight_addr_in_pe <= 9'd0;
                                    weight_pe_index   <= weight_pe_index + 1'b1;
                                end else begin
                                    weight_addr_in_pe <= weight_addr_in_pe + 1'b1;
                                end
                            end
                        end

                        default: ;
                    endcase

                    if (last_byte_of_stream) begin
                        state <= STATE_FINISH_STREAM;
                    end else begin
                        stream_byte_index <= stream_byte_index + 1'b1;
                        if (byte_phase) begin
                            byte_phase       <= 1'b0;
                            stream_word_addr <= stream_word_addr + 1'b1;
                            state            <= STATE_REQUEST_WORD;
                        end else begin
                            byte_phase <= 1'b1;
                        end
                    end
                end

                STATE_FINISH_STREAM: begin
                    case (fetch_mode)
                        MODE_HEADER: begin
                            if (magic_mismatch) begin
                                ERROR_BAD_MAGIC <= 1'b1;
                                state           <= STATE_FINISH_PARAMETERS;
                            end else begin
                                fetch_mode         <= MODE_LUT;
                                stream_word_addr   <= cnn_base_byte_addr[16:1] + LUT_WORD_OFFSET;
                                stream_total_bytes <= LUT_BYTE_COUNT;
                                stream_byte_index  <= 16'd0;
                                byte_phase         <= 1'b0;
                                state              <= STATE_REQUEST_WORD;
                            end
                        end
                        MODE_LUT: begin
                            fetch_mode         <= MODE_LCAM_TABLE;
                            stream_word_addr   <= cnn_base_byte_addr[16:1] + LCAM_WORD_OFFSET;
                            stream_total_bytes <= LCAM_BYTE_COUNT;
                            stream_byte_index  <= 16'd0;
                            byte_phase         <= 1'b0;
                            state              <= STATE_REQUEST_WORD;
                        end
                        MODE_LCAM_TABLE: state <= STATE_FINISH_PARAMETERS;
                        MODE_GROUP:      state <= STATE_FINISH_WEIGHTS;
                        MODE_CLASS_ROW:  state <= STATE_FINISH_CLASS_ROW;
                        default:    state <= STATE_IDLE;
                    endcase
                end

                // Them 1 chu ky de lan ghi cuoi (LUT / bias / weight) xong truoc khi bao DONE
                STATE_FINISH_PARAMETERS: begin
                    DONE_FETCH_PARAMETERS <= 1'b1;
                    state                 <= STATE_IDLE;
                end

                STATE_FINISH_WEIGHTS: begin
                    DONE_FETCH_WEIGHTS <= 1'b1;
                    state              <= STATE_IDLE;
                end

                STATE_FINISH_CLASS_ROW: begin
                    DONE_FETCH_CLASS_ROW <= 1'b1;
                    state                <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
