// ROM to hop: hang so tung lop theo bang muc 3 va muc 6 cua dac ta.
// LAYER_INDEX 0..4 = conv1..conv5, 5 = FC, 6 = BWD1 (LayerCAM, 48->32), 7 = BWD2 (LayerCAM, 32->24).
//
// Kiem tra: LAYER_OFFSET[L+1] = LAYER_OFFSET[L] + GROUP_COUNT[L]*GROUP_BYTES[L]
//   1536 + 2*248  = 2032 ; 2032 + 3*1184 = 5584 ; 5584 + 3*1760 = 10864
//   10864 + 4*1760 = 17904 ; 17904 + 6*2336 = 31920 ; 31920 + 416 = 32336 = 0x7E50
//   LayerCAM (bang LCAM 24 byte o +0x8000, nen cac lop nguoc bat dau o +0x8100):
//   33024 + 4*3488 = 46976 (0xB780) ; 46976 + 3*2336 = 53984 (0xD2E0)
// So byte trong so moi PE cua 1 khoi nhom = (GROUP_BYTES - 32) / 8:
//   27, 144, 216, 216, 288, 48, 432 (BWD1), 288 (BWD2) (FC chi co 48 trong so / PE vi khong co kernel 3x3).
module LAYER_CONFIG_ROM (
    input  wire [2:0]  LAYER_INDEX,
    output reg  [6:0]  IN_SIZE,              // canh anh/feature vao (64, 32, 16, 8, 8, 1)
    output reg  [2:0]  IN_SIZE_LOG2,         // log2(IN_SIZE)
    output reg  [5:0]  IN_CHANNELS,
    output reg  [5:0]  OUT_SIZE,             // canh sau pool (32, 16, 8, 8, 8, 1)
    output reg  [2:0]  OUT_SIZE_LOG2,
    output reg  [5:0]  OUT_CHANNELS,         // FC: 8 (5 that + 3 dem)
    output reg         POOL_ENABLE,
    output reg  [2:0]  GROUP_COUNT,
    output reg  [11:0] GROUP_BYTES,
    output reg  [15:0] LAYER_OFFSET,
    output reg         IS_FULLY_CONNECTED
);
    always @(*) begin
        IN_SIZE            = 7'd0;
        IN_SIZE_LOG2       = 3'd0;
        IN_CHANNELS        = 6'd0;
        OUT_SIZE           = 6'd0;
        OUT_SIZE_LOG2      = 3'd0;
        OUT_CHANNELS       = 6'd0;
        POOL_ENABLE        = 1'b0;
        GROUP_COUNT        = 3'd0;
        GROUP_BYTES        = 12'd0;
        LAYER_OFFSET       = 16'd0;
        IS_FULLY_CONNECTED = 1'b0;

        case (LAYER_INDEX)
            3'd0: begin   // Conv1: 64x64x3 -> 32x32x16 (pool)
                IN_SIZE = 7'd64;  IN_SIZE_LOG2 = 3'd6;  IN_CHANNELS = 6'd3;
                OUT_SIZE = 6'd32; OUT_SIZE_LOG2 = 3'd5; OUT_CHANNELS = 6'd16;
                POOL_ENABLE = 1'b1; GROUP_COUNT = 3'd2;
                GROUP_BYTES = 12'd248; LAYER_OFFSET = 16'd1536;
            end
            3'd1: begin   // Conv2: 32x32x16 -> 16x16x24 (pool)
                IN_SIZE = 7'd32;  IN_SIZE_LOG2 = 3'd5;  IN_CHANNELS = 6'd16;
                OUT_SIZE = 6'd16; OUT_SIZE_LOG2 = 3'd4; OUT_CHANNELS = 6'd24;
                POOL_ENABLE = 1'b1; GROUP_COUNT = 3'd3;
                GROUP_BYTES = 12'd1184; LAYER_OFFSET = 16'd2032;
            end
            3'd2: begin   // Conv3: 16x16x24 -> 8x8x24 (pool)
                IN_SIZE = 7'd16;  IN_SIZE_LOG2 = 3'd4;  IN_CHANNELS = 6'd24;
                OUT_SIZE = 6'd8;  OUT_SIZE_LOG2 = 3'd3; OUT_CHANNELS = 6'd24;
                POOL_ENABLE = 1'b1; GROUP_COUNT = 3'd3;
                GROUP_BYTES = 12'd1760; LAYER_OFFSET = 16'd5584;
            end
            3'd3: begin   // Conv4: 8x8x24 -> 8x8x32
                IN_SIZE = 7'd8;   IN_SIZE_LOG2 = 3'd3;  IN_CHANNELS = 6'd24;
                OUT_SIZE = 6'd8;  OUT_SIZE_LOG2 = 3'd3; OUT_CHANNELS = 6'd32;
                POOL_ENABLE = 1'b0; GROUP_COUNT = 3'd4;
                GROUP_BYTES = 12'd1760; LAYER_OFFSET = 16'd10864;
            end
            3'd4: begin   // Conv5: 8x8x32 -> 8x8x48
                IN_SIZE = 7'd8;   IN_SIZE_LOG2 = 3'd3;  IN_CHANNELS = 6'd32;
                OUT_SIZE = 6'd8;  OUT_SIZE_LOG2 = 3'd3; OUT_CHANNELS = 6'd48;
                POOL_ENABLE = 1'b0; GROUP_COUNT = 3'd6;
                GROUP_BYTES = 12'd2336; LAYER_OFFSET = 16'd17904;
            end
            3'd5: begin   // FC: 48 -> 5 (dem 8)
                IN_SIZE = 7'd1;   IN_SIZE_LOG2 = 3'd0;  IN_CHANNELS = 6'd48;
                OUT_SIZE = 6'd1;  OUT_SIZE_LOG2 = 3'd0; OUT_CHANNELS = 6'd8;
                POOL_ENABLE = 1'b0; GROUP_COUNT = 3'd1;
                GROUP_BYTES = 12'd416; LAYER_OFFSET = 16'd31920;
                IS_FULLY_CONNECTED = 1'b1;
            end
            3'd6: begin   // BWD1 (LayerCAM): 8x8x48 -> 8x8x32, khong pool, bias = 0
                IN_SIZE = 7'd8;   IN_SIZE_LOG2 = 3'd3;  IN_CHANNELS = 6'd48;
                OUT_SIZE = 6'd8;  OUT_SIZE_LOG2 = 3'd3; OUT_CHANNELS = 6'd32;
                POOL_ENABLE = 1'b0; GROUP_COUNT = 3'd4;
                GROUP_BYTES = 12'd3488; LAYER_OFFSET = 16'd33024;
                IS_FULLY_CONNECTED = 1'b0;
            end
            3'd7: begin   // BWD2 (LayerCAM): 8x8x32 -> 8x8x24, khong pool, bias = 0
                IN_SIZE = 7'd8;   IN_SIZE_LOG2 = 3'd3;  IN_CHANNELS = 6'd32;
                OUT_SIZE = 6'd8;  OUT_SIZE_LOG2 = 3'd3; OUT_CHANNELS = 6'd24;
                POOL_ENABLE = 1'b0; GROUP_COUNT = 3'd3;
                GROUP_BYTES = 12'd2336; LAYER_OFFSET = 16'd46976;
                IS_FULLY_CONNECTED = 1'b0;
            end
            default: ;
        endcase
    end
endmodule
