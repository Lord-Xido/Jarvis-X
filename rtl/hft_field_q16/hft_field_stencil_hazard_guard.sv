module hft_field_stencil_hazard_guard #(
    parameter integer COORD_WIDTH = 10,
    parameter integer LATENCY_CYCLES = 17
) (
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   in_valid,
    input  logic [COORD_WIDTH-1:0] center_coord_i,
    input  logic [COORD_WIDTH-1:0] xm_coord_i,
    input  logic [COORD_WIDTH-1:0] xp_coord_i,
    input  logic [COORD_WIDTH-1:0] ym_coord_i,
    input  logic [COORD_WIDTH-1:0] yp_coord_i,
    input  logic [COORD_WIDTH-1:0] zm_coord_i,
    input  logic [COORD_WIDTH-1:0] zp_coord_i,

    output logic                   in_ready,
    output logic                   accept_o,
    output logic                   conflict_o,
    output logic                   commit_valid_o,
    output logic [COORD_WIDTH-1:0] commit_coord_o
);

    // Each accepted transaction reads a seven-cell Psi stencil but writes only
    // its center coordinate. To preserve sequential field semantics, a new
    // transaction must not read any coordinate whose prior center write is
    // still pending. This generalizes the earlier same-center RAW scoreboard.

    logic                   busy_valid [0:LATENCY_CYCLES];
    logic [COORD_WIDTH-1:0] busy_coord [0:LATENCY_CYCLES];

    integer i;
    always_comb begin
        conflict_o = 1'b0;
        for (i = 0; i <= LATENCY_CYCLES; i = i + 1) begin
            if (busy_valid[i] && (
                busy_coord[i] == center_coord_i ||
                busy_coord[i] == xm_coord_i ||
                busy_coord[i] == xp_coord_i ||
                busy_coord[i] == ym_coord_i ||
                busy_coord[i] == yp_coord_i ||
                busy_coord[i] == zm_coord_i ||
                busy_coord[i] == zp_coord_i)) begin
                conflict_o = 1'b1;
            end
        end

        in_ready = !conflict_o;
        accept_o = in_valid && in_ready;
        commit_valid_o = busy_valid[LATENCY_CYCLES];
        commit_coord_o = busy_coord[LATENCY_CYCLES];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i <= LATENCY_CYCLES; i = i + 1) begin
                busy_valid[i] <= 1'b0;
                busy_coord[i] <= '0;
            end
        end else begin
            for (i = LATENCY_CYCLES; i > 0; i = i - 1) begin
                busy_valid[i] <= busy_valid[i - 1];
                busy_coord[i] <= busy_coord[i - 1];
            end

            busy_valid[0] <= accept_o;
            if (accept_o)
                busy_coord[0] <= center_coord_i;
        end
    end

    initial begin
        if (COORD_WIDTH < 1)
            $error("COORD_WIDTH must be >= 1");
        if (LATENCY_CYCLES < 1)
            $error("LATENCY_CYCLES must be >= 1");
    end

endmodule
