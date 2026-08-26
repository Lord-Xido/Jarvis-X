module hft_field_hazard_guard #(
    parameter integer COORD_WIDTH = 16,
    parameter integer LATENCY_CYCLES = 17
) (
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   in_valid,
    input  logic [COORD_WIDTH-1:0] in_coord,

    output logic                   in_ready,
    output logic                   accept_o,
    output logic                   conflict_o,
    output logic                   commit_valid_o,
    output logic [COORD_WIDTH-1:0] commit_coord_o
);

    // Conservative RAW-hazard scoreboard. A coordinate cannot be re-issued
    // while an earlier transaction for that coordinate remains in flight.
    // Independent coordinates are still accepted at II=1.

    logic                   busy_valid [0:LATENCY_CYCLES];
    logic [COORD_WIDTH-1:0] busy_coord [0:LATENCY_CYCLES];

    integer i;
    always_comb begin
        conflict_o = 1'b0;
        for (i = 0; i <= LATENCY_CYCLES; i = i + 1) begin
            if (busy_valid[i] && busy_coord[i] == in_coord)
                conflict_o = 1'b1;
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
                busy_coord[0] <= in_coord;
        end
    end

    initial begin
        if (COORD_WIDTH < 1)
            $error("COORD_WIDTH must be >= 1");
        if (LATENCY_CYCLES < 1)
            $error("LATENCY_CYCLES must be >= 1");
    end

endmodule
