module hft_field_guarded_pipeline #(
    parameter integer COORD_WIDTH = 16,
    parameter integer LATENCY_CYCLES = 17
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    in_valid,
    input  logic [COORD_WIDTH-1:0]  in_coord,

    input  logic signed [31:0]      psi_i,
    input  logic signed [31:0]      omega_i,
    input  logic signed [31:0]      flow_i,
    input  logic signed [31:0]      inventory_i,
    input  logic signed [31:0]      delta_quantity_i,
    input  logic                    side_bid_i,
    input  logic signed [31:0]      psi_xm_i,
    input  logic signed [31:0]      psi_xp_i,
    input  logic signed [31:0]      psi_ym_i,
    input  logic signed [31:0]      psi_yp_i,
    input  logic signed [31:0]      psi_zm_i,
    input  logic signed [31:0]      psi_zp_i,

    output logic                    in_ready,
    output logic                    conflict_o,
    output logic                    out_valid,
    output logic [COORD_WIDTH-1:0]  out_coord,
    output logic signed [31:0]      psi_o,
    output logic signed [31:0]      omega_o,
    output logic signed [31:0]      flow_o,
    output logic signed [31:0]      laplacian_o,
    output logic signed [31:0]      score_o,
    output logic signed [1:0]       action_o,
    output logic signed [31:0]      quantity_o,
    output logic                    risk_accepted_o,
    output logic                    alignment_error_o
);

    // Transaction integration boundary:
    //   1. reject a coordinate already in flight (RAW hazard),
    //   2. launch accepted independent coordinates into the staged core,
    //   3. retire coordinate and arithmetic result on the same cycle.
    //
    // This block still consumes state values supplied by an external store.
    // It does not yet implement BRAM/URAM banking or neighbour fetch.

    logic accept;
    logic commit_valid;
    logic [COORD_WIDTH-1:0] commit_coord;
    logic core_out_valid;

    hft_field_hazard_guard #(
        .COORD_WIDTH(COORD_WIDTH),
        .LATENCY_CYCLES(LATENCY_CYCLES)
    ) guard (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_coord(in_coord),
        .in_ready(in_ready),
        .accept_o(accept),
        .conflict_o(conflict_o),
        .commit_valid_o(commit_valid),
        .commit_coord_o(commit_coord)
    );

    hft_field_cell_staged arithmetic (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(accept),
        .psi_i(psi_i),
        .omega_i(omega_i),
        .flow_i(flow_i),
        .inventory_i(inventory_i),
        .delta_quantity_i(delta_quantity_i),
        .side_bid_i(side_bid_i),
        .psi_xm_i(psi_xm_i),
        .psi_xp_i(psi_xp_i),
        .psi_ym_i(psi_ym_i),
        .psi_yp_i(psi_yp_i),
        .psi_zm_i(psi_zm_i),
        .psi_zp_i(psi_zp_i),
        .out_valid(core_out_valid),
        .psi_o(psi_o),
        .omega_o(omega_o),
        .flow_o(flow_o),
        .laplacian_o(laplacian_o),
        .score_o(score_o),
        .action_o(action_o),
        .quantity_o(quantity_o),
        .risk_accepted_o(risk_accepted_o)
    );

    always_comb begin
        out_valid = core_out_valid && commit_valid;
        out_coord = commit_coord;
        alignment_error_o = core_out_valid ^ commit_valid;
    end

    initial begin
        if (LATENCY_CYCLES != 17)
            $error("hft_field_cell_staged currently has a fixed 17-cycle latency");
    end

endmodule
