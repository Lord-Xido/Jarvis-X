module hft_field_stateful_center #(
    parameter integer ADDR_WIDTH = 6,
    parameter integer LATENCY_CYCLES = 17
) (
    input  logic                    clk,
    input  logic                    rst_n,

    // Initialization/configuration write port. Use before streaming events.
    input  logic                    cfg_write_valid,
    input  logic [ADDR_WIDTH-1:0]   cfg_write_addr,
    input  logic signed [31:0]      cfg_psi,
    input  logic signed [31:0]      cfg_omega,
    input  logic signed [31:0]      cfg_flow,
    output logic                    cfg_write_ready,

    // Event transaction. Center Psi/Omega/Flow are loaded from the store.
    input  logic                    in_valid,
    input  logic [ADDR_WIDTH-1:0]   in_coord,
    input  logic signed [31:0]      inventory_i,
    input  logic signed [31:0]      delta_quantity_i,
    input  logic                    side_bid_i,

    // Neighbour Psi values remain external until the seven-read banking
    // architecture is implemented.
    input  logic signed [31:0]      psi_xm_i,
    input  logic signed [31:0]      psi_xp_i,
    input  logic signed [31:0]      psi_ym_i,
    input  logic signed [31:0]      psi_yp_i,
    input  logic signed [31:0]      psi_zm_i,
    input  logic signed [31:0]      psi_zp_i,

    output logic                    in_ready,
    output logic                    conflict_o,
    output logic                    out_valid,
    output logic [ADDR_WIDTH-1:0]   out_coord,
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

    logic signed [31:0] state_psi;
    logic signed [31:0] state_omega;
    logic signed [31:0] state_flow;

    logic store_write_valid;
    logic [ADDR_WIDTH-1:0] store_write_addr;
    logic signed [31:0] store_psi_w;
    logic signed [31:0] store_omega_w;
    logic signed [31:0] store_flow_w;

    // Pipeline retirement has priority over configuration so a committed
    // market-state transition cannot be overwritten by a simultaneous host
    // write. The host can observe cfg_write_ready and retry deterministically.
    always_comb begin
        cfg_write_ready = !out_valid;

        if (out_valid) begin
            store_write_valid = 1'b1;
            store_write_addr = out_coord;
            store_psi_w = psi_o;
            store_omega_w = omega_o;
            store_flow_w = flow_o;
        end else begin
            store_write_valid = cfg_write_valid;
            store_write_addr = cfg_write_addr;
            store_psi_w = cfg_psi;
            store_omega_w = cfg_omega;
            store_flow_w = cfg_flow;
        end
    end

    hft_field_state_store #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DEPTH(1 << ADDR_WIDTH)
    ) state_store (
        .clk(clk),
        .read_addr(in_coord),
        .psi_r(state_psi),
        .omega_r(state_omega),
        .flow_r(state_flow),
        .write_valid(store_write_valid),
        .write_addr(store_write_addr),
        .psi_w(store_psi_w),
        .omega_w(store_omega_w),
        .flow_w(store_flow_w)
    );

    hft_field_guarded_pipeline #(
        .COORD_WIDTH(ADDR_WIDTH),
        .LATENCY_CYCLES(LATENCY_CYCLES)
    ) guarded_pipeline (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_coord(in_coord),
        .psi_i(state_psi),
        .omega_i(state_omega),
        .flow_i(state_flow),
        .inventory_i(inventory_i),
        .delta_quantity_i(delta_quantity_i),
        .side_bid_i(side_bid_i),
        .psi_xm_i(psi_xm_i),
        .psi_xp_i(psi_xp_i),
        .psi_ym_i(psi_ym_i),
        .psi_yp_i(psi_yp_i),
        .psi_zm_i(psi_zm_i),
        .psi_zp_i(psi_zp_i),
        .in_ready(in_ready),
        .conflict_o(conflict_o),
        .out_valid(out_valid),
        .out_coord(out_coord),
        .psi_o(psi_o),
        .omega_o(omega_o),
        .flow_o(flow_o),
        .laplacian_o(laplacian_o),
        .score_o(score_o),
        .action_o(action_o),
        .quantity_o(quantity_o),
        .risk_accepted_o(risk_accepted_o),
        .alignment_error_o(alignment_error_o)
    );

endmodule
