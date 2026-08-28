module hft_field_cell_pipeline #(
    parameter integer LATENCY_CYCLES = 17
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    in_valid,

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

    output logic                    out_valid,
    output logic signed [31:0]      psi_o,
    output logic signed [31:0]      omega_o,
    output logic signed [31:0]      flow_o,
    output logic signed [31:0]      laplacian_o,
    output logic signed [31:0]      score_o,
    output logic signed [1:0]       action_o,
    output logic signed [31:0]      quantity_o,
    output logic                    risk_accepted_o
);

    // This module establishes the fixed valid-to-valid latency contract around
    // the bit-exact multiplier-free arithmetic core. It intentionally does not
    // claim timing closure: hft_field_cell_pow2 remains combinational here and
    // is the correctness oracle to be repartitioned across registers later.

    logic signed [31:0] comb_psi;
    logic signed [31:0] comb_omega;
    logic signed [31:0] comb_flow;
    logic signed [31:0] comb_laplacian;
    logic signed [31:0] comb_score;
    logic signed [1:0]  comb_action;
    logic signed [31:0] comb_quantity;
    logic               comb_risk;

    hft_field_cell_pow2 arithmetic_core (
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
        .psi_o(comb_psi),
        .omega_o(comb_omega),
        .flow_o(comb_flow),
        .laplacian_o(comb_laplacian),
        .score_o(comb_score),
        .action_o(comb_action),
        .quantity_o(comb_quantity),
        .risk_accepted_o(comb_risk)
    );

    logic                    valid_pipe [0:LATENCY_CYCLES];
    logic signed [31:0]      psi_pipe [0:LATENCY_CYCLES];
    logic signed [31:0]      omega_pipe [0:LATENCY_CYCLES];
    logic signed [31:0]      flow_pipe [0:LATENCY_CYCLES];
    logic signed [31:0]      lap_pipe [0:LATENCY_CYCLES];
    logic signed [31:0]      score_pipe [0:LATENCY_CYCLES];
    logic signed [1:0]       action_pipe [0:LATENCY_CYCLES];
    logic signed [31:0]      quantity_pipe [0:LATENCY_CYCLES];
    logic                    risk_pipe [0:LATENCY_CYCLES];

    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i <= LATENCY_CYCLES; i = i + 1) begin
                valid_pipe[i] <= 1'b0;
                psi_pipe[i] <= '0;
                omega_pipe[i] <= '0;
                flow_pipe[i] <= '0;
                lap_pipe[i] <= '0;
                score_pipe[i] <= '0;
                action_pipe[i] <= '0;
                quantity_pipe[i] <= '0;
                risk_pipe[i] <= 1'b0;
            end
        end else begin
            valid_pipe[0] <= in_valid;
            if (in_valid) begin
                psi_pipe[0] <= comb_psi;
                omega_pipe[0] <= comb_omega;
                flow_pipe[0] <= comb_flow;
                lap_pipe[0] <= comb_laplacian;
                score_pipe[0] <= comb_score;
                action_pipe[0] <= comb_action;
                quantity_pipe[0] <= comb_quantity;
                risk_pipe[0] <= comb_risk;
            end

            for (i = 1; i <= LATENCY_CYCLES; i = i + 1) begin
                valid_pipe[i] <= valid_pipe[i - 1];
                if (valid_pipe[i - 1]) begin
                    psi_pipe[i] <= psi_pipe[i - 1];
                    omega_pipe[i] <= omega_pipe[i - 1];
                    flow_pipe[i] <= flow_pipe[i - 1];
                    lap_pipe[i] <= lap_pipe[i - 1];
                    score_pipe[i] <= score_pipe[i - 1];
                    action_pipe[i] <= action_pipe[i - 1];
                    quantity_pipe[i] <= quantity_pipe[i - 1];
                    risk_pipe[i] <= risk_pipe[i - 1];
                end
            end
        end
    end

    always_comb begin
        out_valid = valid_pipe[LATENCY_CYCLES];
        psi_o = psi_pipe[LATENCY_CYCLES];
        omega_o = omega_pipe[LATENCY_CYCLES];
        flow_o = flow_pipe[LATENCY_CYCLES];
        laplacian_o = lap_pipe[LATENCY_CYCLES];
        score_o = score_pipe[LATENCY_CYCLES];
        action_o = action_pipe[LATENCY_CYCLES];
        quantity_o = quantity_pipe[LATENCY_CYCLES];
        risk_accepted_o = risk_pipe[LATENCY_CYCLES];
    end

    initial begin
        if (LATENCY_CYCLES < 1)
            $error("LATENCY_CYCLES must be >= 1");
    end

endmodule
