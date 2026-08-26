`timescale 1ns/1ps

module tb_hft_field_pow2_equiv;
    logic signed [31:0] psi_i;
    logic signed [31:0] omega_i;
    logic signed [31:0] flow_i;
    logic signed [31:0] inventory_i;
    logic signed [31:0] delta_quantity_i;
    logic               side_bid_i;
    logic signed [31:0] psi_xm_i;
    logic signed [31:0] psi_xp_i;
    logic signed [31:0] psi_ym_i;
    logic signed [31:0] psi_yp_i;
    logic signed [31:0] psi_zm_i;
    logic signed [31:0] psi_zp_i;

    logic signed [31:0] ref_psi;
    logic signed [31:0] ref_omega;
    logic signed [31:0] ref_flow;
    logic signed [31:0] ref_lap;
    logic signed [31:0] ref_score;
    logic signed [1:0]  ref_action;
    logic signed [31:0] ref_quantity;
    logic               ref_risk;

    logic signed [31:0] opt_psi;
    logic signed [31:0] opt_omega;
    logic signed [31:0] opt_flow;
    logic signed [31:0] opt_lap;
    logic signed [31:0] opt_score;
    logic signed [1:0]  opt_action;
    logic signed [31:0] opt_quantity;
    logic               opt_risk;

    integer vectors_checked;
    logic [31:0] rng;

    hft_field_cell_core reference (
        .psi_i, .omega_i, .flow_i, .inventory_i, .delta_quantity_i,
        .side_bid_i, .psi_xm_i, .psi_xp_i, .psi_ym_i, .psi_yp_i,
        .psi_zm_i, .psi_zp_i,
        .psi_o(ref_psi), .omega_o(ref_omega), .flow_o(ref_flow),
        .laplacian_o(ref_lap), .score_o(ref_score), .action_o(ref_action),
        .quantity_o(ref_quantity), .risk_accepted_o(ref_risk)
    );

    hft_field_cell_pow2 optimized (
        .psi_i, .omega_i, .flow_i, .inventory_i, .delta_quantity_i,
        .side_bid_i, .psi_xm_i, .psi_xp_i, .psi_ym_i, .psi_yp_i,
        .psi_zm_i, .psi_zp_i,
        .psi_o(opt_psi), .omega_o(opt_omega), .flow_o(opt_flow),
        .laplacian_o(opt_lap), .score_o(opt_score), .action_o(opt_action),
        .quantity_o(opt_quantity), .risk_accepted_o(opt_risk)
    );

    function automatic logic [31:0] xorshift32(input logic [31:0] x);
        logic [31:0] y;
        begin
            y = x;
            y = y ^ (y << 13);
            y = y ^ (y >> 17);
            y = y ^ (y << 5);
            xorshift32 = y;
        end
    endfunction

    task automatic next_word(output logic signed [31:0] value);
        begin
            rng = xorshift32(rng);
            value = $signed(rng);
        end
    endtask

    task automatic check_current(input string name);
        begin
            #1;
            vectors_checked = vectors_checked + 1;
            if (ref_psi !== opt_psi ||
                ref_omega !== opt_omega ||
                ref_flow !== opt_flow ||
                ref_lap !== opt_lap ||
                ref_score !== opt_score ||
                ref_action !== opt_action ||
                ref_quantity !== opt_quantity ||
                ref_risk !== opt_risk) begin
                $display("EQUIV FAIL %s vector=%0d", name, vectors_checked);
                $display(" inputs psi=%0d omega=%0d flow=%0d inv=%0d dq=%0d bid=%0d",
                         psi_i, omega_i, flow_i, inventory_i, delta_quantity_i,
                         side_bid_i);
                $display(" neighbors xm=%0d xp=%0d ym=%0d yp=%0d zm=%0d zp=%0d",
                         psi_xm_i, psi_xp_i, psi_ym_i, psi_yp_i, psi_zm_i, psi_zp_i);
                $display(" ref psi=%0d omega=%0d flow=%0d lap=%0d score=%0d action=%0d qty=%0d risk=%0d",
                         ref_psi, ref_omega, ref_flow, ref_lap, ref_score,
                         ref_action, ref_quantity, ref_risk);
                $display(" opt psi=%0d omega=%0d flow=%0d lap=%0d score=%0d action=%0d qty=%0d risk=%0d",
                         opt_psi, opt_omega, opt_flow, opt_lap, opt_score,
                         opt_action, opt_quantity, opt_risk);
                $fatal(1, "multiplier-free core is not bit-exact");
            end
        end
    endtask

    task automatic drive_extreme(
        input logic signed [31:0] a,
        input logic signed [31:0] b
    );
        begin
            psi_i = a;
            omega_i = b;
            flow_i = a;
            inventory_i = b;
            delta_quantity_i = a;
            side_bid_i = 1'b1;
            psi_xm_i = a;
            psi_xp_i = b;
            psi_ym_i = a;
            psi_yp_i = b;
            psi_zm_i = a;
            psi_zp_i = b;
            check_current("extreme-bid");

            side_bid_i = 1'b0;
            check_current("extreme-ask");
        end
    endtask

    integer i;
    initial begin
        vectors_checked = 0;
        rng = 32'h4A58_4831;

        // Directed saturation/zero/sign boundary cases.
        drive_extreme(32'sh7fffffff, 32'sh80000000);
        drive_extreme(32'sh80000000, 32'sh7fffffff);
        drive_extreme(32'sd0, 32'sd0);
        drive_extreme(32'sd1, -32'sd1);
        drive_extreme(32'sd65536, -32'sd65536);
        drive_extreme(32'sd4194304, -32'sd4194304);

        // Deterministic full-range stress. This exercises saturation as well as
        // truncation-toward-zero behavior for negative dyadic products.
        for (i = 0; i < 10000; i = i + 1) begin
            next_word(psi_i);
            next_word(omega_i);
            next_word(flow_i);
            next_word(inventory_i);
            next_word(delta_quantity_i);
            rng = xorshift32(rng);
            side_bid_i = rng[0];
            next_word(psi_xm_i);
            next_word(psi_xp_i);
            next_word(psi_ym_i);
            next_word(psi_yp_i);
            next_word(psi_zm_i);
            next_word(psi_zp_i);
            check_current("random");
        end

        $display("PASS multiplier-free equivalence: %0d vectors bit-exact", vectors_checked);
        $finish;
    end
endmodule
