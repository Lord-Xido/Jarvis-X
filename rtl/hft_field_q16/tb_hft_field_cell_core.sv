`timescale 1ns/1ps

module tb_hft_field_cell_core;
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

    logic signed [31:0] psi_o;
    logic signed [31:0] omega_o;
    logic signed [31:0] flow_o;
    logic signed [31:0] laplacian_o;
    logic signed [31:0] score_o;
    logic signed [1:0]  action_o;
    logic signed [31:0] quantity_o;
    logic               risk_accepted_o;

    hft_field_cell_core dut (
        .psi_i,
        .omega_i,
        .flow_i,
        .inventory_i,
        .delta_quantity_i,
        .side_bid_i,
        .psi_xm_i,
        .psi_xp_i,
        .psi_ym_i,
        .psi_yp_i,
        .psi_zm_i,
        .psi_zp_i,
        .psi_o,
        .omega_o,
        .flow_o,
        .laplacian_o,
        .score_o,
        .action_o,
        .quantity_o,
        .risk_accepted_o
    );

    task automatic check_vector(
        input string name,
        input logic signed [31:0] in_psi,
        input logic signed [31:0] in_omega,
        input logic signed [31:0] in_flow,
        input logic signed [31:0] in_inventory,
        input logic signed [31:0] in_delta,
        input logic in_side_bid,
        input logic signed [31:0] in_xm,
        input logic signed [31:0] in_xp,
        input logic signed [31:0] in_ym,
        input logic signed [31:0] in_yp,
        input logic signed [31:0] in_zm,
        input logic signed [31:0] in_zp,
        input logic signed [31:0] exp_psi,
        input logic signed [31:0] exp_omega,
        input logic signed [31:0] exp_flow,
        input logic signed [31:0] exp_lap,
        input logic signed [31:0] exp_score,
        input logic signed [1:0]  exp_action,
        input logic signed [31:0] exp_quantity,
        input logic exp_risk
    );
        begin
            psi_i = in_psi;
            omega_i = in_omega;
            flow_i = in_flow;
            inventory_i = in_inventory;
            delta_quantity_i = in_delta;
            side_bid_i = in_side_bid;
            psi_xm_i = in_xm;
            psi_xp_i = in_xp;
            psi_ym_i = in_ym;
            psi_yp_i = in_yp;
            psi_zm_i = in_zm;
            psi_zp_i = in_zp;
            #1;

            if (psi_o !== exp_psi)
                $fatal(1, "%s psi: got %0d expected %0d", name, psi_o, exp_psi);
            if (omega_o !== exp_omega)
                $fatal(1, "%s omega: got %0d expected %0d", name, omega_o, exp_omega);
            if (flow_o !== exp_flow)
                $fatal(1, "%s flow: got %0d expected %0d", name, flow_o, exp_flow);
            if (laplacian_o !== exp_lap)
                $fatal(1, "%s lap: got %0d expected %0d", name, laplacian_o, exp_lap);
            if (score_o !== exp_score)
                $fatal(1, "%s score: got %0d expected %0d", name, score_o, exp_score);
            if (action_o !== exp_action)
                $fatal(1, "%s action: got %0d expected %0d", name, action_o, exp_action);
            if (quantity_o !== exp_quantity)
                $fatal(1, "%s quantity: got %0d expected %0d", name, quantity_o, exp_quantity);
            if (risk_accepted_o !== exp_risk)
                $fatal(1, "%s risk: got %0d expected %0d", name, risk_accepted_o, exp_risk);

            $display("PASS %-16s psi=%0d omega=%0d flow=%0d lap=%0d score=%0d action=%0d qty=%0d risk=%0d",
                     name, psi_o, omega_o, flow_o, laplacian_o, score_o,
                     action_o, quantity_o, risk_accepted_o);
        end
    endtask

    initial begin
        // Golden vectors were generated from the C++ Q16.16 reference semantics.
        check_vector(
            "bid1",
            32'sd0, 32'sd0, 32'sd0, 32'sd0, 32'sd65536, 1'b1,
            32'sd0, 32'sd0, 32'sd0, 32'sd0, 32'sd0, 32'sd0,
            32'sd4096, 32'sd256, 32'sd65536, 32'sd0, 32'sd36992,
            2'sd1, 32'sd65536, 1'b1
        );

        check_vector(
            "ask1",
            32'sd0, 32'sd0, 32'sd0, 32'sd0, 32'sd65536, 1'b0,
            32'sd0, 32'sd0, 32'sd0, 32'sd0, 32'sd0, 32'sd0,
            -32'sd4096, -32'sd256, -32'sd65536, 32'sd0, -32'sd36992,
            -2'sd1, 32'sd65536, 1'b1
        );

        check_vector(
            "mixed",
            32'sd32768, 32'sd16384, -32'sd8192, 32'sd131072,
            32'sd32768, 1'b1,
            32'sd16384, -32'sd16384, 32'sd8192, 32'sd0,
            32'sd4096, -32'sd4096,
            32'sd33568, 32'sd17458, 32'sd25600, -32'sd188416,
            -32'sd1223, 2'sd0, 32'sd0, 1'b0
        );

        check_vector(
            "risk_reject",
            32'sd0, 32'sd0, 32'sd0, 32'sd4194304,
            32'sd2097152, 1'b1,
            32'sd0, 32'sd0, 32'sd0, 32'sd0, 32'sd0, 32'sd0,
            32'sd131072, 32'sd8192, 32'sd2097152, 32'sd0,
            32'sd135168, 2'sd0, 32'sd0, 1'b0
        );

        check_vector(
            "zeroqty_forced",
            32'sd65536, 32'sd0, 32'sd0, 32'sd0, 32'sd0, 1'b1,
            32'sd0, 32'sd0, 32'sd0, 32'sd0, 32'sd0, 32'sd0,
            32'sd61952, 32'sd3872, 32'sd0, -32'sd393216,
            32'sd14736, 2'sd1, 32'sd262144, 1'b1
        );

        $display("hft_field_cell_core RTL equivalence vectors passed");
        $finish;
    end
endmodule
