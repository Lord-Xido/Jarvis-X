`timescale 1ns/1ps

module tb_hft_field_cell_staged;
    localparam integer LATENCY = 17;
    localparam integer MAX_CYCLES = 256;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic in_valid = 1'b0;

    logic signed [31:0] psi_i;
    logic signed [31:0] omega_i;
    logic signed [31:0] flow_i;
    logic signed [31:0] inventory_i;
    logic signed [31:0] delta_quantity_i;
    logic side_bid_i;
    logic signed [31:0] psi_xm_i;
    logic signed [31:0] psi_xp_i;
    logic signed [31:0] psi_ym_i;
    logic signed [31:0] psi_yp_i;
    logic signed [31:0] psi_zm_i;
    logic signed [31:0] psi_zp_i;

    logic signed [31:0] ref_psi, ref_omega, ref_flow, ref_lap, ref_score, ref_quantity;
    logic signed [1:0] ref_action;
    logic ref_risk;

    logic out_valid;
    logic signed [31:0] dut_psi, dut_omega, dut_flow, dut_lap, dut_score, dut_quantity;
    logic signed [1:0] dut_action;
    logic dut_risk;

    logic exp_valid [0:MAX_CYCLES];
    logic signed [31:0] exp_psi [0:MAX_CYCLES];
    logic signed [31:0] exp_omega [0:MAX_CYCLES];
    logic signed [31:0] exp_flow [0:MAX_CYCLES];
    logic signed [31:0] exp_lap [0:MAX_CYCLES];
    logic signed [31:0] exp_score [0:MAX_CYCLES];
    logic signed [1:0] exp_action [0:MAX_CYCLES];
    logic signed [31:0] exp_quantity [0:MAX_CYCLES];
    logic exp_risk [0:MAX_CYCLES];

    integer cycle;
    integer n;
    integer i;
    integer failures;

    always #5 clk = ~clk;

    hft_field_cell_pow2 ref_core (
        .psi_i(psi_i), .omega_i(omega_i), .flow_i(flow_i),
        .inventory_i(inventory_i), .delta_quantity_i(delta_quantity_i),
        .side_bid_i(side_bid_i),
        .psi_xm_i(psi_xm_i), .psi_xp_i(psi_xp_i),
        .psi_ym_i(psi_ym_i), .psi_yp_i(psi_yp_i),
        .psi_zm_i(psi_zm_i), .psi_zp_i(psi_zp_i),
        .psi_o(ref_psi), .omega_o(ref_omega), .flow_o(ref_flow),
        .laplacian_o(ref_lap), .score_o(ref_score),
        .action_o(ref_action), .quantity_o(ref_quantity),
        .risk_accepted_o(ref_risk)
    );

    hft_field_cell_staged dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
        .psi_i(psi_i), .omega_i(omega_i), .flow_i(flow_i),
        .inventory_i(inventory_i), .delta_quantity_i(delta_quantity_i),
        .side_bid_i(side_bid_i),
        .psi_xm_i(psi_xm_i), .psi_xp_i(psi_xp_i),
        .psi_ym_i(psi_ym_i), .psi_yp_i(psi_yp_i),
        .psi_zm_i(psi_zm_i), .psi_zp_i(psi_zp_i),
        .out_valid(out_valid), .psi_o(dut_psi), .omega_o(dut_omega),
        .flow_o(dut_flow), .laplacian_o(dut_lap), .score_o(dut_score),
        .action_o(dut_action), .quantity_o(dut_quantity),
        .risk_accepted_o(dut_risk)
    );

    task automatic drive_vector(input integer k);
        begin
            psi_i = $signed((k * 104729) % 3000000) - 32'sd1500000;
            omega_i = $signed((k * 65537) % 2200000) - 32'sd1100000;
            flow_i = $signed((k * 8191) % 1600000) - 32'sd800000;
            inventory_i = $signed((k * 32771) % 8200000) - 32'sd4100000;
            delta_quantity_i = $signed(((k + 3) * 12289) % 700000) - 32'sd350000;
            side_bid_i = k[0];
            psi_xm_i = psi_i + 32'sd17000;
            psi_xp_i = psi_i - 32'sd29000;
            psi_ym_i = psi_i + 32'sd43000;
            psi_yp_i = psi_i - 32'sd51000;
            psi_zm_i = psi_i + 32'sd61000;
            psi_zp_i = psi_i - 32'sd73000;
        end
    endtask

    task automatic check_cycle;
        begin
            if (exp_valid[cycle]) begin
                if (!out_valid ||
                    dut_psi !== exp_psi[cycle] ||
                    dut_omega !== exp_omega[cycle] ||
                    dut_flow !== exp_flow[cycle] ||
                    dut_lap !== exp_lap[cycle] ||
                    dut_score !== exp_score[cycle] ||
                    dut_action !== exp_action[cycle] ||
                    dut_quantity !== exp_quantity[cycle] ||
                    dut_risk !== exp_risk[cycle]) begin
                    $display("FAIL cycle=%0d valid=%0b psi=%0d/%0d omega=%0d/%0d flow=%0d/%0d lap=%0d/%0d score=%0d/%0d action=%0d/%0d qty=%0d/%0d risk=%0b/%0b",
                             cycle, out_valid,
                             dut_psi, exp_psi[cycle], dut_omega, exp_omega[cycle],
                             dut_flow, exp_flow[cycle], dut_lap, exp_lap[cycle],
                             dut_score, exp_score[cycle], dut_action, exp_action[cycle],
                             dut_quantity, exp_quantity[cycle], dut_risk, exp_risk[cycle]);
                    failures = failures + 1;
                end
            end else if (out_valid) begin
                $display("FAIL unexpected out_valid at cycle=%0d", cycle);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        cycle = 0;
        failures = 0;
        psi_i = '0; omega_i = '0; flow_i = '0; inventory_i = '0;
        delta_quantity_i = '0; side_bid_i = 1'b1;
        psi_xm_i = '0; psi_xp_i = '0; psi_ym_i = '0; psi_yp_i = '0; psi_zm_i = '0; psi_zp_i = '0;

        for (i = 0; i <= MAX_CYCLES; i = i + 1) begin
            exp_valid[i] = 1'b0;
            exp_psi[i] = '0; exp_omega[i] = '0; exp_flow[i] = '0; exp_lap[i] = '0;
            exp_score[i] = '0; exp_action[i] = '0; exp_quantity[i] = '0; exp_risk[i] = 1'b0;
        end

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // Consecutive transactions prove II=1 arithmetic throughput.
        for (n = 0; n < 96; n = n + 1) begin
            @(negedge clk);
            drive_vector(n);
            in_valid = 1'b1;

            @(posedge clk);
            #1;
            cycle = cycle + 1;
            check_cycle();
            exp_valid[cycle + LATENCY] = 1'b1;
            exp_psi[cycle + LATENCY] = ref_psi;
            exp_omega[cycle + LATENCY] = ref_omega;
            exp_flow[cycle + LATENCY] = ref_flow;
            exp_lap[cycle + LATENCY] = ref_lap;
            exp_score[cycle + LATENCY] = ref_score;
            exp_action[cycle + LATENCY] = ref_action;
            exp_quantity[cycle + LATENCY] = ref_quantity;
            exp_risk[cycle + LATENCY] = ref_risk;
        end

        @(negedge clk);
        in_valid = 1'b0;

        repeat (LATENCY + 4) begin
            @(posedge clk);
            #1;
            cycle = cycle + 1;
            check_cycle();
        end

        if (failures == 0)
            $display("PASS: staged arithmetic pipeline matched multiplier-free oracle for 96 II=1 transactions at 17-cycle latency");
        else begin
            $display("FAIL: %0d staged-pipeline mismatches", failures);
            $fatal(1);
        end

        $finish;
    end
endmodule
