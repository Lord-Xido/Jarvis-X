`timescale 1ns/1ps

module tb_hft_field_cell_pipeline;
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

    logic signed [31:0] ref_psi;
    logic signed [31:0] ref_omega;
    logic signed [31:0] ref_flow;
    logic signed [31:0] ref_lap;
    logic signed [31:0] ref_score;
    logic signed [1:0] ref_action;
    logic signed [31:0] ref_quantity;
    logic ref_risk;

    logic out_valid;
    logic signed [31:0] pipe_psi;
    logic signed [31:0] pipe_omega;
    logic signed [31:0] pipe_flow;
    logic signed [31:0] pipe_lap;
    logic signed [31:0] pipe_score;
    logic signed [1:0] pipe_action;
    logic signed [31:0] pipe_quantity;
    logic pipe_risk;

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

    hft_field_cell_pipeline #(.LATENCY_CYCLES(LATENCY)) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
        .psi_i(psi_i), .omega_i(omega_i), .flow_i(flow_i),
        .inventory_i(inventory_i), .delta_quantity_i(delta_quantity_i),
        .side_bid_i(side_bid_i),
        .psi_xm_i(psi_xm_i), .psi_xp_i(psi_xp_i),
        .psi_ym_i(psi_ym_i), .psi_yp_i(psi_yp_i),
        .psi_zm_i(psi_zm_i), .psi_zp_i(psi_zp_i),
        .out_valid(out_valid), .psi_o(pipe_psi), .omega_o(pipe_omega),
        .flow_o(pipe_flow), .laplacian_o(pipe_lap), .score_o(pipe_score),
        .action_o(pipe_action), .quantity_o(pipe_quantity),
        .risk_accepted_o(pipe_risk)
    );

    task automatic drive_vector(input integer k);
        begin
            psi_i = $signed((k * 7919) % 1200000) - 32'sd600000;
            omega_i = $signed((k * 3571) % 800000) - 32'sd400000;
            flow_i = $signed((k * 1237) % 500000) - 32'sd250000;
            inventory_i = $signed((k * 6151) % 6000000) - 32'sd3000000;
            delta_quantity_i = $signed(((k + 1) * 4099) % 350000) - 32'sd175000;
            side_bid_i = k[0];
            psi_xm_i = psi_i + 32'sd1100;
            psi_xp_i = psi_i - 32'sd2300;
            psi_ym_i = psi_i + 32'sd3700;
            psi_yp_i = psi_i - 32'sd4100;
            psi_zm_i = psi_i + 32'sd5300;
            psi_zp_i = psi_i - 32'sd6700;
        end
    endtask

    task automatic check_cycle;
        begin
            if (exp_valid[cycle]) begin
                if (!out_valid ||
                    pipe_psi !== exp_psi[cycle] ||
                    pipe_omega !== exp_omega[cycle] ||
                    pipe_flow !== exp_flow[cycle] ||
                    pipe_lap !== exp_lap[cycle] ||
                    pipe_score !== exp_score[cycle] ||
                    pipe_action !== exp_action[cycle] ||
                    pipe_quantity !== exp_quantity[cycle] ||
                    pipe_risk !== exp_risk[cycle]) begin
                    $display("FAIL cycle=%0d out_valid=%0b psi=%0d/%0d score=%0d/%0d action=%0d/%0d risk=%0b/%0b",
                             cycle, out_valid, pipe_psi, exp_psi[cycle],
                             pipe_score, exp_score[cycle], pipe_action,
                             exp_action[cycle], pipe_risk, exp_risk[cycle]);
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
        psi_i = '0;
        omega_i = '0;
        flow_i = '0;
        inventory_i = '0;
        delta_quantity_i = '0;
        side_bid_i = 1'b1;
        psi_xm_i = '0;
        psi_xp_i = '0;
        psi_ym_i = '0;
        psi_yp_i = '0;
        psi_zm_i = '0;
        psi_zp_i = '0;

        for (i = 0; i <= MAX_CYCLES; i = i + 1) begin
            exp_valid[i] = 1'b0;
            exp_psi[i] = '0;
            exp_omega[i] = '0;
            exp_flow[i] = '0;
            exp_lap[i] = '0;
            exp_score[i] = '0;
            exp_action[i] = '0;
            exp_quantity[i] = '0;
            exp_risk[i] = 1'b0;
        end

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // Stream 64 independent arithmetic transactions at II=1.
        for (n = 0; n < 64; n = n + 1) begin
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
            $display("PASS: fixed-latency pipeline matched reference for 64 II=1 transactions at %0d cycles", LATENCY);
        else begin
            $display("FAIL: %0d pipeline mismatches", failures);
            $fatal(1);
        end

        $finish;
    end
endmodule
