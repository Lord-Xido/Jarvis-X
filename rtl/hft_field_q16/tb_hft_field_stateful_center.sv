`timescale 1ns/1ps

module tb_hft_field_stateful_center;
    localparam integer ADDR_WIDTH = 6;
    localparam integer LATENCY = 17;

    logic clk = 1'b0;
    logic rst_n = 1'b0;

    logic cfg_write_valid = 1'b0;
    logic [ADDR_WIDTH-1:0] cfg_write_addr = '0;
    logic signed [31:0] cfg_psi = '0, cfg_omega = '0, cfg_flow = '0;
    logic cfg_write_ready;

    logic in_valid = 1'b0;
    logic [ADDR_WIDTH-1:0] in_coord = '0;
    logic signed [31:0] inventory_i = '0, delta_quantity_i = '0;
    logic side_bid_i = 1'b1;
    logic signed [31:0] psi_xm_i = '0, psi_xp_i = '0, psi_ym_i = '0;
    logic signed [31:0] psi_yp_i = '0, psi_zm_i = '0, psi_zp_i = '0;

    logic in_ready, conflict_o, out_valid, risk_accepted_o, alignment_error_o;
    logic [ADDR_WIDTH-1:0] out_coord;
    logic signed [31:0] psi_o, omega_o, flow_o, laplacian_o, score_o, quantity_o;
    logic signed [1:0] action_o;

    logic signed [31:0] model_psi, model_omega, model_flow;
    logic signed [31:0] ref_psi, ref_omega, ref_flow, ref_lap, ref_score, ref_quantity;
    logic signed [1:0] ref_action;
    logic ref_risk;

    logic signed [31:0] exp_psi, exp_omega, exp_flow, exp_lap, exp_score, exp_quantity;
    logic signed [1:0] exp_action;
    logic exp_risk;

    integer failures;
    integer cycles_waited;

    localparam logic [ADDR_WIDTH-1:0] A = 6'h15;

    always #5 clk = ~clk;

    hft_field_cell_pow2 ref_core (
        .psi_i(model_psi), .omega_i(model_omega), .flow_i(model_flow),
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

    hft_field_stateful_center #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .LATENCY_CYCLES(LATENCY)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_write_valid(cfg_write_valid), .cfg_write_addr(cfg_write_addr),
        .cfg_psi(cfg_psi), .cfg_omega(cfg_omega), .cfg_flow(cfg_flow),
        .cfg_write_ready(cfg_write_ready),
        .in_valid(in_valid), .in_coord(in_coord),
        .inventory_i(inventory_i), .delta_quantity_i(delta_quantity_i),
        .side_bid_i(side_bid_i),
        .psi_xm_i(psi_xm_i), .psi_xp_i(psi_xp_i),
        .psi_ym_i(psi_ym_i), .psi_yp_i(psi_yp_i),
        .psi_zm_i(psi_zm_i), .psi_zp_i(psi_zp_i),
        .in_ready(in_ready), .conflict_o(conflict_o),
        .out_valid(out_valid), .out_coord(out_coord),
        .psi_o(psi_o), .omega_o(omega_o), .flow_o(flow_o),
        .laplacian_o(laplacian_o), .score_o(score_o),
        .action_o(action_o), .quantity_o(quantity_o),
        .risk_accepted_o(risk_accepted_o),
        .alignment_error_o(alignment_error_o)
    );

    task automatic configure_state(
        input logic [ADDR_WIDTH-1:0] coord,
        input logic signed [31:0] psi,
        input logic signed [31:0] omega,
        input logic signed [31:0] flow
    );
        begin
            @(negedge clk);
            in_valid = 1'b0;
            cfg_write_addr = coord;
            cfg_psi = psi;
            cfg_omega = omega;
            cfg_flow = flow;
            cfg_write_valid = 1'b1;
            #1;
            if (!cfg_write_ready) begin
                $display("FAIL config unexpectedly not ready");
                failures = failures + 1;
            end
            @(posedge clk);
            #1;
            @(negedge clk);
            cfg_write_valid = 1'b0;
        end
    endtask

    task automatic prepare_event(
        input logic signed [31:0] inventory,
        input logic signed [31:0] delta,
        input logic side_bid,
        input logic signed [31:0] neighbour_bias
    );
        begin
            inventory_i = inventory;
            delta_quantity_i = delta;
            side_bid_i = side_bid;
            psi_xm_i = model_psi + neighbour_bias;
            psi_xp_i = model_psi - neighbour_bias - 32'sd1700;
            psi_ym_i = model_psi + neighbour_bias + 32'sd2300;
            psi_yp_i = model_psi - neighbour_bias - 32'sd3100;
            psi_zm_i = model_psi + neighbour_bias + 32'sd4300;
            psi_zp_i = model_psi - neighbour_bias - 32'sd5900;
        end
    endtask

    task automatic capture_expected;
        begin
            #1;
            exp_psi = ref_psi;
            exp_omega = ref_omega;
            exp_flow = ref_flow;
            exp_lap = ref_lap;
            exp_score = ref_score;
            exp_action = ref_action;
            exp_quantity = ref_quantity;
            exp_risk = ref_risk;
        end
    endtask

    task automatic check_result;
        begin
            if (out_coord !== A ||
                psi_o !== exp_psi || omega_o !== exp_omega || flow_o !== exp_flow ||
                laplacian_o !== exp_lap || score_o !== exp_score ||
                action_o !== exp_action || quantity_o !== exp_quantity ||
                risk_accepted_o !== exp_risk) begin
                $display("FAIL stateful result coord=%h/%h psi=%0d/%0d omega=%0d/%0d flow=%0d/%0d score=%0d/%0d",
                         out_coord, A, psi_o, exp_psi, omega_o, exp_omega,
                         flow_o, exp_flow, score_o, exp_score);
                failures = failures + 1;
            end
            if (alignment_error_o) begin
                $display("FAIL alignment_error_o asserted");
                failures = failures + 1;
            end
        end
    endtask

    task automatic wait_for_result;
        begin
            cycles_waited = 0;
            while (!out_valid && cycles_waited < LATENCY + 4) begin
                @(posedge clk);
                #1;
                cycles_waited = cycles_waited + 1;
            end
            if (!out_valid) begin
                $display("FAIL timed out waiting for stateful result");
                failures = failures + 1;
            end else begin
                check_result();
            end
        end
    endtask

    initial begin
        failures = 0;
        model_psi = 32'sd210000;
        model_omega = -32'sd70000;
        model_flow = 32'sd33000;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        configure_state(A, model_psi, model_omega, model_flow);

        // First transition from configured persistent state.
        @(negedge clk);
        in_coord = A;
        prepare_event(32'sd500000, 32'sd85000, 1'b1, 32'sd7000);
        capture_expected();
        in_valid = 1'b1;
        #1;
        if (!in_ready || conflict_o) begin
            $display("FAIL first A transaction not accepted");
            failures = failures + 1;
        end
        @(posedge clk);
        #1;

        // Same coordinate must remain blocked while the first update is in flight.
        @(negedge clk);
        in_valid = 1'b1;
        #1;
        if (in_ready || !conflict_o) begin
            $display("FAIL same-coordinate recurrence was not blocked in flight");
            failures = failures + 1;
        end
        @(posedge clk);
        #1;
        @(negedge clk);
        in_valid = 1'b0;

        wait_for_result();

        // Update the software model to the committed result. The RTL store
        // writes this same state on the next rising edge.
        model_psi = exp_psi;
        model_omega = exp_omega;
        model_flow = exp_flow;

        @(posedge clk);
        #1;

        // Second A transition must reload the newly committed state, not the
        // original configured state. Matching the oracle proves recurrence.
        @(negedge clk);
        prepare_event(-32'sd350000, 32'sd62000, 1'b0, 32'sd9000);
        capture_expected();
        in_coord = A;
        in_valid = 1'b1;
        #1;
        if (!in_ready || conflict_o) begin
            $display("FAIL A was not accepted after prior commit");
            failures = failures + 1;
        end
        @(posedge clk);
        #1;
        @(negedge clk);
        in_valid = 1'b0;

        wait_for_result();

        // Configuration/event collision must fail closed: neither interface
        // claims readiness for the ambiguous simultaneous operation.
        @(posedge clk);
        #1;
        @(negedge clk);
        in_coord = A;
        in_valid = 1'b1;
        cfg_write_valid = 1'b1;
        cfg_write_addr = A;
        cfg_psi = 32'sd1;
        cfg_omega = 32'sd2;
        cfg_flow = 32'sd3;
        #1;
        if (in_ready || cfg_write_ready || !conflict_o) begin
            $display("FAIL config/event collision did not fail closed");
            failures = failures + 1;
        end
        @(posedge clk);
        #1;
        @(negedge clk);
        in_valid = 1'b0;
        cfg_write_valid = 1'b0;

        if (failures == 0)
            $display("PASS: persistent center state commits and reloads bit-exactly across same-cell recurrence; config/event collision fails closed");
        else begin
            $display("FAIL: %0d stateful-center failures", failures);
            $fatal(1);
        end

        $finish;
    end
endmodule
