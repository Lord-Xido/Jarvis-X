`timescale 1ns/1ps

module tb_hft_field_guarded_pipeline;
    localparam integer COORD_WIDTH = 12;
    localparam integer LATENCY = 17;
    localparam integer MAX_TX = 64;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic in_valid = 1'b0;
    logic [COORD_WIDTH-1:0] in_coord = '0;

    logic signed [31:0] psi_i, omega_i, flow_i, inventory_i, delta_quantity_i;
    logic side_bid_i;
    logic signed [31:0] psi_xm_i, psi_xp_i, psi_ym_i, psi_yp_i, psi_zm_i, psi_zp_i;

    logic in_ready, conflict_o, out_valid, risk_accepted_o, alignment_error_o;
    logic [COORD_WIDTH-1:0] out_coord;
    logic signed [31:0] psi_o, omega_o, flow_o, laplacian_o, score_o, quantity_o;
    logic signed [1:0] action_o;

    logic signed [31:0] ref_psi, ref_omega, ref_flow, ref_lap, ref_score, ref_quantity;
    logic signed [1:0] ref_action;
    logic ref_risk;

    logic [COORD_WIDTH-1:0] exp_coord [0:MAX_TX-1];
    logic signed [31:0] exp_psi [0:MAX_TX-1];
    logic signed [31:0] exp_omega [0:MAX_TX-1];
    logic signed [31:0] exp_flow [0:MAX_TX-1];
    logic signed [31:0] exp_lap [0:MAX_TX-1];
    logic signed [31:0] exp_score [0:MAX_TX-1];
    logic signed [1:0] exp_action [0:MAX_TX-1];
    logic signed [31:0] exp_quantity [0:MAX_TX-1];
    logic exp_risk [0:MAX_TX-1];
    integer exp_due [0:MAX_TX-1];

    integer head;
    integer tail;
    integer failures;
    integer cycle_count;
    integer k;

    localparam logic [COORD_WIDTH-1:0] A = 12'h123;
    localparam logic [COORD_WIDTH-1:0] B = 12'h456;
    localparam logic [COORD_WIDTH-1:0] C = 12'h789;

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

    hft_field_guarded_pipeline #(
        .COORD_WIDTH(COORD_WIDTH),
        .LATENCY_CYCLES(LATENCY)
    ) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_coord(in_coord),
        .psi_i(psi_i), .omega_i(omega_i), .flow_i(flow_i),
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

    task automatic drive_vector(input integer seed);
        begin
            psi_i = $signed((seed * 65537) % 2000000) - 32'sd1000000;
            omega_i = $signed((seed * 32771) % 1400000) - 32'sd700000;
            flow_i = $signed((seed * 12289) % 900000) - 32'sd450000;
            inventory_i = $signed((seed * 8191) % 7000000) - 32'sd3500000;
            delta_quantity_i = $signed(((seed + 5) * 4099) % 500000) - 32'sd250000;
            side_bid_i = seed[0];
            psi_xm_i = psi_i + 32'sd11000;
            psi_xp_i = psi_i - 32'sd13000;
            psi_ym_i = psi_i + 32'sd17000;
            psi_yp_i = psi_i - 32'sd19000;
            psi_zm_i = psi_i + 32'sd23000;
            psi_zp_i = psi_i - 32'sd29000;
        end
    endtask

    task automatic issue(
        input logic [COORD_WIDTH-1:0] coord,
        input integer seed,
        input logic expect_ready
    );
        begin
            @(negedge clk);
            in_coord = coord;
            drive_vector(seed);
            in_valid = 1'b1;
            #1;

            if (expect_ready) begin
                if (!in_ready || conflict_o) begin
                    $display("FAIL expected ready coord=%h ready=%0b conflict=%0b", coord, in_ready, conflict_o);
                    failures = failures + 1;
                end else begin
                    exp_coord[tail] = coord;
                    exp_psi[tail] = ref_psi;
                    exp_omega[tail] = ref_omega;
                    exp_flow[tail] = ref_flow;
                    exp_lap[tail] = ref_lap;
                    exp_score[tail] = ref_score;
                    exp_action[tail] = ref_action;
                    exp_quantity[tail] = ref_quantity;
                    exp_risk[tail] = ref_risk;
                    exp_due[tail] = cycle_count + 1 + LATENCY;
                    tail = tail + 1;
                end
            end else begin
                if (in_ready || !conflict_o) begin
                    $display("FAIL expected RAW block coord=%h ready=%0b conflict=%0b", coord, in_ready, conflict_o);
                    failures = failures + 1;
                end
            end

            @(posedge clk);
            #1;
        end
    endtask

    always @(posedge clk) begin
        #1;
        if (rst_n) begin
            cycle_count = cycle_count + 1;

            if (alignment_error_o) begin
                $display("FAIL guard/arithmetic valid misalignment at cycle=%0d", cycle_count);
                failures = failures + 1;
            end

            if (out_valid) begin
                if (head >= tail) begin
                    $display("FAIL unexpected output at cycle=%0d coord=%h", cycle_count, out_coord);
                    failures = failures + 1;
                end else begin
                    if (cycle_count != exp_due[head]) begin
                        $display("FAIL latency tx=%0d got_cycle=%0d expected_cycle=%0d", head, cycle_count, exp_due[head]);
                        failures = failures + 1;
                    end
                    if (out_coord !== exp_coord[head] ||
                        psi_o !== exp_psi[head] || omega_o !== exp_omega[head] ||
                        flow_o !== exp_flow[head] || laplacian_o !== exp_lap[head] ||
                        score_o !== exp_score[head] || action_o !== exp_action[head] ||
                        quantity_o !== exp_quantity[head] || risk_accepted_o !== exp_risk[head]) begin
                        $display("FAIL tx=%0d coord=%h/%h psi=%0d/%0d score=%0d/%0d action=%0d/%0d risk=%0b/%0b",
                                 head, out_coord, exp_coord[head], psi_o, exp_psi[head],
                                 score_o, exp_score[head], action_o, exp_action[head],
                                 risk_accepted_o, exp_risk[head]);
                        failures = failures + 1;
                    end
                    head = head + 1;
                end
            end
        end
    end

    initial begin
        head = 0;
        tail = 0;
        failures = 0;
        cycle_count = 0;
        psi_i = '0; omega_i = '0; flow_i = '0; inventory_i = '0;
        delta_quantity_i = '0; side_bid_i = 1'b1;
        psi_xm_i = '0; psi_xp_i = '0; psi_ym_i = '0; psi_yp_i = '0; psi_zm_i = '0; psi_zp_i = '0;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // A/B/C demonstrate independent-coordinate issue plus same-A rejection.
        issue(A, 1, 1'b1);
        issue(B, 2, 1'b1);
        issue(A, 3, 1'b0);
        issue(C, 4, 1'b1);

        // Additional unique coordinates demonstrate sustained independent II=1 issue.
        for (k = 0; k < 8; k = k + 1)
            issue(12'h800 + k, 10 + k, 1'b1);

        @(negedge clk);
        in_valid = 1'b0;

        repeat (LATENCY + 5) @(posedge clk);

        // A must be issuable again once its earlier transaction has retired.
        issue(A, 40, 1'b1);
        @(negedge clk);
        in_valid = 1'b0;
        repeat (LATENCY + 3) @(posedge clk);
        #2;

        if (head != tail) begin
            $display("FAIL missing outputs head=%0d tail=%0d", head, tail);
            failures = failures + 1;
        end

        if (failures == 0)
            $display("PASS: guarded staged pipeline preserves 17-cycle coordinate/result alignment, blocks same-cell RAW hazards, and accepts independent II=1 traffic");
        else begin
            $display("FAIL: %0d guarded-pipeline failures", failures);
            $fatal(1);
        end

        $finish;
    end
endmodule
