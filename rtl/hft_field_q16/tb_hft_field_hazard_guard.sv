`timescale 1ns/1ps

module tb_hft_field_hazard_guard;
    localparam integer COORD_WIDTH = 12;
    localparam integer LATENCY = 17;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic in_valid = 1'b0;
    logic [COORD_WIDTH-1:0] in_coord = '0;
    logic in_ready;
    logic accept_o;
    logic conflict_o;
    logic commit_valid_o;
    logic [COORD_WIDTH-1:0] commit_coord_o;

    integer failures;
    integer cycles;
    integer accepted_independent;
    logic seen_a_commit;

    localparam logic [COORD_WIDTH-1:0] A = 12'h123;
    localparam logic [COORD_WIDTH-1:0] B = 12'h456;
    localparam logic [COORD_WIDTH-1:0] C = 12'h789;

    always #5 clk = ~clk;

    hft_field_hazard_guard #(
        .COORD_WIDTH(COORD_WIDTH),
        .LATENCY_CYCLES(LATENCY)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_coord(in_coord),
        .in_ready(in_ready), .accept_o(accept_o), .conflict_o(conflict_o),
        .commit_valid_o(commit_valid_o), .commit_coord_o(commit_coord_o)
    );

    task automatic expect_accept(input logic [COORD_WIDTH-1:0] coord);
        begin
            @(negedge clk);
            in_coord = coord;
            in_valid = 1'b1;
            #1;
            if (!in_ready || !accept_o || conflict_o) begin
                $display("FAIL expected accept coord=%h ready=%0b accept=%0b conflict=%0b",
                         coord, in_ready, accept_o, conflict_o);
                failures = failures + 1;
            end
            @(posedge clk);
            #1;
        end
    endtask

    task automatic expect_block(input logic [COORD_WIDTH-1:0] coord);
        begin
            @(negedge clk);
            in_coord = coord;
            in_valid = 1'b1;
            #1;
            if (in_ready || accept_o || !conflict_o) begin
                $display("FAIL expected block coord=%h ready=%0b accept=%0b conflict=%0b",
                         coord, in_ready, accept_o, conflict_o);
                failures = failures + 1;
            end
            @(posedge clk);
            #1;
        end
    endtask

    initial begin
        failures = 0;
        cycles = 0;
        accepted_independent = 0;
        seen_a_commit = 1'b0;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // First A enters.
        expect_accept(A);

        // Independent coordinates can still enter on consecutive clocks.
        expect_accept(B);
        accepted_independent = accepted_independent + 1;
        expect_accept(C);
        accepted_independent = accepted_independent + 1;

        // Re-issuing A while A is in flight must fail closed.
        expect_block(A);

        // Stop injecting and observe the A commit.
        @(negedge clk);
        in_valid = 1'b0;

        for (cycles = 0; cycles < LATENCY + 4; cycles = cycles + 1) begin
            @(posedge clk);
            #1;
            if (commit_valid_o && commit_coord_o == A)
                seen_a_commit = 1'b1;
        end

        if (!seen_a_commit) begin
            $display("FAIL never observed commit for A");
            failures = failures + 1;
        end

        // After the scoreboard drains, A must be issuable again.
        expect_accept(A);

        @(negedge clk);
        in_valid = 1'b0;

        if (accepted_independent != 2) begin
            $display("FAIL independent acceptance accounting");
            failures = failures + 1;
        end

        if (failures == 0)
            $display("PASS: hazard guard blocks same-coordinate RAW hazards and permits independent II=1 traffic");
        else begin
            $display("FAIL: %0d hazard-guard failures", failures);
            $fatal(1);
        end

        $finish;
    end
endmodule
