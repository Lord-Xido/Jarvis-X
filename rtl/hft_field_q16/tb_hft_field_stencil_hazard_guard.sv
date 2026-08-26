`timescale 1ns/1ps

module tb_hft_field_stencil_hazard_guard;
    localparam integer X_BITS = 6;
    localparam integer Y_BITS = 2;
    localparam integer Z_BITS = 2;
    localparam integer COORD_WIDTH = X_BITS + Y_BITS + Z_BITS;
    localparam integer LATENCY = 17;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic in_valid = 1'b0;
    logic [X_BITS-1:0] x_i = '0;
    logic [Y_BITS-1:0] y_i = '0;
    logic [Z_BITS-1:0] z_i = '0;

    logic [COORD_WIDTH-1:0] center_addr;
    logic [COORD_WIDTH-1:0] xm_addr, xp_addr, ym_addr, yp_addr, zm_addr, zp_addr;
    logic in_ready, accept_o, conflict_o, commit_valid_o;
    logic [COORD_WIDTH-1:0] commit_coord_o;

    integer failures;
    integer i;
    logic saw_a_commit;

    always #5 clk = ~clk;

    hft_field_neighbor_addr #(
        .X_BITS(X_BITS), .Y_BITS(Y_BITS), .Z_BITS(Z_BITS),
        .ADDR_WIDTH(COORD_WIDTH)
    ) addr_gen (
        .x_i(x_i), .y_i(y_i), .z_i(z_i),
        .center_addr_o(center_addr),
        .xm_addr_o(xm_addr), .xp_addr_o(xp_addr),
        .ym_addr_o(ym_addr), .yp_addr_o(yp_addr),
        .zm_addr_o(zm_addr), .zp_addr_o(zp_addr)
    );

    hft_field_stencil_hazard_guard #(
        .COORD_WIDTH(COORD_WIDTH),
        .LATENCY_CYCLES(LATENCY)
    ) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
        .center_coord_i(center_addr),
        .xm_coord_i(xm_addr), .xp_coord_i(xp_addr),
        .ym_coord_i(ym_addr), .yp_coord_i(yp_addr),
        .zm_coord_i(zm_addr), .zp_coord_i(zp_addr),
        .in_ready(in_ready), .accept_o(accept_o), .conflict_o(conflict_o),
        .commit_valid_o(commit_valid_o), .commit_coord_o(commit_coord_o)
    );

    task automatic drive_coord(
        input integer x,
        input integer y,
        input integer z,
        input logic expect_accept
    );
        begin
            @(negedge clk);
            x_i = x[X_BITS-1:0];
            y_i = y[Y_BITS-1:0];
            z_i = z[Z_BITS-1:0];
            in_valid = 1'b1;
            #1;
            if (expect_accept) begin
                if (!in_ready || !accept_o || conflict_o) begin
                    $display("FAIL expected accept (%0d,%0d,%0d) ready=%0b accept=%0b conflict=%0b",
                             x, y, z, in_ready, accept_o, conflict_o);
                    failures = failures + 1;
                end
            end else begin
                if (in_ready || accept_o || !conflict_o) begin
                    $display("FAIL expected stencil block (%0d,%0d,%0d) ready=%0b accept=%0b conflict=%0b",
                             x, y, z, in_ready, accept_o, conflict_o);
                    failures = failures + 1;
                end
            end
            @(posedge clk);
            #1;
        end
    endtask

    initial begin
        failures = 0;
        saw_a_commit = 1'b0;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // A=(0,0,0) becomes the pending write coordinate.
        drive_coord(0, 0, 0, 1'b1);

        // Both direct x neighbours read A, so both must block. x=63 verifies
        // that the dependency check also respects toroidal wrap.
        drive_coord(1, 0, 0, 1'b0);
        drive_coord(63, 0, 0, 1'b0);

        // A y-neighbour and z-neighbour also read A and must block.
        drive_coord(0, 1, 0, 1'b0);
        drive_coord(0, 0, 3, 1'b0);

        // Spatially disjoint stencils remain issuable on consecutive clocks.
        drive_coord(10, 2, 2, 1'b1);
        drive_coord(20, 2, 2, 1'b1);

        // A new event adjacent to the pending center at x=10 must block even
        // though its own center is different.
        drive_coord(11, 2, 2, 1'b0);

        @(negedge clk);
        in_valid = 1'b0;

        for (i = 0; i < LATENCY + 8; i = i + 1) begin
            @(posedge clk);
            #1;
            if (commit_valid_o && commit_coord_o == {2'b00, 2'b00, 6'b000000})
                saw_a_commit = 1'b1;
        end

        if (!saw_a_commit) begin
            $display("FAIL never observed A commit");
            failures = failures + 1;
        end

        // Once all pending writes retire, A is legal again.
        drive_coord(0, 0, 0, 1'b1);
        @(negedge clk);
        in_valid = 1'b0;

        if (failures == 0)
            $display("PASS: stencil scoreboard blocks center and six-neighbour RAW dependencies, including toroidal wrap, while allowing disjoint II=1 traffic");
        else begin
            $display("FAIL: %0d stencil-hazard failures", failures);
            $fatal(1);
        end

        $finish;
    end
endmodule
