`timescale 1ns/1ps

module tb_hft_field_neighbor_addr;
    localparam integer X_BITS = 6;
    localparam integer Y_BITS = 2;
    localparam integer Z_BITS = 2;
    localparam integer ADDR_WIDTH = X_BITS + Y_BITS + Z_BITS;
    localparam integer X_SIZE = 1 << X_BITS;
    localparam integer Y_SIZE = 1 << Y_BITS;
    localparam integer Z_SIZE = 1 << Z_BITS;

    logic [X_BITS-1:0] x_i;
    logic [Y_BITS-1:0] y_i;
    logic [Z_BITS-1:0] z_i;

    logic [ADDR_WIDTH-1:0] center_addr_o;
    logic [ADDR_WIDTH-1:0] xm_addr_o, xp_addr_o;
    logic [ADDR_WIDTH-1:0] ym_addr_o, yp_addr_o;
    logic [ADDR_WIDTH-1:0] zm_addr_o, zp_addr_o;

    integer failures;
    integer x;
    integer y;
    integer z;
    integer expected_center;
    integer expected_xm;
    integer expected_xp;
    integer expected_ym;
    integer expected_yp;
    integer expected_zm;
    integer expected_zp;

    hft_field_neighbor_addr #(
        .X_BITS(X_BITS),
        .Y_BITS(Y_BITS),
        .Z_BITS(Z_BITS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .x_i(x_i), .y_i(y_i), .z_i(z_i),
        .center_addr_o(center_addr_o),
        .xm_addr_o(xm_addr_o), .xp_addr_o(xp_addr_o),
        .ym_addr_o(ym_addr_o), .yp_addr_o(yp_addr_o),
        .zm_addr_o(zm_addr_o), .zp_addr_o(zp_addr_o)
    );

    task automatic check_coord(input integer xi, input integer yi, input integer zi);
        begin
            x_i = xi[X_BITS-1:0];
            y_i = yi[Y_BITS-1:0];
            z_i = zi[Z_BITS-1:0];
            #1;

            expected_center = xi + X_SIZE * (yi + Y_SIZE * zi);
            expected_xm = ((xi + X_SIZE - 1) % X_SIZE) + X_SIZE * (yi + Y_SIZE * zi);
            expected_xp = ((xi + 1) % X_SIZE) + X_SIZE * (yi + Y_SIZE * zi);
            expected_ym = xi + X_SIZE * (((yi + Y_SIZE - 1) % Y_SIZE) + Y_SIZE * zi);
            expected_yp = xi + X_SIZE * (((yi + 1) % Y_SIZE) + Y_SIZE * zi);
            expected_zm = xi + X_SIZE * (yi + Y_SIZE * ((zi + Z_SIZE - 1) % Z_SIZE));
            expected_zp = xi + X_SIZE * (yi + Y_SIZE * ((zi + 1) % Z_SIZE));

            if (center_addr_o !== expected_center[ADDR_WIDTH-1:0] ||
                xm_addr_o !== expected_xm[ADDR_WIDTH-1:0] ||
                xp_addr_o !== expected_xp[ADDR_WIDTH-1:0] ||
                ym_addr_o !== expected_ym[ADDR_WIDTH-1:0] ||
                yp_addr_o !== expected_yp[ADDR_WIDTH-1:0] ||
                zm_addr_o !== expected_zm[ADDR_WIDTH-1:0] ||
                zp_addr_o !== expected_zp[ADDR_WIDTH-1:0]) begin
                $display("FAIL (%0d,%0d,%0d) center=%0d/%0d xm=%0d/%0d xp=%0d/%0d ym=%0d/%0d yp=%0d/%0d zm=%0d/%0d zp=%0d/%0d",
                         xi, yi, zi,
                         center_addr_o, expected_center,
                         xm_addr_o, expected_xm,
                         xp_addr_o, expected_xp,
                         ym_addr_o, expected_ym,
                         yp_addr_o, expected_yp,
                         zm_addr_o, expected_zm,
                         zp_addr_o, expected_zp);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        failures = 0;
        x_i = '0; y_i = '0; z_i = '0;

        // Exhaust the full default 64 x 4 x 4 lattice. This covers all six
        // toroidal boundary wraps and every flattened address exactly.
        for (z = 0; z < Z_SIZE; z = z + 1)
            for (y = 0; y < Y_SIZE; y = y + 1)
                for (x = 0; x < X_SIZE; x = x + 1)
                    check_coord(x, y, z);

        if (failures == 0)
            $display("PASS: all %0d coordinates matched C++ flattening and toroidal six-neighbour wrapping", X_SIZE * Y_SIZE * Z_SIZE);
        else begin
            $display("FAIL: %0d neighbour-address mismatches", failures);
            $fatal(1);
        end

        $finish;
    end
endmodule
