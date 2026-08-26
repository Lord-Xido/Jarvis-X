module hft_field_neighbor_addr #(
    parameter integer X_BITS = 6,
    parameter integer Y_BITS = 2,
    parameter integer Z_BITS = 2,
    parameter integer ADDR_WIDTH = X_BITS + Y_BITS + Z_BITS
) (
    input  logic [X_BITS-1:0] x_i,
    input  logic [Y_BITS-1:0] y_i,
    input  logic [Z_BITS-1:0] z_i,

    output logic [ADDR_WIDTH-1:0] center_addr_o,
    output logic [ADDR_WIDTH-1:0] xm_addr_o,
    output logic [ADDR_WIDTH-1:0] xp_addr_o,
    output logic [ADDR_WIDTH-1:0] ym_addr_o,
    output logic [ADDR_WIDTH-1:0] yp_addr_o,
    output logic [ADDR_WIDTH-1:0] zm_addr_o,
    output logic [ADDR_WIDTH-1:0] zp_addr_o
);

    // Exact hardware lowering of the C++ default coordinate/index policy for
    // power-of-two dimensions:
    //
    //   index = x + 2^X_BITS * (y + 2^Y_BITS * z)
    //
    // which is bitwise concatenation {z,y,x}. Fixed-width +/- 1 arithmetic
    // naturally implements the C++ mask-based toroidal wrap on each axis.

    logic [X_BITS-1:0] xm;
    logic [X_BITS-1:0] xp;
    logic [Y_BITS-1:0] ym;
    logic [Y_BITS-1:0] yp;
    logic [Z_BITS-1:0] zm;
    logic [Z_BITS-1:0] zp;

    always_comb begin
        xm = x_i - {{(X_BITS-1){1'b0}}, 1'b1};
        xp = x_i + {{(X_BITS-1){1'b0}}, 1'b1};
        ym = y_i - {{(Y_BITS-1){1'b0}}, 1'b1};
        yp = y_i + {{(Y_BITS-1){1'b0}}, 1'b1};
        zm = z_i - {{(Z_BITS-1){1'b0}}, 1'b1};
        zp = z_i + {{(Z_BITS-1){1'b0}}, 1'b1};

        center_addr_o = {z_i, y_i, x_i};
        xm_addr_o = {z_i, y_i, xm};
        xp_addr_o = {z_i, y_i, xp};
        ym_addr_o = {z_i, ym, x_i};
        yp_addr_o = {z_i, yp, x_i};
        zm_addr_o = {zm, y_i, x_i};
        zp_addr_o = {zp, y_i, x_i};
    end

    initial begin
        if (X_BITS < 1 || Y_BITS < 1 || Z_BITS < 1)
            $error("All coordinate widths must be >= 1");
        if (ADDR_WIDTH != X_BITS + Y_BITS + Z_BITS)
            $error("ADDR_WIDTH must equal X_BITS + Y_BITS + Z_BITS");
    end

endmodule
