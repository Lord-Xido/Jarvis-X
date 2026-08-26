module hft_field_stencil_store_replicated #(
    parameter integer ADDR_WIDTH = 10,
    parameter integer DEPTH = (1 << ADDR_WIDTH)
) (
    input  logic                    clk,

    input  logic                    read_valid_i,
    input  logic [ADDR_WIDTH-1:0]   center_addr_i,
    input  logic [ADDR_WIDTH-1:0]   xm_addr_i,
    input  logic [ADDR_WIDTH-1:0]   xp_addr_i,
    input  logic [ADDR_WIDTH-1:0]   ym_addr_i,
    input  logic [ADDR_WIDTH-1:0]   yp_addr_i,
    input  logic [ADDR_WIDTH-1:0]   zm_addr_i,
    input  logic [ADDR_WIDTH-1:0]   zp_addr_i,

    output logic                    read_valid_o,
    output logic [ADDR_WIDTH-1:0]   center_addr_o,
    output logic signed [31:0]      psi_center_o,
    output logic signed [31:0]      psi_xm_o,
    output logic signed [31:0]      psi_xp_o,
    output logic signed [31:0]      psi_ym_o,
    output logic signed [31:0]      psi_yp_o,
    output logic signed [31:0]      psi_zm_o,
    output logic signed [31:0]      psi_zp_o,
    output logic signed [31:0]      omega_center_o,
    output logic signed [31:0]      flow_center_o,

    input  logic                    write_valid_i,
    input  logic [ADDR_WIDTH-1:0]   write_addr_i,
    input  logic signed [31:0]      psi_write_i,
    input  logic signed [31:0]      omega_write_i,
    input  logic signed [31:0]      flow_write_i
);

    // Physically realizable reference architecture for seven simultaneous Psi
    // reads: seven identical single-read replicas, with every center-state Psi
    // write broadcast to all copies. Omega and Flow need only the center read.
    //
    // Synchronous reads intentionally model FPGA block-memory timing. Vendor
    // BRAM/URAM inference and exact read-during-write behavior remain target-
    // dependent and must be proven in device-specific synthesis/P&R.

    (* ram_style = "block" *) logic signed [31:0] psi_center_mem [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] psi_xm_mem     [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] psi_xp_mem     [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] psi_ym_mem     [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] psi_yp_mem     [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] psi_zm_mem     [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] psi_zp_mem     [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] omega_mem      [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] flow_mem       [0:DEPTH-1];

    always_ff @(posedge clk) begin
        read_valid_o <= read_valid_i;
        if (read_valid_i) begin
            center_addr_o <= center_addr_i;
            psi_center_o <= psi_center_mem[center_addr_i];
            psi_xm_o <= psi_xm_mem[xm_addr_i];
            psi_xp_o <= psi_xp_mem[xp_addr_i];
            psi_ym_o <= psi_ym_mem[ym_addr_i];
            psi_yp_o <= psi_yp_mem[yp_addr_i];
            psi_zm_o <= psi_zm_mem[zm_addr_i];
            psi_zp_o <= psi_zp_mem[zp_addr_i];
            omega_center_o <= omega_mem[center_addr_i];
            flow_center_o <= flow_mem[center_addr_i];
        end

        if (write_valid_i) begin
            psi_center_mem[write_addr_i] <= psi_write_i;
            psi_xm_mem[write_addr_i] <= psi_write_i;
            psi_xp_mem[write_addr_i] <= psi_write_i;
            psi_ym_mem[write_addr_i] <= psi_write_i;
            psi_yp_mem[write_addr_i] <= psi_write_i;
            psi_zm_mem[write_addr_i] <= psi_write_i;
            psi_zp_mem[write_addr_i] <= psi_write_i;
            omega_mem[write_addr_i] <= omega_write_i;
            flow_mem[write_addr_i] <= flow_write_i;
        end
    end

    initial begin
        read_valid_o = 1'b0;
        center_addr_o = '0;
        psi_center_o = '0;
        psi_xm_o = '0;
        psi_xp_o = '0;
        psi_ym_o = '0;
        psi_yp_o = '0;
        psi_zm_o = '0;
        psi_zp_o = '0;
        omega_center_o = '0;
        flow_center_o = '0;

        if (ADDR_WIDTH < 1)
            $error("ADDR_WIDTH must be >= 1");
        if (DEPTH < 2)
            $error("DEPTH must be >= 2");
    end

endmodule
