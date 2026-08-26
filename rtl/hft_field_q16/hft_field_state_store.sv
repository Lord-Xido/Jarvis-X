module hft_field_state_store #(
    parameter integer ADDR_WIDTH = 6,
    parameter integer DEPTH = (1 << ADDR_WIDTH)
) (
    input  logic                    clk,

    input  logic [ADDR_WIDTH-1:0]   read_addr,
    output logic signed [31:0]      psi_r,
    output logic signed [31:0]      omega_r,
    output logic signed [31:0]      flow_r,

    input  logic                    write_valid,
    input  logic [ADDR_WIDTH-1:0]   write_addr,
    input  logic signed [31:0]      psi_w,
    input  logic signed [31:0]      omega_w,
    input  logic signed [31:0]      flow_w
);

    // Center-cell state store reference. No reset is applied to the memory
    // arrays so a future FPGA implementation can preserve RAM inference.
    // Deterministic initialization is performed through the explicit write
    // port before event processing begins.
    //
    // This module intentionally provides only one center-state read. Six-
    // neighbour Psi fetch/banking is a separate hardware problem and is not
    // hidden behind an unrealistic multi-read RAM abstraction here.

    logic signed [31:0] psi_mem   [0:DEPTH-1];
    logic signed [31:0] omega_mem [0:DEPTH-1];
    logic signed [31:0] flow_mem  [0:DEPTH-1];

    always_comb begin
        psi_r = psi_mem[read_addr];
        omega_r = omega_mem[read_addr];
        flow_r = flow_mem[read_addr];
    end

    always_ff @(posedge clk) begin
        if (write_valid) begin
            psi_mem[write_addr] <= psi_w;
            omega_mem[write_addr] <= omega_w;
            flow_mem[write_addr] <= flow_w;
        end
    end

    initial begin
        if (ADDR_WIDTH < 1)
            $error("ADDR_WIDTH must be >= 1");
        if (DEPTH < 2)
            $error("DEPTH must be >= 2");
    end

endmodule
