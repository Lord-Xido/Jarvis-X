module omega3_electronic_core #(
    parameter logic [7:0] REQUIRED_LAMBDA_MASK = 8'hFF
) (
    input  logic               clk,
    input  logic               reset_n,
    input  logic [63:0]        candidate_word,
    input  logic [7:0]         lambda_mask,
    input  logic signed [15:0] error_q15,
    input  logic signed [15:0] rho_q15,
    input  logic signed [15:0] gain_q15,
    input  logic [6:0]         convergence_threshold,
    output logic [63:0]        state_word,
    output logic signed [15:0] omega_q15,
    output logic [63:0]        selected_word,
    output logic               lambda_approved,
    output logic [6:0]         hamming_delta,
    output logic               converged,
    output logic [63:0]        cycle
);

    logic [63:0] commit_mask;
    logic signed [31:0] omega_product;
    logic signed [31:0] error_product;
    logic signed [31:0] omega_sum;
    logic signed [15:0] omega_candidate;

    function automatic [6:0] popcount64(input logic [63:0] value);
        integer i;
        begin
            popcount64 = 7'd0;
            for (i = 0; i < 64; i = i + 1)
                popcount64 = popcount64 + value[i];
        end
    endfunction

    function automatic logic signed [15:0] saturate_q15(
        input logic signed [31:0] value
    );
        begin
            if (value > 32'sd32767)
                saturate_q15 = 16'sh7FFF;
            else if (value < -32'sd32768)
                saturate_q15 = -16'sd32768;
            else
                saturate_q15 = value[15:0];
        end
    endfunction

    always_comb begin
        lambda_approved =
            (lambda_mask & REQUIRED_LAMBDA_MASK) == REQUIRED_LAMBDA_MASK;

        commit_mask = {64{lambda_approved}};
        selected_word =
            (candidate_word & commit_mask) |
            (state_word & ~commit_mask);

        hamming_delta = popcount64(state_word ^ candidate_word);
        converged = hamming_delta <= convergence_threshold;

        omega_product = $signed(rho_q15) * $signed(omega_q15);
        error_product = $signed(gain_q15) * $signed(error_q15);
        omega_sum = (omega_product >>> 15) + (error_product >>> 15);
        omega_candidate = saturate_q15(omega_sum);
    end

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state_word <= 64'd0;
            omega_q15 <= 16'sd0;
            cycle <= 64'd0;
        end else begin
            cycle <= cycle + 64'd1;
            if (lambda_approved) begin
                state_word <= selected_word;
                omega_q15 <= omega_candidate;
            end
        end
    end

endmodule
