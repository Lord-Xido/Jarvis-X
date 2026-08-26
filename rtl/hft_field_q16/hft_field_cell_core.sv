module hft_field_cell_core (
    input  logic signed [31:0] psi_i,
    input  logic signed [31:0] omega_i,
    input  logic signed [31:0] flow_i,
    input  logic signed [31:0] inventory_i,
    input  logic signed [31:0] delta_quantity_i,
    input  logic               side_bid_i,
    input  logic signed [31:0] psi_xm_i,
    input  logic signed [31:0] psi_xp_i,
    input  logic signed [31:0] psi_ym_i,
    input  logic signed [31:0] psi_yp_i,
    input  logic signed [31:0] psi_zm_i,
    input  logic signed [31:0] psi_zp_i,

    output logic signed [31:0] psi_o,
    output logic signed [31:0] omega_o,
    output logic signed [31:0] flow_o,
    output logic signed [31:0] laplacian_o,
    output logic signed [31:0] score_o,
    output logic signed [1:0]  action_o,
    output logic signed [31:0] quantity_o,
    output logic               risk_accepted_o
);

    // Q16.16 raw constants. These must match cpp_runtime/include/jarvisx/hft_field.hpp.
    localparam logic signed [31:0] Q_ONE            = 32'sd65536;
    localparam logic signed [31:0] Q_SIX            = 32'sd393216;
    localparam logic signed [31:0] ALPHA            = 32'sd16384;   // 1/4
    localparam logic signed [31:0] LAMBDA           = 32'sd2048;    // 1/32
    localparam logic signed [31:0] ETA              = 32'sd32768;   // 1/2
    localparam logic signed [31:0] DT               = 32'sd8192;    // 1/8
    localparam logic signed [31:0] RHO              = 32'sd61440;   // 15/16
    localparam logic signed [31:0] ONE_MINUS_RHO    = 32'sd4096;    // 1/16
    localparam logic signed [31:0] FLOW_DECAY       = 32'sd57344;   // 7/8
    localparam logic signed [31:0] W_PSI            = 32'sd65536;   // 1
    localparam logic signed [31:0] W_OMEGA          = 32'sd32768;   // 1/2
    localparam logic signed [31:0] W_FLOW           = 32'sd32768;   // 1/2
    localparam logic signed [31:0] W_LAPLACIAN      = 32'sd8192;    // 1/8
    localparam logic signed [31:0] W_INVENTORY      = 32'sd16384;   // 1/4
    localparam logic signed [31:0] DECISION_THRESH  = 32'sd4096;    // 1/16
    localparam logic signed [31:0] MAX_ABS_FIELD    = 32'sd2097152; // 32
    localparam logic signed [31:0] MAX_INVENTORY    = 32'sd4194304; // 64
    localparam logic signed [31:0] MAX_ORDER_QTY    = 32'sd262144;  // 4

    function automatic logic signed [31:0] sat32(input logic signed [63:0] x);
        begin
            if (x > 64'sd2147483647)
                sat32 = 32'sh7fffffff;
            else if (x < -64'sd2147483648)
                sat32 = 32'sh80000000;
            else
                sat32 = x[31:0];
        end
    endfunction

    function automatic logic signed [31:0] qadd(
        input logic signed [31:0] a,
        input logic signed [31:0] b
    );
        logic signed [63:0] aa;
        logic signed [63:0] bb;
        begin
            aa = a;
            bb = b;
            qadd = sat32(aa + bb);
        end
    endfunction

    function automatic logic signed [31:0] qsub(
        input logic signed [31:0] a,
        input logic signed [31:0] b
    );
        logic signed [63:0] aa;
        logic signed [63:0] bb;
        begin
            aa = a;
            bb = b;
            qsub = sat32(aa - bb);
        end
    endfunction

    function automatic logic signed [31:0] qneg(input logic signed [31:0] a);
        logic signed [63:0] aa;
        begin
            aa = a;
            qneg = sat32(-aa);
        end
    endfunction

    // Q16.16 multiplication with a 64-bit product and truncation toward zero.
    // The magnitude/shift form is intentional: arithmetic right shift alone would
    // round negative products toward -infinity rather than toward zero.
    function automatic logic signed [31:0] qmul(
        input logic signed [31:0] a,
        input logic signed [31:0] b
    );
        logic signed [63:0] product;
        logic signed [63:0] magnitude;
        logic signed [63:0] scaled;
        begin
            product = $signed(a) * $signed(b);
            if (product < 0) begin
                magnitude = -product;
                scaled = -(magnitude >>> 16);
            end else begin
                scaled = product >>> 16;
            end
            qmul = sat32(scaled);
        end
    endfunction

    function automatic logic signed [31:0] qabs(input logic signed [31:0] a);
        begin
            qabs = a >= 0 ? a : qneg(a);
        end
    endfunction

    function automatic logic signed [31:0] qclamp(
        input logic signed [31:0] v,
        input logic signed [31:0] lo,
        input logic signed [31:0] hi
    );
        begin
            if (v < lo)
                qclamp = lo;
            else if (v > hi)
                qclamp = hi;
            else
                qclamp = v;
        end
    endfunction

    logic signed [31:0] impulse;
    logic signed [31:0] neighbours;
    logic signed [31:0] lap;
    logic signed [31:0] residual;
    logic signed [31:0] rhs;
    logic signed [31:0] candidate;
    logic signed [31:0] omega_next;
    logic signed [31:0] flow_next;
    logic signed [31:0] score;
    logic signed [31:0] quantity;
    logic signed [31:0] signed_order;
    logic signed [31:0] projected_inventory;
    logic signed [1:0]  action;
    logic               risk;

    always_comb begin
        impulse = side_bid_i ? delta_quantity_i : qneg(delta_quantity_i);
        flow_next = qadd(qmul(FLOW_DECAY, flow_i), impulse);

        // Preserve the C++ left-associative saturating reduction order exactly.
        neighbours = qadd(psi_xm_i, psi_xp_i);
        neighbours = qadd(neighbours, psi_ym_i);
        neighbours = qadd(neighbours, psi_yp_i);
        neighbours = qadd(neighbours, psi_zm_i);
        neighbours = qadd(neighbours, psi_zp_i);
        lap = qsub(neighbours, qmul(Q_SIX, psi_i));

        residual = qsub(psi_i, omega_i);
        rhs = qneg(qmul(ALPHA, residual));
        rhs = qadd(rhs, qmul(LAMBDA, lap));
        rhs = qadd(rhs, qmul(ETA, impulse));

        candidate = qclamp(
            qadd(psi_i, qmul(DT, rhs)),
            qneg(MAX_ABS_FIELD),
            MAX_ABS_FIELD
        );

        omega_next = qadd(qmul(RHO, omega_i), qmul(ONE_MINUS_RHO, candidate));

        score = qmul(W_PSI, candidate);
        score = qadd(score, qmul(W_OMEGA, omega_next));
        score = qadd(score, qmul(W_FLOW, flow_next));
        score = qadd(score, qmul(W_LAPLACIAN, lap));
        score = qsub(score, qmul(W_INVENTORY, inventory_i));

        action = 2'sd0;
        quantity = 32'sd0;
        risk = 1'b0;

        if (score > DECISION_THRESH)
            action = 2'sd1;
        else if (score < qneg(DECISION_THRESH))
            action = -2'sd1;

        if (action != 0) begin
            quantity = qabs(delta_quantity_i);
            if (quantity > MAX_ORDER_QTY)
                quantity = MAX_ORDER_QTY;
            if (quantity == 0)
                quantity = MAX_ORDER_QTY;

            signed_order = action == 2'sd1 ? quantity : qneg(quantity);
            projected_inventory = qadd(inventory_i, signed_order);
            risk = qabs(projected_inventory) <= MAX_INVENTORY;

            if (!risk) begin
                action = 2'sd0;
                quantity = 32'sd0;
            end
        end else begin
            signed_order = 32'sd0;
            projected_inventory = inventory_i;
        end

        psi_o = candidate;
        omega_o = omega_next;
        flow_o = flow_next;
        laplacian_o = lap;
        score_o = score;
        action_o = action;
        quantity_o = quantity;
        risk_accepted_o = risk;
    end

endmodule
