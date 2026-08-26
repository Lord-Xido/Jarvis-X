module hft_field_cell_pow2 (
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

    // Fixed default HFT configuration. All coefficients are dyadic rationals,
    // so the exact C++ Q16.16 products can be lowered to shift/add operations.
    localparam logic signed [31:0] DECISION_THRESH = 32'sd4096;    // 1/16
    localparam logic signed [31:0] MAX_ABS_FIELD   = 32'sd2097152; // 32
    localparam logic signed [31:0] MAX_INVENTORY   = 32'sd4194304; // 64
    localparam logic signed [31:0] MAX_ORDER_QTY   = 32'sd262144;  // 4

    function automatic logic signed [63:0] sx32(input logic signed [31:0] a);
        begin
            sx32 = {{32{a[31]}}, a};
        end
    endfunction

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
        begin
            qadd = sat32(sx32(a) + sx32(b));
        end
    endfunction

    function automatic logic signed [31:0] qsub(
        input logic signed [31:0] a,
        input logic signed [31:0] b
    );
        begin
            qsub = sat32(sx32(a) - sx32(b));
        end
    endfunction

    function automatic logic signed [31:0] qneg(input logic signed [31:0] a);
        begin
            qneg = sat32(-sx32(a));
        end
    endfunction

    // Exact signed division by 2^shift with C++ truncation toward zero.
    function automatic logic signed [31:0] div_pow2_tz(
        input logic signed [63:0] value,
        input integer shift
    );
        logic signed [63:0] magnitude;
        logic signed [63:0] scaled;
        begin
            if (value < 0) begin
                magnitude = -value;
                scaled = -(magnitude >>> shift);
            end else begin
                scaled = value >>> shift;
            end
            div_pow2_tz = sat32(scaled);
        end
    endfunction

    function automatic logic signed [31:0] qdiv2(input logic signed [31:0] a);
        begin qdiv2 = div_pow2_tz(sx32(a), 1); end
    endfunction

    function automatic logic signed [31:0] qdiv4(input logic signed [31:0] a);
        begin qdiv4 = div_pow2_tz(sx32(a), 2); end
    endfunction

    function automatic logic signed [31:0] qdiv8(input logic signed [31:0] a);
        begin qdiv8 = div_pow2_tz(sx32(a), 3); end
    endfunction

    function automatic logic signed [31:0] qdiv16(input logic signed [31:0] a);
        begin qdiv16 = div_pow2_tz(sx32(a), 4); end
    endfunction

    function automatic logic signed [31:0] qdiv32(input logic signed [31:0] a);
        begin qdiv32 = div_pow2_tz(sx32(a), 5); end
    endfunction

    function automatic logic signed [31:0] qmul7div8(input logic signed [31:0] a);
        logic signed [63:0] v;
        begin
            v = (sx32(a) <<< 3) - sx32(a);
            qmul7div8 = div_pow2_tz(v, 3);
        end
    endfunction

    function automatic logic signed [31:0] qmul15div16(input logic signed [31:0] a);
        logic signed [63:0] v;
        begin
            v = (sx32(a) <<< 4) - sx32(a);
            qmul15div16 = div_pow2_tz(v, 4);
        end
    endfunction

    function automatic logic signed [31:0] qmul6(input logic signed [31:0] a);
        logic signed [63:0] v;
        begin
            v = (sx32(a) <<< 2) + (sx32(a) <<< 1);
            qmul6 = sat32(v);
        end
    endfunction

    function automatic logic signed [31:0] qabs(input logic signed [31:0] a);
        begin qabs = a >= 0 ? a : qneg(a); end
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
        flow_next = qadd(qmul7div8(flow_i), impulse);

        // Exact left-associative saturated six-neighbour reduction.
        neighbours = qadd(psi_xm_i, psi_xp_i);
        neighbours = qadd(neighbours, psi_ym_i);
        neighbours = qadd(neighbours, psi_yp_i);
        neighbours = qadd(neighbours, psi_zm_i);
        neighbours = qadd(neighbours, psi_zp_i);
        lap = qsub(neighbours, qmul6(psi_i));

        residual = qsub(psi_i, omega_i);
        rhs = qneg(qdiv4(residual));
        rhs = qadd(rhs, qdiv32(lap));
        rhs = qadd(rhs, qdiv2(impulse));

        candidate = qclamp(
            qadd(psi_i, qdiv8(rhs)),
            qneg(MAX_ABS_FIELD),
            MAX_ABS_FIELD
        );

        omega_next = qadd(qmul15div16(omega_i), qdiv16(candidate));

        score = candidate;
        score = qadd(score, qdiv2(omega_next));
        score = qadd(score, qdiv2(flow_next));
        score = qadd(score, qdiv8(lap));
        score = qsub(score, qdiv4(inventory_i));

        action = 2'sd0;
        quantity = 32'sd0;
        risk = 1'b0;
        signed_order = 32'sd0;
        projected_inventory = inventory_i;

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
