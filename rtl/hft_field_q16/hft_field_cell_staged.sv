module hft_field_cell_staged (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    in_valid,

    input  logic signed [31:0]      psi_i,
    input  logic signed [31:0]      omega_i,
    input  logic signed [31:0]      flow_i,
    input  logic signed [31:0]      inventory_i,
    input  logic signed [31:0]      delta_quantity_i,
    input  logic                    side_bid_i,
    input  logic signed [31:0]      psi_xm_i,
    input  logic signed [31:0]      psi_xp_i,
    input  logic signed [31:0]      psi_ym_i,
    input  logic signed [31:0]      psi_yp_i,
    input  logic signed [31:0]      psi_zm_i,
    input  logic signed [31:0]      psi_zp_i,

    output logic                    out_valid,
    output logic signed [31:0]      psi_o,
    output logic signed [31:0]      omega_o,
    output logic signed [31:0]      flow_o,
    output logic signed [31:0]      laplacian_o,
    output logic signed [31:0]      score_o,
    output logic signed [1:0]       action_o,
    output logic signed [31:0]      quantity_o,
    output logic                    risk_accepted_o
);

    localparam logic signed [31:0] DECISION_THRESH = 32'sd4096;
    localparam logic signed [31:0] MAX_ABS_FIELD   = 32'sd2097152;
    localparam logic signed [31:0] MAX_INVENTORY   = 32'sd4194304;
    localparam logic signed [31:0] MAX_ORDER_QTY   = 32'sd262144;

    function automatic logic signed [63:0] sx32(input logic signed [31:0] a);
        begin sx32 = {{32{a[31]}}, a}; end
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
        begin qadd = sat32(sx32(a) + sx32(b)); end
    endfunction

    function automatic logic signed [31:0] qsub(
        input logic signed [31:0] a,
        input logic signed [31:0] b
    );
        begin qsub = sat32(sx32(a) - sx32(b)); end
    endfunction

    function automatic logic signed [31:0] qneg(input logic signed [31:0] a);
        begin qneg = sat32(-sx32(a)); end
    endfunction

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

    logic v0, v1, v2, v3, v4, v5, v6, v7, v8;
    logic v9, v10, v11, v12, v13, v14, v15, v16;

    // Stage 0: captured inputs.
    logic signed [31:0] s0_psi, s0_omega, s0_flow, s0_inventory, s0_delta;
    logic s0_side_bid;
    logic signed [31:0] s0_xm, s0_xp, s0_ym, s0_yp, s0_zm, s0_zp;

    // Stage 1: impulse, flow, residual and first neighbour reduction.
    logic signed [31:0] s1_psi, s1_omega, s1_inventory, s1_delta;
    logic signed [31:0] s1_impulse, s1_flow_next, s1_residual, s1_neighbours;
    logic signed [31:0] s1_ym, s1_yp, s1_zm, s1_zp;

    logic signed [31:0] s2_psi, s2_omega, s2_inventory, s2_delta;
    logic signed [31:0] s2_impulse, s2_flow_next, s2_residual, s2_neighbours;
    logic signed [31:0] s2_yp, s2_zm, s2_zp;

    logic signed [31:0] s3_psi, s3_omega, s3_inventory, s3_delta;
    logic signed [31:0] s3_impulse, s3_flow_next, s3_residual, s3_neighbours;
    logic signed [31:0] s3_zm, s3_zp;

    logic signed [31:0] s4_psi, s4_omega, s4_inventory, s4_delta;
    logic signed [31:0] s4_impulse, s4_flow_next, s4_residual, s4_neighbours;
    logic signed [31:0] s4_zp;

    logic signed [31:0] s5_psi, s5_omega, s5_inventory, s5_delta;
    logic signed [31:0] s5_impulse, s5_flow_next, s5_residual, s5_neighbours;

    logic signed [31:0] s6_psi, s6_omega, s6_inventory, s6_delta;
    logic signed [31:0] s6_impulse, s6_flow_next, s6_residual, s6_lap;

    logic signed [31:0] s7_psi, s7_omega, s7_inventory, s7_delta;
    logic signed [31:0] s7_impulse, s7_flow_next, s7_lap, s7_rhs;

    logic signed [31:0] s8_psi, s8_omega, s8_inventory, s8_delta;
    logic signed [31:0] s8_impulse, s8_flow_next, s8_lap, s8_rhs;

    logic signed [31:0] s9_psi, s9_omega, s9_inventory, s9_delta;
    logic signed [31:0] s9_flow_next, s9_lap, s9_rhs;

    logic signed [31:0] s10_omega, s10_inventory, s10_delta;
    logic signed [31:0] s10_flow_next, s10_lap, s10_candidate;

    logic signed [31:0] s11_inventory, s11_delta, s11_flow_next, s11_lap;
    logic signed [31:0] s11_candidate, s11_omega_next;

    logic signed [31:0] s12_inventory, s12_delta, s12_flow_next, s12_lap;
    logic signed [31:0] s12_candidate, s12_omega_next, s12_score;

    logic signed [31:0] s13_inventory, s13_delta, s13_lap;
    logic signed [31:0] s13_candidate, s13_omega_next, s13_flow_next, s13_score;

    logic signed [31:0] s14_inventory, s14_delta, s14_lap;
    logic signed [31:0] s14_candidate, s14_omega_next, s14_flow_next, s14_score;

    logic signed [31:0] s15_inventory, s15_delta, s15_lap;
    logic signed [31:0] s15_candidate, s15_omega_next, s15_flow_next, s15_score;

    logic signed [31:0] s16_candidate, s16_omega_next, s16_flow_next, s16_lap, s16_score;
    logic signed [1:0]  s16_action;
    logic signed [31:0] s16_quantity;
    logic               s16_risk;

    logic signed [1:0]  final_action_c;
    logic signed [31:0] final_quantity_c;
    logic signed [31:0] final_signed_order_c;
    logic signed [31:0] final_projected_inventory_c;
    logic               final_risk_c;

    always_comb begin
        final_action_c = 2'sd0;
        final_quantity_c = 32'sd0;
        final_signed_order_c = 32'sd0;
        final_projected_inventory_c = s15_inventory;
        final_risk_c = 1'b0;

        if (s15_score > DECISION_THRESH)
            final_action_c = 2'sd1;
        else if (s15_score < qneg(DECISION_THRESH))
            final_action_c = -2'sd1;

        if (final_action_c != 0) begin
            final_quantity_c = qabs(s15_delta);
            if (final_quantity_c > MAX_ORDER_QTY)
                final_quantity_c = MAX_ORDER_QTY;
            if (final_quantity_c == 0)
                final_quantity_c = MAX_ORDER_QTY;

            final_signed_order_c = final_action_c == 2'sd1
                ? final_quantity_c
                : qneg(final_quantity_c);
            final_projected_inventory_c = qadd(s15_inventory, final_signed_order_c);
            final_risk_c = qabs(final_projected_inventory_c) <= MAX_INVENTORY;

            if (!final_risk_c) begin
                final_action_c = 2'sd0;
                final_quantity_c = 32'sd0;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v0 <= 1'b0; v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
            v5 <= 1'b0; v6 <= 1'b0; v7 <= 1'b0; v8 <= 1'b0; v9 <= 1'b0;
            v10 <= 1'b0; v11 <= 1'b0; v12 <= 1'b0; v13 <= 1'b0;
            v14 <= 1'b0; v15 <= 1'b0; v16 <= 1'b0;
            out_valid <= 1'b0;
            psi_o <= '0; omega_o <= '0; flow_o <= '0; laplacian_o <= '0;
            score_o <= '0; action_o <= '0; quantity_o <= '0; risk_accepted_o <= 1'b0;

            s0_psi <= '0; s0_omega <= '0; s0_flow <= '0; s0_inventory <= '0; s0_delta <= '0;
            s0_side_bid <= 1'b0; s0_xm <= '0; s0_xp <= '0; s0_ym <= '0; s0_yp <= '0; s0_zm <= '0; s0_zp <= '0;
            s1_psi <= '0; s1_omega <= '0; s1_inventory <= '0; s1_delta <= '0; s1_impulse <= '0; s1_flow_next <= '0; s1_residual <= '0; s1_neighbours <= '0; s1_ym <= '0; s1_yp <= '0; s1_zm <= '0; s1_zp <= '0;
            s2_psi <= '0; s2_omega <= '0; s2_inventory <= '0; s2_delta <= '0; s2_impulse <= '0; s2_flow_next <= '0; s2_residual <= '0; s2_neighbours <= '0; s2_yp <= '0; s2_zm <= '0; s2_zp <= '0;
            s3_psi <= '0; s3_omega <= '0; s3_inventory <= '0; s3_delta <= '0; s3_impulse <= '0; s3_flow_next <= '0; s3_residual <= '0; s3_neighbours <= '0; s3_zm <= '0; s3_zp <= '0;
            s4_psi <= '0; s4_omega <= '0; s4_inventory <= '0; s4_delta <= '0; s4_impulse <= '0; s4_flow_next <= '0; s4_residual <= '0; s4_neighbours <= '0; s4_zp <= '0;
            s5_psi <= '0; s5_omega <= '0; s5_inventory <= '0; s5_delta <= '0; s5_impulse <= '0; s5_flow_next <= '0; s5_residual <= '0; s5_neighbours <= '0;
            s6_psi <= '0; s6_omega <= '0; s6_inventory <= '0; s6_delta <= '0; s6_impulse <= '0; s6_flow_next <= '0; s6_residual <= '0; s6_lap <= '0;
            s7_psi <= '0; s7_omega <= '0; s7_inventory <= '0; s7_delta <= '0; s7_impulse <= '0; s7_flow_next <= '0; s7_lap <= '0; s7_rhs <= '0;
            s8_psi <= '0; s8_omega <= '0; s8_inventory <= '0; s8_delta <= '0; s8_impulse <= '0; s8_flow_next <= '0; s8_lap <= '0; s8_rhs <= '0;
            s9_psi <= '0; s9_omega <= '0; s9_inventory <= '0; s9_delta <= '0; s9_flow_next <= '0; s9_lap <= '0; s9_rhs <= '0;
            s10_omega <= '0; s10_inventory <= '0; s10_delta <= '0; s10_flow_next <= '0; s10_lap <= '0; s10_candidate <= '0;
            s11_inventory <= '0; s11_delta <= '0; s11_flow_next <= '0; s11_lap <= '0; s11_candidate <= '0; s11_omega_next <= '0;
            s12_inventory <= '0; s12_delta <= '0; s12_flow_next <= '0; s12_lap <= '0; s12_candidate <= '0; s12_omega_next <= '0; s12_score <= '0;
            s13_inventory <= '0; s13_delta <= '0; s13_lap <= '0; s13_candidate <= '0; s13_omega_next <= '0; s13_flow_next <= '0; s13_score <= '0;
            s14_inventory <= '0; s14_delta <= '0; s14_lap <= '0; s14_candidate <= '0; s14_omega_next <= '0; s14_flow_next <= '0; s14_score <= '0;
            s15_inventory <= '0; s15_delta <= '0; s15_lap <= '0; s15_candidate <= '0; s15_omega_next <= '0; s15_flow_next <= '0; s15_score <= '0;
            s16_candidate <= '0; s16_omega_next <= '0; s16_flow_next <= '0; s16_lap <= '0; s16_score <= '0; s16_action <= '0; s16_quantity <= '0; s16_risk <= 1'b0;
        end else begin
            v0 <= in_valid;
            v1 <= v0; v2 <= v1; v3 <= v2; v4 <= v3; v5 <= v4; v6 <= v5;
            v7 <= v6; v8 <= v7; v9 <= v8; v10 <= v9; v11 <= v10;
            v12 <= v11; v13 <= v12; v14 <= v13; v15 <= v14; v16 <= v15;
            out_valid <= v16;

            if (in_valid) begin
                s0_psi <= psi_i; s0_omega <= omega_i; s0_flow <= flow_i;
                s0_inventory <= inventory_i; s0_delta <= delta_quantity_i; s0_side_bid <= side_bid_i;
                s0_xm <= psi_xm_i; s0_xp <= psi_xp_i; s0_ym <= psi_ym_i;
                s0_yp <= psi_yp_i; s0_zm <= psi_zm_i; s0_zp <= psi_zp_i;
            end

            if (v0) begin
                s1_psi <= s0_psi; s1_omega <= s0_omega; s1_inventory <= s0_inventory; s1_delta <= s0_delta;
                s1_impulse <= s0_side_bid ? s0_delta : qneg(s0_delta);
                s1_flow_next <= qadd(qmul7div8(s0_flow), s0_side_bid ? s0_delta : qneg(s0_delta));
                s1_residual <= qsub(s0_psi, s0_omega);
                s1_neighbours <= qadd(s0_xm, s0_xp);
                s1_ym <= s0_ym; s1_yp <= s0_yp; s1_zm <= s0_zm; s1_zp <= s0_zp;
            end

            if (v1) begin
                s2_psi <= s1_psi; s2_omega <= s1_omega; s2_inventory <= s1_inventory; s2_delta <= s1_delta;
                s2_impulse <= s1_impulse; s2_flow_next <= s1_flow_next; s2_residual <= s1_residual;
                s2_neighbours <= qadd(s1_neighbours, s1_ym);
                s2_yp <= s1_yp; s2_zm <= s1_zm; s2_zp <= s1_zp;
            end

            if (v2) begin
                s3_psi <= s2_psi; s3_omega <= s2_omega; s3_inventory <= s2_inventory; s3_delta <= s2_delta;
                s3_impulse <= s2_impulse; s3_flow_next <= s2_flow_next; s3_residual <= s2_residual;
                s3_neighbours <= qadd(s2_neighbours, s2_yp);
                s3_zm <= s2_zm; s3_zp <= s2_zp;
            end

            if (v3) begin
                s4_psi <= s3_psi; s4_omega <= s3_omega; s4_inventory <= s3_inventory; s4_delta <= s3_delta;
                s4_impulse <= s3_impulse; s4_flow_next <= s3_flow_next; s4_residual <= s3_residual;
                s4_neighbours <= qadd(s3_neighbours, s3_zm);
                s4_zp <= s3_zp;
            end

            if (v4) begin
                s5_psi <= s4_psi; s5_omega <= s4_omega; s5_inventory <= s4_inventory; s5_delta <= s4_delta;
                s5_impulse <= s4_impulse; s5_flow_next <= s4_flow_next; s5_residual <= s4_residual;
                s5_neighbours <= qadd(s4_neighbours, s4_zp);
            end

            if (v5) begin
                s6_psi <= s5_psi; s6_omega <= s5_omega; s6_inventory <= s5_inventory; s6_delta <= s5_delta;
                s6_impulse <= s5_impulse; s6_flow_next <= s5_flow_next; s6_residual <= s5_residual;
                s6_lap <= qsub(s5_neighbours, qmul6(s5_psi));
            end

            if (v6) begin
                s7_psi <= s6_psi; s7_omega <= s6_omega; s7_inventory <= s6_inventory; s7_delta <= s6_delta;
                s7_impulse <= s6_impulse; s7_flow_next <= s6_flow_next; s7_lap <= s6_lap;
                s7_rhs <= qneg(qdiv4(s6_residual));
            end

            if (v7) begin
                s8_psi <= s7_psi; s8_omega <= s7_omega; s8_inventory <= s7_inventory; s8_delta <= s7_delta;
                s8_impulse <= s7_impulse; s8_flow_next <= s7_flow_next; s8_lap <= s7_lap;
                s8_rhs <= qadd(s7_rhs, qdiv32(s7_lap));
            end

            if (v8) begin
                s9_psi <= s8_psi; s9_omega <= s8_omega; s9_inventory <= s8_inventory; s9_delta <= s8_delta;
                s9_flow_next <= s8_flow_next; s9_lap <= s8_lap;
                s9_rhs <= qadd(s8_rhs, qdiv2(s8_impulse));
            end

            if (v9) begin
                s10_omega <= s9_omega; s10_inventory <= s9_inventory; s10_delta <= s9_delta;
                s10_flow_next <= s9_flow_next; s10_lap <= s9_lap;
                s10_candidate <= qclamp(qadd(s9_psi, qdiv8(s9_rhs)), qneg(MAX_ABS_FIELD), MAX_ABS_FIELD);
            end

            if (v10) begin
                s11_inventory <= s10_inventory; s11_delta <= s10_delta; s11_flow_next <= s10_flow_next; s11_lap <= s10_lap;
                s11_candidate <= s10_candidate;
                s11_omega_next <= qadd(qmul15div16(s10_omega), qdiv16(s10_candidate));
            end

            if (v11) begin
                s12_inventory <= s11_inventory; s12_delta <= s11_delta; s12_flow_next <= s11_flow_next; s12_lap <= s11_lap;
                s12_candidate <= s11_candidate; s12_omega_next <= s11_omega_next;
                s12_score <= qadd(s11_candidate, qdiv2(s11_omega_next));
            end

            if (v12) begin
                s13_inventory <= s12_inventory; s13_delta <= s12_delta; s13_lap <= s12_lap;
                s13_candidate <= s12_candidate; s13_omega_next <= s12_omega_next; s13_flow_next <= s12_flow_next;
                s13_score <= qadd(s12_score, qdiv2(s12_flow_next));
            end

            if (v13) begin
                s14_inventory <= s13_inventory; s14_delta <= s13_delta; s14_lap <= s13_lap;
                s14_candidate <= s13_candidate; s14_omega_next <= s13_omega_next; s14_flow_next <= s13_flow_next;
                s14_score <= qadd(s13_score, qdiv8(s13_lap));
            end

            if (v14) begin
                s15_inventory <= s14_inventory; s15_delta <= s14_delta; s15_lap <= s14_lap;
                s15_candidate <= s14_candidate; s15_omega_next <= s14_omega_next; s15_flow_next <= s14_flow_next;
                s15_score <= qsub(s14_score, qdiv4(s14_inventory));
            end

            if (v15) begin
                s16_candidate <= s15_candidate; s16_omega_next <= s15_omega_next;
                s16_flow_next <= s15_flow_next; s16_lap <= s15_lap; s16_score <= s15_score;
                s16_action <= final_action_c; s16_quantity <= final_quantity_c; s16_risk <= final_risk_c;
            end

            if (v16) begin
                psi_o <= s16_candidate;
                omega_o <= s16_omega_next;
                flow_o <= s16_flow_next;
                laplacian_o <= s16_lap;
                score_o <= s16_score;
                action_o <= s16_action;
                quantity_o <= s16_quantity;
                risk_accepted_o <= s16_risk;
            end
        end
    end

endmodule
