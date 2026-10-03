#version 300 es
precision highp float;

layout(location = 0) in vec3 a_spatialVertex;
layout(location = 1) in vec4 a_quaternionToken;

uniform mat4 u_viewProjection;
uniform float u_time;
uniform float u_deltaContext;
uniform float u_hbarSemantic;
uniform float u_pointScale;

out vec3 v_cognitiveState;
out float v_stability;
out float v_iterationEnergy;

vec3 phiDescription(vec3 pos, vec4 token) {
    vec4 q = normalize(token);
    return q.w * pos + cross(q.xyz, pos);
}

vec3 executePsiPhiLambdaStack(vec3 pos, vec4 token, float deltaCtx,
                              out float stability, out float energy) {
    vec4 q = normalize(token);
    float alpha = clamp(exp(-max(deltaCtx, 0.0001)), 0.0, 0.9999);

    vec3 attractor = 0.16 * q.xyz * vec3(
        sin(u_time * 0.37 + q.w * 2.0),
        cos(u_time * 0.29 + q.x * 2.0),
        sin(u_time * 0.23 + q.y * 2.0)
    );

    vec3 state = pos;
    energy = 0.0;
    stability = 1.0;

    for (int k = 0; k < 12; ++k) {
        vec3 spatialDescription = phiDescription(state, q);
        float localStability =
            smoothstep(u_hbarSemantic, 1.0, length(spatialDescription));

        vec3 nextState =
            attractor + alpha * (spatialDescription - attractor);

        energy += length(nextState - state);
        stability *= mix(0.92, 1.0, localStability);
        state = nextState;
    }

    energy /= 12.0;
    return state;
}

void main() {
    float stability;
    float energy;
    vec3 cognitiveState = executePsiPhiLambdaStack(
        a_spatialVertex, a_quaternionToken, u_deltaContext, stability, energy);

    v_cognitiveState = cognitiveState;
    v_stability = stability;
    v_iterationEnergy = energy;

    gl_Position = u_viewProjection * vec4(cognitiveState, 1.0);
    gl_PointSize = u_pointScale * (1.0 + 2.2 * stability);
}
