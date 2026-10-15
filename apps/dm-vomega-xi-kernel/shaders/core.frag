#version 300 es
precision highp float;

in vec3 v_cognitiveState;
in float v_stability;
in float v_iterationEnergy;

uniform float u_time;

out vec4 fragColor;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;

    vec3 shaped = (2.25 * v_cognitiveState) /\n                  (1.0 + abs(2.25 * v_cognitiveState));\n    vec3 spectrum = 0.5 + 0.5 * shaped;
    float pulse = 0.82 + 0.18 * sin(
        u_time * 2.0 + v_iterationEnergy * 18.0 +
        dot(v_cognitiveState, vec3(3.1, 4.7, 5.9)));

    vec3 cyan = vec3(0.12, 0.84, 1.0);
    vec3 magenta = vec3(0.95, 0.20, 1.0);
    vec3 amber = vec3(1.0, 0.58, 0.12);
    vec3 color = mix(cyan, magenta, spectrum.y);
    color = mix(color, amber, spectrum.x * spectrum.z * 0.62);
    color *= pulse * mix(0.38, 1.25, v_stability);

    float alpha = smoothstep(1.0, 0.05, r2) *
                  mix(0.30, 0.92, v_stability);
    fragColor = vec4(color, alpha);
}
