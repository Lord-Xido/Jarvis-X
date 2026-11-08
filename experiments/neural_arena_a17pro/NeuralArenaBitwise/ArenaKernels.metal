// NEURAL ARENA v1.3: exact Q12 fixed-point six-neighbor operator.
// Data layout: 3 contiguous signed int16 channels per voxel, x-major.
// Both kernels implement: avg = rounded sum / valid neighbors;
// out = clamp(round((17*center + 3*avg)/20), -4096, 4096).
#include <metal_stdlib>
using namespace metal;

constant uint3 TG_SIZE = uint3(8, 8, 4);
constant uint SX = 10;
constant uint SY = 10;
constant uint SZ = 6;

inline int signed_round_div(int n, int d) {
    return n >= 0 ? (n + d / 2) / d : -((-n + d / 2) / d);
}
inline int output_q12(int center, int sum, int neighbors) {
    int avg = neighbors > 0 ? signed_round_div(sum, neighbors) : center;
    return clamp(signed_round_div(17 * center + 3 * avg, 20), -4096, 4096);
}
inline uint point_index(uint3 p, uint3 dims) {
    return p.x + dims.x * (p.y + dims.y * p.z);
}

kernel void arena_q12_direct(device const short* src [[buffer(0)]],
                             device short* dst [[buffer(1)]],
                             constant uint3& dims [[buffer(2)]],
                             uint3 gid [[thread_position_in_grid]]) {
    if (any(gid >= dims)) return;
    uint p = point_index(gid, dims);
    for (uint c = 0; c < 3; ++c) {
        int center = int(src[3*p+c]);
        int sum = 0; int n = 0;
        if (gid.x > 0)       { sum += int(src[3*(p-1)+c]); n++; }
        if (gid.x+1 < dims.x) { sum += int(src[3*(p+1)+c]); n++; }
        if (gid.y > 0)       { sum += int(src[3*(p-dims.x)+c]); n++; }
        if (gid.y+1 < dims.y) { sum += int(src[3*(p+dims.x)+c]); n++; }
        uint xy = dims.x * dims.y;
        if (gid.z > 0)       { sum += int(src[3*(p-xy)+c]); n++; }
        if (gid.z+1 < dims.z) { sum += int(src[3*(p+xy)+c]); n++; }
        dst[3*p+c] = short(output_q12(center, sum, n));
    }
}

kernel void arena_q12_tiled(device const short* src [[buffer(0)]],
                            device short* dst [[buffer(1)]],
                            constant uint3& dims [[buffer(2)]],
                            uint3 gid [[thread_position_in_grid]],
                            uint3 tid [[thread_position_in_threadgroup]],
                            uint3 tgp [[threadgroup_position_in_grid]]) {
    threadgroup short cache[SX*SY*SZ*3]; // 3600 B, all allocated/initialized
    uint local_linear = tid.x + TG_SIZE.x*(tid.y + TG_SIZE.y*tid.z);
    uint3 base = tgp * TG_SIZE;
    // Cooperative load of the ENTIRE halo tile; padded threads participate.
    for (uint cell = local_linear; cell < SX*SY*SZ; cell += TG_SIZE.x*TG_SIZE.y*TG_SIZE.z) {
        uint lx = cell % SX;
        uint ly = (cell / SX) % SY;
        uint lz = cell / (SX*SY);
        uint3 g = uint3(
            uint(clamp(int(base.x)+int(lx)-1, 0, int(dims.x)-1)),
            uint(clamp(int(base.y)+int(ly)-1, 0, int(dims.y)-1)),
            uint(clamp(int(base.z)+int(lz)-1, 0, int(dims.z)-1))
        );
        uint gi = 3*point_index(g, dims);
        for (uint c = 0; c < 3; ++c) cache[3*cell+c] = src[gi+c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Never exit before the barrier: partially populated groups are supported.
    if (any(gid >= dims)) return;
    uint cell = (tid.x+1) + SX*((tid.y+1)+SY*(tid.z+1));
    uint global = 3*point_index(gid, dims);
    for (uint c = 0; c < 3; ++c) {
        int center = int(cache[3*cell+c]);
        int sum=0, n=0;
        if (gid.x > 0)       { sum += int(cache[3*(cell-1)+c]); n++; }
        if (gid.x+1 < dims.x) { sum += int(cache[3*(cell+1)+c]); n++; }
        if (gid.y > 0)       { sum += int(cache[3*(cell-SX)+c]); n++; }
        if (gid.y+1 < dims.y) { sum += int(cache[3*(cell+SX)+c]); n++; }
        if (gid.z > 0)       { sum += int(cache[3*(cell-SX*SY)+c]); n++; }
        if (gid.z+1 < dims.z) { sum += int(cache[3*(cell+SX*SY)+c]); n++; }
        dst[global+c] = short(output_q12(center, sum, n));
    }
}
