#pragma once
// DM3D finite hidden-fiber runtime: exact Z11 x Z6 x Z4 cubed.
// A constant visible collapse is NOT a learned encoder or a contraction.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace dm3d::fiber {

inline constexpr std::uint64_t BASE_SIZE = 11ull * 6ull * 4ull;
inline constexpr std::uint64_t STATE_COUNT = BASE_SIZE * BASE_SIZE * BASE_SIZE;
inline constexpr std::size_t LATENT_DIM = 512;
inline constexpr std::array<int, 9> RADIX = {11, 6, 4, 11, 6, 4, 11, 6, 4};
static_assert(BASE_SIZE == 264 && STATE_COUNT == 18399744, "finite fiber count");

struct State {
    std::array<std::uint8_t, 9> coord{};
    bool operator==(const State& other) const noexcept { return coord == other.coord; }
    bool operator!=(const State& other) const noexcept { return !(*this == other); }
};
struct XYZ { std::uint16_t x{}, y{}, z{}; };

inline void validate(const State& s) {
    for (std::size_t i = 0; i < RADIX.size(); ++i)
        if (s.coord[i] >= RADIX[i]) throw std::invalid_argument("group digit exceeds its modulus");
}
inline XYZ to_xyz(const State& s) {
    validate(s);
    auto axis = [&](int off) {
        return std::uint16_t(s.coord[off] + 11 * (s.coord[off+1] + 6 * s.coord[off+2]));
    };
    return {axis(0), axis(3), axis(6)};
}
inline State from_xyz(XYZ p) {
    if (p.x >= BASE_SIZE || p.y >= BASE_SIZE || p.z >= BASE_SIZE)
        throw std::out_of_range("XYZ coordinate outside the 264-sided virtual array");
    State s;
    const std::array<std::uint16_t,3> axes = {p.x, p.y, p.z};
    for (int a = 0; a < 3; ++a) {
        auto v = axes[std::size_t(a)];
        s.coord[3*a]   = std::uint8_t(v % 11); v /= 11;
        s.coord[3*a+1] = std::uint8_t(v % 6);  v /= 6;
        s.coord[3*a+2] = std::uint8_t(v % 4);
    }
    return s;
}
inline std::uint64_t to_index(const State& s) {
    const auto p=to_xyz(s);
    return std::uint64_t(p.x) + BASE_SIZE * (std::uint64_t(p.y) + BASE_SIZE * p.z);
}
inline State from_index(std::uint64_t i) {
    if(i>=STATE_COUNT) throw std::out_of_range("index outside hidden fiber");
    XYZ p{std::uint16_t(i % BASE_SIZE), std::uint16_t((i/BASE_SIZE)%BASE_SIZE),
          std::uint16_t(i/(BASE_SIZE*BASE_SIZE))};
    return from_xyz(p);
}
// Group arithmetic MUST use componentwise moduli; the XYZ array is only a set bijection.
inline State add(const State& a,const State& b) {
    validate(a);validate(b);State r;
    for(std::size_t i=0;i<RADIX.size();++i)
        r.coord[i]=std::uint8_t((int(a.coord[i])+int(b.coord[i]))%RADIX[i]);
    return r;
}
inline State inverse(const State& a) {
    validate(a);State r;
    for(std::size_t i=0;i<RADIX.size();++i)
        r.coord[i]=std::uint8_t((RADIX[i]-int(a.coord[i]))%RADIX[i]);
    return r;
}
inline int translation_order(const State& shift) {
    validate(shift);int order=1;
    for(std::size_t i=0;i<RADIX.size();++i)
        order=std::lcm(order,RADIX[i]/std::gcd(RADIX[i],int(shift.coord[i])));
    return order; // Divides lcm(11,6,4) = 132.
}
inline State control_shift(std::uint64_t theta,std::uint64_t q) {
    State g;
    // Deterministic, bounded control mapping, not a trainable controller.
    for(std::size_t i=0;i<RADIX.size();++i) {
        std::uint64_t k=theta + (q ^ (0x9E3779B97F4A7C15ull + i * 13));
        g.coord[i]=std::uint8_t((k ^ (k>>11) ^ (k>>29)) % std::uint64_t(RADIX[i]));
    }
    return g;
}

inline const State V_STAR{}; // zero in the *same* space X; Fix(C) is well-defined.
inline State collapse(const State&) noexcept {return V_STAR;}
inline State psi(double /*r_x*/,double /*r_y*/,double /*r_z*/,double /*t*/) noexcept {
    return V_STAR;
}
inline bool ctr_invariant(const State& x,const State& g) {
    return collapse(x)==V_STAR && collapse(add(x,g))==V_STAR
        && add(add(x,g),inverse(g))==x
        && from_index(to_index(x))==x;
}
inline std::uint64_t one_cycle_permutation(std::uint64_t index) {
    if(index>=STATE_COUNT) throw std::out_of_range("invalid permutation index");
    // One permutation cycle of all STATE_COUNT elements; this is NOT a group translation.
    return (index+1==STATE_COUNT)?0:index+1;
}
inline std::array<double,9> readout(const State& x) {
    validate(x);std::array<double,9> r{};
    for(std::size_t i=0;i<9;++i)r[i]=double(x.coord[i])/double(RADIX[i]-1);
    return r; // Injective nonconstant observation, unlike collapse().
}
inline State decode_readout(const std::array<double,9>& r) {
    State x;
    for(std::size_t i=0;i<9;++i){
        if(!std::isfinite(r[i]))throw std::invalid_argument("non-finite observation");
        x.coord[i]=std::uint8_t(std::lround(std::clamp(r[i],0.0,1.0)*double(RADIX[i]-1)));
    }
    return x;
}
using Latent512 = std::array<double,LATENT_DIM>;
inline Latent512 embed512(const State& x) {
    const auto r=readout(x);
    Latent512 z{};
    for(std::size_t i=0;i<9;++i) z[i]=r[i];
    for(std::size_t i=9;i<LATENT_DIM;++i) {
        const std::size_t j=i%9;
        z[i]=std::sin((double((i%11)+1)*r[j]+double((i%7)+1)*r[(j+4)%9])*0.5);
    }
    return z; // Deterministic feature map, NOT a pretrained ANN embedding.
}
inline State decode512(const Latent512& z) {
    std::array<double,9> first{};
    for(std::size_t i=0;i<9;++i)first[i]=z[i];
    return decode_readout(first);
}
inline double mse(const Latent512& a,const Latent512& b) {
    double s=0;
    for(std::size_t i=0;i<LATENT_DIM;++i){const double d=a[i]-b[i];s+=d*d;}
    return s/double(LATENT_DIM);
}
inline double max_residual(const Latent512& a,const Latent512& b){
    double e=0;for(std::size_t i=0;i<LATENT_DIM;++i)e=std::max(e,std::abs(a[i]-b[i]));
    return e;
}
struct Refinement {
    Latent512 latent{};
    int iterations{};
    double final_delta{};
    double final_mse{};
    bool converged{};
};
// Anchored continuous contraction z_(k+1)=lambda*z_k+(1-lambda)*target.
// Never conflate this convergence with the independent periodic hidden group action.
inline Refinement refine(Latent512 current,const Latent512& target,double lambda,
                         int max_steps=100,double tolerance=1e-8) {
    if(!(lambda>=0.0 && lambda<1.0) || !std::isfinite(lambda)
        || !(tolerance>0.0) || !std::isfinite(tolerance) || max_steps<1)
        throw std::invalid_argument("invalid contraction controls");
    for(double v:target)if(!std::isfinite(v))throw std::invalid_argument("non-finite target");
    Refinement r{};
    for(int k=0;k<max_steps;++k){
        for(std::size_t i=0;i<LATENT_DIM;++i)
            current[i]=lambda*current[i]+(1.0-lambda)*target[i];
        r.iterations=k+1;
        // Residual of the *new* state under a hypothetical next iteration.
        // For lambda=0 the fixed point is reached in one step (residual zero).
        r.final_delta=(1.0-lambda)*max_residual(current,target);
        if(r.final_delta<=tolerance){r.converged=true;break;}
    }
    r.latent=current;r.final_mse=mse(current,target);return r;
}
} // namespace dm3d::fiber
