#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace jx64 {
constexpr int W=64, H=64, D=64, N=W*H*D;
constexpr std::uint32_t MASK = N-1;
using Field = std::vector<float>;
static_assert((N&(N-1))==0);
struct Config {
  std::uint64_t seed=0xDEADBEEF12345678ULL;
  std::uint32_t stride=73;
  int iterations=8;
  float feedback_gain=0.5f;
};
struct Metrics {
  double mse_to_seed=0;
  double mse_to_encoded=0;
  double checksum32=0;
  std::uint64_t input_hash=0;
  std::uint64_t output_hash=0;
  std::string decoder;
};
struct AutoEncoding3DEngine {
  Field R0, R1, R2, R3, R4, seed_field, scratch;
  AutoEncoding3DEngine();
  static std::size_t idx(int x,int y,int z);
  static float activate(float x);
  static std::uint32_t normalized_stride(std::uint32_t stride);
  static std::uint32_t inverse_stride(std::uint32_t stride);
  static std::uint64_t hash_field(const Field& v);
  void generate_seed(std::uint64_t seed);
  void encode();
  void permute(std::uint32_t stride);
  void undo_permute(std::uint32_t stride, const Field& in, Field& out) const;
  void feedback(int iterations,float gain);
  void propagate();
  void decode_reference(Field& out) const;
  void decode_separable(Field& out);
  Metrics execute(Config cfg, const std::string& decoder);
};
}
