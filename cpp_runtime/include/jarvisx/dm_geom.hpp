#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace jarvisx::dmgeom {

struct Vec3 {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};

    [[nodiscard]] Vec3 operator+(const Vec3& other) const noexcept;
    [[nodiscard]] Vec3 operator-(const Vec3& other) const noexcept;
    [[nodiscard]] Vec3 operator*(float scalar) const noexcept;
    [[nodiscard]] float dot(const Vec3& other) const noexcept;
    [[nodiscard]] Vec3 cross(const Vec3& other) const noexcept;
    [[nodiscard]] float length() const noexcept;
    [[nodiscard]] Vec3 normalized() const noexcept;
};

struct Triangle {
    std::uint32_t i0{0U};
    std::uint32_t i1{0U};
    std::uint32_t i2{0U};
};

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
};

struct Latent3D {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};
};

using FeatureVector = std::array<float, 9U>;

struct LossBreakdown {
    float position{0.0F};
    float radius{0.0F};
    float area{0.0F};
    float total{0.0F};
};

struct EvolutionConfig {
    int steps{100};
    int segments{32};
    float learning_rate{0.05F};
    float finite_difference_epsilon{1.0e-3F};
    float gradient_clip{5.0F};
    float tolerance{1.0e-7F};
    bool verbose{true};
};

class Encoder {
public:
    [[nodiscard]] static FeatureVector extract_features(const Mesh& mesh);
    [[nodiscard]] static Latent3D encode(const Mesh& mesh);
};

class Decoder {
public:
    [[nodiscard]] static Mesh decode(const Latent3D& latent, int segments = 32);
};

[[nodiscard]] bool validate_mesh(const Mesh& mesh, std::string* error = nullptr);
[[nodiscard]] float surface_area(const Mesh& mesh);
[[nodiscard]] float mean_radius(const Mesh& mesh);
[[nodiscard]] LossBreakdown reconstruction_loss(const Mesh& target, const Mesh& decoded);
[[nodiscard]] Latent3D evolve(const Mesh& target, Latent3D start, const EvolutionConfig& config = {});
[[nodiscard]] Mesh make_uv_sphere(float radius, int segments);
void export_binary_stl(const Mesh& mesh, const std::string& path);

}  // namespace jarvisx::dmgeom
