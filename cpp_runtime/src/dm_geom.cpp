#include "jarvisx/dm_geom.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace jarvisx::dmgeom {
namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kMinRadius = 1.0e-4F;
constexpr float kMinLogRadius = -8.0F;
constexpr float kMaxLogRadius = 8.0F;

[[nodiscard]] float square(float value) noexcept { return value * value; }

[[nodiscard]] float clamp_finite(float value, float low, float high, float fallback) noexcept {
    return std::isfinite(value) ? std::clamp(value, low, high) : fallback;
}

[[nodiscard]] float sigmoid(float value) noexcept {
    const float bounded = std::clamp(value, -20.0F, 20.0F);
    return 1.0F / (1.0F + std::exp(-bounded));
}

[[nodiscard]] Latent3D project(Latent3D latent) noexcept {
    latent.x = clamp_finite(latent.x, kMinLogRadius, kMaxLogRadius, 0.0F);
    latent.y = clamp_finite(latent.y, -8.0F, 8.0F, 0.0F);
    latent.z = clamp_finite(latent.z, -8.0F, 8.0F, 0.0F);
    return latent;
}

[[nodiscard]] float decoded_radius(const Latent3D& latent) noexcept {
    return std::exp(project(latent).x);
}

[[nodiscard]] float decoded_amplitude(const Latent3D& latent) noexcept {
    return 0.35F * std::tanh(project(latent).y);
}

[[nodiscard]] float decoded_frequency(const Latent3D& latent) noexcept {
    return 2.0F + (6.0F * sigmoid(project(latent).z));
}

[[nodiscard]] float relative_error(float actual, float expected) noexcept {
    return (actual - expected) / std::max(std::fabs(expected), 1.0e-6F);
}

void write_u16_le(std::ofstream& stream, std::uint16_t value) {
    const std::array<char, 2U> bytes{
        static_cast<char>(value & 0xFFU),
        static_cast<char>((value >> 8U) & 0xFFU),
    };
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_u32_le(std::ofstream& stream, std::uint32_t value) {
    const std::array<char, 4U> bytes{
        static_cast<char>(value & 0xFFU),
        static_cast<char>((value >> 8U) & 0xFFU),
        static_cast<char>((value >> 16U) & 0xFFU),
        static_cast<char>((value >> 24U) & 0xFFU),
    };
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_f32_le(std::ofstream& stream, float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "binary STL requires 32-bit float");
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32_le(stream, bits);
}

[[nodiscard]] float radial_stddev(const Mesh& mesh, float mean) {
    double sum = 0.0;
    for (const Vec3& vertex : mesh.vertices) {
        const double delta = static_cast<double>(vertex.length() - mean);
        sum += delta * delta;
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(mesh.vertices.size())));
}

[[nodiscard]] std::vector<float> edge_lengths(const Mesh& mesh) {
    std::vector<float> lengths;
    lengths.reserve(mesh.triangles.size() * 3U);
    for (const Triangle& triangle : mesh.triangles) {
        const Vec3& a = mesh.vertices[triangle.i0];
        const Vec3& b = mesh.vertices[triangle.i1];
        const Vec3& c = mesh.vertices[triangle.i2];
        lengths.push_back((b - a).length());
        lengths.push_back((c - b).length());
        lengths.push_back((a - c).length());
    }
    return lengths;
}

}  // namespace

Vec3 Vec3::operator+(const Vec3& other) const noexcept { return {x + other.x, y + other.y, z + other.z}; }
Vec3 Vec3::operator-(const Vec3& other) const noexcept { return {x - other.x, y - other.y, z - other.z}; }
Vec3 Vec3::operator*(float scalar) const noexcept { return {x * scalar, y * scalar, z * scalar}; }
float Vec3::dot(const Vec3& other) const noexcept { return (x * other.x) + (y * other.y) + (z * other.z); }
Vec3 Vec3::cross(const Vec3& other) const noexcept {
    return {(y * other.z) - (z * other.y), (z * other.x) - (x * other.z), (x * other.y) - (y * other.x)};
}
float Vec3::length() const noexcept { return std::sqrt(dot(*this)); }
Vec3 Vec3::normalized() const noexcept {
    const float magnitude = length();
    return magnitude > 0.0F ? (*this) * (1.0F / magnitude) : Vec3{};
}

bool validate_mesh(const Mesh& mesh, std::string* error) {
    const auto fail = [error](const std::string& message) {
        if (error != nullptr) { *error = message; }
        return false;
    };
    if (mesh.vertices.empty()) { return fail("mesh contains no vertices"); }
    if (mesh.triangles.empty()) { return fail("mesh contains no triangles"); }
    if (mesh.vertices.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return fail("mesh exceeds 32-bit index capacity");
    }
    for (const Vec3& vertex : mesh.vertices) {
        if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) || !std::isfinite(vertex.z)) {
            return fail("mesh contains a non-finite vertex");
        }
    }
    const std::uint32_t count = static_cast<std::uint32_t>(mesh.vertices.size());
    for (const Triangle& triangle : mesh.triangles) {
        if (triangle.i0 >= count || triangle.i1 >= count || triangle.i2 >= count) {
            return fail("triangle index lies outside the vertex array");
        }
        if (triangle.i0 == triangle.i1 || triangle.i1 == triangle.i2 || triangle.i0 == triangle.i2) {
            return fail("triangle contains repeated indices");
        }
    }
    if (error != nullptr) { error->clear(); }
    return true;
}

float surface_area(const Mesh& mesh) {
    std::string error;
    if (!validate_mesh(mesh, &error)) { throw std::invalid_argument("surface_area: " + error); }
    double area = 0.0;
    for (const Triangle& triangle : mesh.triangles) {
        const Vec3 a = mesh.vertices[triangle.i1] - mesh.vertices[triangle.i0];
        const Vec3 b = mesh.vertices[triangle.i2] - mesh.vertices[triangle.i0];
        area += 0.5 * static_cast<double>(a.cross(b).length());
    }
    return static_cast<float>(area);
}

float mean_radius(const Mesh& mesh) {
    if (mesh.vertices.empty()) { throw std::invalid_argument("mean_radius: mesh contains no vertices"); }
    double total = 0.0;
    for (const Vec3& vertex : mesh.vertices) { total += static_cast<double>(vertex.length()); }
    return static_cast<float>(total / static_cast<double>(mesh.vertices.size()));
}

FeatureVector Encoder::extract_features(const Mesh& mesh) {
    std::string error;
    if (!validate_mesh(mesh, &error)) { throw std::invalid_argument("extract_features: " + error); }

    const std::vector<float> edges = edge_lengths(mesh);
    double edge_sum = 0.0;
    for (float edge : edges) { edge_sum += static_cast<double>(edge); }
    const float edge_mean = static_cast<float>(edge_sum / static_cast<double>(edges.size()));
    double variance_sum = 0.0;
    for (float edge : edges) {
        const double delta = static_cast<double>(edge - edge_mean);
        variance_sum += delta * delta;
    }
    const float edge_stddev = static_cast<float>(std::sqrt(variance_sum / static_cast<double>(edges.size())));
    const float edge_cv = edge_mean > 1.0e-6F ? edge_stddev / edge_mean : 0.0F;

    const float radius_mean = mean_radius(mesh);
    const float radius_cv = radius_mean > 1.0e-6F ? radial_stddev(mesh, radius_mean) / radius_mean : 0.0F;
    Vec3 minimum = mesh.vertices.front();
    Vec3 maximum = mesh.vertices.front();
    for (const Vec3& vertex : mesh.vertices) {
        minimum.x = std::min(minimum.x, vertex.x); minimum.y = std::min(minimum.y, vertex.y); minimum.z = std::min(minimum.z, vertex.z);
        maximum.x = std::max(maximum.x, vertex.x); maximum.y = std::max(maximum.y, vertex.y); maximum.z = std::max(maximum.z, vertex.z);
    }

    return {
        static_cast<float>(mesh.vertices.size()) / 1000.0F,
        edge_cv,
        surface_area(mesh) / 100.0F,
        radius_mean,
        radius_cv,
        maximum.x - minimum.x,
        maximum.y - minimum.y,
        maximum.z - minimum.z,
        static_cast<float>(mesh.triangles.size()) / static_cast<float>(mesh.vertices.size()),
    };
}

Latent3D Encoder::encode(const Mesh& mesh) {
    const FeatureVector features = extract_features(mesh);
    const float radius = std::max(features[3], kMinRadius);
    const float normalized_deformation = std::clamp(features[4] / 0.35F, -0.95F, 0.95F);
    return project({std::log(radius), std::atanh(normalized_deformation), 0.0F});
}

Mesh make_uv_sphere(float radius, int segments) {
    if (!std::isfinite(radius) || radius <= 0.0F) {
        throw std::invalid_argument("make_uv_sphere: radius must be finite and positive");
    }
    if (segments < 4 || segments > 512) {
        throw std::invalid_argument("make_uv_sphere: segments must be in [4, 512]");
    }

    const int longitudes = segments * 2;
    Mesh mesh;
    mesh.vertices.reserve(2U + static_cast<std::size_t>(segments - 1) * static_cast<std::size_t>(longitudes));
    mesh.triangles.reserve(static_cast<std::size_t>(4 * longitudes * (segments - 1)));
    mesh.vertices.push_back({0.0F, radius, 0.0F});

    for (int latitude = 1; latitude < segments; ++latitude) {
        const float phi = kPi * static_cast<float>(latitude) / static_cast<float>(segments);
        for (int longitude = 0; longitude < longitudes; ++longitude) {
            const float theta = 2.0F * kPi * static_cast<float>(longitude) / static_cast<float>(longitudes);
            mesh.vertices.push_back({
                radius * std::sin(phi) * std::cos(theta),
                radius * std::cos(phi),
                radius * std::sin(phi) * std::sin(theta),
            });
        }
    }

    const std::uint32_t bottom = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({0.0F, -radius, 0.0F});
    const std::uint32_t width = static_cast<std::uint32_t>(longitudes);
    for (std::uint32_t longitude = 0U; longitude < width; ++longitude) {
        const std::uint32_t next = (longitude + 1U) % width;
        mesh.triangles.push_back({0U, 1U + next, 1U + longitude});
    }
    for (int latitude = 0; latitude < segments - 2; ++latitude) {
        const std::uint32_t ring_a = 1U + static_cast<std::uint32_t>(latitude * longitudes);
        const std::uint32_t ring_b = ring_a + width;
        for (std::uint32_t longitude = 0U; longitude < width; ++longitude) {
            const std::uint32_t next = (longitude + 1U) % width;
            mesh.triangles.push_back({ring_a + longitude, ring_b + longitude, ring_a + next});
            mesh.triangles.push_back({ring_a + next, ring_b + longitude, ring_b + next});
        }
    }
    const std::uint32_t last_ring = bottom - width;
    for (std::uint32_t longitude = 0U; longitude < width; ++longitude) {
        const std::uint32_t next = (longitude + 1U) % width;
        mesh.triangles.push_back({last_ring + longitude, last_ring + next, bottom});
    }
    return mesh;
}

Mesh Decoder::decode(const Latent3D& latent, int segments) {
    const float radius = decoded_radius(latent);
    const float amplitude = decoded_amplitude(latent);
    const float frequency = decoded_frequency(latent);
    Mesh mesh = make_uv_sphere(radius, segments);
    for (Vec3& vertex : mesh.vertices) {
        const float base_radius = vertex.length();
        const float theta = std::atan2(vertex.z, vertex.x);
        const float phi = std::acos(std::clamp(vertex.y / base_radius, -1.0F, 1.0F));
        const float scale = 1.0F + (amplitude * std::sin(frequency * theta) * square(std::sin(phi)));
        vertex = vertex.normalized() * (radius * std::max(scale, 0.1F));
    }
    return mesh;
}

LossBreakdown reconstruction_loss(const Mesh& target, const Mesh& decoded) {
    std::string error;
    if (!validate_mesh(target, &error)) { throw std::invalid_argument("reconstruction_loss target: " + error); }
    if (!validate_mesh(decoded, &error)) { throw std::invalid_argument("reconstruction_loss decoded: " + error); }
    if (target.vertices.size() != decoded.vertices.size() || target.triangles.size() != decoded.triangles.size()) {
        throw std::invalid_argument("reconstruction_loss requires matching topology");
    }

    const float target_radius = std::max(mean_radius(target), kMinRadius);
    double position_sum = 0.0;
    for (std::size_t index = 0U; index < target.vertices.size(); ++index) {
        const Vec3 delta = target.vertices[index] - decoded.vertices[index];
        position_sum += static_cast<double>(delta.dot(delta));
    }

    LossBreakdown loss{};
    loss.position = static_cast<float>(position_sum / static_cast<double>(target.vertices.size())) / square(target_radius);
    loss.radius = square(relative_error(mean_radius(decoded), target_radius));
    loss.area = square(relative_error(surface_area(decoded), surface_area(target)));
    loss.total = loss.position + (0.25F * loss.radius) + (0.10F * loss.area);
    return loss;
}

Latent3D evolve(const Mesh& target, Latent3D start, const EvolutionConfig& config) {
    if (config.steps < 0) { throw std::invalid_argument("evolve: steps must be non-negative"); }
    if (config.segments < 4 || config.segments > 512) { throw std::invalid_argument("evolve: segments must be in [4, 512]"); }
    if (!(config.learning_rate > 0.0F) || !std::isfinite(config.learning_rate)) {
        throw std::invalid_argument("evolve: learning_rate must be finite and positive");
    }
    if (!(config.finite_difference_epsilon > 0.0F) || !std::isfinite(config.finite_difference_epsilon)) {
        throw std::invalid_argument("evolve: finite_difference_epsilon must be finite and positive");
    }
    if (!(config.gradient_clip > 0.0F) || !std::isfinite(config.gradient_clip)) {
        throw std::invalid_argument("evolve: gradient_clip must be finite and positive");
    }

    Latent3D latent = project(start);
    const float epsilon = config.finite_difference_epsilon;
    const auto loss_at = [&target, &config](const Latent3D& candidate) {
        return reconstruction_loss(target, Decoder::decode(candidate, config.segments)).total;
    };

    for (int step = 0; step < config.steps; ++step) {
        const float loss = loss_at(latent);
        if (loss <= config.tolerance) { break; }
        float gradient_x = (loss_at({latent.x + epsilon, latent.y, latent.z}) - loss_at({latent.x - epsilon, latent.y, latent.z})) / (2.0F * epsilon);
        float gradient_y = (loss_at({latent.x, latent.y + epsilon, latent.z}) - loss_at({latent.x, latent.y - epsilon, latent.z})) / (2.0F * epsilon);
        float gradient_z = (loss_at({latent.x, latent.y, latent.z + epsilon}) - loss_at({latent.x, latent.y, latent.z - epsilon})) / (2.0F * epsilon);
        gradient_x = std::clamp(gradient_x, -config.gradient_clip, config.gradient_clip);
        gradient_y = std::clamp(gradient_y, -config.gradient_clip, config.gradient_clip);
        gradient_z = std::clamp(gradient_z, -config.gradient_clip, config.gradient_clip);
        latent = project({
            latent.x - (config.learning_rate * gradient_x),
            latent.y - (config.learning_rate * gradient_y),
            latent.z - (config.learning_rate * gradient_z),
        });
        if (config.verbose && (((step % 20) == 0) || step == config.steps - 1)) {
            std::cout << "Step " << step << " Loss: " << loss << " z=("
                      << latent.x << ',' << latent.y << ',' << latent.z << ")\n";
        }
    }
    return latent;
}

void export_binary_stl(const Mesh& mesh, const std::string& path) {
    std::string error;
    if (!validate_mesh(mesh, &error)) { throw std::invalid_argument("export_binary_stl: " + error); }
    if (mesh.triangles.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("export_binary_stl: triangle count exceeds STL limit");
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) { throw std::runtime_error("export_binary_stl: unable to open " + path); }
    std::array<char, 80U> header{};
    constexpr char label[] = "DM-GEOM deterministic binary STL";
    std::copy(label, label + (sizeof(label) - 1U), header.begin());
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    write_u32_le(stream, static_cast<std::uint32_t>(mesh.triangles.size()));
    for (const Triangle& triangle : mesh.triangles) {
        const Vec3& v0 = mesh.vertices[triangle.i0];
        const Vec3& v1 = mesh.vertices[triangle.i1];
        const Vec3& v2 = mesh.vertices[triangle.i2];
        const Vec3 normal = (v1 - v0).cross(v2 - v0).normalized();
        for (float value : std::array<float, 12U>{
                 normal.x, normal.y, normal.z,
                 v0.x, v0.y, v0.z,
                 v1.x, v1.y, v1.z,
                 v2.x, v2.y, v2.z}) {
            write_f32_le(stream, value);
        }
        write_u16_le(stream, 0U);
    }
    if (!stream) { throw std::runtime_error("export_binary_stl: write failed for " + path); }
}

}  // namespace jarvisx::dmgeom
