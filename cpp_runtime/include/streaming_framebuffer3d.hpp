#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace dm3d_frame {

constexpr std::size_t tile_edge = 64;
constexpr std::size_t torus_edge = 8;
constexpr std::size_t voxels = torus_edge * torus_edge * torus_edge;
constexpr std::size_t channels = 3;
constexpr std::size_t latent_scalars = voxels * channels;
static_assert(7680ULL * 4320ULL == 33177600ULL, "8K UHD pixel contract");

struct Config {
    std::size_t width = 7680;
    std::size_t height = 4320;
    std::size_t iterations = 5;
    double gamma = 0.60;
    double diffusion = 0.01;
    double memory_beta = 0.88;
};

struct Report {
    std::uint64_t pixels = 0;
    double baseline_mse = 0;
    double authoritative_mse = 0;
    std::uint64_t accepted_tiles = 0;
    std::uint64_t rejected_steps = 0;
    double elapsed_ms = 0;
    std::size_t working_bytes_upper_bound = 0;
};

inline std::size_t at(std::size_t x, std::size_t y, std::size_t z) {
    return (z * torus_edge + y) * torus_edge + x;
}

inline std::size_t index_wrap(int x, int y, int z) {
    const auto wrap = [](int v) { return static_cast<std::size_t>((v + 8) % 8); };
    return at(wrap(x), wrap(y), wrap(z));
}

inline std::array<std::uint8_t, 3> demo_pixel(std::size_t x, std::size_t y, std::size_t frame) {
    return {{static_cast<std::uint8_t>((3*x + 5*y + 17*frame) % 256),
             static_cast<std::uint8_t>((7*x + 11*y + 13*frame) % 256),
             static_cast<std::uint8_t>((13*x + 17*y + 19*frame) % 256)}};
}

class Engine {
    Config config_;
    std::size_t tiles_x_ = 0;
    std::vector<std::uint8_t> omega_; // persistent compressed RGB8 latent per tile

public:
    explicit Engine(Config cfg) : config_(cfg) {
        if (cfg.width == 0 || cfg.height == 0 || cfg.width > 16384 || cfg.height > 16384 ||
            cfg.iterations == 0 || cfg.iterations > 32 ||
            !std::isfinite(cfg.gamma) || cfg.gamma <= 0 || cfg.gamma > 1 ||
            !std::isfinite(cfg.diffusion) || cfg.diffusion < 0 ||
            !std::isfinite(cfg.memory_beta) || cfg.memory_beta < 0 || cfg.memory_beta >= 1 ||
            1.0 - cfg.gamma + 12.0 * cfg.diffusion >= 1.0) {
            throw std::invalid_argument("invalid dimensions, iterations or non-contractive parameters");
        }
        tiles_x_ = (cfg.width + tile_edge - 1) / tile_edge;
        const std::size_t tiles_y = (cfg.height + tile_edge - 1) / tile_edge;
        omega_.assign(tiles_x_ * tiles_y * latent_scalars, 128);
    }

    std::size_t persistent_bytes() const { return omega_.size(); }

    Report process_frame(std::size_t frame, std::istream* input = nullptr, std::ostream* output = nullptr) {
        const auto start = std::chrono::steady_clock::now();
        Report report;
        std::vector<std::uint8_t> strip;
        std::vector<std::uint8_t> decoded;
        double baseline_sum = 0.0, authoritative_sum = 0.0;

        for (std::size_t top = 0; top < config_.height; top += tile_edge) {
            const std::size_t rows = std::min(tile_edge, config_.height - top);
            const std::size_t bytes = rows * config_.width * channels;
            strip.resize(bytes);
            if (input != nullptr) {
                if (!input->read(reinterpret_cast<char*>(strip.data()),
                                 static_cast<std::streamsize>(bytes))) {
                    throw std::runtime_error("short RGB24 framebuffer input");
                }
            } else {
                for (std::size_t y = 0; y < rows; ++y) {
                    for (std::size_t x = 0; x < config_.width; ++x) {
                        const auto rgb = demo_pixel(x, top + y, frame);
                        for (std::size_t c = 0; c < channels; ++c)
                            strip[(y * config_.width + x) * channels + c] = rgb[c];
                    }
                }
            }
            if (output != nullptr) decoded.resize(bytes);

            for (std::size_t left = 0; left < config_.width; left += tile_edge) {
                const std::size_t cols = std::min(tile_edge, config_.width - left);
                const std::size_t tile = (top / tile_edge) * tiles_x_ + (left / tile_edge);
                const std::size_t offset = tile * latent_scalars;
                std::array<double, latent_scalars> sums{}, squared{}, target{};
                std::array<std::uint32_t, voxels> count{};
                std::array<double, latent_scalars> state{}, trial{};

                for (std::size_t y = 0; y < rows; ++y) {
                    for (std::size_t x = 0; x < cols; ++x) {
                        const std::size_t voxel = (y * cols + x) % voxels;
                        ++count[voxel];
                        for (std::size_t c = 0; c < channels; ++c) {
                            const double v = strip[(y * config_.width + left + x)*channels+c] / 255.0;
                            sums[voxel*channels+c] += v;
                            squared[voxel*channels+c] += v*v;
                        }
                    }
                }
                for (std::size_t v = 0; v < voxels; ++v) {
                    for (std::size_t c = 0; c < channels; ++c) {
                        const std::size_t i = v*channels+c;
                        target[i] = count[v] ? sums[i] / count[v] : 0.5;
                        state[i] = omega_[offset+i] / 255.0;
                    }
                }
                const auto loss = [&](const std::array<double, latent_scalars>& z) {
                    double total = 0.0;
                    for (std::size_t v = 0; v < voxels; ++v) {
                        for (std::size_t c = 0; c < channels; ++c) {
                            const std::size_t i = v*channels+c;
                            total += squared[i] - 2.0*z[i]*sums[i] + count[v]*z[i]*z[i];
                        }
                    }
                    return std::max(0.0, total);
                };
                const double initial_loss = loss(state);
                double best_loss = initial_loss;
                bool improved = false;

                for (std::size_t step = 0; step < config_.iterations; ++step) {
                    for (std::size_t z = 0; z < torus_edge; ++z)
                        for (std::size_t y = 0; y < torus_edge; ++y)
                            for (std::size_t x = 0; x < torus_edge; ++x) {
                                const std::size_t v = at(x,y,z);
                                const std::array<std::size_t,6> nb{{
                                    index_wrap(static_cast<int>(x)+1,static_cast<int>(y),static_cast<int>(z)),
                                    index_wrap(static_cast<int>(x)-1,static_cast<int>(y),static_cast<int>(z)),
                                    index_wrap(static_cast<int>(x),static_cast<int>(y)+1,static_cast<int>(z)),
                                    index_wrap(static_cast<int>(x),static_cast<int>(y)-1,static_cast<int>(z)),
                                    index_wrap(static_cast<int>(x),static_cast<int>(y),static_cast<int>(z)+1),
                                    index_wrap(static_cast<int>(x),static_cast<int>(y),static_cast<int>(z)-1)}};
                                for (std::size_t c = 0; c < channels; ++c) {
                                    const std::size_t i = v*channels+c;
                                    double lap = -6.0*state[i];
                                    for (std::size_t n : nb) lap += state[n*channels+c];
                                    trial[i] = std::clamp(state[i] + config_.gamma*(target[i]-state[i]) +
                                                          config_.diffusion*lap, 0.0, 1.0);
                                }
                            }
                    const double candidate_loss = loss(trial);
                    if (!std::isfinite(candidate_loss) || candidate_loss > best_loss + 1e-10) {
                        ++report.rejected_steps;
                        break; // CTR: authoritative state cannot regress
                    }
                    improved = improved || candidate_loss < best_loss - 1e-10;
                    best_loss = candidate_loss;
                    state.swap(trial);
                }
                report.accepted_tiles += improved ? 1ULL : 0ULL;
                baseline_sum += initial_loss;
                authoritative_sum += best_loss;

                for (std::size_t i = 0; i < latent_scalars; ++i) {
                    const double updated = config_.memory_beta*(omega_[offset+i]/255.0) +
                                           (1.0-config_.memory_beta)*state[i];
                    omega_[offset+i] = static_cast<std::uint8_t>(std::lround(255.0*std::clamp(updated,0.0,1.0)));
                }
                if (output != nullptr) {
                    for (std::size_t y = 0; y < rows; ++y)
                        for (std::size_t x = 0; x < cols; ++x) {
                            const std::size_t voxel = (y*cols+x) % voxels;
                            for (std::size_t c = 0; c < channels; ++c)
                                decoded[(y*config_.width+left+x)*channels+c] =
                                    static_cast<std::uint8_t>(std::lround(255.0*state[voxel*channels+c]));
                        }
                }
            }
            if (output != nullptr && !output->write(reinterpret_cast<const char*>(decoded.data()),
                                                    static_cast<std::streamsize>(bytes)))
                throw std::runtime_error("failed writing RGB24 framebuffer output");
        }
        report.pixels = static_cast<std::uint64_t>(config_.width) * config_.height;
        report.baseline_mse = baseline_sum / (static_cast<double>(report.pixels)*channels);
        report.authoritative_mse = authoritative_sum / (static_cast<double>(report.pixels)*channels);
        report.working_bytes_upper_bound = omega_.size() + 2*tile_edge*config_.width*channels +
                                            5*latent_scalars*sizeof(double) + voxels*sizeof(std::uint32_t);
        report.elapsed_ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return report;
    }
};
} // namespace dm3d_frame
