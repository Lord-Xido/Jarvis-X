#include "jarvisx/volumetric_80k_mp4.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void write_ppm(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& rgb,
    std::uint32_t width,
    std::uint32_t height) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open frame output: " + path.string());
    }
    out << "P6\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()),
              static_cast<std::streamsize>(rgb.size()));
}

void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " [--cycles N] [--active-bricks N] [--width N] [--height N]"
           " [--frames DIR] [--fps N] [--quiet]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::uint32_t cycles = 16u;
        std::uint32_t width = 256u;
        std::uint32_t height = 256u;
        std::uint32_t fps = 30u;
        bool quiet = false;
        std::filesystem::path frame_dir;
        jarvisx::volumetric_80k::RuntimePolicy policy{};

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(arg + " requires a value");
                }
                return argv[++i];
            };

            if (arg == "--cycles") {
                const auto v = std::stoul(next());
                if (v == 0ul || v > 100000ul) {
                    throw std::invalid_argument("--cycles must be in [1,100000]");
                }
                cycles = static_cast<std::uint32_t>(v);
            } else if (arg == "--active-bricks") {
                const auto v = std::stoul(next());
                if (v == 0ul || v > 4096ul) {
                    throw std::invalid_argument("--active-bricks must be in [1,4096]");
                }
                policy.max_active_bricks = static_cast<std::uint32_t>(v);
            } else if (arg == "--width") {
                width = static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--height") {
                height = static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--fps") {
                fps = static_cast<std::uint32_t>(std::stoul(next()));
                if (fps == 0u || fps > 240u) {
                    throw std::invalid_argument("--fps must be in [1,240]");
                }
            } else if (arg == "--frames") {
                frame_dir = next();
            } else if (arg == "--quiet") {
                quiet = true;
            } else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }

        if (width == 0u || height == 0u || width > 4096u || height > 4096u) {
            throw std::invalid_argument("frame dimensions must be in [1,4096]");
        }

        if (!frame_dir.empty()) {
            std::filesystem::create_directories(frame_dir);
        }

        jarvisx::volumetric_80k::Engine engine(policy);

        if (!quiet) {
            std::cout
                << "Dr Moagi 80K^3 sparse AE/AD video compute runtime\n"
                << "logical field : 80000 x 80000 x 80000 = "
                << jarvisx::volumetric_80k::kLogicalVoxels << " voxels\n"
                << "brick         : 32^3\n"
                << "active cap    : " << policy.max_active_bricks << "\n"
                << "projection    : " << width << 'x' << height << " RGB\n";
        }

        for (std::uint32_t cycle = 0; cycle < cycles; ++cycle) {
            const auto receipt = engine.step(cycle);

            if (!frame_dir.empty()) {
                const auto rgb = engine.render_rgb(width, height);
                std::ostringstream name;
                name << "frame_" << std::setfill('0') << std::setw(6) << cycle << ".ppm";
                write_ppm(frame_dir / name.str(), rgb, width, height);
            }

            if (!quiet) {
                std::cout
                    << "cycle=" << receipt.cycle
                    << " mse_before=" << receipt.ctr.mse_before
                    << " mse_after=" << receipt.ctr.mse_after
                    << " fp=" << receipt.ctr.fixed_point_relative
                    << " commit=" << (receipt.ctr.committed ? "yes" : "no")
                    << '\n';
            }
        }

        const auto& stats = engine.stats();
        if (!quiet) {
            std::cout
                << "\ncycles         : " << stats.cycles
                << "\ncommits        : " << stats.commits
                << "\nrollbacks      : " << stats.rollbacks
                << "\nactive bricks  : " << stats.active_bricks
                << "\nresident bytes : " << stats.resident_bytes
                << "\nlast MSE       : " << stats.last_mse
                << "\nlast FP rel    : " << stats.last_fixed_point_relative
                << "\n";
            if (!frame_dir.empty()) {
                std::cout
                    << "\nMP4 adapter command:\n"
                    << "ffmpeg -y -framerate " << fps
                    << " -i " << (frame_dir / "frame_%06d.ppm").string()
                    << " -c:v libx264 -pix_fmt yuv420p output.mp4\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << '\n';
        return 1;
    }
}
