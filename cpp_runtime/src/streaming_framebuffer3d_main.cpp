#include "streaming_framebuffer3d.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
std::size_t positive(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("positive integer required");
    const auto number = std::stoull(value);
    if (!number || number > 16384) throw std::out_of_range("integer outside supported bounds");
    return static_cast<std::size_t>(number);
}
}

int main(int argc, char** argv) {
    try {
        dm3d_frame::Config config;
        std::size_t frames = 1;
        std::string in_path, out_path;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("option missing argument: " + option);
            const std::string value = argv[++i];
            if (option == "--width") config.width = positive(value);
            else if (option == "--height") config.height = positive(value);
            else if (option == "--iterations") config.iterations = positive(value);
            else if (option == "--frames") frames = positive(value);
            else if (option == "--input") in_path = value;
            else if (option == "--output") out_path = value;
            else throw std::invalid_argument("unknown option: " + option);
        }
        if (frames > 8 || (frames != 1 && (!in_path.empty() || !out_path.empty())))
            throw std::invalid_argument("raw RGB24 file I/O requires one frame; maximum 8 demo frames");
        std::ifstream in;
        std::ofstream out;
        if (!in_path.empty()) {
            in.open(in_path, std::ios::binary);
            if (!in) throw std::runtime_error("cannot open RGB24 input");
        }
        if (!out_path.empty()) {
            out.open(out_path, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot open RGB24 output");
        }
        dm3d_frame::Engine engine(config);
        std::cout << "DM3D 8K streaming framebuffer reference (RGB24, torus 8^3)\n"
                  << "frame=" << config.width << "x" << config.height
                  << " latent=512x3 per 64x64 tile; fixed-point bound="
                  << 1.0-config.gamma+12.0*config.diffusion << '\n';
        for (std::size_t frame = 0; frame < frames; ++frame) {
            const auto report = engine.process_frame(frame, in_path.empty()?nullptr:&in,
                                                       out_path.empty()?nullptr:&out);
            const bool ok = report.authoritative_mse <= report.baseline_mse + 1e-9;
            std::cout << std::setprecision(10) << "frame=" << frame
                      << " pixels=" << report.pixels
                      << " baseline_mse=" << report.baseline_mse
                      << " mse=" << report.authoritative_mse
                      << " accepted_tiles=" << report.accepted_tiles
                      << " rejected_steps=" << report.rejected_steps
                      << " elapsed_ms=" << report.elapsed_ms
                      << " bounded_bytes=" << report.working_bytes_upper_bound
                      << " verify=" << (ok?"PASS":"FAIL") << '\n';
            if (!ok) return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
