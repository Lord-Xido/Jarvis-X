#include "spatial_engine3d.hpp"
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        std::size_t count = JarvisX::MAX_SPARSE_POINTS;
        int steps = 1;
        double scale = 0.5;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a.rfind("--count=", 0) == 0) count = std::stoull(a.substr(8));
            else if (a.rfind("--steps=", 0) == 0) steps = std::stoi(a.substr(8));
            else if (a.rfind("--scale=", 0) == 0) scale = std::stod(a.substr(8));
            else throw std::invalid_argument("unexpected argument: " + a);
        }
        if (steps < 0 || steps > 10000) throw std::invalid_argument("steps must be in [0,10000]");
        JarvisX::SpatialEngine3D engine;
        engine.generate_sparse_grid(count);
        std::cout << std::fixed << std::setprecision(6);
        auto print = [&](const char* label) {
            std::cout << label << " count=" << engine.size();
            if (engine.size() > 0) {
                const auto& first = engine.points().front();
                const auto& last = engine.points().back();
                std::cout << " first=(" << JarvisX::to_double(first.x) << ","
                          << JarvisX::to_double(first.y) << ","
                          << JarvisX::to_double(first.z) << ")"
                          << " last=(" << JarvisX::to_double(last.x) << ","
                          << JarvisX::to_double(last.y) << ","
                          << JarvisX::to_double(last.z) << ")";
            }
            std::cout << " fnv1a64=" << std::hex << engine.fnv1a64() << std::dec << "\n";
        };
        print("initial");
        for (int i = 0; i < steps; ++i) engine.apply_contraction(scale);
        print("contracted");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "spatial3d error: " << e.what() << "\n";
        return 2;
    }
}
