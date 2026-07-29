#include "jarvisx/runtime.hpp"

int main(int argc, char** argv) {
    try {
        jarvisx::Options options = jarvisx::parse_options(argc, argv);
        jarvisx::Packet packet = jarvisx::resolve_packet(argc, argv, options);
        jarvisx::Runtime runtime(std::move(options), std::move(packet));
        return runtime.run();
    } catch (const std::exception& error) {
        std::cerr << "Jarvis X runtime failure: " << error.what() << '\n';
        return 1;
    }
}
