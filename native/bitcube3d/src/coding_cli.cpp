#include "bitcube3d/program.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace bitcube3d;
static void usage() {
    std::cerr << "BitCube3D C++ Coding Engine v2\n"
              << "Usage:\n"
              << "  bitcube3d-cpp run SCRIPT.b3ds\n"
              << "  bitcube3d-cpp repl\n"
              << "  bitcube3d-cpp codegen SCRIPT.b3ds OUTPUT.cpp\n"
              << "  bitcube3d-cpp help\n";
}
int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 2; }
        const std::string mode(argv[1]);
        if (mode == "run" && argc == 3) {
            std::ifstream script(argv[2]);
            if (!script) throw std::runtime_error("cannot read script");
            Program program(std::cout);
            program.run_script(script);
            std::cout << "DONE commands=" << program.executed_commands() << '\n';
        } else if (mode == "repl" && argc == 2) {
            Program program(std::cout);
            std::cout << "BitCube3D native REPL. HELP for commands; EXIT to quit.\n";
            std::string line;
            while (true) {
                std::cout << "b3d> " << std::flush;
                if (!std::getline(std::cin,line) || line == "EXIT" || line == "QUIT") break;
                try { program.run_line(line); }
                catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; }
            }
        } else if (mode == "codegen" && argc == 4) {
            std::ifstream source(argv[2]);
            if (!source) throw std::runtime_error("cannot read script");
            std::ofstream target(argv[3]);
            if (!target) throw std::runtime_error("cannot create output C++ file");
            generate_cpp(source,target);
            std::cout << "Generated " << argv[3] << '\n';
        } else if (mode == "help" && argc == 2) usage();
        else { usage(); return 2; }
        return 0;
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
