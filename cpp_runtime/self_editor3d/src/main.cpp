#include "jarvisx/self_editor3d.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using jarvisx::selfedit::WorkspaceMetrics3D;

void usage() {
    std::cout
        << "Jarvis-X Inward 3D C++ Self-Editing Engine\n\n"
        << "Usage:\n"
        << "  jarvisx-self-editor3d scan <workspace>\n"
        << "  jarvisx-self-editor3d analyze-3d <workspace>\n"
        << "  jarvisx-self-editor3d field-3d <workspace>\n"
        << "  jarvisx-self-editor3d propose-3d <workspace>\n"
        << "  jarvisx-self-editor3d optimize-3d <workspace> <max-passes> [validation-command]\n"
        << "  jarvisx-self-editor3d replace <workspace> <file> <needle> <replacement> [validation-command]\n"
        << "  jarvisx-self-editor3d insert-after <workspace> <file> <anchor> <text> [validation-command]\n\n"
        << "Safe self-fold example:\n"
        << "  jarvisx-self-editor3d optimize-3d cpp_runtime 16 \"cmake --build ../build/cpp-runtime --config Release\"\n";
}

std::optional<std::string> arg_optional(int argc, char** argv, int index) {
    if (argc > index) return std::string(argv[index]);
    return std::nullopt;
}

std::size_t parse_size(const char* text) {
    const auto value = std::stoull(text);
    if (value > 10000ULL) throw std::runtime_error("max-passes must be <= 10000");
    return static_cast<std::size_t>(value);
}

void print_metrics(const WorkspaceMetrics3D& m) {
    std::cout << std::fixed << std::setprecision(8)
              << "files=" << m.files
              << " bytes=" << m.bytes
              << " lines=" << m.lines
              << " trailing_ws=" << m.trailing_whitespace_bytes
              << " excessive_blank=" << m.excessive_blank_lines
              << " tabs=" << m.tab_bytes
              << " missing_final_newline=" << m.missing_final_newline_files
              << " mean_hotspot=" << m.mean_hotspot
              << " objective=" << m.objective << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            usage();
            return 1;
        }

        const std::string command = argv[1];
        const std::filesystem::path root = argv[2];

        if (command == "analyze-3d" || command == "field-3d" ||
            command == "propose-3d" || command == "optimize-3d") {
            jarvisx::selfedit::Inward3DSelfOptimizer inward(root);

            if (command == "analyze-3d") {
                print_metrics(inward.analyze());
                return 0;
            }

            if (command == "field-3d") {
                for (const auto& field : inward.encode_workspace()) {
                    std::cout << field.file.generic_string()
                              << " bytes=" << field.source_bytes
                              << " levels=" << field.levels.size() << " sides=";
                    for (std::size_t i = 0; i < field.levels.size(); ++i) {
                        if (i) std::cout << "->";
                        std::cout << field.levels[i].side << '^' << 3;
                    }
                    std::cout << " core=[byte=" << field.core.byte_value
                              << ", syntax=" << field.core.syntax_energy
                              << ", whitespace=" << field.core.whitespace_energy
                              << ", hotspot=" << field.core.hotspot << "]\n";
                }
                return 0;
            }

            if (command == "propose-3d") {
                const auto mutations = inward.propose();
                if (mutations.empty()) {
                    std::cout << "fixed point: no admissible mutations\n";
                    return 0;
                }
                for (const auto& m : mutations) {
                    std::cout << m.file.generic_string()
                              << " range=[" << m.edit.begin << ',' << m.edit.end << ")"
                              << " gain=" << m.predicted_gain
                              << " :: " << m.rationale << '\n';
                }
                return 0;
            }

            if (argc < 4) {
                usage();
                return 1;
            }
            const auto max_passes = parse_size(argv[3]);
            const auto report = inward.optimize(max_passes, arg_optional(argc, argv, 4));
            std::cout << "INITIAL ";
            print_metrics(report.initial);
            for (std::size_t i = 0; i < report.passes.size(); ++i) {
                const auto& pass = report.passes[i];
                std::cout << "PASS " << (i + 1) << ' '
                          << (pass.accepted ? "ACCEPT " : "REJECT ")
                          << pass.message;
                if (pass.mutation) {
                    std::cout << " :: " << pass.mutation->file.generic_string()
                              << " :: " << pass.mutation->rationale;
                }
                std::cout << '\n';
            }
            std::cout << "FINAL   ";
            print_metrics(report.final);
            return 0;
        }

        jarvisx::selfedit::SelfRefiningEditor engine(root);

        if (command == "scan") {
            for (const auto& file : engine.workspace().cpp_files()) {
                const auto state = engine.workspace().read(file);
                std::cout << file.generic_string()
                          << " bytes=" << state.content.size()
                          << " fingerprint=" << state.fingerprint << '\n';
            }
            return 0;
        }

        if (command == "replace") {
            if (argc < 6) {
                usage();
                return 1;
            }
            const auto result = engine.replace_once(argv[3], argv[4], argv[5], arg_optional(argc, argv, 6));
            std::cout << (result.ok ? "OK: " : "ERROR: ") << result.message << '\n';
            return result.ok ? 0 : 2;
        }

        if (command == "insert-after") {
            if (argc < 6) {
                usage();
                return 1;
            }
            const auto result = engine.insert_after(argv[3], argv[4], argv[5], arg_optional(argc, argv, 6));
            std::cout << (result.ok ? "OK: " : "ERROR: ") << result.message << '\n';
            return result.ok ? 0 : 2;
        }

        usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << '\n';
        return 3;
    }
}
