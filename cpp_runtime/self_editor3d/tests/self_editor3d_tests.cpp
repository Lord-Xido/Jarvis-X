#include "jarvisx/self_editor3d.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace jarvisx::selfedit;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct TempWorkspace {
    fs::path root;

    TempWorkspace() {
        const auto tick = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        root = fs::temp_directory_path() / ("jarvisx-selfedit-" + std::to_string(tick));
        fs::create_directories(root / "src");
    }

    ~TempWorkspace() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    void write(const fs::path& relative, const std::string& text) const {
        fs::create_directories((root / relative).parent_path());
        std::ofstream out(root / relative, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create fixture");
        out << text;
    }
};

void test_transaction_guards() {
    TempWorkspace fixture;
    fixture.write("src/a.cpp", "int x = 1;\nint x2 = 1;\n");
    Workspace workspace(fixture.root);

    bool escaped = false;
    try {
        (void)workspace.resolve("../outside.cpp");
    } catch (const std::exception&) {
        escaped = true;
    }
    require(escaped, "workspace escape must be rejected");

    Transaction ambiguous(workspace);
    ambiguous.add(Edit{Edit::Kind::ReplaceOnce, "src/a.cpp", "= 1", "= 2", 0, 0});
    require(!ambiguous.apply().ok, "ambiguous replace must be rejected");

    Transaction noop(workspace);
    noop.add(Edit{Edit::Kind::ReplaceRange, "src/a.cpp", {}, {}, 0, 0});
    require(!noop.apply().ok, "no-op edit must not commit");
}

void test_inward_fold_and_optimization() {
    TempWorkspace fixture;
    fixture.write("src/dirty.cpp",
                  "#include <iostream>   \n"
                  "\n"
                  "\n"
                  "\n"
                  "int main() {\t\n"
                  "    std::cout << 42;    \n"
                  "}");

    Workspace workspace(fixture.root);
    Inward3DEncoder encoder;
    const auto field = encoder.encode(workspace.read("src/dirty.cpp"));
    require(!field.levels.empty(), "3D field must contain at least one level");
    require(field.levels.back().side == 1, "3D fold must terminate at a 1^3 core");
    require(field.source_bytes > 0, "source bytes must be represented");

    Inward3DSelfOptimizer optimizer(fixture.root);
    const auto before = optimizer.analyze();
    require(before.trailing_whitespace_bytes > 0, "fixture must have trailing whitespace");
    require(before.excessive_blank_lines > 0, "fixture must have excessive blank lines");
    require(before.missing_final_newline_files == 1, "fixture must lack a final newline");

    const auto report = optimizer.optimize(32);
    require(report.final.objective <= report.initial.objective, "optimization objective must not increase");
    require(report.final.trailing_whitespace_bytes == 0, "optimizer must remove trailing whitespace");
    require(report.final.excessive_blank_lines == 0, "optimizer must collapse excessive blank lines");
    require(report.final.missing_final_newline_files == 0, "optimizer must canonicalize source termination");

    const auto proposals = optimizer.propose();
    require(proposals.empty(), "clean fixture should converge to a fixed point");
}

} // namespace

int main() {
    try {
        test_transaction_guards();
        test_inward_fold_and_optimization();
        std::cout << "self_editor3d_regressions: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "self_editor3d_regressions: FAIL: " << e.what() << '\n';
        return 1;
    }
}
