#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jarvisx {
namespace selfedit {

struct FileState {
    std::filesystem::path path;
    std::string content;
    std::uint64_t fingerprint{};
};

struct Edit {
    enum class Kind { ReplaceOnce, InsertAfter, ReplaceRange };
    Kind kind{Kind::ReplaceOnce};
    std::filesystem::path file;
    std::string needle;
    std::string replacement;
    std::size_t begin{};
    std::size_t end{};
};

struct EditResult {
    bool ok{false};
    std::string message;
    std::vector<std::filesystem::path> changed_files;
};

class Workspace {
public:
    explicit Workspace(std::filesystem::path root);

    const std::filesystem::path& root() const noexcept { return root_; }
    std::filesystem::path resolve(const std::filesystem::path& relative) const;
    std::vector<std::filesystem::path> cpp_files() const;
    FileState read(const std::filesystem::path& relative) const;

private:
    std::filesystem::path root_;
};

class Transaction {
public:
    explicit Transaction(const Workspace& workspace);

    void add(Edit edit);
    EditResult apply();
    void rollback() noexcept;
    bool committed() const noexcept { return committed_; }

private:
    const Workspace& workspace_;
    std::vector<Edit> edits_;
    std::vector<FileState> before_;
    bool committed_{false};

    static std::string apply_one(std::string current, const Edit& edit);
    static void atomic_write(const std::filesystem::path& path, std::string_view content);
};

struct ValidationResult {
    bool ok{false};
    int exit_code{-1};
    std::string command;
};

class Validator {
public:
    explicit Validator(std::filesystem::path workspace_root);
    ValidationResult run(const std::string& command) const;

private:
    std::filesystem::path root_;
};

class SelfRefiningEditor {
public:
    explicit SelfRefiningEditor(std::filesystem::path root);

    EditResult replace_once(const std::filesystem::path& file,
                            std::string needle,
                            std::string replacement,
                            const std::optional<std::string>& validation_command = std::nullopt);

    EditResult insert_after(const std::filesystem::path& file,
                            std::string anchor,
                            std::string text,
                            const std::optional<std::string>& validation_command = std::nullopt);

    const Workspace& workspace() const noexcept { return workspace_; }

private:
    Workspace workspace_;
    Validator validator_;
};

struct Voxel3D {
    float byte_value{};
    float syntax_energy{};
    float whitespace_energy{};
    float alpha_energy{};
    float digit_energy{};
    float newline_energy{};
    float hotspot{};
    std::size_t source_index{static_cast<std::size_t>(-1)};
};

struct InwardLevel3D {
    std::size_t side{};
    std::vector<Voxel3D> voxels;
};

struct SourceField3D {
    std::filesystem::path file;
    std::size_t source_bytes{};
    std::vector<InwardLevel3D> levels;
    Voxel3D core;
};

struct WorkspaceMetrics3D {
    std::size_t files{};
    std::size_t bytes{};
    std::size_t lines{};
    std::size_t trailing_whitespace_bytes{};
    std::size_t excessive_blank_lines{};
    std::size_t tab_bytes{};
    std::size_t missing_final_newline_files{};
    double mean_hotspot{};
    double objective{};
};

struct Mutation3D {
    std::filesystem::path file;
    Edit edit;
    double predicted_gain{};
    std::string rationale;
};

struct OptimizationPass3D {
    bool changed{false};
    bool accepted{false};
    std::string message;
    std::optional<Mutation3D> mutation;
    WorkspaceMetrics3D before;
    WorkspaceMetrics3D after;
};

struct OptimizationReport3D {
    std::vector<OptimizationPass3D> passes;
    WorkspaceMetrics3D initial;
    WorkspaceMetrics3D final;
};

class Inward3DEncoder {
public:
    SourceField3D encode(const FileState& source) const;

private:
    static Voxel3D voxelize(unsigned char byte, std::size_t source_index) noexcept;
    static Voxel3D pool8(const std::vector<Voxel3D>& children) noexcept;
};

class Inward3DSelfOptimizer {
public:
    explicit Inward3DSelfOptimizer(std::filesystem::path root);

    WorkspaceMetrics3D analyze() const;
    std::vector<SourceField3D> encode_workspace() const;
    std::vector<Mutation3D> propose() const;

    OptimizationReport3D optimize(
        std::size_t max_passes,
        const std::optional<std::string>& validation_command = std::nullopt);

private:
    Workspace workspace_;
    Validator validator_;
    Inward3DEncoder encoder_;

    static double objective(const WorkspaceMetrics3D& metrics) noexcept;
    static std::vector<Mutation3D> propose_for_file(const FileState& state,
                                                    const SourceField3D& field);
};

std::uint64_t fingerprint(std::string_view data) noexcept;

} // namespace selfedit
} // namespace jarvisx
