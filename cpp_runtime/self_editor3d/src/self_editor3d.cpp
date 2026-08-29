#include "jarvisx/self_editor3d.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace fs = std::filesystem;

namespace jarvisx {
namespace selfedit {
namespace {

bool is_contained_path(const fs::path& root, const fs::path& candidate) {
    auto root_it = root.begin();
    auto candidate_it = candidate.begin();
    for (; root_it != root.end(); ++root_it, ++candidate_it) {
        if (candidate_it == candidate.end() || *root_it != *candidate_it) return false;
    }
    return true;
}

std::size_t cube_side_for(std::size_t n) {
    if (n == 0) return 1;
    auto side = static_cast<std::size_t>(std::ceil(std::cbrt(static_cast<double>(n))));
    while (static_cast<long double>(side) * static_cast<long double>(side) *
           static_cast<long double>(side) < static_cast<long double>(n)) {
        ++side;
    }
    return std::max<std::size_t>(1, side);
}

std::size_t index3(std::size_t x, std::size_t y, std::size_t z, std::size_t side) {
    return (z * side + y) * side + x;
}

bool is_syntax(unsigned char c) noexcept {
    constexpr std::string_view syntax = "{}[]();,:<>+-=*/%&|!^~?.#\\\"'";
    return syntax.find(static_cast<char>(c)) != std::string_view::npos;
}

std::size_t count_lines(std::string_view s) noexcept {
    if (s.empty()) return 0;
    return 1 + static_cast<std::size_t>(std::count(s.begin(), s.end(), '\n'));
}

struct TextPenalties {
    std::size_t trailing{};
    std::size_t excessive_blank{};
    std::size_t tabs{};
};

TextPenalties inspect_text(std::string_view s) {
    TextPenalties p;
    p.tabs = static_cast<std::size_t>(std::count(s.begin(), s.end(), '\t'));

    std::size_t line_start = 0;
    std::size_t consecutive_blank = 0;
    while (line_start < s.size()) {
        auto line_end = s.find('\n', line_start);
        if (line_end == std::string_view::npos) line_end = s.size();

        std::size_t trimmed = line_end;
        while (trimmed > line_start && (s[trimmed - 1] == ' ' || s[trimmed - 1] == '\t')) {
            --trimmed;
        }
        p.trailing += line_end - trimmed;

        const bool blank = trimmed == line_start;
        if (blank) {
            ++consecutive_blank;
            if (consecutive_blank > 2) ++p.excessive_blank;
        } else {
            consecutive_blank = 0;
        }

        if (line_end == s.size()) break;
        line_start = line_end + 1;
    }
    return p;
}

} // namespace

std::uint64_t fingerprint(std::string_view data) noexcept {
    constexpr std::uint64_t offset = 1469598103934665603ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = offset;
    for (unsigned char c : data) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= prime;
    }
    return hash;
}

Workspace::Workspace(fs::path root) : root_(fs::weakly_canonical(std::move(root))) {
    if (!fs::exists(root_) || !fs::is_directory(root_)) {
        throw std::runtime_error("workspace root is not a directory: " + root_.string());
    }
}

fs::path Workspace::resolve(const fs::path& relative) const {
    if (relative.is_absolute()) {
        throw std::runtime_error("absolute edit paths are not allowed: " + relative.string());
    }
    const auto candidate = fs::weakly_canonical(root_ / relative);
    if (!is_contained_path(root_, candidate)) {
        throw std::runtime_error("path escapes workspace: " + relative.string());
    }
    return candidate;
}

std::vector<fs::path> Workspace::cpp_files() const {
    std::vector<fs::path> out;
    fs::recursive_directory_iterator it(root_), end;
    while (it != end) {
        const auto& entry = *it;
        if (entry.is_directory()) {
            const auto name = entry.path().filename().string();
            if (name == ".git" || name == "build" || name == "out" || name == ".cache") {
                it.disable_recursion_pending();
            }
        } else if (entry.is_regular_file()) {
            const auto ext = entry.path().extension().string();
            if (ext == ".cpp" || ext == ".cc" || ext == ".cxx" || ext == ".hpp" || ext == ".h") {
                out.push_back(fs::relative(entry.path(), root_));
            }
        }
        ++it;
    }
    std::sort(out.begin(), out.end());
    return out;
}

FileState Workspace::read(const fs::path& relative) const {
    const auto path = resolve(relative);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read file: " + path.string());
    std::ostringstream buffer;
    buffer << in.rdbuf();
    FileState state{relative, buffer.str(), 0};
    state.fingerprint = fingerprint(state.content);
    return state;
}

Transaction::Transaction(const Workspace& workspace) : workspace_(workspace) {}

void Transaction::add(Edit edit) {
    if (committed_) throw std::runtime_error("cannot add edits after commit");
    edits_.push_back(std::move(edit));
}

std::string Transaction::apply_one(std::string current, const Edit& edit) {
    switch (edit.kind) {
        case Edit::Kind::ReplaceOnce: {
            if (edit.needle.empty()) throw std::runtime_error("replace target must not be empty");
            const auto pos = current.find(edit.needle);
            if (pos == std::string::npos) throw std::runtime_error("replace target not found");
            if (current.find(edit.needle, pos + edit.needle.size()) != std::string::npos) {
                throw std::runtime_error("replace target is ambiguous; expected exactly one match");
            }
            current.replace(pos, edit.needle.size(), edit.replacement);
            return current;
        }
        case Edit::Kind::InsertAfter: {
            if (edit.needle.empty()) throw std::runtime_error("insert anchor must not be empty");
            const auto pos = current.find(edit.needle);
            if (pos == std::string::npos) throw std::runtime_error("insert anchor not found");
            if (current.find(edit.needle, pos + edit.needle.size()) != std::string::npos) {
                throw std::runtime_error("insert anchor is ambiguous; expected exactly one match");
            }
            current.insert(pos + edit.needle.size(), edit.replacement);
            return current;
        }
        case Edit::Kind::ReplaceRange: {
            if (edit.begin > edit.end || edit.end > current.size()) {
                throw std::runtime_error("invalid replacement range");
            }
            current.replace(edit.begin, edit.end - edit.begin, edit.replacement);
            return current;
        }
    }
    throw std::runtime_error("unknown edit kind");
}

void Transaction::atomic_write(const fs::path& path, std::string_view content) {
    const fs::path tmp = path.string() + ".jarvisx-selfedit.tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open temp file: " + tmp.string());
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!out) throw std::runtime_error("failed writing temp file: " + tmp.string());
    }

    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        std::error_code remove_ec;
        fs::remove(path, remove_ec);
        ec.clear();
        fs::rename(tmp, path, ec);
    }
    if (ec) {
        std::error_code cleanup_ec;
        fs::remove(tmp, cleanup_ec);
        throw std::runtime_error("atomic rename failed: " + ec.message());
    }
}

EditResult Transaction::apply() {
    if (committed_) return {false, "transaction already committed", {}};
    if (edits_.empty()) return {false, "no edits supplied", {}};

    try {
        std::unordered_map<std::string, std::string> staged;
        std::unordered_map<std::string, fs::path> relpaths;

        for (const auto& edit : edits_) {
            const auto key = edit.file.generic_string();
            auto staged_it = staged.find(key);
            if (staged_it == staged.end()) {
                auto state = workspace_.read(edit.file);
                before_.push_back(state);
                staged.emplace(key, state.content);
                relpaths.emplace(key, edit.file);
                staged_it = staged.find(key);
            }
            staged_it->second = apply_one(std::move(staged_it->second), edit);
        }

        std::unordered_map<std::string, std::string> originals;
        for (const auto& state : before_) {
            originals.emplace(state.path.generic_string(), state.content);
        }

        std::vector<fs::path> changed;
        for (const auto& pair : staged) {
            const auto& key = pair.first;
            const auto& content = pair.second;
            if (content == originals.at(key)) continue;
            atomic_write(workspace_.resolve(relpaths.at(key)), content);
            changed.push_back(relpaths.at(key));
        }
        if (changed.empty()) {
            before_.clear();
            return {false, "edit produced no source change", {}};
        }
        committed_ = true;
        return {true, "edits committed", std::move(changed)};
    } catch (const std::exception& e) {
        rollback();
        return {false, e.what(), {}};
    }
}

void Transaction::rollback() noexcept {
    for (const auto& state : before_) {
        try {
            atomic_write(workspace_.resolve(state.path), state.content);
        } catch (...) {
        }
    }
    committed_ = false;
}

Validator::Validator(fs::path workspace_root) : root_(fs::weakly_canonical(std::move(workspace_root))) {}

ValidationResult Validator::run(const std::string& command) const {
    if (command.empty()) return {true, 0, command};
#if defined(_WIN32)
    const std::string wrapped = "cd /d \"" + root_.string() + "\" && " + command;
#else
    const std::string wrapped = "cd \"" + root_.string() + "\" && " + command;
#endif
    const int rc = std::system(wrapped.c_str());
    return {rc == 0, rc, command};
}

SelfRefiningEditor::SelfRefiningEditor(fs::path root)
    : workspace_(std::move(root)), validator_(workspace_.root()) {}

EditResult SelfRefiningEditor::replace_once(const fs::path& file,
                                            std::string needle,
                                            std::string replacement,
                                            const std::optional<std::string>& validation_command) {
    Transaction tx(workspace_);
    tx.add(Edit{Edit::Kind::ReplaceOnce, file, std::move(needle), std::move(replacement), 0, 0});
    auto result = tx.apply();
    if (!result.ok) return result;

    if (validation_command) {
        const auto validation = validator_.run(*validation_command);
        if (!validation.ok) {
            tx.rollback();
            return {false, "validation failed; edit rolled back", {}};
        }
    }
    return result;
}

EditResult SelfRefiningEditor::insert_after(const fs::path& file,
                                            std::string anchor,
                                            std::string text,
                                            const std::optional<std::string>& validation_command) {
    Transaction tx(workspace_);
    tx.add(Edit{Edit::Kind::InsertAfter, file, std::move(anchor), std::move(text), 0, 0});
    auto result = tx.apply();
    if (!result.ok) return result;

    if (validation_command) {
        const auto validation = validator_.run(*validation_command);
        if (!validation.ok) {
            tx.rollback();
            return {false, "validation failed; edit rolled back", {}};
        }
    }
    return result;
}

Voxel3D Inward3DEncoder::voxelize(unsigned char byte, std::size_t source_index) noexcept {
    const float value = static_cast<float>(byte) / 255.0F;
    const float syntax = is_syntax(byte) ? 1.0F : 0.0F;
    const float whitespace = (byte == ' ' || byte == '\t' || byte == '\r') ? 1.0F : 0.0F;
    const float alpha = std::isalpha(byte) != 0 ? 1.0F : 0.0F;
    const float digit = std::isdigit(byte) != 0 ? 1.0F : 0.0F;
    const float newline = byte == '\n' ? 1.0F : 0.0F;
    const float hotspot = 0.34F * syntax + 0.22F * whitespace + 0.18F * newline
                        + 0.14F * digit + 0.12F * std::abs(value - 0.5F);
    return {value, syntax, whitespace, alpha, digit, newline, hotspot, source_index};
}

Voxel3D Inward3DEncoder::pool8(const std::vector<Voxel3D>& children) noexcept {
    if (children.empty()) return {};
    Voxel3D out{};
    float max_hotspot = -1.0F;
    for (const auto& v : children) {
        out.byte_value += v.byte_value;
        out.syntax_energy += v.syntax_energy;
        out.whitespace_energy += v.whitespace_energy;
        out.alpha_energy += v.alpha_energy;
        out.digit_energy += v.digit_energy;
        out.newline_energy += v.newline_energy;
        out.hotspot += v.hotspot;
        if (v.hotspot > max_hotspot && v.source_index != static_cast<std::size_t>(-1)) {
            max_hotspot = v.hotspot;
            out.source_index = v.source_index;
        }
    }
    const float inv = 1.0F / static_cast<float>(children.size());
    out.byte_value *= inv;
    out.syntax_energy *= inv;
    out.whitespace_energy *= inv;
    out.alpha_energy *= inv;
    out.digit_energy *= inv;
    out.newline_energy *= inv;
    out.hotspot *= inv;
    return out;
}

SourceField3D Inward3DEncoder::encode(const FileState& source) const {
    SourceField3D field;
    field.file = source.path;
    field.source_bytes = source.content.size();

    const std::size_t side = cube_side_for(source.content.size());
    InwardLevel3D base;
    base.side = side;
    base.voxels.resize(side * side * side);

    for (std::size_t i = 0; i < source.content.size(); ++i) {
        base.voxels[i] = voxelize(static_cast<unsigned char>(source.content[i]), i);
    }
    field.levels.push_back(std::move(base));

    while (field.levels.back().side > 1) {
        const auto& prev = field.levels.back();
        const std::size_t next_side = (prev.side + 1) / 2;
        InwardLevel3D next;
        next.side = next_side;
        next.voxels.resize(next_side * next_side * next_side);

        for (std::size_t z = 0; z < next_side; ++z) {
            for (std::size_t y = 0; y < next_side; ++y) {
                for (std::size_t x = 0; x < next_side; ++x) {
                    std::vector<Voxel3D> children;
                    children.reserve(8);
                    for (std::size_t dz = 0; dz < 2; ++dz) {
                        for (std::size_t dy = 0; dy < 2; ++dy) {
                            for (std::size_t dx = 0; dx < 2; ++dx) {
                                const auto px = x * 2 + dx;
                                const auto py = y * 2 + dy;
                                const auto pz = z * 2 + dz;
                                if (px < prev.side && py < prev.side && pz < prev.side) {
                                    children.push_back(prev.voxels[index3(px, py, pz, prev.side)]);
                                }
                            }
                        }
                    }
                    next.voxels[index3(x, y, z, next_side)] = pool8(children);
                }
            }
        }
        field.levels.push_back(std::move(next));
    }

    field.core = field.levels.back().voxels.front();
    return field;
}

Inward3DSelfOptimizer::Inward3DSelfOptimizer(fs::path root)
    : workspace_(std::move(root)), validator_(workspace_.root()) {}

std::vector<SourceField3D> Inward3DSelfOptimizer::encode_workspace() const {
    std::vector<SourceField3D> fields;
    const auto files = workspace_.cpp_files();
    fields.reserve(files.size());
    for (const auto& file : files) {
        fields.push_back(encoder_.encode(workspace_.read(file)));
    }
    return fields;
}

double Inward3DSelfOptimizer::objective(const WorkspaceMetrics3D& m) noexcept {
    if (m.bytes == 0) return 0.0;
    const double bytes = static_cast<double>(m.bytes);
    const double trailing = static_cast<double>(m.trailing_whitespace_bytes) / bytes;
    const double blanks = static_cast<double>(m.excessive_blank_lines) /
                          static_cast<double>(std::max<std::size_t>(1, m.lines));
    const double termination = static_cast<double>(m.missing_final_newline_files) /
                               static_cast<double>(std::max<std::size_t>(1, m.files));
    return 4.0 * trailing + 2.5 * blanks + 0.25 * termination + 0.15 * m.mean_hotspot;
}

WorkspaceMetrics3D Inward3DSelfOptimizer::analyze() const {
    WorkspaceMetrics3D m;
    double hotspot_sum = 0.0;
    std::size_t hotspot_count = 0;

    for (const auto& file : workspace_.cpp_files()) {
        const auto state = workspace_.read(file);
        const auto penalties = inspect_text(state.content);
        const auto field = encoder_.encode(state);

        ++m.files;
        m.bytes += state.content.size();
        m.lines += count_lines(state.content);
        m.trailing_whitespace_bytes += penalties.trailing;
        m.excessive_blank_lines += penalties.excessive_blank;
        m.tab_bytes += penalties.tabs;
        if (!state.content.empty() && state.content.back() != '\n') {
            ++m.missing_final_newline_files;
        }

        for (const auto& v : field.levels.front().voxels) {
            if (v.source_index != static_cast<std::size_t>(-1)) {
                hotspot_sum += v.hotspot;
                ++hotspot_count;
            }
        }
    }

    m.mean_hotspot = hotspot_count == 0 ? 0.0 : hotspot_sum / static_cast<double>(hotspot_count);
    m.objective = objective(m);
    return m;
}

std::vector<Mutation3D> Inward3DSelfOptimizer::propose_for_file(const FileState& state,
                                                                 const SourceField3D& field) {
    std::vector<Mutation3D> out;
    const auto& s = state.content;

    std::size_t line_start = 0;
    while (line_start <= s.size()) {
        auto line_end = s.find('\n', line_start);
        if (line_end == std::string::npos) line_end = s.size();
        std::size_t trimmed = line_end;
        while (trimmed > line_start && (s[trimmed - 1] == ' ' || s[trimmed - 1] == '\t')) --trimmed;
        if (trimmed < line_end) {
            Edit edit;
            edit.kind = Edit::Kind::ReplaceRange;
            edit.file = state.path;
            edit.begin = trimmed;
            edit.end = line_end;
            out.push_back({state.path, std::move(edit),
                           static_cast<double>(line_end - trimmed),
                           "remove trailing whitespace from a 3D hotspot line"});
        }
        if (line_end == s.size()) break;
        line_start = line_end + 1;
    }

    std::size_t blank_line_start = 0;
    std::size_t consecutive_blank = 0;
    while (blank_line_start < s.size()) {
        auto blank_line_end = s.find('\n', blank_line_start);
        const bool has_newline = blank_line_end != std::string::npos;
        if (!has_newline) blank_line_end = s.size();

        std::size_t trimmed = blank_line_end;
        while (trimmed > blank_line_start &&
               (s[trimmed - 1] == ' ' || s[trimmed - 1] == '\t' || s[trimmed - 1] == '\r')) {
            --trimmed;
        }
        const bool blank = trimmed == blank_line_start;
        if (blank) {
            ++consecutive_blank;
            if (consecutive_blank > 2) {
                const std::size_t erase_end = has_newline ? blank_line_end + 1 : blank_line_end;
                if (erase_end > blank_line_start) {
                    Edit edit;
                    edit.kind = Edit::Kind::ReplaceRange;
                    edit.file = state.path;
                    edit.begin = blank_line_start;
                    edit.end = erase_end;
                    out.push_back({state.path, std::move(edit), 1.0,
                                   "collapse excessive vertical whitespace in the inward source field"});
                }
                break;
            }
        } else {
            consecutive_blank = 0;
        }

        if (!has_newline) break;
        blank_line_start = blank_line_end + 1;
    }

    if (!s.empty() && s.back() != '\n') {
        Edit edit;
        edit.kind = Edit::Kind::ReplaceRange;
        edit.file = state.path;
        edit.begin = s.size();
        edit.end = s.size();
        edit.replacement = "\n";
        out.push_back({state.path, std::move(edit), 0.5,
                       "canonicalize source termination for deterministic self-rewrites"});
    }

    const double core_bias = 1.0 + static_cast<double>(field.core.hotspot);
    for (auto& mutation : out) mutation.predicted_gain *= core_bias;
    return out;
}

std::vector<Mutation3D> Inward3DSelfOptimizer::propose() const {
    std::vector<Mutation3D> all;
    for (const auto& file : workspace_.cpp_files()) {
        const auto state = workspace_.read(file);
        const auto field = encoder_.encode(state);
        auto local = propose_for_file(state, field);
        all.insert(all.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
    }

    std::sort(all.begin(), all.end(), [](const Mutation3D& a, const Mutation3D& b) {
        if (a.predicted_gain != b.predicted_gain) return a.predicted_gain > b.predicted_gain;
        if (a.file != b.file) return a.file.generic_string() < b.file.generic_string();
        return a.edit.begin < b.edit.begin;
    });
    return all;
}

OptimizationReport3D Inward3DSelfOptimizer::optimize(
    std::size_t max_passes,
    const std::optional<std::string>& validation_command) {

    OptimizationReport3D report;
    report.initial = analyze();

    for (std::size_t pass = 0; pass < max_passes; ++pass) {
        OptimizationPass3D result;
        result.before = analyze();

        const auto candidates = propose();
        if (candidates.empty()) {
            result.message = "fixed point reached: no admissible source mutations remain";
            result.after = result.before;
            report.passes.push_back(std::move(result));
            break;
        }

        result.mutation = candidates.front();
        result.changed = true;

        Transaction tx(workspace_);
        tx.add(result.mutation->edit);
        const auto applied = tx.apply();
        if (!applied.ok) {
            result.message = "mutation rejected during transaction: " + applied.message;
            result.after = analyze();
            report.passes.push_back(std::move(result));
            continue;
        }

        if (validation_command) {
            const auto validation = validator_.run(*validation_command);
            if (!validation.ok) {
                tx.rollback();
                result.message = "mutation failed compile/test validation and was rolled back";
                result.after = analyze();
                report.passes.push_back(std::move(result));
                continue;
            }
        }

        result.after = analyze();
        constexpr double epsilon = 1e-12;
        if (result.after.objective > result.before.objective + epsilon) {
            tx.rollback();
            result.after = analyze();
            result.message = "mutation increased the 3D objective and was rolled back";
            report.passes.push_back(std::move(result));
            continue;
        }

        result.accepted = true;
        result.message = "mutation accepted into the next inward source state";
        report.passes.push_back(std::move(result));
    }

    report.final = analyze();
    return report;
}

} // namespace selfedit
} // namespace jarvisx
