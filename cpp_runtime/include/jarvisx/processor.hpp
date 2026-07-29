#pragma once

#include "jarvisx/core.hpp"

namespace jarvisx {
struct Metrics {
    std::uint64_t cycles{};
    std::uint64_t committed{};
    std::uint64_t rejected{};
    float mse{std::numeric_limits<float>::infinity()};
    float coherence{};
    float energy{};
};

class Processor {
public:
    Processor(Genome genome, Packet packet)
        : genome_(std::move(genome)), packet_(std::move(packet)),
          omega_(genome_.feature_dim, 0.0F), rom_(synthesize_rom(genome_)) {
        genome_.clamp();
        if (!packet_.valid()) throw std::runtime_error("invalid packet");
    }

    void run() {
        while (!halted_ && metrics_.cycles < 32ULL + genome_.iterations * 16ULL) {
            if (pc_ >= rom_.size()) throw std::runtime_error("ROM PC overflow");
            execute(rom_[pc_]);
            ++metrics_.cycles;
        }
        if (!halted_) throw std::runtime_error("processor cycle limit reached");
    }

    const Metrics& metrics() const noexcept { return metrics_; }
    const SparseLattice& lattice() const noexcept { return lattice_; }

private:
    Genome genome_;
    Packet packet_;
    SparseLattice lattice_;
    std::vector<float> features_;
    std::vector<std::int8_t> latent_;
    std::vector<float> reconstruction_;
    std::vector<float> omega_;
    std::vector<Instruction> rom_;
    Metrics metrics_{};
    std::size_t pc_{};
    std::uint64_t iteration_{};
    bool halted_{};
    bool admissible_{};

    Vec3u coordinate(std::size_t index) const noexcept {
        const std::uint64_t a = mix64(index ^ (iteration_ << 32U) ^ genome_.seed);
        const std::uint64_t b = mix64(a);
        const std::uint64_t c = mix64(b);
        return {static_cast<std::uint32_t>(a & (kWorldEdge - 1U)),
                static_cast<std::uint32_t>(b & (kWorldEdge - 1U)),
                static_cast<std::uint32_t>(c & (kWorldEdge - 1U))};
    }

    void execute(const Instruction& instruction) {
        switch (instruction.op()) {
        case Op::Extract:
            features_ = extract_features(packet_, genome_.feature_dim);
            ++pc_;
            break;
        case Op::Encode:
            encode();
            ++pc_;
            break;
        case Op::Scatter:
            scatter();
            ++pc_;
            break;
        case Op::Diffuse:
            diffuse(std::max<std::uint16_t>(1, instruction.a()));
            ++pc_;
            break;
        case Op::Decode:
            decode();
            ++pc_;
            break;
        case Op::Learn:
            learn(instruction.a(), instruction.b());
            ++pc_;
            break;
        case Op::Project:
            project(instruction.a(), instruction.b(), instruction.c());
            ++pc_;
            break;
        case Op::Commit:
            admissible_ ? ++metrics_.committed : ++metrics_.rejected;
            ++pc_;
            break;
        case Op::Loop:
            ++iteration_;
            pc_ = iteration_ < instruction.a() ? instruction.b() : pc_ + 1;
            break;
        case Op::Halt:
            halted_ = true;
            break;
        }
    }

    void encode() {
        latent_.assign(genome_.latent_dim, 0);
        double energy = 0.0;
        for (std::size_t j = 0; j < latent_.size(); ++j) {
            float sum = 0.0F;
            for (std::size_t i = 0; i < features_.size(); ++i) {
                const float weight = 0.125F * signed_unit(
                    genome_.seed ^ (j * 0x9E3779B97F4A7C15ULL + i));
                sum += weight * features_[i];
            }
            latent_[j] = quantize_q3(std::tanh(sum));
            const double x = dequantize_q3(latent_[j]);
            energy += x * x;
        }
        metrics_.energy = static_cast<float>(energy / latent_.size());
    }

    void scatter() {
        for (std::size_t i = 0; i < latent_.size(); ++i) {
            Cell cell;
            cell.latent = latent_[i];
            cell.prediction = latent_[i];
            cell.modality = static_cast<std::uint8_t>(packet_.modality);
            cell.generation = static_cast<std::uint16_t>(
                genome_.generation & 0xFFFFU);
            cell.flags = 1U;
            lattice_.write(coordinate(i), cell);
        }
    }

    void diffuse(std::uint16_t radius) {
        std::vector<std::int8_t> evolved(latent_.size());
        for (std::size_t i = 0; i < latent_.size(); ++i) {
            const std::size_t left =
                (i + latent_.size() - radius % latent_.size()) % latent_.size();
            const std::size_t right = (i + radius) % latent_.size();
            const int value = latent_[left] + 2 * latent_[i] + latent_[right];
            evolved[i] = static_cast<std::int8_t>(
                std::max(-4, std::min(3, int(std::lround(value / 4.0)))));
            Cell cell = lattice_.read(coordinate(i));
            cell.prediction = evolved[i];
            cell.residual = static_cast<std::int8_t>(
                std::max(-4, std::min(3, int(cell.latent) - int(evolved[i]))));
            cell.flags |= 2U;
            lattice_.write(coordinate(i), cell);
        }
        latent_.swap(evolved);
    }

    void decode() {
        reconstruction_.assign(genome_.feature_dim, 0.0F);
        for (std::size_t i = 0; i < reconstruction_.size(); ++i) {
            float sum = 0.0F;
            for (std::size_t j = 0; j < latent_.size(); ++j) {
                const float weight = 0.125F * signed_unit(
                    (genome_.seed ^ 0xC0DEC0DEC0DEULL) ^
                    (i * 0xD1B54A32D192ED03ULL + j));
                sum += weight * dequantize_q3(latent_[j]);
            }
            reconstruction_[i] = clampf(std::tanh(sum) + omega_[i], -1.0F, 1.0F);
        }
    }

    void learn(std::uint16_t learning_units, std::uint16_t omega_units) {
        const float learning = learning_units / 100000.0F;
        const float omega_rate = omega_units / 100000.0F;
        double error2 = 0.0;
        for (std::size_t i = 0; i < features_.size(); ++i) {
            const float error = features_[i] - reconstruction_[i];
            error2 += error * error;
            omega_[i] = clampf(omega_[i] + (learning + omega_rate) * error,
                               -0.25F, 0.25F);
        }
        metrics_.mse = static_cast<float>(error2 / features_.size());
    }

    void project(std::uint16_t mse_units, std::uint16_t energy_units,
                 std::uint16_t coherence_units) {
        const float mse_limit = std::max(kEpsilon, mse_units / 10000.0F);
        const float energy_limit = std::max(kEpsilon, energy_units / 10000.0F);
        const float mse_score = 1.0F - clampf(metrics_.mse / mse_limit, 0.0F, 1.0F);
        const float energy_score =
            1.0F - clampf(metrics_.energy / energy_limit, 0.0F, 1.0F);
        metrics_.coherence = 0.7F * mse_score + 0.3F * energy_score;
        admissible_ = std::isfinite(metrics_.mse) &&
            metrics_.coherence >= coherence_units / 10000.0F;
        if (!admissible_) {
            for (std::int8_t& q : latent_) q = static_cast<std::int8_t>(q / 2);
        }
    }
};

struct Evaluation {
    Genome genome;
    Metrics metrics;
    std::size_t tiles{};
    std::uint64_t memory_bytes{};
    double elapsed_ms{};
    double fitness{-std::numeric_limits<double>::infinity()};
    bool valid{};
    std::string error;
};

double fitness(const Evaluation& e) noexcept {
    if (!e.valid || !std::isfinite(e.metrics.mse)) {
        return -std::numeric_limits<double>::infinity();
    }
    return 8.0 * clampf(e.metrics.coherence, 0.0F, 1.0F)
         - 2.0 * std::log1p(1000.0 * std::max(0.0F, e.metrics.mse))
         - 0.25 * e.metrics.energy
         - 0.15 * e.metrics.rejected
         - 0.0005 * e.elapsed_ms
         - 0.00000001 * static_cast<double>(e.memory_bytes);
}

Evaluation evaluate(Genome genome, const Packet& packet) {
    Evaluation result;
    result.genome = genome;
    result.genome.clamp();
    try {
        const auto start = std::chrono::steady_clock::now();
        Processor processor(result.genome, packet);
        processor.run();
        const auto stop = std::chrono::steady_clock::now();
        result.elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        result.metrics = processor.metrics();
        result.tiles = processor.lattice().tile_count();
        result.memory_bytes = processor.lattice().estimated_bytes();
        result.valid = result.metrics.committed + result.metrics.rejected >=
                       result.genome.iterations;
        result.fitness = fitness(result);
    } catch (const std::exception& error) {
        result.error = error.what();
    }
    return result;
}

} // namespace jarvisx
