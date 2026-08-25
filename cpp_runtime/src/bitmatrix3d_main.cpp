#include "jarvisx/bitmatrix3d.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::size_t edge{8U};
    std::size_t channels{4U};
    std::size_t epochs{160U};
    float learning_rate{0.01F};
    float threshold{0.25F};
    std::uint64_t seed{42U};
    std::string pattern{"sphere"};
    fs::path output_dir{".jarvisx-bitmatrix3d"};
    bool quiet{false};
};

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string(name) + " requires a value");
            return argv[++i];
        };
        if (arg == "--edge") options.edge = static_cast<std::size_t>(std::stoull(require_value("--edge")));
        else if (arg == "--channels") options.channels = static_cast<std::size_t>(std::stoull(require_value("--channels")));
        else if (arg == "--epochs") options.epochs = static_cast<std::size_t>(std::stoull(require_value("--epochs")));
        else if (arg == "--learning-rate") options.learning_rate = std::stof(require_value("--learning-rate"));
        else if (arg == "--threshold") options.threshold = std::stof(require_value("--threshold"));
        else if (arg == "--seed") options.seed = static_cast<std::uint64_t>(std::stoull(require_value("--seed")));
        else if (arg == "--pattern") options.pattern = require_value("--pattern");
        else if (arg == "--output-dir") options.output_dir = require_value("--output-dir");
        else if (arg == "--quiet") options.quiet = true;
        else if (arg == "--help") {
            std::cout << "jarvisx-bitmatrix3d [--edge N] [--channels N] [--epochs N] "
                         "[--learning-rate F] [--threshold F] [--seed N] "
                         "[--pattern sphere|shell|checker|wave|noise] [--output-dir PATH] [--quiet]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.epochs == 0U || options.epochs > 100000U)
        throw std::invalid_argument("epochs must be in [1, 100000]");
    return options;
}

void write_words(std::ostream& out, const char* name, const std::vector<std::uint64_t>& words) {
    out << name << "_count=" << words.size() << '\n';
    for (std::size_t i = 0; i < words.size(); ++i) {
        out << name << '[' << i << "]=0x" << std::hex << std::setw(16) << std::setfill('0')
            << words[i] << std::dec << std::setfill(' ') << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        jarvisx::BitMatrix3DConfig config;
        config.input_edge = options.edge;
        config.latent_channels = options.channels;
        config.learning_rate = options.learning_rate;
        config.ternary_threshold = options.threshold;
        config.seed = options.seed;
        jarvisx::BitMatrix3DEngine engine(config);
        const auto input = jarvisx::make_volume(options.edge, options.pattern, options.seed);

        fs::create_directories(options.output_dir);
        std::ofstream metrics(options.output_dir / "metrics.csv", std::ios::trunc);
        if (!metrics) throw std::runtime_error("cannot open metrics.csv");
        metrics << "step,mse,mae,max_abs,fixed_point_residual,encoder_density,decoder_density,"
                   "latent_density,gradient_l2,shadow_weight_bytes,packed_weight_bytes,"
                   "packed_latent_bytes,self_description_valid,fixed_point_converged\n";

        for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
            const auto m = engine.train_step(input);
            metrics << m.step << ',' << m.mse << ',' << m.mae << ',' << m.max_abs_error << ','
                    << m.fixed_point_residual << ',' << m.encoder_weight_density << ','
                    << m.decoder_weight_density << ',' << m.latent_density << ',' << m.gradient_l2 << ','
                    << m.shadow_weight_bytes << ',' << m.packed_weight_bytes << ',' << m.packed_latent_bytes << ','
                    << (m.self_description_valid ? 1 : 0) << ',' << (m.fixed_point_converged ? 1 : 0) << '\n';
            if (!options.quiet && (epoch == 0U || epoch + 1U == options.epochs || (epoch + 1U) % 25U == 0U)) {
                std::cout << "step=" << m.step << " mse=" << m.mse
                          << " fp_residual=" << m.fixed_point_residual
                          << " latent_density=" << m.latent_density
                          << " packed_weights=" << m.packed_weight_bytes << "B\n";
            }
        }

        const auto final_forward = engine.forward(input);
        const auto final_metrics = engine.evaluate_inward(input);
        std::ofstream snapshot(options.output_dir / "bitplanes.txt", std::ios::trunc);
        if (!snapshot) throw std::runtime_error("cannot open bitplanes.txt");
        snapshot << "DMVX_BITMATRIX3D_V1\n";
        snapshot << "steps=" << engine.steps() << '\n';
        snapshot << "encoder_scale=" << final_forward.encoder_weights.scale << '\n';
        snapshot << "decoder_scale=" << final_forward.decoder_weights.scale << '\n';
        snapshot << "latent_elements=" << final_forward.latent_packed.size() << '\n';
        snapshot << "latent_nonzero=" << final_forward.latent_packed.nonzero_count() << '\n';
        write_words(snapshot, "encoder_sign", final_forward.encoder_weights.packed.sign_words());
        write_words(snapshot, "encoder_mask", final_forward.encoder_weights.packed.nonzero_words());
        write_words(snapshot, "decoder_sign", final_forward.decoder_weights.packed.sign_words());
        write_words(snapshot, "decoder_mask", final_forward.decoder_weights.packed.nonzero_words());
        write_words(snapshot, "latent_sign", final_forward.latent_packed.sign_words());
        write_words(snapshot, "latent_mask", final_forward.latent_packed.nonzero_words());

        std::ofstream report(options.output_dir / "report.txt", std::ios::trunc);
        if (!report) throw std::runtime_error("cannot open report.txt");
        report << "DM-vOmegaXi+ Bit-Matrix Engine reference report\n"
               << "mse=" << final_metrics.mse << '\n'
               << "fixed_point_residual=" << final_metrics.fixed_point_residual << '\n'
               << "self_description_valid=" << (final_metrics.self_description_valid ? "true" : "false") << '\n'
               << "shadow_weight_bytes=" << final_metrics.shadow_weight_bytes << '\n'
               << "packed_weight_bytes=" << final_metrics.packed_weight_bytes << '\n'
               << "packed_latent_bytes=" << final_metrics.packed_latent_bytes << '\n'
               << "physical_weight_compression="
               << (final_metrics.packed_weight_bytes == 0U ? 0.0 :
                   static_cast<double>(final_metrics.shadow_weight_bytes) /
                   static_cast<double>(final_metrics.packed_weight_bytes)) << "x\n"
               << "note=reference scalar mixed-precision kernel; no AVX/CUDA speedup claim\n";

        if (!final_metrics.self_description_valid)
            throw std::runtime_error("packed bitplanes failed self-description verification");
        if (!options.quiet)
            std::cout << "artifacts=" << fs::absolute(options.output_dir).string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bit-matrix engine failure: " << error.what() << '\n';
        return 1;
    }
}
