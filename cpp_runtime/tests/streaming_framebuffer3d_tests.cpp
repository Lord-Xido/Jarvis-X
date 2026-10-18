#include "streaming_framebuffer3d.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    using namespace dm3d_frame;
    require(index_wrap(-1,0,0) == at(7,0,0), "regression condition failed");
    require(index_wrap(0,8,0) == at(0,0,0), "regression condition failed");
    require(index_wrap(0,0,-1) == at(0,0,7), "regression condition failed");
    {
        Config cfg;
        cfg.width = 128; cfg.height = 96;
        Engine engine(cfg);
        const Report first = engine.process_frame(0);
        const Report second = engine.process_frame(1);
        require(first.pixels == 12288 && second.pixels == 12288, "regression condition failed");
        require(first.accepted_tiles > 0, "regression condition failed");
        require(first.authoritative_mse <= first.baseline_mse + 1e-9, "regression condition failed");
        require(second.authoritative_mse <= second.baseline_mse + 1e-9, "regression condition failed");
        require(engine.persistent_bytes() == 4 * latent_scalars, "regression condition failed");
    }
    {
        Config cfg;
        cfg.width = 7; cfg.height = 5; cfg.gamma=1.0; cfg.diffusion=0.0;
        cfg.iterations = 1;
        Engine engine(cfg);
        std::string original(7 * 5 * 3, '\0');
        for (std::size_t i = 0; i < original.size(); i += 3) {
            original[i] = static_cast<char>(12);
            original[i+1] = static_cast<char>(67);
            original[i+2] = static_cast<char>(201);
        }
        std::istringstream input(original);
        std::ostringstream output;
        const Report report = engine.process_frame(0,&input,&output);
        require(output.str() == original, "regression condition failed"); // exact uniform RGB24 round trip
        require(report.authoritative_mse < 1e-12, "regression condition failed");
        require(report.pixels == 35, "regression condition failed");
    }
    {
        Config cfg;
        cfg.width=17;cfg.height=9;
        Engine engine(cfg);
        std::istringstream short_input(std::string("short"));
        bool threw = false;
        try { (void)engine.process_frame(0,&short_input); }
        catch (const std::runtime_error&) { threw=true; }
        require(threw, "regression condition failed");
    }
    {
        Config invalid;
        invalid.gamma=0.01;invalid.diffusion=0.01;
        bool threw = false;
        try { Engine unused(invalid); (void)unused; }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "regression condition failed"); // Banach sufficient condition not met
        invalid.gamma=0.6;invalid.diffusion=0.01;invalid.width=0;
        threw=false;
        try { Engine unused(invalid); (void)unused; }
        catch (const std::invalid_argument&) { threw=true; }
        require(threw, "regression condition failed");
    }
}
