#define ANN_OS_NO_MAIN
#include "ann_os.cpp"
#include <stdexcept>
#include <string>
#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(std::string("CHECK failed: ") + #expr); } while (false)

int main() {
    // Full range byte-to-3D-to-byte exact reconstruction including binary bytes.
    std::string bytes;
    for (int i = 0; i < 256; ++i) bytes.push_back(static_cast<char>(i));
    auto cube = CodeCube::encode(bytes);
    CHECK(cube.decode() == bytes);
    CHECK(cube.index(0, 0, 0) == 0);
    CHECK(cube.index(1, 0, 0) == 1);
    CHECK(cube.index(0, 1, 0) == cube.D);
    CHECK(CodeCube::encode("").decode().empty());
    try { cube.index(cube.D, 0, 0); CHECK(false); }
    catch (const std::out_of_range&) {}
    CubeFS filesystem;
    filesystem.write("/test", cube);
    CHECK(filesystem.read("/test").decode() == bytes);
    try { (void)filesystem.read("/missing"); CHECK(false); }
    catch (const std::out_of_range&) {}

    // Priority must be distance to nonzero goal, NOT norm to origin.
    ANNScheduler scheduler({10, 0, 0});
    scheduler.push({0, 0, 0}); // lower norm but farther from goal
    scheduler.push({9, 0, 0});
    auto first = scheduler.pop();
    CHECK(first && first->id == 1);
    CHECK(scheduler.pop()->id == 0);
    CHECK(!scheduler.pop());
    try { scheduler.push({1, 2}); CHECK(false); }
    catch (const std::invalid_argument&) {}
    CHECK(app_template("hello").find("Hello from ANN OS") != std::string::npos);
    CHECK(app_template("xor").find("ANN net") != std::string::npos);
    try { (void)app_template("make me a compiler"); CHECK(false); }
    catch (const std::invalid_argument&) {}

    // Real ANN backprop gradient audit and transactional held-out optimization.
    ANN net({8, 12, 3, 12, 8},
        {Activation::Tanh, Activation::Tanh, Activation::Tanh, Activation::Tanh}, 42);
    const double grad_error = net.gradient_check(
        {0.0, 0.2, -0.1, 0.5, 0.6, -0.3, 0.8, 0.1},
        {0.3, 0.2, 0.1, -0.1, 0.4, 0.5, -0.2, -0.3});
    CHECK(grad_error < 1e-6);

    ANN_OS os("ann_os.cpp");
    std::ostringstream output;
    os.boot(output);
    const double before = os.validation_loss();
    const bool accepted = os.auto_optimize(output);
    const double after = os.validation_loss();
    CHECK(std::isfinite(before) && std::isfinite(after));
    CHECK(after <= before + 1e-12); // rejection is real rollback
    CHECK(accepted ? after < before : std::abs(after - before) <= 1e-12);
    os.dispatch("schedule 0.1 0.2 0.3", output);
    os.dispatch("run", output);
    os.dispatch("image", output);
    os.dispatch("doc", output);
    os.dispatch("generate app hello", output);
    os.dispatch("replicate", output);
    CHECK(read_text("app/replica.cpp") == read_text("ann_os.cpp"));
    CHECK(fs::exists("os_arch.svg"));
    CHECK(fs::exists("os_manual.md"));
    os.dispatch("self-evolve", output);
    os.dispatch("audio", output);
    CHECK(fs::file_size("app/tone.wav") == 44 + 8000);
    os.dispatch("save", output);
    CHECK(fs::exists("app/autoencoder.ann17"));
    const auto restored = ANN::load("app/autoencoder.ann17");
    CHECK(restored.layers.size() == 4);
    std::cout << "PASS: exact cube round-trip, scheduler, gradient check, "
              << "transactional training, shell outputs, WAV, checkpoint\n";
}
