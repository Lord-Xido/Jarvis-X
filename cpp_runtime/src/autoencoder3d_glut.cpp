#include "jarvisx/autoencoder3d.hpp"

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kPipelineExtent = 2.15F;
constexpr std::size_t kMaxStreams = 96U;
constexpr unsigned int kTimerMilliseconds = 16U;
constexpr float kTrainingInterval = 0.075F;

struct Vec3 {
    float x{};
    float y{};
    float z{};

    Vec3 operator+(const Vec3& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vec3 operator-(const Vec3& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vec3 operator*(float scalar) const noexcept {
        return {x * scalar, y * scalar, z * scalar};
    }
};

struct Color {
    float r{};
    float g{};
    float b{};
};

enum class OperationType : std::uint8_t {
    Encode,
    Decode,
    Residual
};

enum class ViewMode : std::uint8_t {
    Composite,
    Input,
    Latent,
    Reconstruction,
    Residual
};

Vec3 quadratic_bezier(const Vec3& start, const Vec3& control,
                      const Vec3& end, float t) noexcept {
    const float one_minus_t = 1.0F - t;
    return start * (one_minus_t * one_minus_t) +
           control * (2.0F * one_minus_t * t) + end * (t * t);
}

struct DataStream {
    Vec3 start;
    Vec3 control;
    Vec3 end;
    Vec3 position;
    Color color;
    OperationType operation{OperationType::Encode};
    float progress{};
    float speed{0.8F};

    void update(float delta_seconds) noexcept {
        progress = std::min(1.0F, progress + speed * delta_seconds);
        position = quadratic_bezier(start, control, end, progress);
    }

    bool alive() const noexcept { return progress < 1.0F; }

    void render() const {
        const float alpha = 0.35F + 0.65F * (1.0F - progress);
        glColor4f(color.r, color.g, color.b, alpha);
        glPushMatrix();
        glTranslatef(position.x, position.y, position.z);
        const double radius = static_cast<double>(0.035F + 0.025F * alpha);
        glutSolidSphere(radius, 8, 8);
        glPopMatrix();
    }
};

struct Options {
    std::size_t edge{16U};
    std::size_t channels{4U};
    float learning_rate{0.015F};
    float l2_penalty{1.0e-4F};
    float gradient_clip{1.0F};
    std::uint64_t seed{0x4A415256495358ULL};
    std::string pattern{"sphere"};
    std::filesystem::path load_model;
    bool start_paused{};
    bool quantized{};
};

std::uint64_t parse_u64(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0U;
    const std::uint64_t parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) {
        throw std::invalid_argument("invalid integer after " + flag);
    }
    return parsed;
}

float parse_float(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0U;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument("invalid floating-point value after " + flag);
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    auto next_value = [&](int& index, const std::string& flag) -> std::string {
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value after " + flag);
        }
        ++index;
        return argv[index];
    };

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--edge") {
            options.edge = static_cast<std::size_t>(
                parse_u64(next_value(index, argument), argument));
        } else if (argument == "--channels") {
            options.channels = static_cast<std::size_t>(
                parse_u64(next_value(index, argument), argument));
        } else if (argument == "--learning-rate") {
            options.learning_rate = parse_float(next_value(index, argument), argument);
        } else if (argument == "--l2") {
            options.l2_penalty = parse_float(next_value(index, argument), argument);
        } else if (argument == "--gradient-clip") {
            options.gradient_clip = parse_float(next_value(index, argument), argument);
        } else if (argument == "--seed") {
            options.seed = parse_u64(next_value(index, argument), argument);
        } else if (argument == "--pattern") {
            options.pattern = next_value(index, argument);
        } else if (argument == "--load-model") {
            options.load_model = next_value(index, argument);
        } else if (argument == "--paused") {
            options.start_paused = true;
        } else if (argument == "--quantized") {
            options.quantized = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: jarvisx-autoencoder3d-gl [options]\n"
                << "  --edge N              even input edge in [4,64]\n"
                << "  --channels N          latent channels in [1,32]\n"
                << "  --learning-rate X     SGD learning rate\n"
                << "  --l2 X                L2 regularization\n"
                << "  --gradient-clip X     elementwise gradient clip\n"
                << "  --seed N              deterministic model seed\n"
                << "  --pattern NAME        sphere|shell|checker|wave|noise\n"
                << "  --load-model PATH     load a JX3D checkpoint\n"
                << "  --quantized           render signed 3-bit latent inference\n"
                << "  --paused              start with online training paused\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}

jarvisx::Autoencoder3D make_model(const Options& options) {
    if (!options.load_model.empty()) {
        return jarvisx::Autoencoder3D::load(options.load_model);
    }
    return jarvisx::Autoencoder3D({
        options.edge,
        options.channels,
        options.learning_rate,
        options.l2_penalty,
        options.gradient_clip,
        options.seed
    });
}

class GeometricAutoencoderEngine {
public:
    explicit GeometricAutoencoderEngine(Options options)
        : options_(std::move(options)),
          model_(make_model(options_)),
          input_(jarvisx::make_volume(model_.config().input_edge,
                                      options_.pattern, model_.config().seed)),
          latent_(model_.encode(input_, options_.quantized)),
          reconstruction_(model_.decode(latent_)),
          rng_(model_.config().seed),
          unit_distribution_(-1.0F, 1.0F),
          training_(!options_.start_paused),
          quantized_(options_.quantized) {
        const auto found = std::find(patterns_.begin(), patterns_.end(),
                                     options_.pattern);
        if (found == patterns_.end()) {
            throw std::invalid_argument("unknown visualizer pattern: " +
                                        options_.pattern);
        }
        pattern_index_ = static_cast<std::size_t>(
            std::distance(patterns_.begin(), found));
        refresh_snapshots(0.0F);
        spawn_pipeline_burst();
    }

    void update(float delta_seconds) {
        const float dt = std::clamp(delta_seconds, 0.0F, 0.1F);
        if (auto_rotate_) {
            camera_angle_y_ += dt * 0.22F;
        }

        if (training_) {
            training_accumulator_ += dt;
            unsigned int steps_this_frame = 0U;
            while (training_accumulator_ >= kTrainingInterval &&
                   steps_this_frame < 2U) {
                const jarvisx::Autoencoder3DMetrics training_metrics =
                    model_.train_step(input_);
                refresh_snapshots(training_metrics.gradient_l2);
                training_accumulator_ -= kTrainingInterval;
                ++steps_this_frame;
                spawn_pipeline_burst();
            }
        }

        stream_spawn_accumulator_ += dt;
        if (stream_spawn_accumulator_ >= 0.11F) {
            spawn_stream(next_stream_is_encode_ ? OperationType::Encode
                                                : OperationType::Decode);
            next_stream_is_encode_ = !next_stream_is_encode_;
            stream_spawn_accumulator_ = 0.0F;
        }

        for (auto iterator = streams_.begin(); iterator != streams_.end();) {
            iterator->update(dt);
            if (!iterator->alive()) {
                iterator = streams_.erase(iterator);
            } else {
                ++iterator;
            }
        }

        pulse_ = std::max(0.0F, pulse_ - dt * 1.7F);
        status_time_ = std::max(0.0F, status_time_ - dt);
    }

    void render(int window_width, int window_height) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0F, 0.0F, camera_distance_);
        glRotatef(camera_angle_x_ * 180.0F / kPi, 1.0F, 0.0F, 0.0F);
        glRotatef(camera_angle_y_ * 180.0F / kPi, 0.0F, 1.0F, 0.0F);

        const GLfloat light_position[] = {0.0F, 5.0F, 7.0F, 1.0F};
        glLightfv(GL_LIGHT0, GL_POSITION, light_position);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(GL_FALSE);

        visible_voxels_ = 0U;
        switch (view_mode_) {
        case ViewMode::Composite:
            render_composite();
            break;
        case ViewMode::Input:
            visible_voxels_ += render_scalar_tensor(
                input_, {0.0F, 0.0F, 0.0F}, 4.5F, threshold_,
                {0.0F, 1.0F, 0.8F}, false);
            render_wire_cube({0.0F, 0.0F, 0.0F}, 2.25F,
                             {0.0F, 1.0F, 0.8F}, 0.45F);
            break;
        case ViewMode::Latent:
            visible_voxels_ += render_latent_tensor(
                {0.0F, 0.0F, 0.0F}, 4.5F, threshold_ * 0.65F);
            render_wire_cube({0.0F, 0.0F, 0.0F}, 2.25F,
                             {0.55F, 0.2F, 1.0F}, 0.45F);
            break;
        case ViewMode::Reconstruction:
            visible_voxels_ += render_scalar_tensor(
                reconstruction_, {0.0F, 0.0F, 0.0F}, 4.5F, threshold_,
                {1.0F, 0.35F, 0.8F}, false);
            render_wire_cube({0.0F, 0.0F, 0.0F}, 2.25F,
                             {1.0F, 0.35F, 0.8F}, 0.45F);
            break;
        case ViewMode::Residual:
            visible_voxels_ += render_residual_tensor(
                {0.0F, 0.0F, 0.0F}, 4.5F,
                std::max(0.08F, threshold_ * 0.45F));
            render_wire_cube({0.0F, 0.0F, 0.0F}, 2.25F,
                             {1.0F, 0.75F, 0.15F}, 0.45F);
            break;
        }

        for (const DataStream& stream : streams_) {
            stream.render();
        }
        render_base_ring();

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        render_hud(window_width, window_height);
    }

    void rotate_camera(int delta_x, int delta_y) noexcept {
        camera_angle_y_ += static_cast<float>(delta_x) * 0.005F;
        camera_angle_x_ += static_cast<float>(delta_y) * 0.005F;
        camera_angle_x_ = std::clamp(camera_angle_x_, -1.2F, 1.2F);
        auto_rotate_ = false;
    }

    void zoom(float delta) noexcept {
        camera_distance_ = std::clamp(camera_distance_ + delta, -18.0F, -6.5F);
    }

    void special_key(int key) noexcept {
        constexpr float increment = 0.08F;
        if (key == GLUT_KEY_LEFT) camera_angle_y_ -= increment;
        if (key == GLUT_KEY_RIGHT) camera_angle_y_ += increment;
        if (key == GLUT_KEY_UP) camera_angle_x_ -= increment;
        if (key == GLUT_KEY_DOWN) camera_angle_x_ += increment;
        camera_angle_x_ = std::clamp(camera_angle_x_, -1.2F, 1.2F);
        auto_rotate_ = false;
    }

    void keyboard(unsigned char key) {
        switch (key) {
        case '0':
            view_mode_ = ViewMode::Composite;
            break;
        case '1':
            view_mode_ = ViewMode::Input;
            break;
        case '2':
            view_mode_ = ViewMode::Latent;
            break;
        case '3':
            view_mode_ = ViewMode::Reconstruction;
            break;
        case '4':
            view_mode_ = ViewMode::Residual;
            break;
        case 'r':
        case 'R':
            auto_rotate_ = !auto_rotate_;
            break;
        case 't':
        case 'T':
        case ' ':
            training_ = !training_;
            set_status(training_ ? "online SGD resumed" : "online SGD paused");
            break;
        case 'q':
        case 'Q':
            quantized_ = !quantized_;
            refresh_snapshots(metrics_.gradient_l2);
            set_status(quantized_ ? "Q3 latent inference enabled"
                                  : "floating latent inference enabled");
            break;
        case 'p':
        case 'P':
            cycle_pattern();
            break;
        case 'n':
        case 'N':
            reset_model();
            break;
        case 's':
        case 'S':
            save_checkpoint();
            break;
        case 'l':
        case 'L':
            load_checkpoint();
            break;
        case '+':
        case '=':
            threshold_ = std::min(0.95F, threshold_ + 0.05F);
            break;
        case '-':
        case '_':
            threshold_ = std::max(0.0F, threshold_ - 0.05F);
            break;
        case 27:
            std::exit(0);
        default:
            break;
        }
    }

    std::size_t stream_count() const noexcept { return streams_.size(); }
    std::size_t visible_voxel_count() const noexcept { return visible_voxels_; }
    const jarvisx::Autoencoder3DMetrics& metrics() const noexcept { return metrics_; }

private:
    Options options_;
    jarvisx::Autoencoder3D model_;
    jarvisx::Tensor4D input_;
    jarvisx::Tensor4D latent_;
    jarvisx::Tensor4D reconstruction_;
    jarvisx::Autoencoder3DMetrics metrics_{};
    std::vector<DataStream> streams_;
    std::vector<std::string> patterns_{"sphere", "shell", "checker", "wave", "noise"};
    std::size_t pattern_index_{};
    std::mt19937_64 rng_;
    std::uniform_real_distribution<float> unit_distribution_;
    ViewMode view_mode_{ViewMode::Composite};
    bool auto_rotate_{true};
    bool training_{true};
    bool quantized_{};
    bool next_stream_is_encode_{true};
    float camera_angle_x_{0.28F};
    float camera_angle_y_{};
    float camera_distance_{-11.5F};
    float threshold_{0.30F};
    float training_accumulator_{};
    float stream_spawn_accumulator_{};
    float pulse_{};
    float status_time_{};
    float last_gradient_l2_{};
    std::size_t visible_voxels_{};
    std::string status_message_;
    const std::filesystem::path checkpoint_path_{
        ".jarvisx-autoencoder3d/visualizer-model.jx3d"};

    static constexpr Vec3 input_origin() noexcept { return {-3.25F, 0.0F, 0.0F}; }
    static constexpr Vec3 latent_origin() noexcept { return {0.0F, 0.0F, 0.0F}; }
    static constexpr Vec3 output_origin() noexcept { return {3.25F, 0.0F, 0.0F}; }

    void refresh_snapshots(float gradient_l2) {
        latent_ = model_.encode(input_, quantized_);
        reconstruction_ = model_.decode(latent_);
        metrics_ = jarvisx::measure_reconstruction(
            input_, latent_, reconstruction_, model_.steps());
        metrics_.gradient_l2 = gradient_l2;
        last_gradient_l2_ = gradient_l2;
        pulse_ = 1.0F;
    }

    void spawn_pipeline_burst() {
        for (int index = 0; index < 3; ++index) {
            spawn_stream(OperationType::Encode);
            spawn_stream(OperationType::Decode);
        }
        spawn_stream(OperationType::Residual);
    }

    void spawn_stream(OperationType operation) {
        if (streams_.size() >= kMaxStreams) return;

        const float jitter_y = unit_distribution_(rng_) * 0.85F;
        const float jitter_z = unit_distribution_(rng_) * 0.85F;
        DataStream stream;
        stream.operation = operation;
        stream.speed = 0.65F + 0.55F *
            (0.5F + 0.5F * unit_distribution_(rng_));

        if (operation == OperationType::Encode) {
            stream.start = input_origin() + Vec3{0.8F, jitter_y, jitter_z};
            stream.end = latent_origin() + Vec3{-0.65F, -0.25F * jitter_y,
                                                -0.25F * jitter_z};
            stream.control = (stream.start + stream.end) * 0.5F +
                             Vec3{0.0F, 0.65F, 0.0F};
            stream.color = {0.0F, 1.0F, 0.8F};
        } else if (operation == OperationType::Decode) {
            stream.start = latent_origin() + Vec3{0.65F, 0.25F * jitter_y,
                                                  0.25F * jitter_z};
            stream.end = output_origin() + Vec3{-0.8F, jitter_y, jitter_z};
            stream.control = (stream.start + stream.end) * 0.5F +
                             Vec3{0.0F, -0.65F, 0.0F};
            stream.color = {1.0F, 0.35F, 0.8F};
        } else {
            stream.start = output_origin() + Vec3{0.0F, jitter_y, jitter_z};
            stream.end = input_origin() + Vec3{0.0F, -jitter_y, -jitter_z};
            stream.control = {0.0F, 3.0F + 0.4F * jitter_y, 0.0F};
            stream.color = {1.0F, 0.75F, 0.15F};
            stream.speed *= 0.65F;
        }
        stream.position = stream.start;
        streams_.push_back(stream);
    }

    void cycle_pattern() {
        pattern_index_ = (pattern_index_ + 1U) % patterns_.size();
        options_.pattern = patterns_[pattern_index_];
        input_ = jarvisx::make_volume(model_.config().input_edge,
                                      options_.pattern, model_.config().seed);
        refresh_snapshots(last_gradient_l2_);
        set_status("input pattern: " + options_.pattern);
        spawn_pipeline_burst();
    }

    void reset_model() {
        const jarvisx::Autoencoder3DConfig config = model_.config();
        model_ = jarvisx::Autoencoder3D(config);
        refresh_snapshots(0.0F);
        set_status("model reinitialized from deterministic seed");
        spawn_pipeline_burst();
    }

    void save_checkpoint() {
        try {
            model_.save(checkpoint_path_);
            set_status("checkpoint saved: " + checkpoint_path_.string());
        } catch (const std::exception& error) {
            set_status(std::string("save failed: ") + error.what());
        }
    }

    void load_checkpoint() {
        try {
            model_ = jarvisx::Autoencoder3D::load(checkpoint_path_);
            input_ = jarvisx::make_volume(model_.config().input_edge,
                                          options_.pattern,
                                          model_.config().seed);
            refresh_snapshots(0.0F);
            set_status("checkpoint loaded: " + checkpoint_path_.string());
            spawn_pipeline_burst();
        } catch (const std::exception& error) {
            set_status(std::string("load failed: ") + error.what());
        }
    }

    void set_status(std::string message) {
        status_message_ = std::move(message);
        status_time_ = 4.0F;
        std::cout << status_message_ << '\n';
    }

    void render_composite() {
        const float dynamic_extent = kPipelineExtent + 0.08F * pulse_;
        visible_voxels_ += render_scalar_tensor(
            input_, input_origin(), dynamic_extent, threshold_,
            {0.0F, 1.0F, 0.8F}, false);
        visible_voxels_ += render_latent_tensor(
            latent_origin(), dynamic_extent * 0.92F, threshold_ * 0.65F);
        visible_voxels_ += render_scalar_tensor(
            reconstruction_, output_origin(), dynamic_extent, threshold_,
            {1.0F, 0.35F, 0.8F}, false);
        visible_voxels_ += render_residual_tensor(
            output_origin(), dynamic_extent,
            std::max(0.10F, threshold_ * 0.55F));

        render_wire_cube(input_origin(), dynamic_extent * 0.5F,
                         {0.0F, 1.0F, 0.8F}, 0.32F);
        render_wire_cube(latent_origin(), dynamic_extent * 0.46F,
                         {0.55F, 0.2F, 1.0F}, 0.36F);
        render_wire_cube(output_origin(), dynamic_extent * 0.5F,
                         {1.0F, 0.35F, 0.8F}, 0.32F);
        render_pipeline_edges();
    }

    std::size_t render_scalar_tensor(const jarvisx::Tensor4D& tensor,
                                     const Vec3& origin, float extent,
                                     float threshold, const Color& color,
                                     bool absolute_value) const {
        const jarvisx::TensorShape4D shape = tensor.shape();
        const float edge = static_cast<float>(shape.width);
        const float spacing = extent / edge;
        const float cube_size = spacing * 0.58F;
        std::size_t count = 0U;

        for (std::size_t z = 0; z < shape.depth; ++z) {
            for (std::size_t y = 0; y < shape.height; ++y) {
                for (std::size_t x = 0; x < shape.width; ++x) {
                    const float raw = tensor(0U, z, y, x);
                    const float activation = absolute_value ? std::fabs(raw) : raw;
                    if (activation <= threshold) continue;
                    const float intensity = std::clamp(
                        (activation - threshold) /
                            std::max(0.001F, 1.0F - threshold),
                        0.0F, 1.0F);
                    glColor4f(color.r, color.g, color.b,
                              0.16F + 0.70F * intensity);
                    glPushMatrix();
                    glTranslatef(
                        origin.x + (static_cast<float>(x) + 0.5F - edge * 0.5F) * spacing,
                        origin.y + (static_cast<float>(y) + 0.5F - edge * 0.5F) * spacing,
                        origin.z + (static_cast<float>(z) + 0.5F - edge * 0.5F) * spacing);
                    glutSolidCube(static_cast<double>(cube_size));
                    glPopMatrix();
                    ++count;
                }
            }
        }
        return count;
    }

    std::size_t render_latent_tensor(const Vec3& origin, float extent,
                                     float threshold) const {
        const jarvisx::TensorShape4D shape = latent_.shape();
        const float edge = static_cast<float>(shape.width);
        const float spacing = extent / edge;
        const float cube_size = spacing * 0.62F;
        const float inverse_channels = 1.0F /
            static_cast<float>(shape.channels);
        std::size_t count = 0U;

        for (std::size_t z = 0; z < shape.depth; ++z) {
            for (std::size_t y = 0; y < shape.height; ++y) {
                for (std::size_t x = 0; x < shape.width; ++x) {
                    float signed_sum = 0.0F;
                    float square_sum = 0.0F;
                    for (std::size_t channel = 0; channel < shape.channels;
                         ++channel) {
                        const float value = latent_(channel, z, y, x);
                        signed_sum += value;
                        square_sum += value * value;
                    }
                    const float energy = std::sqrt(square_sum * inverse_channels);
                    if (energy <= threshold) continue;
                    const float intensity = std::clamp(
                        (energy - threshold) /
                            std::max(0.001F, 1.0F - threshold),
                        0.0F, 1.0F);
                    const bool positive = signed_sum >= 0.0F;
                    const Color color = positive
                        ? Color{0.15F, 0.55F, 1.0F}
                        : Color{0.65F, 0.15F, 1.0F};
                    glColor4f(color.r, color.g, color.b,
                              0.22F + 0.74F * intensity);
                    glPushMatrix();
                    glTranslatef(
                        origin.x + (static_cast<float>(x) + 0.5F - edge * 0.5F) * spacing,
                        origin.y + (static_cast<float>(y) + 0.5F - edge * 0.5F) * spacing,
                        origin.z + (static_cast<float>(z) + 0.5F - edge * 0.5F) * spacing);
                    glutSolidCube(static_cast<double>(cube_size));
                    glPopMatrix();
                    ++count;
                }
            }
        }
        return count;
    }

    std::size_t render_residual_tensor(const Vec3& origin, float extent,
                                       float threshold) const {
        const jarvisx::TensorShape4D shape = input_.shape();
        const float edge = static_cast<float>(shape.width);
        const float spacing = extent / edge;
        const float cube_size = spacing * 0.31F;
        std::size_t count = 0U;

        for (std::size_t z = 0; z < shape.depth; ++z) {
            for (std::size_t y = 0; y < shape.height; ++y) {
                for (std::size_t x = 0; x < shape.width; ++x) {
                    const float error = std::fabs(
                        input_(0U, z, y, x) - reconstruction_(0U, z, y, x));
                    if (error <= threshold) continue;
                    const float intensity = std::clamp(error * 0.5F, 0.0F, 1.0F);
                    glColor4f(1.0F, 0.72F, 0.10F, 0.20F + 0.78F * intensity);
                    glPushMatrix();
                    glTranslatef(
                        origin.x + (static_cast<float>(x) + 0.5F - edge * 0.5F) * spacing,
                        origin.y + (static_cast<float>(y) + 0.5F - edge * 0.5F) * spacing,
                        origin.z + (static_cast<float>(z) + 0.5F - edge * 0.5F) * spacing);
                    glutSolidCube(static_cast<double>(cube_size));
                    glPopMatrix();
                    ++count;
                }
            }
        }
        return count;
    }

    static void render_wire_cube(const Vec3& origin, float half_extent,
                                 const Color& color, float alpha) {
        glColor4f(color.r, color.g, color.b, alpha);
        glLineWidth(1.35F);
        glPushMatrix();
        glTranslatef(origin.x, origin.y, origin.z);
        glutWireCube(static_cast<double>(2.0F * half_extent));
        glPopMatrix();
    }

    static void render_pipeline_edges() {
        glLineWidth(2.0F);
        glBegin(GL_LINES);
        glColor4f(0.0F, 1.0F, 0.8F, 0.55F);
        glVertex3f(-2.15F, 0.0F, 0.0F);
        glVertex3f(-1.05F, 0.0F, 0.0F);
        glColor4f(1.0F, 0.35F, 0.8F, 0.55F);
        glVertex3f(1.05F, 0.0F, 0.0F);
        glVertex3f(2.15F, 0.0F, 0.0F);
        glEnd();
    }

    static void render_base_ring() {
        glPushMatrix();
        glTranslatef(0.0F, -2.15F, 0.0F);
        glRotatef(90.0F, 1.0F, 0.0F, 0.0F);
        glColor4f(0.0F, 0.65F, 1.0F, 0.30F);
        glutWireTorus(0.06, 3.65, 18, 72);
        glColor4f(0.65F, 0.20F, 1.0F, 0.22F);
        glutWireTorus(0.04, 4.15, 14, 72);
        glPopMatrix();
    }

    static const char* view_name(ViewMode mode) noexcept {
        switch (mode) {
        case ViewMode::Composite: return "composite pipeline";
        case ViewMode::Input: return "input tensor X";
        case ViewMode::Latent: return "latent tensor Z";
        case ViewMode::Reconstruction: return "reconstruction X-hat";
        case ViewMode::Residual: return "residual |X-X-hat|";
        }
        return "unknown";
    }

    static void draw_text(float x, float y, const std::string& text,
                          void* font = GLUT_BITMAP_8_BY_13) {
        glRasterPos2f(x, y);
        for (const unsigned char character : text) {
            glutBitmapCharacter(font, static_cast<int>(character));
        }
    }

    void render_hud(int window_width, int window_height) const {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0.0, static_cast<double>(window_width), 0.0,
                static_cast<double>(window_height), -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);

        const float top = static_cast<float>(window_height) - 28.0F;
        glColor3f(0.25F, 0.90F, 1.0F);
        draw_text(18.0F, top,
                  "DR MOAGI 3D AUTO-ENCODING / DECODING ANN CORE",
                  GLUT_BITMAP_HELVETICA_18);

        std::ostringstream telemetry;
        telemetry << std::fixed << std::setprecision(6)
                  << "step=" << metrics_.step
                  << "  mse=" << metrics_.mse
                  << "  mae=" << metrics_.mae
                  << "  latent_energy=" << metrics_.latent_energy
                  << "  grad_l2=" << metrics_.gradient_l2;
        glColor3f(0.75F, 0.88F, 1.0F);
        draw_text(18.0F, top - 24.0F, telemetry.str());

        std::ostringstream state;
        state << "X=" << model_.config().input_edge << "^3"
              << "  Z=" << model_.config().latent_channels << "x"
              << model_.config().input_edge / 2U << "^3"
              << "  pattern=" << patterns_[pattern_index_]
              << "  view=" << view_name(view_mode_)
              << "  training=" << (training_ ? "ON" : "PAUSED")
              << "  Q3=" << (quantized_ ? "ON" : "OFF")
              << "  threshold=" << std::setprecision(2) << threshold_;
        glColor3f(0.65F, 1.0F, 0.78F);
        draw_text(18.0F, top - 44.0F, state.str());

        std::ostringstream render_state;
        render_state << "visible_voxels=" << visible_voxels_
                     << "  streams=" << streams_.size()
                     << "  encode -> latent -> decode -> residual -> SGD";
        glColor3f(1.0F, 0.72F, 0.90F);
        draw_text(18.0F, top - 64.0F, render_state.str());

        glColor3f(0.65F, 0.75F, 0.88F);
        draw_text(18.0F, 18.0F,
                  "Mouse/Arrows rotate | Wheel zoom | T/Space train | Q Q3 | P pattern | N reset | S/L checkpoint | 0-4 views | +/- threshold | R auto | Esc exit");

        if (status_time_ > 0.0F && !status_message_.empty()) {
            glColor3f(1.0F, 0.82F, 0.20F);
            draw_text(18.0F, 42.0F, status_message_, GLUT_BITMAP_HELVETICA_12);
        }

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }
};

std::unique_ptr<GeometricAutoencoderEngine> g_engine;
int g_last_time = 0;
int g_window_width = 1280;
int g_window_height = 720;
bool g_mouse_down = false;
int g_last_mouse_x = 0;
int g_last_mouse_y = 0;

void initialize_opengl() {
    glClearColor(0.0F, 0.0F, 0.025F, 1.0F);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_NORMALIZE);

    const GLfloat ambient[] = {0.18F, 0.18F, 0.25F, 1.0F};
    const GLfloat diffuse[] = {0.90F, 0.90F, 1.0F, 1.0F};
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
}

void display() {
    g_engine->render(g_window_width, g_window_height);
    glutSwapBuffers();
}

void reshape(int width, int height) {
    if (height <= 0) height = 1;
    g_window_width = width;
    g_window_height = height;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    constexpr double near_plane = 0.1;
    constexpr double far_plane = 100.0;
    constexpr double field_of_view_degrees = 60.0;
    const double aspect = static_cast<double>(width) / static_cast<double>(height);
    const double top = near_plane * std::tan(
        field_of_view_degrees * static_cast<double>(kPi) / 360.0);
    const double right = top * aspect;
    glFrustum(-right, right, -top, top, near_plane, far_plane);
    glMatrixMode(GL_MODELVIEW);
}

void timer(int) {
    const int current_time = glutGet(GLUT_ELAPSED_TIME);
    const int elapsed_milliseconds = std::max(0, current_time - g_last_time);
    g_last_time = current_time;
    g_engine->update(static_cast<float>(elapsed_milliseconds) / 1000.0F);
    glutPostRedisplay();
    glutTimerFunc(kTimerMilliseconds, timer, 0);
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        g_mouse_down = state == GLUT_DOWN;
        g_last_mouse_x = x;
        g_last_mouse_y = y;
    }
    if (state == GLUT_DOWN && button == 3) g_engine->zoom(0.55F);
    if (state == GLUT_DOWN && button == 4) g_engine->zoom(-0.55F);
}

void motion(int x, int y) {
    if (!g_mouse_down) return;
    g_engine->rotate_camera(x - g_last_mouse_x, y - g_last_mouse_y);
    g_last_mouse_x = x;
    g_last_mouse_y = y;
}

void keyboard(unsigned char key, int, int) {
    g_engine->keyboard(key);
}

void special(int key, int, int) {
    g_engine->special_key(key);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(g_window_width, g_window_height);
        glutCreateWindow("Dr Moagi 3D Auto-Encoding and Decoding ANN Core");

        initialize_opengl();
        g_engine = std::make_unique<GeometricAutoencoderEngine>(options);

        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutTimerFunc(kTimerMilliseconds, timer, 0);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutKeyboardFunc(keyboard);
        glutSpecialFunc(special);

        g_last_time = glutGet(GLUT_ELAPSED_TIME);
        std::cout
            << "=== Dr Moagi 3D Auto-Encoding / Decoding ANN Core ===\n"
            << "The visual field is driven by the real C++17 convolutional "
               "autoencoder state.\n"
            << "Press T to train, Q for signed 3-bit latent inference, P to "
               "change the fixture, and 0-4 to change views.\n";

        glutMainLoop();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Jarvis-X OpenGL autoencoder failure: " << error.what() << '\n';
        return 1;
    }
}
