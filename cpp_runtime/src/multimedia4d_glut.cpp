#if defined(__APPLE__)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "jarvisx/multimedia4d.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr std::size_t kMaxStreams = 96U;

struct Vec3 {
    float x{};
    float y{};
    float z{};
    Vec3 operator+(const Vec3& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }
    Vec3 operator*(float scale) const noexcept {
        return {x * scale, y * scale, z * scale};
    }
};

struct Color { float r{}, g{}, b{}; };

Color media_color(jarvisx::MediaType media) noexcept {
    switch (media) {
    case jarvisx::MediaType::Visual: return {0.0F, 1.0F, 0.8F};
    case jarvisx::MediaType::Audio: return {1.0F, 0.35F, 0.8F};
    case jarvisx::MediaType::Text: return {1.0F, 0.82F, 0.18F};
    case jarvisx::MediaType::Generic: return {0.35F, 0.55F, 1.0F};
    }
    return {1.0F, 1.0F, 1.0F};
}

enum class StreamType : std::uint8_t { Encode, Decode, Feedback };

struct DataStream {
    Vec3 start;
    Vec3 control;
    Vec3 end;
    Vec3 position;
    Color color;
    float progress{};
    float speed{1.0F};
    StreamType type{StreamType::Encode};

    bool update(float dt) noexcept {
        progress = std::min(1.0F, progress + speed * dt);
        const float u = 1.0F - progress;
        position = start * (u * u) + control * (2.0F * u * progress) +
                   end * (progress * progress);
        return progress < 1.0F;
    }

    void render() const {
        glColor4f(color.r, color.g, color.b, 1.0F - 0.45F * progress);
        glPushMatrix();
        glTranslatef(position.x, position.y, position.z);
        const float radius = type == StreamType::Feedback ? 0.055F : 0.045F;
        glutSolidSphere(static_cast<double>(radius), 8, 8);
        glPopMatrix();
    }
};

class FrameBudgetController {
public:
    void record(float milliseconds) {
        history_.push_back(milliseconds);
        if (history_.size() > 60U) history_.pop_front();
        double sum = 0.0;
        for (const float sample : history_) sum += sample;
        average_ms_ = history_.empty() ? 0.0F :
            static_cast<float>(sum / static_cast<double>(history_.size()));
        if (history_.size() == 60U) adapt();
    }

    float average_ms() const noexcept { return average_ms_; }
    float threshold_bias() const noexcept { return threshold_bias_; }
    std::size_t temporal_layers(std::size_t available) const noexcept {
        return std::min(available, temporal_layers_);
    }
    float training_interval() const noexcept { return training_interval_; }

private:
    std::deque<float> history_;
    float average_ms_{16.67F};
    float threshold_bias_{};
    std::size_t temporal_layers_{8U};
    float training_interval_{0.075F};

    void adapt() noexcept {
        if (average_ms_ > 22.0F) {
            threshold_bias_ = std::min(0.35F, threshold_bias_ + 0.025F);
            temporal_layers_ = std::max<std::size_t>(1U, temporal_layers_ - 1U);
            training_interval_ = std::min(0.25F, training_interval_ * 1.08F);
        } else if (average_ms_ < 13.0F) {
            threshold_bias_ = std::max(0.0F, threshold_bias_ - 0.015F);
            temporal_layers_ = std::min<std::size_t>(16U, temporal_layers_ + 1U);
            training_interval_ = std::max(0.035F, training_interval_ * 0.96F);
        }
    }
};

struct Options {
    std::size_t edge{16U};
    std::size_t channels{4U};
    std::size_t temporal_depth{8U};
    std::uint16_t proposal_steps{2U};
    float learning_rate{0.015F};
    float temporal_decay{0.72F};
    std::uint64_t seed{0x4A415256495358ULL};
    bool quantized{};
    bool paused{};
};

std::uint64_t parse_u64(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0U;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) {
        throw std::invalid_argument("invalid integer after " + flag);
    }
    return parsed;
}

float parse_float(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0U;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument("invalid float after " + flag);
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    auto value = [&](int& index, const std::string& flag) {
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value after " + flag);
        }
        return std::string(argv[++index]);
    };
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--edge") {
            options.edge = static_cast<std::size_t>(
                parse_u64(value(index, argument), argument));
        } else if (argument == "--channels") {
            options.channels = static_cast<std::size_t>(
                parse_u64(value(index, argument), argument));
        } else if (argument == "--temporal-depth") {
            options.temporal_depth = static_cast<std::size_t>(
                parse_u64(value(index, argument), argument));
        } else if (argument == "--proposal-steps") {
            options.proposal_steps = static_cast<std::uint16_t>(
                parse_u64(value(index, argument), argument));
        } else if (argument == "--learning-rate") {
            options.learning_rate = parse_float(value(index, argument), argument);
        } else if (argument == "--temporal-decay") {
            options.temporal_decay = parse_float(value(index, argument), argument);
        } else if (argument == "--seed") {
            options.seed = parse_u64(value(index, argument), argument);
        } else if (argument == "--quantized") {
            options.quantized = true;
        } else if (argument == "--paused") {
            options.paused = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: jarvisx-multimedia4d-gl [options]\n"
                << "  --edge N --channels N --temporal-depth N\n"
                << "  --proposal-steps N --learning-rate X\n"
                << "  --temporal-decay X --seed N --quantized --paused\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}

jarvisx::Multimedia4DConfig make_config(const Options& options) {
    jarvisx::Multimedia4DConfig config;
    config.model.input_edge = options.edge;
    config.model.latent_channels = options.channels;
    config.model.learning_rate = options.learning_rate;
    config.model.seed = options.seed;
    config.temporal_depth = options.temporal_depth;
    config.temporal_decay = options.temporal_decay;
    config.proposal_steps = options.proposal_steps;
    config.quantized_inference = options.quantized;
    return config;
}

class MultimediaVisualizer {
public:
    explicit MultimediaVisualizer(Options options)
        : options_(std::move(options)), engine_(make_config(options_)),
          selected_(jarvisx::MediaType::Visual), training_(!options_.paused),
          rng_(options_.seed), unit_(-1.0F, 1.0F) {
        spawn_burst();
    }

    void update(float dt) {
        const auto start = std::chrono::steady_clock::now();
        dt = std::clamp(dt, 0.0F, 0.1F);
        if (auto_rotate_) camera_y_ += dt * 0.22F;

        if (training_) {
            training_accumulator_ += dt;
            if (training_accumulator_ >= budget_.training_interval()) {
                const auto metrics = engine_.step();
                if (follow_optimizer_) selected_ = metrics.selected;
                training_accumulator_ = 0.0F;
                spawn_burst();
            }
        }

        stream_accumulator_ += dt;
        if (stream_accumulator_ >= 0.12F) {
            spawn_stream(next_encode_ ? StreamType::Encode : StreamType::Decode);
            next_encode_ = !next_encode_;
            stream_accumulator_ = 0.0F;
        }
        for (auto iterator = streams_.begin(); iterator != streams_.end();) {
            if (!iterator->update(dt)) iterator = streams_.erase(iterator);
            else ++iterator;
        }

        const auto stop = std::chrono::steady_clock::now();
        budget_.record(static_cast<float>(
            std::chrono::duration<double, std::milli>(stop - start).count()));
    }

    void render(int width, int height) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0F, 0.0F, camera_distance_);
        glRotatef(camera_x_ * 180.0F / kPi, 1.0F, 0.0F, 0.0F);
        glRotatef(camera_y_ * 180.0F / kPi, 0.0F, 1.0F, 0.0F);

        const GLfloat light_position[] = {2.0F, 6.0F, 8.0F, 1.0F};
        glLightfv(GL_LIGHT0, GL_POSITION, light_position);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(GL_FALSE);

        render_pipeline();
        for (const DataStream& stream : streams_) stream.render();
        render_base_ring();

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        render_hud(width, height);
    }

    void keyboard(unsigned char key) {
        switch (key) {
        case '1': selected_ = jarvisx::MediaType::Visual; break;
        case '2': selected_ = jarvisx::MediaType::Audio; break;
        case '3': selected_ = jarvisx::MediaType::Text; break;
        case '4': selected_ = jarvisx::MediaType::Generic; break;
        case ' ': training_ = !training_; break;
        case 'q': case 'Q': engine_.set_quantized(!engine_.quantized()); break;
        case 'r': case 'R': auto_rotate_ = !auto_rotate_; break;
        case 'f': case 'F': follow_optimizer_ = !follow_optimizer_; break;
        case 's': case 'S': save(); break;
        case 'l': case 'L': load(); break;
        case '+': case '=': threshold_ = std::max(0.02F, threshold_ - 0.03F); break;
        case '-': case '_': threshold_ = std::min(0.92F, threshold_ + 0.03F); break;
        case 27: std::exit(0);
        default: break;
        }
        spawn_burst();
    }

    void rotate(int dx, int dy) noexcept {
        camera_y_ += static_cast<float>(dx) * 0.005F;
        camera_x_ = std::clamp(
            camera_x_ + static_cast<float>(dy) * 0.005F, -1.2F, 1.2F);
        auto_rotate_ = false;
    }

    void zoom(float delta) noexcept {
        camera_distance_ = std::clamp(camera_distance_ + delta, -18.0F, -7.5F);
    }

private:
    Options options_;
    jarvisx::MultimediaAutoencoder4D engine_;
    jarvisx::MediaType selected_;
    bool training_{true};
    bool auto_rotate_{true};
    bool follow_optimizer_{true};
    bool next_encode_{true};
    float camera_x_{0.28F};
    float camera_y_{};
    float camera_distance_{-12.5F};
    float threshold_{0.30F};
    float training_accumulator_{};
    float stream_accumulator_{};
    std::mt19937_64 rng_;
    std::uniform_real_distribution<float> unit_;
    std::vector<DataStream> streams_;
    FrameBudgetController budget_;
    const std::filesystem::path checkpoint_{
        ".jarvisx-multimedia4d/visualizer-checkpoint"};

    static Vec3 input_origin() noexcept { return {-3.4F, 0.0F, 0.0F}; }
    static Vec3 latent_origin() noexcept { return {0.0F, 0.0F, 0.0F}; }
    static Vec3 output_origin() noexcept { return {3.4F, 0.0F, 0.0F}; }

    void render_pipeline() {
        const Color color = media_color(selected_);
        const float threshold = std::min(0.95F,
            threshold_ + budget_.threshold_bias());
        render_scalar(engine_.input(selected_), input_origin(), 3.4F,
                      threshold, color, false);
        render_temporal_latent(latent_origin(), 3.0F,
                               threshold * 0.65F);
        render_scalar(engine_.reconstruction(selected_), output_origin(), 3.4F,
                      threshold, {1.0F, 0.35F, 0.8F}, false);
        render_residual(output_origin(), 3.4F,
                        std::max(0.08F, threshold * 0.45F));
        render_wire_cube(input_origin(), 1.8F, color);
        render_wire_cube(latent_origin(), 1.65F, {0.45F, 0.25F, 1.0F});
        render_wire_cube(output_origin(), 1.8F, {1.0F, 0.35F, 0.8F});
        render_edges(color);
    }

    static void render_scalar(const jarvisx::Tensor4D& tensor,
                              const Vec3& origin, float extent, float threshold,
                              const Color& color, bool absolute) {
        const auto shape = tensor.shape();
        const float edge = static_cast<float>(shape.width);
        const float spacing = extent / edge;
        const float cube = spacing * 0.58F;
        for (std::size_t z = 0; z < shape.depth; ++z) {
            for (std::size_t y = 0; y < shape.height; ++y) {
                for (std::size_t x = 0; x < shape.width; ++x) {
                    const float raw = tensor(0U, z, y, x);
                    const float value = absolute ? std::fabs(raw) : raw;
                    if (value <= threshold) continue;
                    const float intensity = std::clamp(
                        (value - threshold) / std::max(0.001F, 1.0F - threshold),
                        0.0F, 1.0F);
                    glColor4f(color.r, color.g, color.b,
                              0.16F + 0.72F * intensity);
                    glPushMatrix();
                    glTranslatef(
                        origin.x + (static_cast<float>(x) + 0.5F - edge * 0.5F) * spacing,
                        origin.y + (static_cast<float>(y) + 0.5F - edge * 0.5F) * spacing,
                        origin.z + (static_cast<float>(z) + 0.5F - edge * 0.5F) * spacing);
                    glutSolidCube(static_cast<double>(cube));
                    glPopMatrix();
                }
            }
        }
    }

    void render_temporal_latent(const Vec3& origin, float extent,
                                float threshold) const {
        const auto& history = engine_.temporal_history(selected_);
        const std::size_t layers = budget_.temporal_layers(history.size());
        std::size_t layer = 0U;
        for (const auto& frame : history) {
            if (layer >= layers) break;
            const float time_fraction = layers <= 1U ? 0.0F :
                static_cast<float>(layer) / static_cast<float>(layers - 1U);
            render_latent_frame(frame,
                {origin.x, origin.y + 0.18F * static_cast<float>(layer),
                 origin.z - 0.22F * static_cast<float>(layer)},
                extent * (1.0F - 0.045F * static_cast<float>(layer)),
                threshold + 0.025F * static_cast<float>(layer),
                1.0F - 0.75F * time_fraction);
            ++layer;
        }
    }

    static void render_latent_frame(const jarvisx::Tensor4D& tensor,
                                    const Vec3& origin, float extent,
                                    float threshold, float alpha_scale) {
        const auto shape = tensor.shape();
        const float edge = static_cast<float>(shape.width);
        const float spacing = extent / edge;
        const float cube = spacing * 0.60F;
        const float inverse_channels = 1.0F /
            static_cast<float>(shape.channels);
        for (std::size_t z = 0; z < shape.depth; ++z) {
            for (std::size_t y = 0; y < shape.height; ++y) {
                for (std::size_t x = 0; x < shape.width; ++x) {
                    float sum = 0.0F;
                    float energy = 0.0F;
                    for (std::size_t channel = 0; channel < shape.channels;
                         ++channel) {
                        const float value = tensor(channel, z, y, x);
                        sum += value;
                        energy += value * value;
                    }
                    energy = std::sqrt(energy * inverse_channels);
                    if (energy <= threshold) continue;
                    const Color color = sum >= 0.0F
                        ? Color{0.12F, 0.58F, 1.0F}
                        : Color{0.68F, 0.18F, 1.0F};
                    glColor4f(color.r, color.g, color.b,
                              alpha_scale * (0.18F + 0.70F * energy));
                    glPushMatrix();
                    glTranslatef(
                        origin.x + (static_cast<float>(x) + 0.5F - edge * 0.5F) * spacing,
                        origin.y + (static_cast<float>(y) + 0.5F - edge * 0.5F) * spacing,
                        origin.z + (static_cast<float>(z) + 0.5F - edge * 0.5F) * spacing);
                    glutSolidCube(static_cast<double>(cube));
                    glPopMatrix();
                }
            }
        }
    }

    void render_residual(const Vec3& origin, float extent, float threshold) const {
        const auto& input = engine_.input(selected_);
        const auto& output = engine_.reconstruction(selected_);
        const auto shape = input.shape();
        const float edge = static_cast<float>(shape.width);
        const float spacing = extent / edge;
        for (std::size_t z = 0; z < shape.depth; ++z) {
            for (std::size_t y = 0; y < shape.height; ++y) {
                for (std::size_t x = 0; x < shape.width; ++x) {
                    const float error = std::fabs(
                        input(0U, z, y, x) - output(0U, z, y, x));
                    if (error <= threshold) continue;
                    glColor4f(1.0F, 0.72F, 0.10F,
                              0.18F + 0.68F * std::min(1.0F, error));
                    glPushMatrix();
                    glTranslatef(
                        origin.x + (static_cast<float>(x) + 0.5F - edge * 0.5F) * spacing,
                        origin.y + (static_cast<float>(y) + 0.5F - edge * 0.5F) * spacing,
                        origin.z + (static_cast<float>(z) + 0.5F - edge * 0.5F) * spacing);
                    glutSolidCube(static_cast<double>(spacing * 0.28F));
                    glPopMatrix();
                }
            }
        }
    }

    static void render_wire_cube(const Vec3& origin, float half_extent,
                                 const Color& color) {
        glColor4f(color.r, color.g, color.b, 0.34F);
        glPushMatrix();
        glTranslatef(origin.x, origin.y, origin.z);
        glutWireCube(static_cast<double>(2.0F * half_extent));
        glPopMatrix();
    }

    static void render_edges(const Color& media) {
        glLineWidth(2.0F);
        glBegin(GL_LINES);
        glColor4f(media.r, media.g, media.b, 0.58F);
        glVertex3f(-2.2F, 0.0F, 0.0F); glVertex3f(-1.1F, 0.0F, 0.0F);
        glColor4f(1.0F, 0.35F, 0.8F, 0.58F);
        glVertex3f(1.1F, 0.0F, 0.0F); glVertex3f(2.2F, 0.0F, 0.0F);
        glColor4f(1.0F, 0.72F, 0.10F, 0.42F);
        glVertex3f(3.4F, 1.7F, 0.0F); glVertex3f(-3.4F, 1.7F, 0.0F);
        glEnd();
    }

    static void render_base_ring() {
        glDisable(GL_LIGHTING);
        glColor4f(0.0F, 0.65F, 1.0F, 0.45F);
        glBegin(GL_LINE_LOOP);
        for (int segment = 0; segment < 96; ++segment) {
            const float angle = 2.0F * kPi * static_cast<float>(segment) / 96.0F;
            glVertex3f(5.3F * std::cos(angle), -2.3F,
                       5.3F * std::sin(angle));
        }
        glEnd();
        glEnable(GL_LIGHTING);
    }

    void spawn_burst() {
        for (int index = 0; index < 2; ++index) {
            spawn_stream(StreamType::Encode);
            spawn_stream(StreamType::Decode);
        }
        spawn_stream(StreamType::Feedback);
    }

    void spawn_stream(StreamType type) {
        if (streams_.size() >= kMaxStreams) return;
        const float jy = unit_(rng_) * 0.75F;
        const float jz = unit_(rng_) * 0.75F;
        DataStream stream;
        stream.type = type;
        stream.speed = 0.65F + 0.28F * (1.0F + unit_(rng_));
        if (type == StreamType::Encode) {
            stream.start = input_origin() + Vec3{0.9F, jy, jz};
            stream.end = latent_origin() + Vec3{-0.7F, -0.2F * jy, -0.2F * jz};
            stream.control = {-1.65F, 1.2F, 0.0F};
            stream.color = media_color(selected_);
        } else if (type == StreamType::Decode) {
            stream.start = latent_origin() + Vec3{0.7F, 0.2F * jy, 0.2F * jz};
            stream.end = output_origin() + Vec3{-0.9F, jy, jz};
            stream.control = {1.65F, -1.2F, 0.0F};
            stream.color = {1.0F, 0.35F, 0.8F};
        } else {
            stream.start = output_origin() + Vec3{0.0F, jy, jz};
            stream.end = input_origin() + Vec3{0.0F, -jy, -jz};
            stream.control = {0.0F, 3.1F, 0.0F};
            stream.color = {1.0F, 0.72F, 0.10F};
            stream.speed *= 0.60F;
        }
        stream.position = stream.start;
        streams_.push_back(stream);
    }

    void save() {
        try {
            engine_.save_checkpoint(checkpoint_);
            std::cout << "checkpoint saved: " << checkpoint_.string() << '\n';
        } catch (const std::exception& error) {
            std::cerr << "save failed: " << error.what() << '\n';
        }
    }

    void load() {
        try {
            engine_.load_checkpoint(checkpoint_);
            std::cout << "checkpoint loaded: " << checkpoint_.string() << '\n';
        } catch (const std::exception& error) {
            std::cerr << "load failed: " << error.what() << '\n';
        }
    }

    static void draw_text(float x, float y, const std::string& text) {
        glRasterPos2f(x, y);
        for (const unsigned char character : text) {
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, character);
        }
    }

    void render_hud(int width, int height) const {
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluOrtho2D(0.0, static_cast<double>(width), 0.0,
                   static_cast<double>(height));
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glColor3f(0.8F, 0.95F, 1.0F);

        const auto& metrics = engine_.metrics();
        const auto& modality = metrics.modalities[
            jarvisx::media_index(selected_)];
        std::ostringstream line;
        line << "JARVIS-X 4D MULTIMODAL ANN | " << jarvisx::media_name(selected_)
             << " | cycle " << metrics.cycle
             << " | " << (training_ ? "TRAINING" : "PAUSED")
             << " | " << (engine_.quantized() ? "Q3" : "FLOAT");
        draw_text(18.0F, static_cast<float>(height - 24), line.str());
        line.str({}); line.clear();
        line << std::fixed << std::setprecision(5)
             << "instant MSE " << modality.instantaneous_mse
             << " | temporal MSE " << modality.temporal_mse
             << " | coherence " << modality.temporal_coherence
             << " | commits " << metrics.accepted
             << " | rollbacks " << metrics.rejected;
        draw_text(18.0F, static_cast<float>(height - 44), line.str());
        line.str({}); line.clear();
        line << std::setprecision(2)
             << "update " << budget_.average_ms() << " ms"
             << " | history " << engine_.temporal_history(selected_).size()
             << "/" << engine_.config().temporal_depth
             << " | follow " << (follow_optimizer_ ? "ON" : "OFF");
        draw_text(18.0F, static_cast<float>(height - 64), line.str());
        draw_text(18.0F, 18.0F,
            "1-4 modality | SPACE train | Q Q3 | F follow | R rotate | S/L checkpoint | +/- density | ESC exit");

        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
    }
};

std::unique_ptr<MultimediaVisualizer> visualizer;
int last_time = 0;
bool mouse_down = false;
int last_x = 0;
int last_y = 0;
int window_width = 1280;
int window_height = 720;

void display() {
    visualizer->render(window_width, window_height);
    glutSwapBuffers();
}

void reshape(int width, int height) {
    window_width = std::max(1, width);
    window_height = std::max(1, height);
    glViewport(0, 0, window_width, window_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(55.0, static_cast<double>(window_width) /
                         static_cast<double>(window_height),
                   0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

void timer(int) {
    const int now = glutGet(GLUT_ELAPSED_TIME);
    const float dt = static_cast<float>(now - last_time) / 1000.0F;
    last_time = now;
    visualizer->update(dt);
    glutPostRedisplay();
    glutTimerFunc(16U, timer, 0);
}

void keyboard(unsigned char key, int, int) { visualizer->keyboard(key); }

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        mouse_down = state == GLUT_DOWN;
        last_x = x;
        last_y = y;
    }
#if defined(GLUT_WHEEL_UP)
    if (button == GLUT_WHEEL_UP && state == GLUT_DOWN) visualizer->zoom(0.7F);
    if (button == GLUT_WHEEL_DOWN && state == GLUT_DOWN) visualizer->zoom(-0.7F);
#endif
}

void motion(int x, int y) {
    if (!mouse_down) return;
    visualizer->rotate(x - last_x, y - last_y);
    last_x = x;
    last_y = y;
}

void initialize_gl() {
    glClearColor(0.0F, 0.0F, 0.02F, 1.0F);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    const GLfloat ambient[] = {0.25F, 0.25F, 0.35F, 1.0F};
    const GLfloat diffuse[] = {0.90F, 0.90F, 1.0F, 1.0F};
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(window_width, window_height);
        glutCreateWindow("Jarvis-X Self-Optimizing 4D Multimedia ANN");
        initialize_gl();
        visualizer = std::make_unique<MultimediaVisualizer>(options);
        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        last_time = glutGet(GLUT_ELAPSED_TIME);
        glutTimerFunc(16U, timer, 0);
        std::cout << "Jarvis-X 4D multimodal visualizer online.\n";
        glutMainLoop();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Jarvis-X multimedia4d OpenGL failure: "
                  << error.what() << '\n';
        return 1;
    }
}
