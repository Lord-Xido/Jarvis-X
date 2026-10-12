#ifndef ANN17_HEADER_ONLY_HPP
#define ANN17_HEADER_ONLY_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

enum class Activation { ReLU, Sigmoid, Tanh, Linear };

class Layer {
public:
    std::size_t input_size;
    std::size_t output_size;
    Activation activation;

    // W is row-major: W[output * input_size + input].
    std::vector<double> W, b;
    std::vector<double> z, a;       // Cached preactivations and activations.
    std::vector<double> dW, db;     // Accumulated (unaveraged) gradients.

    Layer(std::size_t inputs, std::size_t outputs,
          Activation act, std::mt19937& rng)
        : input_size(inputs), output_size(outputs), activation(act),
          W(checked_product(inputs, outputs)), b(outputs, 0.0),
          z(outputs), a(outputs), dW(W.size(), 0.0), db(outputs, 0.0),
          input_cache_(inputs), grad_input_(inputs) {
        const double limit = std::sqrt(6.0 / static_cast<double>(inputs + outputs));
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (double& weight : W) weight = dist(rng);
    }

    const std::vector<double>& forward(const std::vector<double>& input) {
        check_input(input);
        input_cache_ = input;
        for (std::size_t o = 0; o < output_size; ++o) {
            double sum = b[o];
            const std::size_t row = o * input_size;
            for (std::size_t i = 0; i < input_size; ++i)
                sum += W[row + i] * input[i];
            z[o] = sum;
            a[o] = activate(sum, activation);
        }
        return a;
    }

    // Const inference: does not overwrite the caches needed for backprop.
    void infer(const std::vector<double>& input,
               std::vector<double>& output) const {
        check_input(input);
        if (&input == &output)
            throw std::invalid_argument("Layer::infer requires distinct buffers");
        output.resize(output_size);
        for (std::size_t o = 0; o < output_size; ++o) {
            double sum = b[o];
            const std::size_t row = o * input_size;
            for (std::size_t i = 0; i < input_size; ++i)
                sum += W[row + i] * input[i];
            output[o] = activate(sum, activation);
        }
    }

    // Upstream gradient is dL/da. Returns dL/d(input); accumulates dW, db.
    const std::vector<double>& backward(const std::vector<double>& upstream) {
        if (upstream.size() != output_size)
            throw std::invalid_argument("Layer::backward gradient dimension mismatch");
        std::fill(grad_input_.begin(), grad_input_.end(), 0.0);
        for (std::size_t o = 0; o < output_size; ++o) {
            const double delta = upstream[o] * derivative(z[o], a[o], activation);
            db[o] += delta;
            const std::size_t row = o * input_size;
            for (std::size_t i = 0; i < input_size; ++i) {
                dW[row + i] += delta * input_cache_[i];
                grad_input_[i] += W[row + i] * delta;
            }
        }
        return grad_input_;
    }

    void zero_grad() {
        std::fill(dW.begin(), dW.end(), 0.0);
        std::fill(db.begin(), db.end(), 0.0);
    }

    void sgd_step(double learning_rate, std::size_t batch_count) {
        if (batch_count == 0 || !std::isfinite(learning_rate) || learning_rate <= 0.0)
            throw std::invalid_argument("Invalid SGD step arguments");
        const double scale = learning_rate / static_cast<double>(batch_count);
        // Check the entire update before mutating any parameter in this layer.
        for (std::size_t k = 0; k < W.size(); ++k)
            if (!std::isfinite(W[k] - scale * dW[k]))
                throw std::runtime_error("Non-finite SGD weight update");
        for (std::size_t k = 0; k < b.size(); ++k)
            if (!std::isfinite(b[k] - scale * db[k]))
                throw std::runtime_error("Non-finite SGD bias update");
        for (std::size_t k = 0; k < W.size(); ++k) W[k] -= scale * dW[k];
        for (std::size_t k = 0; k < b.size(); ++k) b[k] -= scale * db[k];
    }

private:
    std::vector<double> input_cache_, grad_input_;

    static std::size_t checked_product(std::size_t inputs, std::size_t outputs) {
        if (inputs == 0 || outputs == 0 ||
            inputs > std::numeric_limits<std::size_t>::max() / outputs)
            throw std::invalid_argument("Invalid layer dimensions");
        return inputs * outputs;
    }

    void check_input(const std::vector<double>& input) const {
        if (input.size() != input_size)
            throw std::invalid_argument("Layer input dimension mismatch");
    }

    static double activate(double x, Activation act) {
        switch (act) {
            case Activation::ReLU:   return std::max(0.0, x);
            case Activation::Sigmoid:
                if (x >= 0.0) { const double e = std::exp(-x); return 1.0 / (1.0 + e); }
                else          { const double e = std::exp(x);  return e / (1.0 + e); }
            case Activation::Tanh:   return std::tanh(x);
            case Activation::Linear: return x;
        }
        throw std::invalid_argument("Invalid activation");
    }

    static double derivative(double preactivation, double output, Activation act) {
        switch (act) {
            case Activation::ReLU:   return preactivation > 0.0 ? 1.0 : 0.0;
            case Activation::Sigmoid:return output * (1.0 - output);
            case Activation::Tanh:   return 1.0 - output * output;
            case Activation::Linear: return 1.0;
        }
        throw std::invalid_argument("Invalid activation");
    }
};

class ANN {
public:
    std::vector<Layer> layers;

    ANN(const std::vector<std::size_t>& topology,
        const std::vector<Activation>& activations,
        std::uint32_t seed = 42)
        : rng_(seed) {
        if (topology.size() < 2 || activations.size() != topology.size() - 1)
            throw std::invalid_argument("Provide one activation for each non-input layer");
        for (std::size_t width : topology)
            if (width == 0) throw std::invalid_argument("Zero-width network layer");
        layers.reserve(activations.size());
        for (std::size_t i = 0; i < activations.size(); ++i)
            layers.emplace_back(topology[i], topology[i + 1], activations[i], rng_);
        output_gradient_.resize(topology.back());
    }

    // Mutating forward pass: caches each layer's z and a for backward().
    std::vector<double> forward(const std::vector<double>& input) {
        return forward_cached(input);
    }

    // Read-only inference does not disturb any training caches.
    std::vector<double> predict(const std::vector<double>& input) const {
        std::vector<double> current = input, next;
        for (const Layer& layer : layers) {
            layer.infer(current, next);
            current.swap(next);
        }
        return current;
    }

    // Loss is mean squared error over OUTPUT UNITS for one sample.
    static double mse(const std::vector<double>& predicted,
                      const std::vector<double>& target) {
        if (predicted.empty() || predicted.size() != target.size())
            throw std::invalid_argument("MSE dimension mismatch");
        double total = 0.0;
        for (std::size_t i = 0; i < predicted.size(); ++i) {
            const double difference = predicted[i] - target[i];
            total += difference * difference;
        }
        return total / static_cast<double>(predicted.size());
    }

    void zero_grad() {
        for (Layer& layer : layers) layer.zero_grad();
    }

    // Must follow forward(). Gradients accumulate until zero_grad().
    void backward(const std::vector<double>& target) {
        if (!has_forward_)
            throw std::logic_error("Call forward() before backward()");
        const auto& output = layers.back().a;
        if (target.size() != output.size())
            throw std::invalid_argument("Target dimension mismatch");
        const double scale = 2.0 / static_cast<double>(output.size());
        for (std::size_t k = 0; k < output.size(); ++k)
            output_gradient_[k] = scale * (output[k] - target[k]);
        const std::vector<double>* grad = &output_gradient_;
        for (std::size_t i = layers.size(); i-- > 0;)
            grad = &layers[i].backward(*grad);
    }

    double evaluate_mse(const std::vector<std::vector<double>>& X,
                        const std::vector<std::vector<double>>& Y) const {
        validate_dataset(X, Y);
        double total = 0.0;
        for (std::size_t i = 0; i < X.size(); ++i)
            total += mse(predict(X[i]), Y[i]);
        return total / static_cast<double>(X.size());
    }

    // Mini-batch SGD with deterministic shuffling for a fixed seed.
    // Returns post-training dataset MSE, not a stale pre-update batch loss.
    // stop_mse = 0 disables early stopping.
    double train(const std::vector<std::vector<double>>& X,
                 const std::vector<std::vector<double>>& Y,
                 std::size_t epochs,
                 std::size_t batch_size,
                 double learning_rate,
                 double stop_mse = 0.0) {
        validate_dataset(X, Y);
        if (batch_size == 0 || !std::isfinite(learning_rate) ||
            learning_rate <= 0.0 || !std::isfinite(stop_mse) || stop_mse < 0.0)
            throw std::invalid_argument("Invalid training hyperparameters");

        std::vector<std::size_t> order(X.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), rng_);
            for (std::size_t start = 0; start < X.size();) {
                const std::size_t count = std::min(batch_size, X.size() - start);
                zero_grad();
                for (std::size_t j = 0; j < count; ++j) {
                    const std::size_t index = order[start + j];
                    forward_cached(X[index]);
                    backward(Y[index]);
                }
                for (Layer& layer : layers) layer.sgd_step(learning_rate, count);
                has_forward_ = false;  // Cached activations correspond to older weights.
                start += count;
            }
            if (stop_mse > 0.0 && evaluate_mse(X, Y) <= stop_mse) break;
        }
        return evaluate_mse(X, Y);
    }

    // Central finite-difference gradient audit on one sample.
    // Returns maximum normalized absolute gradient discrepancy.
    double gradient_check(const std::vector<double>& input,
                          const std::vector<double>& target,
                          double epsilon = 1e-5) {
        if (!std::isfinite(epsilon) || epsilon <= 0.0)
            throw std::invalid_argument("Invalid finite-difference epsilon");
        zero_grad();
        forward_cached(input);
        backward(target);
        double worst = 0.0;
        for (Layer& layer : layers) {
            for (std::size_t k = 0; k < layer.W.size(); ++k) {
                const double analytical = layer.dW[k];
                const double numerical = finite_difference(layer.W[k], input, target, epsilon);
                worst = std::max(worst, relative_error(analytical, numerical));
            }
            for (std::size_t k = 0; k < layer.b.size(); ++k) {
                const double analytical = layer.db[k];
                const double numerical = finite_difference(layer.b[k], input, target, epsilon);
                worst = std::max(worst, relative_error(analytical, numerical));
            }
        }
        forward_cached(input); // Restore caches after the probe.
        return worst;
    }

    // Portable, inspectable text checkpoint. Parameters use 17-digit precision.
    void save(const std::string& path) const {
        std::ofstream out(path);
        if (!out) throw std::runtime_error("Cannot open ANN checkpoint for writing");
        out << "ANN17 1\n" << layers.size() + 1 << '\n';
        out << layers.front().input_size;
        for (const Layer& layer : layers) out << ' ' << layer.output_size;
        out << '\n';
        for (const Layer& layer : layers)
            out << static_cast<int>(layer.activation) << ' ';
        out << '\n' << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (const Layer& layer : layers) {
            for (double value : layer.W) out << value << ' ';
            out << '\n';
            for (double value : layer.b) out << value << ' ';
            out << '\n';
        }
        if (!out) throw std::runtime_error("Failed to write ANN checkpoint");
    }

    static ANN load(const std::string& path, std::uint32_t seed = 42) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("Cannot open ANN checkpoint for reading");
        std::string magic;
        int version = 0;
        std::size_t n = 0;
        if (!(in >> magic >> version >> n) || magic != "ANN17" || version != 1 ||
            n < 2 || n > 10000)
            throw std::runtime_error("Invalid ANN checkpoint header");
        std::vector<std::size_t> topology(n);
        std::vector<Activation> activations(n - 1);
        for (std::size_t& size : topology)
            if (!(in >> size) || size == 0)
                throw std::runtime_error("Invalid ANN checkpoint dimensions");
        for (Activation& a : activations) {
            int value = -1;
            if (!(in >> value) || value < 0 || value > 3)
                throw std::runtime_error("Invalid ANN checkpoint activation");
            a = static_cast<Activation>(value);
        }
        ANN network(topology, activations, seed);
        for (Layer& layer : network.layers) {
            for (double& value : layer.W)
                if (!(in >> value) || !std::isfinite(value))
                    throw std::runtime_error("Invalid ANN checkpoint weight");
            for (double& value : layer.b)
                if (!(in >> value) || !std::isfinite(value))
                    throw std::runtime_error("Invalid ANN checkpoint bias");
        }
        std::string extra;
        if (in >> extra) throw std::runtime_error("Trailing ANN checkpoint data");
        return network;
    }

private:
    std::mt19937 rng_;
    std::vector<double> output_gradient_;
    bool has_forward_ = false;

    const std::vector<double>& forward_cached(const std::vector<double>& input) {
        const std::vector<double>* current = &input;
        has_forward_ = false;
        for (Layer& layer : layers) current = &layer.forward(*current);
        has_forward_ = true;
        return *current;
    }

    static double relative_error(double x, double y) {
        return std::abs(x - y) / std::max({1.0, std::abs(x), std::abs(y)});
    }

    double finite_difference(double& parameter,
                             const std::vector<double>& input,
                             const std::vector<double>& target,
                             double epsilon) const {
        const double original = parameter;
        parameter = original + epsilon;
        const double plus = mse(predict(input), target);
        parameter = original - epsilon;
        const double minus = mse(predict(input), target);
        parameter = original;
        return (plus - minus) / (2.0 * epsilon);
    }

    void validate_dataset(const std::vector<std::vector<double>>& X,
                          const std::vector<std::vector<double>>& Y) const {
        if (X.empty() || X.size() != Y.size())
            throw std::invalid_argument("Dataset must contain paired samples");
        for (std::size_t n = 0; n < X.size(); ++n) {
            if (X[n].size() != layers.front().input_size ||
                Y[n].size() != layers.back().output_size)
                throw std::invalid_argument("Dataset sample dimension mismatch");
            for (double value : X[n])
                if (!std::isfinite(value)) throw std::invalid_argument("Non-finite input");
            for (double value : Y[n])
                if (!std::isfinite(value)) throw std::invalid_argument("Non-finite target");
        }
    }
};

#endif // ANN17_HEADER_ONLY_HPP
