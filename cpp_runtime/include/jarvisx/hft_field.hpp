#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace jarvisx::hft {

class Q16 {
public:
    static constexpr std::int64_t kScale = 1LL << 16;

    constexpr Q16() noexcept = default;

    static constexpr Q16 from_raw(std::int32_t raw) noexcept {
        Q16 out;
        out.raw_ = raw;
        return out;
    }

    static constexpr Q16 from_int(std::int32_t value) noexcept {
        return from_raw(saturate(static_cast<std::int64_t>(value) * kScale));
    }

    static constexpr Q16 from_ratio(std::int32_t numerator,
                                    std::int32_t denominator) noexcept {
        return denominator == 0
            ? from_raw(numerator >= 0 ? std::numeric_limits<std::int32_t>::max()
                                      : std::numeric_limits<std::int32_t>::min())
            : from_raw(saturate((static_cast<std::int64_t>(numerator) * kScale) /
                                static_cast<std::int64_t>(denominator)));
    }

    constexpr std::int32_t raw() const noexcept { return raw_; }
    double to_double() const noexcept {
        return static_cast<double>(raw_) / static_cast<double>(kScale);
    }

    friend constexpr Q16 operator+(Q16 a, Q16 b) noexcept {
        return from_raw(saturate(static_cast<std::int64_t>(a.raw_) + b.raw_));
    }

    friend constexpr Q16 operator-(Q16 a, Q16 b) noexcept {
        return from_raw(saturate(static_cast<std::int64_t>(a.raw_) - b.raw_));
    }

    friend constexpr Q16 operator-(Q16 value) noexcept {
        return from_raw(saturate(-static_cast<std::int64_t>(value.raw_)));
    }

    friend constexpr Q16 operator*(Q16 a, Q16 b) noexcept {
        const std::int64_t product = static_cast<std::int64_t>(a.raw_) * b.raw_;
        // Integer division truncates toward zero in C++11+, avoiding signed-shift ambiguity.
        return from_raw(saturate(product / kScale));
    }

    friend constexpr bool operator>(Q16 a, Q16 b) noexcept { return a.raw_ > b.raw_; }
    friend constexpr bool operator<(Q16 a, Q16 b) noexcept { return a.raw_ < b.raw_; }
    friend constexpr bool operator>=(Q16 a, Q16 b) noexcept { return a.raw_ >= b.raw_; }
    friend constexpr bool operator<=(Q16 a, Q16 b) noexcept { return a.raw_ <= b.raw_; }
    friend constexpr bool operator==(Q16 a, Q16 b) noexcept { return a.raw_ == b.raw_; }
    friend constexpr bool operator!=(Q16 a, Q16 b) noexcept { return !(a == b); }

    static constexpr Q16 min(Q16 a, Q16 b) noexcept { return a < b ? a : b; }
    static constexpr Q16 max(Q16 a, Q16 b) noexcept { return a > b ? a : b; }
    static constexpr Q16 clamp(Q16 v, Q16 lo, Q16 hi) noexcept {
        return v < lo ? lo : (v > hi ? hi : v);
    }
    static constexpr Q16 abs(Q16 value) noexcept {
        return value.raw_ >= 0
            ? value
            : from_raw(saturate(-static_cast<std::int64_t>(value.raw_)));
    }

private:
    std::int32_t raw_{0};

    static constexpr std::int32_t saturate(std::int64_t value) noexcept {
        return value > std::numeric_limits<std::int32_t>::max()
            ? std::numeric_limits<std::int32_t>::max()
            : (value < std::numeric_limits<std::int32_t>::min()
                   ? std::numeric_limits<std::int32_t>::min()
                   : static_cast<std::int32_t>(value));
    }
};

enum class Side : std::int8_t { Bid = 1, Ask = -1 };
enum class Action : std::int8_t { None = 0, Buy = 1, Sell = -1 };

struct MarketEvent {
    std::int32_t price_tick{};
    std::uint16_t venue{};
    Side side{Side::Bid};
    Q16 delta_quantity{}; // signed: add > 0, cancel/remove < 0
    std::uint64_t sequence{};
};

struct OrderIntent {
    Action action{Action::None};
    Q16 quantity{};
    Q16 score{};
    bool risk_accepted{false};
    std::uint64_t source_sequence{};
};

struct HftFieldConfig {
    Q16 alpha{Q16::from_ratio(1, 4)};
    Q16 lambda{Q16::from_ratio(1, 32)};
    Q16 eta{Q16::from_ratio(1, 2)};
    Q16 dt{Q16::from_ratio(1, 8)};
    Q16 rho{Q16::from_ratio(15, 16)};
    Q16 flow_decay{Q16::from_ratio(7, 8)};

    Q16 w_psi{Q16::from_ratio(1, 1)};
    Q16 w_omega{Q16::from_ratio(1, 2)};
    Q16 w_flow{Q16::from_ratio(1, 2)};
    Q16 w_laplacian{Q16::from_ratio(1, 8)};
    Q16 w_inventory{Q16::from_ratio(1, 4)};

    Q16 decision_threshold{Q16::from_ratio(1, 16)};
    Q16 max_abs_field{Q16::from_int(32)};
    Q16 max_inventory{Q16::from_int(64)};
    Q16 max_order_quantity{Q16::from_int(4)};
};

struct PipelineBudget {
    std::uint16_t ingress_cutthrough{8};
    std::uint16_t book_update{4};
    std::uint16_t state_load{4};
    std::uint16_t field_gradient{8};
    std::uint16_t memory_update{4};
    std::uint16_t score{6};
    std::uint16_t risk{4};
    std::uint16_t order_encode{6};
    std::uint16_t tx_launch{4};

    constexpr std::uint16_t total_cycles() const noexcept {
        return static_cast<std::uint16_t>(
            ingress_cutthrough + book_update + state_load + field_gradient +
            memory_update + score + risk + order_encode + tx_launch);
    }

    double target_latency_ns(double clock_mhz = 500.0) const noexcept {
        return static_cast<double>(total_cycles()) * 1000.0 / clock_mhz;
    }
};

template <std::size_t PriceBins = 64,
          std::size_t Venues = 4,
          std::size_t Horizons = 4>
class HftFieldEngine {
    static_assert(PriceBins >= 8 && (PriceBins & (PriceBins - 1U)) == 0U,
                  "PriceBins must be a power of two >= 8");
    static_assert(Venues >= 2 && (Venues & (Venues - 1U)) == 0U,
                  "Venues must be a power of two >= 2");
    static_assert(Horizons >= 2 && (Horizons & (Horizons - 1U)) == 0U,
                  "Horizons must be a power of two >= 2");

public:
    struct Cell {
        Q16 psi{};
        Q16 omega{};
        Q16 bid_depth{};
        Q16 ask_depth{};
        Q16 flow{};
    };

    static constexpr std::size_t kCellCount = PriceBins * Venues * Horizons;
    static constexpr std::size_t kStencilReads = 7;

    explicit constexpr HftFieldEngine(HftFieldConfig config = {}) noexcept
        : config_(config) {}

    OrderIntent process(const MarketEvent& event) noexcept {
        const Coord c = map(event);
        Cell& center = cells_[index(c)];

        if (event.side == Side::Bid) {
            center.bid_depth = Q16::max(Q16{}, center.bid_depth + event.delta_quantity);
        } else {
            center.ask_depth = Q16::max(Q16{}, center.ask_depth + event.delta_quantity);
        }

        const Q16 signed_impulse = event.side == Side::Bid
            ? event.delta_quantity
            : -event.delta_quantity;
        center.flow = config_.flow_decay * center.flow + signed_impulse;

        const Q16 lap = laplacian(c);
        const Q16 residual = center.psi - center.omega;
        const Q16 rhs = -(config_.alpha * residual)
                      + config_.lambda * lap
                      + config_.eta * signed_impulse;

        const Q16 candidate = Q16::clamp(
            center.psi + config_.dt * rhs,
            -config_.max_abs_field,
            config_.max_abs_field);

        const Q16 one_minus_rho = Q16::from_int(1) - config_.rho;
        center.omega = config_.rho * center.omega + one_minus_rho * candidate;
        center.psi = candidate;

        const Q16 score = config_.w_psi * center.psi
                        + config_.w_omega * center.omega
                        + config_.w_flow * center.flow
                        + config_.w_laplacian * lap
                        - config_.w_inventory * inventory_;

        OrderIntent intent;
        intent.score = score;
        intent.source_sequence = event.sequence;

        if (score > config_.decision_threshold) {
            intent.action = Action::Buy;
        } else if (score < -config_.decision_threshold) {
            intent.action = Action::Sell;
        } else {
            return intent;
        }

        intent.quantity = Q16::min(Q16::abs(event.delta_quantity),
                                   config_.max_order_quantity);
        if (intent.quantity == Q16{}) {
            intent.quantity = config_.max_order_quantity;
        }

        const Q16 signed_order = intent.action == Action::Buy
            ? intent.quantity
            : -intent.quantity;
        const Q16 projected_inventory = inventory_ + signed_order;
        intent.risk_accepted = Q16::abs(projected_inventory) <= config_.max_inventory;
        if (!intent.risk_accepted) {
            intent.action = Action::None;
            intent.quantity = Q16{};
        }
        return intent;
    }

    void on_fill(Action action, Q16 quantity) noexcept {
        if (action == Action::Buy) {
            inventory_ = inventory_ + quantity;
        } else if (action == Action::Sell) {
            inventory_ = inventory_ - quantity;
        }
        inventory_ = Q16::clamp(inventory_, -config_.max_inventory,
                                config_.max_inventory);
    }

    Q16 inventory() const noexcept { return inventory_; }

    const Cell& cell(std::size_t price, std::size_t venue,
                     std::size_t horizon) const {
        if (price >= PriceBins || venue >= Venues || horizon >= Horizons) {
            throw std::out_of_range("HFT field coordinate out of range");
        }
        return cells_[price + PriceBins * (venue + Venues * horizon)];
    }

    std::uint64_t digest() const noexcept {
        std::uint64_t h = 1469598103934665603ULL;
        for (const Cell& cell_value : cells_) {
            hash_word(h, cell_value.psi.raw());
            hash_word(h, cell_value.omega.raw());
            hash_word(h, cell_value.bid_depth.raw());
            hash_word(h, cell_value.ask_depth.raw());
            hash_word(h, cell_value.flow.raw());
        }
        hash_word(h, inventory_.raw());
        return h;
    }

private:
    struct Coord {
        std::size_t x{};
        std::size_t y{};
        std::size_t z{};
    };

    HftFieldConfig config_{};
    std::array<Cell, kCellCount> cells_{};
    Q16 inventory_{};

    static constexpr std::size_t index(Coord c) noexcept {
        return c.x + PriceBins * (c.y + Venues * c.z);
    }

    static constexpr std::size_t wrap(std::size_t value,
                                      std::size_t mask) noexcept {
        return value & mask;
    }

    static constexpr Coord map(const MarketEvent& event) noexcept {
        const auto unsigned_price = static_cast<std::uint32_t>(event.price_tick);
        return {
            static_cast<std::size_t>(unsigned_price) & (PriceBins - 1U),
            static_cast<std::size_t>(event.venue) & (Venues - 1U),
            0U
        };
    }

    Q16 psi(Coord c) const noexcept { return cells_[index(c)].psi; }

    Q16 laplacian(Coord c) const noexcept {
        const std::size_t xm = wrap(c.x + PriceBins - 1U, PriceBins - 1U);
        const std::size_t xp = wrap(c.x + 1U, PriceBins - 1U);
        const std::size_t ym = wrap(c.y + Venues - 1U, Venues - 1U);
        const std::size_t yp = wrap(c.y + 1U, Venues - 1U);
        const std::size_t zm = wrap(c.z + Horizons - 1U, Horizons - 1U);
        const std::size_t zp = wrap(c.z + 1U, Horizons - 1U);

        const Q16 neighbours = psi({xm, c.y, c.z}) + psi({xp, c.y, c.z})
                             + psi({c.x, ym, c.z}) + psi({c.x, yp, c.z})
                             + psi({c.x, c.y, zm}) + psi({c.x, c.y, zp});
        return neighbours - Q16::from_int(6) * psi(c);
    }

    static void hash_word(std::uint64_t& h, std::int32_t word) noexcept {
        const auto u = static_cast<std::uint32_t>(word);
        for (unsigned shift = 0; shift < 32U; shift += 8U) {
            h ^= static_cast<std::uint8_t>((u >> shift) & 0xFFU);
            h *= 1099511628211ULL;
        }
    }
};

} // namespace jarvisx::hft
