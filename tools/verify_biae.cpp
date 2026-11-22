// Independent 3D-BIAE-Ω v1 seed-vector verifier. C++17 and OpenSSL libcrypto.
#include <openssl/evp.h>

#include <array>
#include <bitset>
#include <charconv>
#include <climits>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
static_assert(CHAR_BIT == 8);
using Bytes = std::vector<std::uint8_t>;
using Digest = std::array<std::uint8_t, 32>;
constexpr char domain[] = "JX/3D-BIAE-Omega/v1";  // Include its trailing NUL on the wire.
constexpr std::uint32_t seed = 0x7FFF0133;

void append_le(Bytes& bytes, std::uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) {
        bytes.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
    }
}

Bytes seeded_field(std::size_t size) {
    Bytes input(domain, domain + sizeof(domain));
    input.insert(input.end(), {'s', 'e', 'e', 'd', 0});
    append_le(input, seed, 4);
    Bytes field(size);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_shake256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx.get(), input.data(), input.size()) != 1 ||
        EVP_DigestFinalXOF(ctx.get(), field.data(), field.size()) != 1) {
        throw std::runtime_error("SHAKE256 seed expansion failed");
    }
    return field;
}

Digest sha3(const Bytes& bytes) {
    Digest digest{};
    unsigned length = 0;
    if (EVP_Digest(bytes.data(), bytes.size(), digest.data(), &length, EVP_sha3_256(), nullptr) != 1 ||
        length != digest.size()) {
        throw std::runtime_error("SHA3-256 failed");
    }
    return digest;
}

std::string hex(const Digest& digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (auto value : digest) {
        out.push_back(digits[value >> 4]);
        out.push_back(digits[value & 15]);
    }
    return out;
}

// Gather eight bits starting at the output byte's bit address plus a circular offset.
std::uint8_t gather(const Bytes& field, std::size_t byte, std::size_t offset) {
    const auto mask = field.size() - 1;
    const auto lo = (byte + (offset >> 3)) & mask;
    const unsigned pair = field[lo] | (static_cast<unsigned>(field[(lo + 1) & mask]) << 8);
    return static_cast<std::uint8_t>(pair >> (offset & 7));
}

std::uint8_t decode_shear(const Bytes& plane, std::size_t byte) {
    return plane[byte] ^ gather(plane, byte, (plane.size() << 3) - 3) ^ gather(plane, byte, 11);
}

Digest verify_tick(unsigned side, const Bytes& field, std::uint64_t tick, const Digest& previous) {
    const auto size = field.size();
    const auto bits = size << 3;
    Bytes latent(size), drift(size), warped(size), residual(size);

    // Derive the coordinate mask from the three low Morton digits for each lane.
    unsigned coordinates = 0;
    for (unsigned m = 0; m < 8; ++m) {
        coordinates |= ((m ^ (m >> 1) ^ (m >> 2)) & 1u) << m;
    }
    for (std::size_t i = 0; i < size; ++i) {
        latent[i] = field[i] ^ gather(field, i, bits - 7) ^ gather(field, i, 13);
        drift[i] = static_cast<std::uint8_t>(field[i] ^ gather(field, i, bits - 3) ^ coordinates);
        warped[i] = latent[i] ^ drift[i];
    }
    std::uint64_t errors = 0;
    for (std::size_t i = 0; i < size; ++i) {
        residual[i] = field[i] ^ decode_shear(warped, i);
        errors += std::bitset<8>(residual[i]).count();
    }
    for (std::size_t i = 0; i < size; ++i) {
        // Apply D to the two planes separately, using linearity over GF(2).
        if ((decode_shear(latent, i) ^ decode_shear(drift, i) ^ residual[i]) != field[i]) {
            throw std::runtime_error("reconstruction mismatch");
        }
    }

    const auto b = sha3(field), z = sha3(latent), e = sha3(residual), d = sha3(drift);
    Bytes record(domain, domain + sizeof(domain));
    record.insert(record.end(), previous.begin(), previous.end());
    append_le(record, side, 4);
    for (const auto& digest : {b, z, e, d}) {
        record.insert(record.end(), digest.begin(), digest.end());
    }
    append_le(record, tick, 8);
    record.push_back(1);
    auto omega = sha3(record);
    for (std::size_t i = 0; i < omega.size(); ++i) {
        omega[i] ^= previous[i];
    }

    // The field order and spacing match Python's sorted JSON replay lines.
    std::cout << "{\"B\": \"" << hex(b) << "\", \"Z\": \"" << hex(z)
              << "\", \"changed_bits\": 0, \"delta\": \"" << hex(d)
              << "\", \"epsilon\": \"" << hex(e) << "\", \"gate_mask\": 1, \"input_tick\": " << tick
              << ", \"omega_in\": \"" << hex(previous) << "\", \"omega_out\": \"" << hex(omega)
              << "\", \"packed_bytes\": " << size << ", \"pre_correction_errors\": " << errors
              << ", \"seed\": " << seed << ", \"side\": " << side << "}\n";
    return omega;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("usage: verify_biae SIDE");
        }
        const std::string_view arg(argv[1]);
        unsigned side = 0;
        const auto result = std::from_chars(arg.data(), arg.data() + arg.size(), side);
        if (result.ec != std::errc{} || result.ptr != arg.data() + arg.size() ||
            side < 4 || side > 1024 || (side & (side - 1)) != 0) {
            throw std::invalid_argument("SIDE must be a power of two from 4 through 1024");
        }
        const auto bits = static_cast<std::size_t>(side) * side * side;
        const auto field = seeded_field(bits >> 3);
        Digest previous{};
        for (std::uint64_t tick = 0; tick < 2; ++tick) {
            previous = verify_tick(side, field, tick, previous);
        }
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
    return 0;
}
