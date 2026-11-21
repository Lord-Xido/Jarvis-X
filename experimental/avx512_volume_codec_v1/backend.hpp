// Copyright (c) 2026 Dr Matladi Maxwell Moagi. See repository LICENSE.
#pragma once
#include <cstddef>
namespace jx::volume {
enum class Backend { Auto, Scalar, AVX512 };
bool avx512_compiled() noexcept;
bool avx512_supported() noexcept;
Backend resolve(Backend requested);
const char* backend_name(Backend backend) noexcept;
namespace detail {
void pool_scalar(const float*, float*, std::size_t, std::size_t, std::size_t);
void expand_scalar(const float*, float*, std::size_t, std::size_t, std::size_t);
double mse_scalar(const float*, const float*, std::size_t);
void pool_avx512(const float*, float*, std::size_t, std::size_t, std::size_t);
void expand_avx512(const float*, float*, std::size_t, std::size_t, std::size_t);
double mse_avx512(const float*, const float*, std::size_t);
}
}
