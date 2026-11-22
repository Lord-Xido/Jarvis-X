// Copyright (c) 2026 Dr Matladi Maxwell Moagi. See repository LICENSE.
// Supplied pooling codec: Dr Moagi. Safe dispatch/SIMD lowering: OpenAI Codex.
#pragma once
#include "backend.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace jx::volume {
template<std::size_t Depth, std::size_t Height, std::size_t Width, std::size_t Channels>
class AVX512VolumeAutoencoderCodec {
    static_assert(Depth > 0 && Height > 0 && Width > 0, "dimensions must be positive");
    static_assert(Depth % 2 == 0 && Height % 2 == 0 && Width % 2 == 0, "dimensions must be even");
    static_assert(Channels > 0, "channels must be positive");
    static_assert(Height == 0 || Depth <= std::numeric_limits<std::size_t>::max()/Height, "shape overflow");
    static_assert(Width == 0 || Depth*Height <= std::numeric_limits<std::size_t>::max()/Width, "shape overflow");
public:
    static constexpr std::size_t InputVolumeSize = Depth*Height*Width;
    static constexpr std::size_t LatentDepth = Depth/2, LatentHeight = Height/2, LatentWidth = Width/2;
    static constexpr std::size_t CoarseVolumeSize = InputVolumeSize/8;
    static_assert(Channels == 0 || CoarseVolumeSize <= std::numeric_limits<std::size_t>::max()/Channels, "latent overflow");
    static constexpr std::size_t LatentVolumeSize = CoarseVolumeSize*Channels;
    static_assert(InputVolumeSize <= std::numeric_limits<std::size_t>::max()/sizeof(float), "input byte overflow");
    static_assert(LatentVolumeSize <= std::numeric_limits<std::size_t>::max()/sizeof(float), "latent byte overflow");
    struct LatentCode { alignas(64) std::array<float,LatentVolumeSize> data{}; bool is_quantized=false; };
    struct Reconstruction { alignas(64) std::array<float,InputVolumeSize> data{}; double reconstruction_mse=0; };
    explicit AVX512VolumeAutoencoderCodec(Backend requested=Backend::Auto):backend_(resolve(requested)){}
    Backend backend() const noexcept {return backend_;}
    LatentCode encode(const std::array<float,InputVolumeSize>& input) const {
        require_finite(input); LatentCode latent;
        if(backend_==Backend::AVX512) detail::pool_avx512(input.data(),latent.data.data(),Depth,Height,Width);
        else detail::pool_scalar(input.data(),latent.data.data(),Depth,Height,Width);
        // Preserve the proposal's repeated channel planes without recomputing them.
        for(std::size_t c=1;c<Channels;++c)
            std::copy_n(latent.data.begin(),CoarseVolumeSize,latent.data.begin()+static_cast<std::ptrdiff_t>(c*CoarseVolumeSize));
        return latent;
    }
    Reconstruction decode(const LatentCode& latent,const std::array<float,InputVolumeSize>& reference) const {
        if(latent.is_quantized) throw std::invalid_argument("quantized latent format is not implemented");
        require_finite(latent.data); require_finite(reference);
        std::vector<float> means(CoarseVolumeSize);
        for(std::size_t i=0;i<CoarseVolumeSize;++i){
            double sum=0;for(std::size_t c=0;c<Channels;++c)sum+=static_cast<double>(latent.data[c*CoarseVolumeSize+i]);
            means[i]=static_cast<float>(sum/static_cast<double>(Channels));
        }
        Reconstruction output;
        if(backend_==Backend::AVX512){
            detail::expand_avx512(means.data(),output.data.data(),Depth,Height,Width);
            output.reconstruction_mse=detail::mse_avx512(reference.data(),output.data.data(),InputVolumeSize);
        }else{
            detail::expand_scalar(means.data(),output.data.data(),Depth,Height,Width);
            output.reconstruction_mse=detail::mse_scalar(reference.data(),output.data.data(),InputVolumeSize);
        }
        return output;
    }
private:
    Backend backend_;
    template<std::size_t N> static void require_finite(const std::array<float,N>& values){
        for(float value:values)if(!std::isfinite(value))throw std::invalid_argument("non-finite voxel or latent");
    }
};
}
