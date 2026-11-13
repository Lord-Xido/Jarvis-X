// Copyright (c) 2026 Dr Matladi Maxwell Moagi. See repository LICENSE.
#include "codec.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
std::string float_bits(float value){std::uint32_t bits;std::memcpy(&bits,&value,4);std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(8)<<bits;return s.str();}
int main(int argc,char** argv){try{
    if(argc!=2){std::cerr<<"Usage: codec_demo OUTPUT_DIRECTORY\n";return 2;}
    using Codec=jx::volume::AVX512VolumeAutoencoderCodec<6,4,34,1>;
    constexpr std::size_t D=6,H=4,W=34;std::array<float,Codec::InputVolumeSize> input{};
    for(std::size_t z=0;z<D;++z)for(std::size_t y=0;y<H;++y)for(std::size_t x=0;x<W;++x)
        input[z*H*W+y*W+x]=static_cast<float>(z)+0.1f*static_cast<float>(y)+0.01f*static_cast<float>(x);
    Codec codec;const auto latent=codec.encode(input);const auto output=codec.decode(latent,input);
    const std::filesystem::path directory=argv[1];std::filesystem::create_directories(directory);
    std::ofstream voxels(directory/"voxel_trace.csv"),blocks(directory/"block_trace.csv"),summary(directory/"summary.json");
    if(!voxels||!blocks||!summary)throw std::runtime_error("cannot create output files");
    voxels<<std::setprecision(17)<<"index,x,y,z,input_bits,reconstruction_bits,input,reconstruction,residual,squared_error\n";
    for(std::size_t z=0;z<D;++z)for(std::size_t y=0;y<H;++y)for(std::size_t x=0;x<W;++x){const auto i=z*H*W+y*W+x;
        const double residual=static_cast<double>(input[i])-output.data[i];
        voxels<<i<<','<<x<<','<<y<<','<<z<<','<<float_bits(input[i])<<','<<float_bits(output.data[i])<<','<<input[i]<<','<<output.data[i]<<','<<residual<<','<<residual*residual<<'\n';}
    blocks<<std::setprecision(17)<<"latent_index,x,y,z,mean_bits,mean\n";
    for(std::size_t z=0;z<D/2;++z)for(std::size_t y=0;y<H/2;++y)for(std::size_t x=0;x<W/2;++x){const auto i=(z*(H/2)+y)*(W/2)+x;
        blocks<<i<<','<<x<<','<<y<<','<<z<<','<<float_bits(latent.data[i])<<','<<latent.data[i]<<'\n';}
    summary<<std::setprecision(17)<<"{\n  \"backend\": \""<<jx::volume::backend_name(codec.backend())<<"\",\n  \"avx512_compiled\": "<<(jx::volume::avx512_compiled()?"true":"false")
        <<",\n  \"avx512_supported\": "<<(jx::volume::avx512_supported()?"true":"false")
        <<",\n  \"shape\": [6,4,34],\n  \"channels\": 1,\n  \"input_voxels\": "<<input.size()<<",\n  \"latent_floats\": "<<latent.data.size()
        <<",\n  \"input_payload_bytes\": "<<input.size()*sizeof(float)<<",\n  \"latent_payload_bytes\": "<<latent.data.size()*sizeof(float)
        <<",\n  \"float_payload_ratio\": 8,\n  \"mse\": "<<output.reconstruction_mse<<",\n  \"quantization_implemented\": false,\n  \"trained_model\": false\n}\n";
    voxels.flush();blocks.flush();summary.flush();if(!voxels||!blocks||!summary)throw std::runtime_error("cannot finish output files");
    std::cout<<"backend="<<jx::volume::backend_name(codec.backend())<<" voxels="<<input.size()<<" latent_floats="<<latent.data.size()<<" mse="<<std::setprecision(17)<<output.reconstruction_mse<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<"Error: "<<e.what()<<'\n';return 1;}}
