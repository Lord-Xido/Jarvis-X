// Copyright (c) 2026 Dr Matladi Maxwell Moagi. See repository LICENSE.
#include "codec.hpp"
#include <cstring>
#include <iostream>
#include <string>
using namespace jx::volume;
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
void near(double a,double b){require(std::abs(a-b)<=1e-12*std::max(1.0,std::abs(b)),"numeric mismatch");}
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"expected rejection");}

template<std::size_t D,std::size_t H,std::size_t W,std::size_t C>
void reference_case(){
    using Codec=AVX512VolumeAutoencoderCodec<D,H,W,C>;
    std::array<float,Codec::InputVolumeSize> input{};
    for(std::size_t i=0;i<input.size();++i)input[i]=static_cast<float>(static_cast<int>((i*37)%113)-56)*0.0625f;
    Codec scalar(Backend::Scalar);const auto latent=scalar.encode(input);const auto output=scalar.decode(latent,input);
    long double squared=0;
    for(std::size_t z=0;z<D/2;++z)for(std::size_t y=0;y<H/2;++y)for(std::size_t x=0;x<W/2;++x){
        long double sum=0;for(std::size_t dz=0;dz<2;++dz)for(std::size_t dy=0;dy<2;++dy)for(std::size_t dx=0;dx<2;++dx)
            sum+=input[(2*z+dz)*H*W+(2*y+dy)*W+2*x+dx];
        const float mean=static_cast<float>(sum/8);
        const auto key=(z*(H/2)+y)*(W/2)+x;
        for(std::size_t c=0;c<C;++c)require(latent.data[c*Codec::CoarseVolumeSize+key]==mean,"channel or block mean mismatch");
        for(std::size_t dz=0;dz<2;++dz)for(std::size_t dy=0;dy<2;++dy)for(std::size_t dx=0;dx<2;++dx){
            const auto i=(2*z+dz)*H*W+(2*y+dy)*W+2*x+dx;
            require(output.data[i]==mean,"block replication mismatch");
            const long double residual=static_cast<long double>(input[i])-output.data[i];squared+=residual*residual;
        }
    }
    near(output.reconstruction_mse,static_cast<double>(squared/static_cast<long double>(input.size())));
    Codec automatic;const auto auto_latent=automatic.encode(input);const auto auto_output=automatic.decode(auto_latent,input);
    require(auto_latent.data==latent.data && auto_output.data==output.data,"automatic dispatch changed state");
    if(avx512_supported()){
        Codec native(Backend::AVX512);const auto nl=native.encode(input);const auto nr=native.decode(nl,input);
        require(nl.data==latent.data && nr.data==output.data,"AVX-512/scalar voxel parity failed");near(nr.reconstruction_mse,output.reconstruction_mse);
    }
}
int main(){try{
    using Small=AVX512VolumeAutoencoderCodec<2,2,2,1>;Small codec(Backend::Scalar);std::array<float,8> input{};
    for(std::size_t i=0;i<8;++i)input[i]=static_cast<float>(i);
    const auto latent=codec.encode(input);const auto recon=codec.decode(latent,input);
    require(latent.data[0]==3.5f,"0..7 mean");near(recon.reconstruction_mse,5.25);
    std::cout<<"PASS analytic 2x2x2 ramp: mean 3.5; MSE 5.25\n";
    input.fill(2.5f);const auto constant=codec.decode(codec.encode(input),input);near(constant.reconstruction_mse,0);
    input.fill(0);input[0]=1;near(codec.decode(codec.encode(input),input).reconstruction_mse,0.109375);
    std::cout<<"PASS constants and single-voxel impulse\n";
    reference_case<2,2,2,1>();reference_case<4,6,16,2>();reference_case<6,4,18,3>();
    reference_case<4,4,32,1>();reference_case<6,4,34,1>();reference_case<4,4,66,8>();
    std::cout<<"PASS independent long-double oracle across six shapes/channel counts and vector tails\n";
    using Multi=AVX512VolumeAutoencoderCodec<2,2,2,2>;Multi multi(Backend::Scalar);Multi::LatentCode external;
    external.data[0]=2;external.data[1]=6;std::array<float,8> zeros{};const auto average=multi.decode(external,zeros);
    for(float value:average.data)require(value==4,"external channel averaging");
    near(average.reconstruction_mse,16);
    std::cout<<"PASS channel-mean decoder contract\n";
    auto invalid=input;invalid[0]=std::numeric_limits<float>::infinity();rejects([&]{codec.encode(invalid);});
    invalid[0]=std::numeric_limits<float>::quiet_NaN();rejects([&]{codec.encode(invalid);});
    auto bad=latent;bad.data[0]=std::numeric_limits<float>::infinity();rejects([&]{codec.decode(bad,input);});
    bad=latent;bad.is_quantized=true;rejects([&]{codec.decode(bad,input);});
    std::cout<<"PASS nonfinite input/latent and unsupported quantization rejection\n";
    input.fill(std::numeric_limits<float>::max());auto large=codec.decode(codec.encode(input),input);
    require(large.data[0]==input[0] && large.reconstruction_mse==0,"finite extreme input overflowed pooling");
    Small::LatentCode negative;negative.data.fill(-std::numeric_limits<float>::max());large=codec.decode(negative,input);
    require(std::isfinite(large.reconstruction_mse),"double MSE overflowed for float32 extremes");
    std::cout<<"PASS finite float32 extremes with double accumulators\n";
    float a[17]{},b[17]{};for(std::size_t i=0;i<17;++i){a[i]=static_cast<float>(i);b[i]=0.25f;}
    const auto mse=detail::mse_scalar(a,b,17);if(avx512_supported())near(detail::mse_avx512(a,b,17),mse);
    near(detail::mse_scalar(a,b,0),0);std::cout<<"PASS generic MSE tail and empty-kernel convention\n";
    require(resolve(Backend::Auto)==(avx512_supported()?Backend::AVX512:Backend::Scalar),"dispatch selection");
    if(!avx512_supported())rejects([]{Small unavailable(Backend::AVX512);});
    rejects([]{(void)resolve(static_cast<Backend>(99));});
    std::cout<<"PASS runtime selection and unavailable/invalid backend rejection\n";
    const auto replay1=codec.encode(input),replay2=codec.encode(input);require(replay1.data==replay2.data,"replay");
    std::cout<<"PASS repeatability\n";
    std::cout<<"AVX512_COMPILED="<<avx512_compiled()<<" AVX512_SUPPORTED="<<avx512_supported()<<'\n';
    std::cout<<(avx512_supported()?"PASS native AVX-512 parity executed\n":"SKIP native AVX-512 execution: CPU/build unavailable\n");
    std::cout<<"ALL 9 RUNTIME TEST GROUPS PASSED\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
