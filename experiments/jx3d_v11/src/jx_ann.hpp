#pragma once
// Deterministic shared-weight 8-4-8 neural encoder/decoder for spatial 2x2x2 tiles.
// A small learned predictor, not a large language model or physical neuromorphic device.
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace jx {
struct TileANN {
    static constexpr unsigned param_count = 76; // Wenc:4*8, benc:4, Wdec:8*4, bdec:8
    std::array<float,param_count> w{};
    void initialize(uint64_t seed) {
        for(unsigned i=0;i<param_count;i++) {
            w[i]=float(0.11*std::sin((i+1)*0.73 + double(seed%100003)*0.00013));
        }
        for(unsigned k=0;k<8;k++) w[68+k]=0.0f;
        for(unsigned k=0;k<4;k++) w[32+k]=0.0f;
    }
    static float sigmoid(double z) {
        if(z>35) return 1.0f;
        if(z< -35) return 0.0f;
        return float(1.0/(1.0+std::exp(-z)));
    }
    void forward(const float* x,float* h,float* y) const {
        for(int a=0;a<4;a++) {
            double z=w[32+a];
            for(int k=0;k<8;k++)z+=double(w[a*8+k])*x[k];
            h[a]=float(std::tanh(z));
        }
        for(int k=0;k<8;k++) {
            double z=w[68+k];
            for(int a=0;a<4;a++)z+=double(w[36+k*4+a])*h[a];
            y[k]=sigmoid(z);
        }
    }
    void predict(const uint8_t* tile,uint8_t* predicted,float* latent=nullptr) const {
        float x[8],h[4],y[8];
        for(int k=0;k<8;k++)x[k]=tile[k]/255.0f;
        forward(x,h,y);
        for(int k=0;k<8;k++)predicted[k]=uint8_t(std::lround(std::clamp(y[k],0.0f,1.0f)*255));
        if(latent)for(int a=0;a<4;a++) latent[a]=h[a];
    }
    void encode_latent(const uint8_t* tile,uint8_t* qlatent) const {
        float x[8],h[4],y[8];
        for(int k=0;k<8;k++)x[k]=tile[k]/255.0f;
        forward(x,h,y);
        for(int a=0;a<4;a++) qlatent[a]=uint8_t(std::lround(std::clamp((double(h[a])+1.0)*127.5,0.0,255.0)));
    }
    void decode_latent(const uint8_t* qlatent,uint8_t* prediction) const {
        double h[4];
        for(int a=0;a<4;a++)h[a]=double(qlatent[a])/127.5-1.0;
        for(int k=0;k<8;k++) {
            double z=w[68+k];
            for(int a=0;a<4;a++)z+=double(w[36+k*4+a])*h[a];
            prediction[k]=uint8_t(std::lround(sigmoid(z)*255.0f));
        }
    }
    double mse(const std::vector<uint8_t>& spatial_tiles) const {
        if(spatial_tiles.empty()||spatial_tiles.size()%8)return 0;
        double loss=0;
        for(size_t i=0;i<spatial_tiles.size();i+=8){
            float x[8],h[4],y[8];
            for(int k=0;k<8;k++) x[k]=spatial_tiles[i+k]/255.0f;
            forward(x,h,y);
            for(int k=0;k<8;k++) {double d=y[k]-x[k];loss+=d*d;}
        }
        return loss/spatial_tiles.size();
    }
    // Full-batch analytical reverse-mode differentiation with non-increasing loss line search.
    // Returns whether a parameter update was accepted.
    bool train_epoch(const std::vector<uint8_t>& tiled, double learning_rate) {
        if(tiled.empty()||tiled.size()%8 || !(learning_rate>0) || !std::isfinite(learning_rate))return false;
        std::array<double,param_count> grad{};
        const double norm=2.0/double(tiled.size());
        for(size_t i=0;i<tiled.size();i+=8) {
            float x[8],h[4],y[8];
            for(int k=0;k<8;k++) x[k]=tiled[i+k]/255.0f;
            forward(x,h,y);
            double dzout[8];
            for(int k=0;k<8;k++) {
                dzout[k]=norm*(y[k]-x[k])*double(y[k])*(1.0-double(y[k]));
                grad[68+k]+=dzout[k];
                for(int a=0;a<4;a++)grad[36+k*4+a]+=dzout[k]*h[a];
            }
            for(int a=0;a<4;a++) {
                double dh=0;
                for(int k=0;k<8;k++)dh+=dzout[k]*w[36+k*4+a];
                const double dz=dh*(1.0-double(h[a])*h[a]);
                grad[32+a]+=dz;
                for(int k=0;k<8;k++)grad[a*8+k]+=dz*x[k];
            }
        }
        const auto original=w;
        const double before=mse(tiled);
        double rate=std::min(learning_rate,10.0);
        for(int attempt=0;attempt<16;attempt++,rate*=0.5) {
            for(unsigned k=0;k<param_count;k++) w[k]=float(std::clamp(double(original[k])-rate*grad[k],-20.0,20.0));
            const double after=mse(tiled);
            if(std::isfinite(after) && after <= before && (after<before || attempt==0))return true;
        }
        w=original;
        return false;
    }
};
}
