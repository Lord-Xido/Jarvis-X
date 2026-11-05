#pragma once
// Imported after the Morton mapping and scalar weight definitions in namespace jx.
// Deterministic per-block 3->8->1 trained MLP; Q6 int8 weights, int16 byte biases.
constexpr int NH = 8;
struct NeuralWeights {
    std::array<std::int8_t, NH*3> w1{};
    std::array<std::int16_t, NH> b1{};
    std::array<std::int8_t, NH> w2{};
    std::int16_t b2=0;
};
inline int roundQ6(int v) { return v >= 0 ? (v+32)/64 : -((-v+32)/64); }
inline U8 neuralPredict(int a,int b,int c,const NeuralWeights& nn) {
    const int x[3]={a,b,c};int total=0;
    for(int j=0;j<NH;++j){int sum=0;
        for(int i=0;i<3;++i)sum+=int(nn.w1[3*j+i])*x[i];
        const int h=std::clamp(roundQ6(sum)+int(nn.b1[j]),0,255);
        total+=int(nn.w2[j])*h;
    }
    return U8(std::clamp(roundQ6(total)+int(nn.b2),0,255));
}
// In pixel mode the 3D coordinates are (x pixel, y pixel, z RGBA channel).
// Access is causal and confined to the current chunk; edges use zero halos.
struct SpatialContext {bool pixels=false; std::size_t rowBytes=0; U64 globalBase=0;};
inline std::array<int,3> pixelNeighbors(const U8* prefix,std::size_t i,const SpatialContext& ctx){
    const U64 global=ctx.globalBase+i;
    const int left=(i>=4 && global%ctx.rowBytes>=4)?prefix[i-4]:0;
    const int above=(i>=ctx.rowBytes)?prefix[i-ctx.rowBytes]:0;
    const int channel=(i>0 && global%4!=0)?prefix[i-1]:0;
    return {left,above,channel};
}
struct TrainingExample {std::array<double,3> x{}; double y=0;};
inline std::vector<TrainingExample> trainingSamples(const U8* src,std::size_t n){
    std::vector<TrainingExample> data;data.reserve((n+15)/16);
    std::array<U8,C> voxel{};const auto& m=mapping();
    for(std::size_t k=0;k<n;++k){const XYZ p=m.xyz[k];
        if(k%16==0){
            const int a=p.x?voxel[(p.z*32+p.y)*32+p.x-1]:0;
            const int b=p.y?voxel[(p.z*32+p.y-1)*32+p.x]:0;
            const int c=p.z?voxel[((p.z-1)*32+p.y)*32+p.x]:0;
            data.push_back({{double(a)/255.0,double(b)/255.0,double(c)/255.0},double(src[k])/255.0});
        }
        voxel[m.idx[k]]=src[k];
    }
    return data;
}
inline std::vector<TrainingExample> trainingSamplesPixels(const U8* src,std::size_t n,SpatialContext ctx){
    std::vector<TrainingExample> d;d.reserve((n+15)/16);
    for(std::size_t i=0;i<n;++i){
        // Round-robin across R,G,B,A; sample 1 of every 16 byte positions.
        if(i%16 == (i/16)%4){
            const auto x=pixelNeighbors(src,i,ctx);
            d.push_back({{double(x[0])/255.0,double(x[1])/255.0,double(x[2])/255.0},double(src[i])/255.0});
        }
    }
    return d;
}
struct FloatMLP {
    std::array<double,NH*3> w1{};
    std::array<double,NH> b1{};
    std::array<double,NH> w2{};
    double b2=0;
    FloatMLP(){
        for(int j=0;j<3;++j){w1[3*j+j]=0.95;b1[j]=0.015;}
        for(int j=3;j<NH;++j){
            for(int i=0;i<3;++i)w1[3*j+i]=0.15*std::sin(double(j*7+i*13));
            b1[j]=0.15;
        }
        w2[0]=0.55;w2[1]=0.25;w2[2]=0.20;
        for(int j=3;j<NH;++j)w2[j]=0.05*std::cos(double(j*11));
        b2=0.02;
    }
    double forward(const std::array<double,3>& x,std::array<double,NH>* hidden=nullptr) const{
        std::array<double,NH> h{};double y=b2;
        for(int j=0;j<NH;++j){double t=b1[j];
            for(int i=0;i<3;++i)t+=w1[3*j+i]*x[i];
            h[j]=std::clamp(t,0.0,1.0);y+=w2[j]*h[j];
        }
        if(hidden)*hidden=h;
        return y;
    }
    double loss(const std::vector<TrainingExample>& data)const{
        double sum=0;
        for(const auto& e:data){const double d=forward(e.x)-e.y;sum+=d*d;}
        return sum/double(std::max<std::size_t>(1,data.size()));
    }
    void train(const std::vector<TrainingExample>& data){
        if(data.empty())return;
        FloatMLP best=*this;
        double bestLoss=loss(data);
        for(int epoch=0;epoch<3;++epoch){const double lr=0.025/(1.0+0.5*epoch);
            for(const auto& e:data){std::array<double,NH> h{};
                const double raw=forward(e.x,&h);
                const double error=std::clamp(raw-e.y,-2.0,2.0);
                const auto oldW2=w2;
                for(int j=0;j<NH;++j)w2[j]=std::clamp(w2[j]-lr*error*h[j],-1.95,1.95);
                b2=std::clamp(b2-lr*error,-1.5,1.5);
                for(int j=0;j<NH;++j){
                    if(h[j]>0.0&&h[j]<1.0){const double g=error*oldW2[j];
                        for(int i=0;i<3;++i)
                            w1[3*j+i]=std::clamp(w1[3*j+i]-lr*g*e.x[i],-1.95,1.95);
                        b1[j]=std::clamp(b1[j]-lr*g,-1.5,1.5);
                    }
                }
            }
            const double err=loss(data);
            if(err<bestLoss){best=*this;bestLoss=err;}
        }
        *this=best;
    }
    NeuralWeights quantize()const{
        NeuralWeights nn{};
        const auto qw=[](double x){return std::int8_t(std::clamp(std::lround(x*64),-127L,127L));};
        const auto qb=[](double x){return std::int16_t(std::clamp(std::lround(x*255),-32768L,32767L));};
        for(int j=0;j<NH;++j){
            for(int i=0;i<3;++i)nn.w1[3*j+i]=qw(w1[3*j+i]);
            nn.b1[j]=qb(b1[j]);nn.w2[j]=qw(w2[j]);
        }
        nn.b2=qb(b2);return nn;
    }
};
inline NeuralWeights learnNeural(const U8* src,std::size_t n,SpatialContext ctx={}){
    const auto data=ctx.pixels?trainingSamplesPixels(src,n,ctx):trainingSamples(src,n);
    FloatMLP m;m.train(data);return m.quantize();
}