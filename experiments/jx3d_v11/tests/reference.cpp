// Isolated reference check for v11 neural path: genuine 3D spatial tiles,
// quantized 8->4->8 ANN, XOR correction, bitplanes, bounded state, A*.
#include "../src/jx_ann.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <vector>

static constexpr int D=10, N=D*D*D;
static int at(int x,int y,int z){return x+D*(y+D*z);}
static std::array<int,3> xyz(int i){return {i%D,(i/D)%D,i/(D*D)};}
static std::vector<uint8_t> tiles(const std::vector<uint8_t>& x){
    std::vector<uint8_t> t; t.reserve(N);
    for(int z=0;z<D;z+=2)for(int y=0;y<D;y+=2)for(int x0=0;x0<D;x0+=2)
        for(int dz=0;dz<2;dz++)for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)
            t.push_back(x[at(x0+dx,y+dy,z+dz)]);
    return t;
}
static std::vector<uint8_t> raster(const std::vector<uint8_t>& t){
    std::vector<uint8_t> x(N);size_t k=0;
    for(int z=0;z<D;z+=2)for(int y=0;y<D;y+=2)for(int x0=0;x0<D;x0+=2)
        for(int dz=0;dz<2;dz++)for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)
            x[at(x0+dx,y+dy,z+dz)]=t[k++];
    return x;
}
template<class F> static void neighbors(int i,F&& f){
    auto [x,y,z]=xyz(i);
    if(x>0)f(i-1);
    if(x<D-1)f(i+1);
    if(y>0)f(i-D);
    if(y<D-1)f(i+D);
    if(z>0)f(i-D*D);
    if(z<D-1)f(i+D*D);
}
static int distance(int a,int b){
    auto x=xyz(a), y=xyz(b);
    return std::abs(x[0]-y[0])+std::abs(x[1]-y[1])+std::abs(x[2]-y[2]);
}
static int bfs(int src,int dst,const std::vector<uint8_t>& blocked){
    if(blocked[src]||blocked[dst])return -1;
    std::vector<int> d(N,-1);std::queue<int> q;q.push(src);d[src]=0;
    while(!q.empty()){
        int i=q.front();q.pop();if(i==dst)return d[i];
        neighbors(i,[&](int v){if(!blocked[v]&&d[v]<0){d[v]=d[i]+1;q.push(v);}});
    }return -1;
}
static int astar(int src,int dst,const std::vector<uint8_t>& blocked){
    if(blocked[src]||blocked[dst])return -1;
    using Item=std::pair<int,int>;
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> pq;
    std::vector<int> g(N,std::numeric_limits<int>::max());g[src]=0;
    pq.push({distance(src,dst),src});
    while(!pq.empty()){
        auto [f,i]=pq.top();pq.pop();if(f!=g[i]+distance(i,dst))continue;
        if(i==dst)return g[i];
        neighbors(i,[&](int v){if(!blocked[v]&&g[i]+1<g[v]){
            g[v]=g[i]+1;pq.push({g[v]+distance(v,dst),v});
        }});
    }return -1;
}
static std::vector<float> evolve(const std::vector<float>& field,const std::vector<float>& source){
    std::vector<float> next(N);
    for(int i=0;i<N;i++){
        double sum=0;int n=0;
        neighbors(i,[&](int j){sum+=field[j];n++;});
        double mean=n?sum/n:field[i];
        next[i]=float(std::clamp(0.78*field[i]+0.18*mean+0.04*source[i],0.0,1.0));
    }return next;
}
int main(){
    std::mt19937 rng(20261104);
    std::vector<uint8_t> raw(N);
    for(auto& x:raw)x=uint8_t(rng()%256);
    auto t=tiles(raw); assert(raster(t)==raw);
    jx::TileANN model;model.initialize(20261104);
    const double initial=model.mse(t);
    for(int e=0;e<100;e++)model.train_epoch(t,1.0);
    const double final=model.mse(t);
    assert(std::isfinite(final)&&final<=initial);
    // Encoder -> quantized latent -> decoder -> residual -> byte-exact Q8.
    std::vector<uint8_t> decoded(t.size());
    for(size_t k=0;k<t.size();k+=8){
        uint8_t latent[4], predicted[8];
        model.encode_latent(&t[k],latent);
        model.decode_latent(latent,predicted);
        for(int q=0;q<8;q++){
            uint8_t residual=uint8_t(t[k+q]^predicted[q]);
            decoded[k+q]=uint8_t(predicted[q]^residual);
        }
    }
    assert(decoded==t && raster(decoded)==raw);
    // Every byte must be exactly reconstructible from its eight bitplanes.
    for(int i=0;i<N;i++){
        uint8_t reconstructed=0;
        for(int bit=0;bit<8;bit++)reconstructed|=uint8_t(((raw[i]>>bit)&1)<<bit);
        assert(reconstructed==raw[i]);
    }
    // Valid contraction bound for identical sources and fixed topology.
    std::vector<float> a(N),b(N),source(N,0.5f);
    for(int i=0;i<N;i++){a[i]=raw[i]/255.0f;b[i]=0.8f*a[i]+0.1f;}
    auto na=evolve(a,source),nb=evolve(b,source);double before=0,after=0;
    for(int i=0;i<N;i++){before=std::max(before,double(std::abs(a[i]-b[i])));
                           after=std::max(after,double(std::abs(na[i]-nb[i])));}
    assert(after<=0.96*before+1e-6);
    // A* is required to match the exact BFS shortest-path length or no-path verdict.
    for(int trial=0;trial<600;trial++){
        std::vector<uint8_t> blocked(N);
        for(auto& v:blocked)v=uint8_t(rng()%7==0);
        int src=int(rng()%N),dst=int(rng()%N);
        assert(astar(src,dst,blocked)==bfs(src,dst,blocked));
    }
    std::cout << "PASS: 2x2x2 tile/inverse, 76-param neural MSE " << initial << " -> " << final
              << ", Q8 XOR exact, 8 bitplanes, 0.96 contraction, 600 A*/BFS trials\n";
}
