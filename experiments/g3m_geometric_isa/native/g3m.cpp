// G3M: bounded 3D point-cloud variational autoencoder, LibTorch/C++17.
// Proof of computational execution, NOT proof of million-fold runtime speedup.
// Input [B,N,3] -> 512D stochastic latent -> reconstructed [B,M,3].
// Source cloud can contain 1M points, but attention operates on a bounded sample.
#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace g3m {
constexpr double PI = 3.14159265358979323846;
using Tensor = torch::Tensor;

struct Normalized {
    Tensor value, center, radius;
};

Normalized normalize(const Tensor& p) {
    TORCH_CHECK(p.dim() == 3 && p.size(2) == 3 && p.size(1) >= 2, "Expected [B,N,3] with N>=2");
    auto center = p.mean(1, true);                       // [B,1,3]
    auto centered = p - center;                         // [B,N,3]
    auto radius = std::get<0>(centered.norm(2, 2).max(1, true)).clamp_min(1e-6).unsqueeze(-1);
    return {centered / radius, center, radius};        // [B,N,3], [B,1,3], [B,1,1]
}

// O(S) index generation and O(S) gather, no 1M-by-1M distance matrix.
// This is deterministic pseudopermutation sampling, not learned curvature sampling.
Tensor support_indices(int64_t n, int64_t sample, torch::Device device) {
    TORCH_CHECK(n >= 2 && sample >= 2 && sample <= n, "Invalid support size");
    int64_t stride = n / 2 + 1;
    while (std::gcd(stride, n) != 1) ++stride;
    std::vector<int64_t> indices(static_cast<size_t>(sample));
    const uint64_t offset = 12345 % static_cast<uint64_t>(n);
    for (int64_t i=0; i<sample; ++i) {
        // Arithmetic deliberately uses uint64_t to avoid signed overflow.
        indices[static_cast<size_t>(i)] = static_cast<int64_t>(
            (offset + static_cast<uint64_t>(i) * static_cast<uint64_t>(stride)) % static_cast<uint64_t>(n));
    }
    return torch::from_blob(indices.data(), {sample}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(device);
}

Tensor nearest_neighbors(const Tensor& points, int64_t k) {
    const auto n = points.size(1);
    TORCH_CHECK(k >= 1 && k < n, "k must be between 1 and S-1");
    auto distances = torch::cdist(points, points);      // [B,S,S], bounded S
    auto diag = torch::eye(n, points.options().dtype(torch::kBool)).unsqueeze(0);
    distances.masked_fill_(diag, 1e9);                // exclude self-neighbors
    return std::get<1>(distances.topk(k, 2, false, true));
}

Tensor gather_neighbors(const Tensor& points, const Tensor& idx) {
    auto b=points.size(0), s=points.size(1), c=points.size(2), k=idx.size(2);
    auto offsets=torch::arange(b, idx.options()).view({b,1,1}) * s;
    auto flat=(idx + offsets).reshape({-1});
    return points.reshape({b*s,c}).index_select(0,flat).reshape({b,s,k,c});
}

Tensor squared_chamfer(const Tensor& a, const Tensor& b) {
    // Exact all-pair Chamfer for bounded training subsets, not a full N=1M cloud.
    TORCH_CHECK(a.size(0) == b.size(0), "Batch mismatch");
    auto d2=torch::cdist(a,b).pow(2);
    return std::get<0>(d2.min(2)).mean() + std::get<0>(d2.min(1)).mean();
}

// O(B*S^2) geometry-aware attention using Euclidean distance proxy.
// This is NOT an intrinsic geodesic distance solver.
struct G3M : torch::nn::Module {
    torch::nn::Linear edge1{nullptr},edge2{nullptr};
    torch::nn::Linear q{nullptr},k{nullptr},v{nullptr};
    torch::nn::Linear fuse{nullptr},mu{nullptr},logvar{nullptr};
    torch::nn::Linear latent_to_64{nullptr},fold1{nullptr},fold2{nullptr},fold3{nullptr};
    int64_t k_neigh = 12;
    int64_t latent_size = 512;
    double geo_strength = 0.5;

    G3M() {
        edge1=register_module("edge1",torch::nn::Linear(6,64));
        edge2=register_module("edge2",torch::nn::Linear(64,64));
        q=register_module("q",torch::nn::Linear(64,64));
        k=register_module("k",torch::nn::Linear(64,64));
        v=register_module("v",torch::nn::Linear(64,64));
        fuse=register_module("fuse",torch::nn::Linear(128,128));
        mu=register_module("mu",torch::nn::Linear(128,512));
        logvar=register_module("logvar",torch::nn::Linear(128,512));
        latent_to_64=register_module("latent_to_64",torch::nn::Linear(512,64));
        fold1=register_module("fold1",torch::nn::Linear(66,64));
        fold2=register_module("fold2",torch::nn::Linear(64,64));
        fold3=register_module("fold3",torch::nn::Linear(64,3));
    }

    // Encoder: [B,S,3] -> z[ B,512 ]; KL from Gaussian prior.
    std::tuple<Tensor,Tensor,Tensor> encode(const Tensor& points, bool stochastic=true) {
        const auto s=points.size(1);
        auto idx=nearest_neighbors(points,std::min(k_neigh,s-1));
        auto pj=gather_neighbors(points,idx);           // [B,S,K,3]
        auto pi=points.unsqueeze(2).expand_as(pj);
        auto edges=torch::cat({pi,pj-pi},-1);           // [B,S,K,6]
        auto h=torch::relu(edge1->forward(edges));
        h=torch::relu(edge2->forward(h));
        h=std::get<0>(h.max(2));                        // [B,S,64]
        auto query=q->forward(h);
        auto key=k->forward(h);
        auto val=v->forward(h);
        auto spatial=torch::cdist(points,points);       // [B,S,S]
        auto scores=torch::matmul(query,key.transpose(1,2))/std::sqrt(64.0) - geo_strength*spatial;
        auto weights=torch::softmax(scores,-1);
        h=h + torch::matmul(weights,val);               // [B,S,64]
        auto global=torch::cat({std::get<0>(h.max(1)),h.mean(1)},-1);
        global=torch::relu(fuse->forward(global));
        auto m=mu->forward(global);                      // [B,512]
        auto lv=logvar->forward(global).clamp(-8.0,5.0);// [B,512]
        auto raw=stochastic ? (m + torch::exp(0.5*lv)*torch::randn_like(m)) : m;
        // Norm bounded to <1; this projection alone is not hyperbolic learning.
        auto norm=raw.norm(2,1,true).clamp_min(1e-8);
        auto z=torch::tanh(norm)*(raw/norm);            // [B,512]
        return {z,m,lv};
    }

    Tensor decode_chunk(const Tensor& z, int64_t start, int64_t count, int64_t total) {
        TORCH_CHECK(count>0 && total>=count && start>=0 && start+count<=total, "Invalid decode chunk");
        auto b=z.size(0);
        const int64_t side=static_cast<int64_t>(std::ceil(std::sqrt(static_cast<double>(total))));
        auto n=torch::arange(start,start+count,z.options().dtype(torch::kFloat));
        auto x=(n - torch::floor(n/static_cast<double>(side))*static_cast<double>(side)) /
               static_cast<double>(std::max<int64_t>(side-1,1));
        auto y=torch::floor(n/static_cast<double>(side)) /
               static_cast<double>(std::max<int64_t>(side-1,1));
        auto uv=torch::stack({2*x-1,2*y-1},-1).unsqueeze(0).expand({b,count,2});
        auto lat=torch::relu(latent_to_64->forward(z)).unsqueeze(1).expand({b,count,64});
        auto inp=torch::cat({uv,lat},-1);                // [B,count,66]
        auto h=torch::relu(fold1->forward(inp));
        h=torch::relu(fold2->forward(h));
        return torch::tanh(fold3->forward(h));          // [B,count,3]
    }

    Tensor decode(const Tensor& z, int64_t total, int64_t chunk=1024) {
        TORCH_CHECK(total>=2 && chunk>=1, "Invalid decoder size");
        std::vector<Tensor> outs;
        for (int64_t i=0;i<total;i+=chunk)
            outs.push_back(decode_chunk(z,i,std::min(chunk,total-i),total));
        return torch::cat(outs,1);                     // [B,total,3]
    }

    std::tuple<Tensor,Tensor,Tensor> forward(const Tensor& support) {
        Tensor z,m,lv;
        std::tie(z,m,lv)=encode(support,true);
        auto output=decode(z,support.size(1),support.size(1));
        auto kl=-0.5*(1+lv-m.pow(2)-lv.exp()).sum(1).mean();
        return {output,kl,z};
    }
};

Tensor make_torus(int64_t n, torch::Device dev) {
    auto opts=torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto u=torch::rand({1,n},opts)*(2*PI), v=torch::rand({1,n},opts)*(2*PI);
    auto rr=1.0+0.35*torch::cos(v);
    return torch::stack({rr*torch::cos(u),rr*torch::sin(u),0.35*torch::sin(v)},2);
}

int64_t flag(int argc, char** argv, const std::string& f, int64_t fallback) {
    for(int i=1;i+1<argc;i++)if(argv[i]==f)return std::stoll(argv[i+1]);
    return fallback;
}

std::string strflag(int argc, char** argv, const std::string& f, const std::string& fallback="") {
    for(int i=1;i+1<argc;i++)if(argv[i]==f)return argv[i+1];
    return fallback;
}

// Little-endian host float32 XYZ array, no header. One cloud [1,N,3].
Tensor read_xyz(const std::string& path) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if (!in) throw std::runtime_error("Failed to open input: "+path);
    auto len=in.tellg();
    if (len<static_cast<std::streampos>(24) || static_cast<uint64_t>(len)%(3*sizeof(float))!=0)
        throw std::runtime_error("Input must contain complete float32 XYZ triples");
    auto n=static_cast<int64_t>(len)/(3*static_cast<int64_t>(sizeof(float)));
    if (n>1000000) throw std::runtime_error("Input exceeds 1M point bounded reference");
    std::vector<float> raw(static_cast<size_t>(3*n));
    in.seekg(0,std::ios::beg);
    if (!in.read(reinterpret_cast<char*>(raw.data()),static_cast<std::streamsize>(len)))
        throw std::runtime_error("Input truncated");
    auto result=torch::from_blob(raw.data(),{1,n,3},torch::TensorOptions().dtype(torch::kFloat32)).clone();
    TORCH_CHECK(torch::isfinite(result).all().item<bool>(),"Non-finite point input");
    return result;
}

void save_xyz(const std::string& path, const Tensor& points) {
    TORCH_CHECK(points.device().is_cpu() && points.dtype()==torch::kFloat32 && points.dim()==3 && points.size(2)==3,
                "Expected CPU float32 XYZ tensor");
    auto c=points.contiguous();
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    if(!out) throw std::runtime_error("Cannot open output: "+path);
    out.write(reinterpret_cast<const char*>(c.data_ptr<float>()),
              static_cast<std::streamsize>(c.numel()*sizeof(float)));
    if (!out.good()) throw std::runtime_error("Write failed: "+path);
}

// Candidate-first latent recurrence F(z)=E(D(z)), accepted only if self-gap
// improves and squared Chamfer against the sampled source does not degrade >1%.
// This is an empirical guard, NOT a proof of contraction or semantic correctness.
Tensor guarded_refine(const std::shared_ptr<G3M>& model, Tensor z,
                      const Tensor& support,int64_t max_steps) {
    if(max_steps<=0) return z;
    auto points=static_cast<int64_t>(support.size(1));
    for(int64_t i=0;i<max_steps;++i) {
        auto pred=model->decode(z,points,points);
        auto feedback=std::get<0>(model->encode(pred,false));
        const double gap=(feedback-z).norm(2,1).mean().item<double>();
        const double geo=squared_chamfer(pred,support).item<double>();
        auto candidate=0.5*z+0.5*feedback;
        auto pred2=model->decode(candidate,points,points);
        auto feedback2=std::get<0>(model->encode(pred2,false));
        const double gap2=(feedback2-candidate).norm(2,1).mean().item<double>();
        const double geo2=squared_chamfer(pred2,support).item<double>();
        bool accept=std::isfinite(gap2)&&std::isfinite(geo2)&&gap2 < gap && geo2 <= geo*1.01;
        std::cout<<"refine="<<(i+1)<<" latent_gap="<<gap<<" candidate_gap="<<gap2
                 <<" chamfer="<<geo<<" candidate_chamfer="<<geo2
                 <<" admitted="<<(accept?1:0)<<"\n";
        if(!accept) break;
        z=candidate;
    }
    return z;
}

} // namespace g3m

int main(int argc,char** argv) {
    try {
        torch::manual_seed(19);
        torch::set_num_threads(2);
        const auto input_file=g3m::strflag(argc,argv,"--input");
        const auto output_file=g3m::strflag(argc,argv,"--output");
        const auto weights_file=g3m::strflag(argc,argv,"--weights");
        const auto save_weights=g3m::strflag(argc,argv,"--save-weights","g3m_weights.pt");
        const int64_t requested_n=g3m::flag(argc,argv,"--points",8192);
        const int64_t support_n=g3m::flag(argc,argv,"--support",128);
        const int64_t epochs=g3m::flag(argc,argv,"--epochs",10);
        const int64_t out_n=g3m::flag(argc,argv,"--decode",512);
        const int64_t refine=g3m::flag(argc,argv,"--refine",2);
        TORCH_CHECK(requested_n>=2 && requested_n<=1000000 && support_n>=16 && support_n<=1024 &&
                    epochs>=0 && epochs<=100000 && out_n>=2 && out_n<=1000000 && refine>=0 && refine<=32,
                    "Parameter limit exceeded");
        const torch::Device device(torch::kCPU);
        const auto t0=std::chrono::steady_clock::now();
        auto original=input_file.empty()?g3m::make_torus(requested_n,device):g3m::read_xyz(input_file);
        const auto n=original.size(1);
        TORCH_CHECK(n>=support_n,"Insufficient points for selected support");
        auto normalized=g3m::normalize(original);
        auto idx=g3m::support_indices(n,support_n,device);
        auto support=normalized.value.index_select(1,idx);
        auto model=std::make_shared<g3m::G3M>();
        model->to(device);
        double first=std::numeric_limits<double>::quiet_NaN();
        double last=std::numeric_limits<double>::quiet_NaN();
        if(!weights_file.empty()) {
            torch::load(model,weights_file);
            std::cout<<"loaded_checkpoint="<<weights_file<<"\n";
        } else {
            TORCH_CHECK(epochs>0,"--epochs must be positive without --weights");
            torch::optim::Adam optimizer(model->parameters(),torch::optim::AdamOptions(0.002));
            for(int64_t epoch=0;epoch<epochs;++epoch){
                model->train();
                optimizer.zero_grad();
                g3m::Tensor pred,kl,z;
                std::tie(pred,kl,z)=model->forward(support);
                auto loss=g3m::squared_chamfer(pred,support)+0.0001*kl;
                loss.backward();
                optimizer.step();
                last=loss.item<double>();
                if(epoch==0)first=last;
                if(epoch==0 || epoch==epochs-1 || (epoch+1)%10==0)
                    std::cout<<"epoch="<<(epoch+1)<<" loss="<<last<<" kl="<<kl.item<double>()<<"\n";
            }
            torch::save(model,save_weights);
            std::cout<<"checkpoint="<<save_weights<<"\n";
        }
        model->eval();
        torch::NoGradGuard guard;
        g3m::Tensor z,m,lv;
        std::tie(z,m,lv)=model->encode(support,false);
        z=g3m::guarded_refine(model,z,support,refine);
        auto reconstructed=model->decode(z,out_n,1024);
        auto z_norm=z.norm(2,1).max().item<double>();
        TORCH_CHECK(reconstructed.size(1)==out_n && z.size(1)==512 && torch::isfinite(reconstructed).all().item<bool>(),
                    "Invalid decoder output");
        if(!output_file.empty()){
            // Undo input normalization to preserve the input's coordinate system.
            g3m::save_xyz(output_file,reconstructed*normalized.radius+normalized.center);
            std::cout<<"output_xyz_f32="<<output_file<<"\n";
        }
        const auto t1=std::chrono::steady_clock::now();
        std::cout<<"INPUT "<<n<<"x3, SUPPORT "<<support_n<<"x3, LATENT "<<z.size(1)
                 <<", RECONSTRUCTION "<<reconstructed.size(1)<<"x3\n";
        std::cout<<"latent_norm="<<z_norm<<" start_loss="<<first<<" end_loss="<<last<<"\n";
        std::cout<<"seconds="<<std::chrono::duration<double>(t1-t0).count()<<"\n";
        return 0;
    }catch(const c10::Error& e){std::cerr<<"TORCH ERROR: "<<e.what()<<"\n";return 1;}
     catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
