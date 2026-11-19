// Jarvis-X: 3D VNAND addressed quantized autoencoder reference VM.
// Research simulation ONLY. No physical NAND, analog MAC, or energy claim.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vnand {
constexpr std::uint32_t kPlanes=2, kBlocksPerPlane=512, kWordlines=48;
constexpr std::uint32_t kColumns=16384, kGridSide=128;
constexpr std::uint32_t kInput=784, kLatent=128;
constexpr std::uint64_t kCells=static_cast<std::uint64_t>(kPlanes)*kBlocksPerPlane*kWordlines*kColumns;
constexpr std::uint64_t kCapacityBytes=kCells*3/8;

struct Address {
    std::uint32_t plane, block, wordline, column;
    std::uint32_t x() const { return column%kGridSide; }
    std::uint32_t y() const { return column/kGridSide; }
    std::uint32_t z() const { return wordline; }
    std::uint64_t index() const {
        if(plane>=kPlanes || block>=kBlocksPerPlane || wordline>=kWordlines || column>=kColumns)
            throw std::out_of_range("VNAND coordinate outside geometry");
        return (((static_cast<std::uint64_t>(plane)*kBlocksPerPlane+block)*kWordlines+wordline)*kColumns+column);
    }
};

struct LayerSpec {
    std::uint32_t plane, first_wl, rows_per_wl, inputs, outputs;
    Address at(std::uint32_t i, std::uint32_t j) const {
        if(i>=inputs || j>=outputs) throw std::out_of_range("weight index");
        Address a{plane,0,first_wl+i/rows_per_wl,(i%rows_per_wl)*outputs+j};
        (void)a.index();
        return a;
    }
};
constexpr std::array<LayerSpec,4> kLayers{{
    {0,0,49,784,256}, {0,16,32,256,128},
    {1,0,16,128,256}, {1,8,16,256,784}
}};
constexpr std::uint64_t kWeights=784ULL*256+256ULL*128+128ULL*256+256ULL*784;

class SparseLevelStore {
    std::unordered_map<std::uint64_t,std::uint8_t> levels_;
public:
    void insert(Address a, std::uint8_t q) {
        if(q>7 || !levels_.emplace(a.index(),q).second)
            throw std::runtime_error("invalid or aliased VNAND initialization");
    }
    std::uint8_t read(Address a) const {
        auto it=levels_.find(a.index());
        if(it==levels_.end()) throw std::runtime_error("unallocated VNAND cell");
        return it->second;
    }
    void shadow_update(Address a, std::uint8_t q) {
        if(q>7) throw std::out_of_range("TLC code");
        auto it=levels_.find(a.index());
        if(it==levels_.end()) throw std::runtime_error("unallocated VNAND cell");
        it->second=q; // Software state, NOT a physical cell program/erase.
    }
    std::size_t size() const { return levels_.size(); }
};

struct Layer {
    LayerSpec spec;
    float scale;
    std::vector<float> master; // high-precision shadow weights for QAT
    std::vector<float> bias;
    std::size_t offset(std::uint32_t i,std::uint32_t j) const {
        return static_cast<std::size_t>(i)*spec.outputs+j;
    }
    std::uint8_t quantize(float w) const {
        if(!std::isfinite(w)) throw std::runtime_error("nonfinite weight");
        int q=static_cast<int>(std::lround(w/scale))+3;
        return static_cast<std::uint8_t>(std::clamp(q,0,7));
    }
    float dequantize(std::uint8_t q) const {
        return static_cast<float>(static_cast<int>(q)-3)*scale;
    }
};

struct Cache {
    std::array<std::vector<float>,5> activations;
    std::array<std::vector<float>,4> pre;
    std::size_t bus_elements=0;
};

enum class Op:std::uint8_t {Load,MatvecQ3,Relu,Transfer128,Sigmoid,Halt};
struct Instruction {Op opcode; std::uint8_t layer;};
constexpr std::array<Instruction,11> kProgram{{
    {Op::Load,0},{Op::MatvecQ3,0},{Op::Relu,0},
    {Op::MatvecQ3,1},{Op::Relu,1},{Op::Transfer128,0},
    {Op::MatvecQ3,2},{Op::Relu,2},{Op::MatvecQ3,3},
    {Op::Sigmoid,3},{Op::Halt,0}
}};

class VM {
    SparseLevelStore nand_;
    std::array<Layer,4> layers_;
    std::uint64_t program_up_=0, erase_down_=0, block_rewrite_requests_=0;
    static float sigmoid(float x) {
        if(x>=0) {float t=std::exp(-x); return 1.F/(1.F+t);}
        float t=std::exp(x); return t/(1.F+t);
    }
    void validate_input(const std::vector<float>& x) const {
        if(x.size()!=kInput) throw std::invalid_argument("expected 28x28 input");
        for(float v:x) if(!std::isfinite(v) || v<0.F || v>1.F)
            throw std::invalid_argument("input pixels must be finite in [0,1]");
    }
public:
    explicit VM(std::uint64_t seed=20261119) {
        std::mt19937_64 rng(seed);
        for(std::size_t l=0;l<layers_.size();++l) {
            auto& layer=layers_[l]; layer.spec=kLayers[l];
            float bound=std::sqrt(6.F/static_cast<float>(layer.spec.inputs+layer.spec.outputs));
            layer.scale=bound/3.F;
            layer.master.resize(static_cast<std::size_t>(layer.spec.inputs)*layer.spec.outputs);
            layer.bias.assign(layer.spec.outputs,0.F);
            std::uniform_real_distribution<float> dist(-bound,bound);
            for(std::uint32_t i=0;i<layer.spec.inputs;++i)
                for(std::uint32_t j=0;j<layer.spec.outputs;++j) {
                    std::size_t index=layer.offset(i,j);
                    layer.master[index]=dist(rng);
                    nand_.insert(layer.spec.at(i,j),layer.quantize(layer.master[index]));
                }
        }
        if(nand_.size()!=kWeights) throw std::logic_error("VNAND address alias detected");
    }
    Cache execute(const std::vector<float>& input) const {
        validate_input(input);
        Cache cache;
        std::vector<float> reg;
        bool loaded=false;
        std::size_t next_layer=0;
        for(const auto inst:kProgram) {
            switch(inst.opcode) {
            case Op::Load:
                if(loaded) throw std::logic_error("duplicate VM load");
                reg=input; cache.activations[0]=reg; loaded=true; break;
            case Op::MatvecQ3: {
                if(!loaded || inst.layer!=next_layer) throw std::logic_error("VM layer order");
                const auto& layer=layers_[inst.layer];
                if(reg.size()!=layer.spec.inputs) throw std::logic_error("VM dimension mismatch");
                std::vector<float> out=layer.bias;
                for(std::uint32_t i=0;i<layer.spec.inputs;++i) {
                    const float x=reg[i];
                    for(std::uint32_t j=0;j<layer.spec.outputs;++j) {
                        float w=layer.dequantize(nand_.read(layer.spec.at(i,j)));
                        out[j]+=x*w; // Digital MAC after idealized TLC read.
                    }
                }
                cache.pre[inst.layer]=out; reg=std::move(out); ++next_layer; break;
            }
            case Op::Relu:
                for(float& v:reg) v=std::max(0.F,v);
                cache.activations[static_cast<std::size_t>(inst.layer)+1]=reg; break;
            case Op::Transfer128:
                if(reg.size()!=kLatent || cache.activations[2]!=reg)
                    throw std::logic_error("latent bus contract");
                cache.bus_elements+=reg.size(); break;
            case Op::Sigmoid:
                for(float& v:reg) v=sigmoid(v);
                cache.activations[4]=reg; break;
            case Op::Halt:
                if(cache.activations[4].size()!=kInput || cache.bus_elements!=kLatent)
                    throw std::logic_error("incomplete VM program");
                return cache;
            }
        }
        throw std::logic_error("program missing HALT");
    }
    static float mse(const std::vector<float>& a,const std::vector<float>& b) {
        if(a.size()!=b.size() || a.empty()) throw std::invalid_argument("MSE shape");
        double sum=0.;
        for(std::size_t i=0;i<a.size();++i) {
            double d=static_cast<double>(a[i])-b[i]; sum+=d*d;
        }
        return static_cast<float>(sum/static_cast<double>(a.size()));
    }
    float train_step(const std::vector<float>& input,float learning_rate=0.5F,float lambda=1.0e-4F) {
        if(!std::isfinite(learning_rate)||learning_rate<=0.F||learning_rate>1.F||
           !std::isfinite(lambda)||lambda<0.F||lambda>1.F)
            throw std::invalid_argument("learning rate or L1 penalty out of range");
        const Cache c=execute(input);
        std::array<std::vector<float>,4> delta;
        for(std::size_t l=0;l<4;++l) delta[l].assign(layers_[l].spec.outputs,0.F);
        for(std::size_t j=0;j<kInput;++j) {
            float o=c.activations[4][j];
            delta[3][j]=2.F/static_cast<float>(kInput)*(o-input[j])*o*(1.F-o);
        }
        for(int l=3;l>=1;--l) {
            const Layer& layer=layers_[static_cast<std::size_t>(l)];
            for(std::uint32_t i=0;i<layer.spec.inputs;++i) {
                float total=0.F;
                for(std::uint32_t j=0;j<layer.spec.outputs;++j)
                    total+=layer.dequantize(nand_.read(layer.spec.at(i,j)))*delta[static_cast<std::size_t>(l)][j];
                if(c.pre[static_cast<std::size_t>(l-1)][i]<=0.F) total=0.F;
                if(l==2 && c.activations[2][i]>0.F)
                    total+=lambda/static_cast<float>(kLatent);
                delta[static_cast<std::size_t>(l-1)][i]=total;
            }
        }
        std::array<bool,kPlanes> needs_erase{{false,false}};
        for(std::size_t l=0;l<layers_.size();++l) {
            Layer& layer=layers_[l];
            for(std::uint32_t j=0;j<layer.spec.outputs;++j) {
                layer.bias[j]-=learning_rate*delta[l][j];
            }
            for(std::uint32_t i=0;i<layer.spec.inputs;++i)
                for(std::uint32_t j=0;j<layer.spec.outputs;++j) {
                    std::size_t ix=layer.offset(i,j);
                    Address a=layer.spec.at(i,j);
                    std::uint8_t old=nand_.read(a);
                    float grad=c.activations[l][i]*delta[l][j];
                    layer.master[ix]=std::clamp(layer.master[ix]-learning_rate*grad,
                        -3.F*layer.scale,4.F*layer.scale);
                    std::uint8_t now=layer.quantize(layer.master[ix]);
                    if(now>old) ++program_up_;
                    if(now<old) {++erase_down_; needs_erase[a.plane]=true;}
                    if(now!=old) nand_.shadow_update(a,now);
                }
        }
        for(bool b:needs_erase) if(b) ++block_rewrite_requests_;
        // A real NAND backend must erase entire affected block and restore other pages.
        return mse(input,c.activations[4]);
    }
    std::uint64_t up_requests() const {return program_up_;}
    std::uint64_t down_requests() const {return erase_down_;}
    std::uint64_t block_rewrite_requests() const {return block_rewrite_requests_;}
    std::size_t stored_weights() const {return nand_.size();}
};

std::vector<float> fixture() {
    std::vector<float> image(kInput);
    for(std::size_t y=0;y<28;++y) for(std::size_t x=0;x<28;++x)
        image[y*28+x]=(x>=7&&x<21&&y>=7&&y<21 && (x==7||x==20||y==7||y==20))?1.F:0.F;
    return image;
}

void self_test() {
    if(kCells!=805306368ULL || kCapacityBytes!=301989888ULL)
        throw std::runtime_error("TLC geometry calculation failed");
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(static_cast<std::size_t>(kWeights));
    for(const auto s:kLayers) {
        for(std::uint32_t i=0;i<s.inputs;++i) for(std::uint32_t j=0;j<s.outputs;++j) {
            auto a=s.at(i,j);
            if(a.x()>=128 || a.y()>=128 || a.z()>=48 || !seen.insert(a.index()).second)
                throw std::runtime_error("invalid/overlapping 3D weight mapping");
        }
    }
    if(seen.size()!=kWeights) throw std::runtime_error("missing mapped weights");
    try { (void)Address{2,0,0,0}.index(); throw std::runtime_error("missing bounds error"); }
    catch(const std::out_of_range&) {}
    VM a(42), b(42);
    auto x=fixture();
    auto first=a.execute(x), twin=b.execute(x);
    if(first.activations[4]!=twin.activations[4] || first.activations[2].size()!=kLatent ||
       first.bus_elements!=128 || a.stored_weights()!=kWeights)
        throw std::runtime_error("VM replay/shape/address contract failed");
    for(float v:first.activations[4]) if(!std::isfinite(v) || v<0.F || v>1.F)
        throw std::runtime_error("output not in [0,1]");
    bool rejected=false;
    try {a.execute(std::vector<float>(782,0.F));}
    catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected) throw std::runtime_error("invalid input dimension accepted");
    float before=VM::mse(x,first.activations[4]);
    for(int epoch=0;epoch<50;++epoch) {a.train_step(x);b.train_step(x);}
    float after=VM::mse(x,a.execute(x).activations[4]);
    if(!(after<before) || !std::isfinite(after) ||
       a.execute(x).activations[4]!=b.execute(x).activations[4])
        throw std::runtime_error("training/replay regression");
    std::cout<<"vnand-vm-self-test=PASS before="<<before<<" after="<<after<<"\n";
}
} // namespace vnand

int main(int argc,char** argv) {
    try {
        bool tests=false;
        int epochs=20;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--self-test") tests=true;
            else if(arg=="--epochs") {
                if(++i>=argc) throw std::invalid_argument("missing epoch count");
                std::string value=argv[i];
                if(value.empty()||value[0]=='-'||value[0]=='+') throw std::invalid_argument("bad epochs");
                std::size_t consumed=0;
                unsigned long n=std::stoul(value,&consumed,10);
                if(consumed!=value.size() || n>10000UL) throw std::invalid_argument("epochs out of bounds");
                epochs=static_cast<int>(n);
            } else if(arg=="--help" || arg=="-h") {
                std::cout<<"Usage: jarvisx-vnand-vm [--self-test] [--epochs 0..10000]\n"; return 0;
            } else throw std::invalid_argument("unknown argument: "+arg);
        }
        if(tests) {vnand::self_test(); return 0;}
        vnand::VM vm;
        auto x=vnand::fixture();
        float before=vnand::VM::mse(x,vm.execute(x).activations[4]);
        for(int i=0;i<epochs;++i) (void)vm.train_step(x);
        float after=vnand::VM::mse(x,vm.execute(x).activations[4]);
        std::cout<<std::setprecision(8)
                 <<"mode=software-reference no_physical_nand=1 no_analog_mac=1\n"
                 <<"geometry_cells="<<vnand::kCells<<" nominal_TLC_MiB="<<vnand::kCapacityBytes/(1024*1024)<<"\n"
                 <<"stored_weights="<<vm.stored_weights()<<" latent=128\n"
                 <<"epochs="<<epochs<<" mse_before="<<before<<" mse_after="<<after<<"\n"
                 <<"program_up_requests="<<vm.up_requests()
                 <<" erase_down_requests="<<vm.down_requests()
                 <<" block_rewrite_requests="<<vm.block_rewrite_requests()<<"\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"VNAND VM failure: "<<e.what()<<'\n';
        return 1;
    }
}