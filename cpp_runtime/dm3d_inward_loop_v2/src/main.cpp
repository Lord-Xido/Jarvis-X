#include "dm3d/core.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
unsigned arg_uint(const std::string& v,unsigned max){
    auto n=dm3d::parse_u128(v);
    if(n>max)throw std::runtime_error("numeric option exceeds maximum");
    return unsigned(n);
}
void usage(){
    std::cout << "Dr Moagi DM3D v2.0 — sparse toroidal C++20 codec & agent orchestrator\n\n"
      << "Commands:\n"
      << "  dm3d loop     INPUT ARCHIVE RESTORED [--threads N] [--folds K] [--passes P]\n"
      << "  dm3d pipeline INPUT ARCHIVE RESTORED [--threads N] [--folds K] [--passes P]\n"
      << "  dm3d encode   INPUT ARCHIVE [--threads N] [--folds K] [--passes P]\n"
      << "  dm3d decode   ARCHIVE OUTPUT\n"
      << "  dm3d verify   ARCHIVE ORIGINAL\n"
      << "  dm3d inspect  INPUT_OR_ARCHIVE\n"
      << "  dm3d coord    X Y Z\n"
      << "  dm3d uncoord  128BIT_LINEAR_INDEX\n"
      << "  dm3d voxel-set VOLUME X Y Z BYTE\n"
      << "  dm3d voxel-get VOLUME X Y Z\n"
      << "  dm3d selftest\n\n"
      << "1000 GB interpreted as 10^12 logical address positions per axis.\n"
      << "Only touched 16x16x16 tiles are stored; archive preserves bytes exactly.\n";
}
struct Options {unsigned threads=4,folds=8,passes=5;};
Options options(int argc,char** argv,int start){
    Options o;
    for(int i=start;i<argc;i++){
        std::string a=argv[i];
        if(a=="--threads"&&i+1<argc)o.threads=arg_uint(argv[++i],128);
        else if(a=="--folds"&&i+1<argc)o.folds=arg_uint(argv[++i],1000);
        else if(a=="--passes"&&i+1<argc)o.passes=arg_uint(argv[++i],5);
        else throw std::runtime_error("unknown option or missing value: "+a);
    }
    return o;
}
dm3d::Coord parse_coord(char** argv,int from){
    dm3d::Coord c{};uint64_t* v[]={&c.x,&c.y,&c.z};
    for(int i=0;i<3;i++){auto n=dm3d::parse_u128(argv[from+i]);
        if(n>=dm3d::AXIS)throw std::runtime_error("coordinate out of bounds");
        *v[i]=uint64_t(n);
    }return c;
}
void print_archive(const dm3d::ArchiveInfo& h){
    std::cout<<"modality="<<dm3d::modality_name(h.modality)
       <<" bytes="<<h.original_size<<" tiles="<<h.tiles
       <<" archive_bytes="<<h.stored_size
       <<" FNV64="<<std::hex<<std::setw(16)<<std::setfill('0')<<h.file_hash
       <<std::dec<<std::setfill(' ')<<'\n';
}
void agent(const std::string& name,const std::string& msg){
    std::cout<<"[AGENT "<<name<<"] "<<msg<<'\n';
}
}
int main(int argc,char** argv){
    try{
        if(argc<2){usage();return 0;}
        std::string cmd=argv[1];
        if(cmd=="--help"||cmd=="help"){usage();return 0;}
        if(cmd=="coord"&&argc==5){
            dm3d::Coord c=parse_coord(argv,2);
            std::cout<<"linear_index="<<dm3d::decimal(dm3d::flatten(c))<<"\n";return 0;
        }
        if(cmd=="uncoord"&&argc==3){
            auto c=dm3d::unflatten(dm3d::parse_u128(argv[2]));
            std::cout<<"x="<<c.x<<" y="<<c.y<<" z="<<c.z<<"\n";return 0;
        }
        if(cmd=="voxel-set"&&argc==7){
            dm3d::Coord c=parse_coord(argv,3);
            uint8_t value=uint8_t(arg_uint(argv[6],255));
            dm3d::voxel_set(argv[2],c,value);
            agent("SPARSE-MEMORY","stored voxel at "+dm3d::decimal(dm3d::flatten(c))+
                  " value="+std::to_string(value));return 0;
        }
        if(cmd=="voxel-get"&&argc==6){
            dm3d::Coord c=parse_coord(argv,3);
            auto value=dm3d::voxel_get(argv[2],c);
            std::cout<<"value="<<unsigned(value.value_or(0))
                     <<" allocated="<<(value?"true":"false")<<"\n";return 0;
        }
        if(cmd=="inspect"&&argc==3){
            std::string p=argv[2];
            // Detect archive via magic before attempting file feature analysis.
            try{auto h=dm3d::inspect_archive(p);print_archive(h);}
            catch(const std::exception&){
                auto f=dm3d::analyze_file(p);
                std::cout<<"modality="<<dm3d::modality_name(f.modality)<<" bytes="<<f.bytes
                <<" sample_entropy="<<std::fixed<<std::setprecision(3)<<f.entropy_bits_per_byte
                <<" bits/byte detail="<<f.detail<<'\n';
            }return 0;
        }
        if(cmd=="encode"&&(argc>=4)){
            auto opt=options(argc,argv,4);
            agent("INGEST","read file and determine modality");
            auto f=dm3d::analyze_file(argv[2]);
            agent("CLASSIFY",dm3d::modality_name(f.modality)+" | "+f.detail);
            agent("PLAN","assign sequential sparse 3D torus tile addresses");
            agent("ENCODE","3D XOR predictor search -> reconstruct -> CTR compare -> improve");
            agent("FOLD","4^3 latent periodic contraction, diagnostics only");
            auto s=dm3d::encode_file(argv[2],argv[3],opt.threads,opt.folds,opt.passes);
            agent("MEMORY","updated Omega EMA; committed lossless archive");
            print_archive(s.archive);
            std::cout<<std::fixed<<std::setprecision(8)
               <<"latent_mean="<<s.mean_latent_value<<" latent_delta="<<s.mean_latent_change
               <<" workers="<<s.threads<<" fold_iterations="<<s.fold_iterations
               <<" proposals_verified="<<s.candidate_trials
               <<" raw_fallback_tiles="<<s.raw_fallback_tiles
               <<" payload_bytes_saved="<<s.saved_payload_bytes<<"\n";
            return 0;
        }
        if(cmd=="decode"&&argc==4){
            agent("DECODER","reconstruct 3D tiles and validate every checksum");
            auto h=dm3d::decode_file(argv[2],argv[3]);
            agent("AUDITOR","validated whole-file digest before output commit");
            print_archive(h);return 0;
        }
        if(cmd=="verify"&&argc==4){
            dm3d::verify_file(argv[2],argv[3]);
            agent("CTR","PASS: archive decodes bit-exactly to supplied original");return 0;
        }
        if((cmd=="pipeline"||cmd=="loop")&&(argc>=5)){
            auto opt=options(argc,argv,5);
            agent("PLANNER","DAG: classify -> address -> encode/fold -> memory -> verify -> decode -> audit");
            auto f=dm3d::analyze_file(argv[2]);
            agent("MULTIMODAL",dm3d::modality_name(f.modality)+" | "+f.detail);
            auto s=dm3d::encode_file(argv[2],argv[3],opt.threads,opt.folds,opt.passes);
            agent("INWARD","completed "+std::to_string(s.candidate_trials)+" exact encode-decode-contrast attempts; "
                  +std::to_string(s.saved_payload_bytes)+" payload bytes saved vs raw tiles");
            agent("ENCODER","committed "+std::to_string(s.archive.tiles)+" 3D tiles, verified "
                  +std::to_string(s.candidate_trials)+" recursive codec proposals");
            dm3d::verify_file(argv[3],argv[2]);
            agent("CTR","original/decoded tile identity verified");
            auto h=dm3d::decode_file(argv[3],argv[4]);
            if(dm3d::hash_file(argv[4])!=h.file_hash)throw std::runtime_error("restored file hash mismatch");
            std::ifstream a(argv[2],std::ios::binary), b(argv[4],std::ios::binary);
            if(!a || !b)throw std::runtime_error("cannot open files for final bytewise audit");
            char left[65536],right[65536];
            do {
                a.read(left,sizeof(left));b.read(right,sizeof(right));
                if(a.gcount()!=b.gcount() ||
                   !std::equal(left,left+a.gcount(),right))
                    throw std::runtime_error("reconstructed file differs bitwise from source");
            }while(a.gcount()>0);
            if(!a.eof() || !b.eof())throw std::runtime_error("incomplete file comparison");
            agent("AUDITOR","PASS: independent restored bytes match original exactly");
            print_archive(h);return 0;
        }
        if(cmd=="selftest"&&argc==2){dm3d::run_selftest();return 0;}
        usage();return 2;
    }catch(const std::exception& e){std::cerr<<"[DM3D ERROR] "<<e.what()<<'\n';return 1;}
}
