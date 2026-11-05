#include "dm3d/core.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace dm3d;
namespace {
void check(bool c,const char* msg){if(!c)throw std::runtime_error(msg);}
void roundtrip(const fs::path& dir,const std::string& name,const std::vector<uint8_t>& data){
    auto src=dir/(name+".bin"), arc=dir/(name+".dm3d"), out=dir/(name+".restored");
    {std::ofstream f(src,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()));}
    auto stats=encode_file(src,arc,3,10);
    check(stats.archive.original_size==data.size(),"archive input size wrong");
    verify_file(arc,src);
    auto info=decode_file(arc,out);
    check(info.file_hash==hash_file(src),"archive hash mismatch");
    check(fs::file_size(src)==fs::file_size(out),"decoded file size mismatch");
    check(hash_file(src)==hash_file(out),"decoded payload mismatch");
    auto bytes=fs::file_size(arc);
    // Mutate a stored checksum byte, requiring decoder and verifier to reject corruption.
    if(!data.empty()){
        auto tampered=dir/(name+".corrupt.dm3d");
        fs::copy_file(arc,tampered,fs::copy_options::overwrite_existing);
        std::fstream fp(tampered,std::ios::in|std::ios::out|std::ios::binary);
        fp.seekg(std::streamoff(bytes-1));char c=0;fp.get(c);
        fp.seekp(std::streamoff(bytes-1));fp.put(char(c^0x5a));fp.close();
        bool rejected=false;
        try{decode_file(tampered,dir/(name+".bad"));}catch(const std::exception&){rejected=true;}
        check(rejected,"mutated archive was not rejected");
    }
    std::cout<<"PASS "<<name<<" "<<data.size()<<" bytes -> "<<bytes<<" archive bytes\n";
}
}
void dm3d::run_selftest(){
    auto tmp=fs::temp_directory_path()/"dm3d_roundtrip_test_v1";
    fs::remove_all(tmp);fs::create_directories(tmp);
    try{
        Coord a{999999999999ULL,999999999999ULL,999999999999ULL};
        check(unflatten(flatten(a)).x==a.x && unflatten(flatten(a)).y==a.y && unflatten(flatten(a)).z==a.z,"3D wide coordinate failed");
        check(unflatten(parse_u128(decimal(flatten(a)))).z==a.z,"128-bit decimal conversion failed");
        check(flatten({0,0,0})==0,"origin mismatch");
        std::cout<<"PASS 120-bit 3D toroidal virtual addressing\n";
        auto voxel_file=tmp/"sparse.vxl";
        check(!voxel_get(voxel_file,a).has_value(),"unset sparse voxel should be absent");
        voxel_set(voxel_file,a,42);
        check(voxel_get(voxel_file,a).value()==42,"sparse write at edge failed");
        voxel_set(voxel_file,a,173);
        check(voxel_get(voxel_file,a).value()==173,"sparse overwrite failed");
        check(!voxel_get(voxel_file,{12,34,56}).has_value(),"wrong voxel read");
        check(fs::file_size(voxel_file)==12+2*33,"sparse file record size wrong");
        std::cout<<"PASS sparse virtual voxel write/read at 10^12 cube boundary\n";
        // Full recursive trial sweep: each proposal reconstructs the entire 3D tile
        // before it is eligible to replace the incumbent lossless codec.
        std::array<uint8_t,TILE_BYTES> patterned{};
        for(size_t z=0;z<16;z++)for(size_t y=0;y<16;y++)for(size_t x=0;x<16;x++)
            patterned[z*256+y*16+x]=uint8_t(x*13+(x%3)*17);
        for(uint8_t remembered=1;remembered<=5;remembered++){
            auto encoded=encode_tile(patterned,TILE_BYTES,0,12,5,remembered);
            check(encoded.tested==5,"not all recursive 3D modes were verified");
            check(decode_tile(encoded)==patterned,"3D codec failed exact involutive roundtrip");
            check(encoded.payload.size()<=TILE_BYTES,"self-loop made best candidate worse");
        }
        auto single=encode_tile(patterned,TILE_BYTES,0,5,1,5);
        check(single.tested==1,"one proposal budget was not honored");
        check(decode_tile(single)==patterned,"one proposal codec failed");
        std::cout<<"PASS self-refining 3D codec: 5 spatial predictors, each independently decoded and contrasted\n";
        roundtrip(tmp,"empty",{});
        roundtrip(tmp,"zeros",std::vector<uint8_t>(8195));
        std::mt19937_64 gen(0xD03A613ULL);
        std::vector<uint8_t> random(20001);
        for(auto& v:random)v=uint8_t(gen());
        roundtrip(tmp,"random",random);
        std::vector<uint8_t> text(13015);
        const std::string s="The Dr Moagi 3D codepath is deterministic and bit-exact.\n";
        for(size_t i=0;i<text.size();i++)text[i]=uint8_t(s[i%s.size()]);
        roundtrip(tmp,"text",text);
        // Actual minimal WAV header and 16-bit mono PCM samples.
        std::vector<uint8_t> wav(44+8000*2,0);
        std::copy_n("RIFF",4,wav.begin());wav[8]='W';wav[9]='A';wav[10]='V';wav[11]='E';
        wav[12]='f';wav[13]='m';wav[14]='t';wav[15]=' ';
        wav[16]=16;wav[20]=1;wav[22]=1;wav[24]=0x40;wav[25]=0x1f;
        wav[28]=0x80;wav[29]=0x3e;wav[32]=2;wav[34]=16;
        wav[36]='d';wav[37]='a';wav[38]='t';wav[39]='a';
        for(size_t i=44;i<wav.size();i++)wav[i]=uint8_t(i*17);
        roundtrip(tmp,"wav",wav);
        // PNG signature and IHDR geometry (fixture is a header probe, not decoded pixels).
        std::vector<uint8_t> png(4099,0);std::array<uint8_t,8> sig{137,80,78,71,13,10,26,10};
        std::copy(sig.begin(),sig.end(),png.begin());png[12]='I';png[13]='H';png[14]='D';png[15]='R';
        png[19]=32;png[23]=32;roundtrip(tmp,"png_fixture",png);
        // MP4 ISO base media ftyp header fixture.
        std::vector<uint8_t> mp4(13005,0);
        mp4[3]=24;mp4[4]='f';mp4[5]='t';mp4[6]='y';mp4[7]='p';
        mp4[8]='i';mp4[9]='s';mp4[10]='o';mp4[11]='m';roundtrip(tmp,"mp4_fixture",mp4);
        // Confirm modality signature recognizer independent of extension.
        check(detect_modality("a.bin",wav)==Modality::Audio,"WAV classification failed");
        check(detect_modality("a.bin",png)==Modality::Image,"PNG classification failed");
        check(detect_modality("a.bin",mp4)==Modality::Video,"MP4 classification failed");
        fs::remove_all(tmp);std::cout<<"ALL TESTS PASSED\n";
    }catch(...){fs::remove_all(tmp);throw;}
}
