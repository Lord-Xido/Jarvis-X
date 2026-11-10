// Original user-submitted AutoEncoding3DEngine, preserved for regression/audit.
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
using u64 = uint64_t;
class AutoEncoding3DEngine {
public:
    static constexpr int W=64, H=64, D=64, N=W*H*D;
    std::vector<float> R0,R1,R2,R3;
    float feedback_gain=0.5f;
    AutoEncoding3DEngine():R0(N),R1(N),R2(N),R3(N){}
    inline int idx(int x,int y,int z)const{return x+y*W+z*W*H;}
    inline float activate(float x)const{
        if(x>3.0f)return 1.0f;
        if(x< -3.0f)return -1.0f;
        float x2=x*x;return x*(27.0f+x2)/(27.0f+9.0f*x2);
    }
    void generateSeed(std::vector<float>&dst,u64 seed){
        for(int i=0;i<N;i++){
            seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;
            dst[i]=(float)(seed&0xFFFF)/32768.0f-1.0f;
        }
    }
    void encode(){constexpr float weight=2.5f,bias=0.5f;
       for(int i=0;i<N;i++)R1[i]=activate(R0[i]*weight+bias);
    }
    void permute(uint32_t stride){stride|=1;
       for(int i=0;i<N;i++){
          uint32_t src=((uint64_t)i*stride)%N;R2[i]=R1[src];
       }
    }
    void feedback(int iterations){
       for(int t=0;t<iterations;t++)for(int i=0;i<N;i++)
          R3[i]=activate(R2[i]+feedback_gain*R3[i]);
    }
    void matrixPropagate(){
       for(int z=1;z<D-1;z++)for(int y=1;y<H-1;y++)for(int x=1;x<W-1;x++){
          int c=idx(x,y,z),center=c;
          float xp=R3[idx(x+1,y,z)],xm=R3[idx(x-1,y,z)];
          float yp=R3[idx(x,y+1,z)],ym=R3[idx(x,y-1,z)];
          float zp=R3[idx(x,y,z+1)],zm=R3[idx(x,y,z-1)];
          R2[c]=0.40f*R3[center]+0.10f*(xp+xm)+0.10f*(yp+ym)+0.10f*(zp+zm);
       }
    }
    void decode(){
       for(int z=1;z<D-1;z++)for(int y=1;y<H-1;y++)for(int x=1;x<W-1;x++){
          int c=idx(x,y,z);float sum=0;
          for(int dz=-1;dz<=1;dz++)for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++)
             sum+=R2[idx(x+dx,y+dy,z+dz)];
          R0[c]=activate(sum/27.0f);
       }
    }
    double reconstructionLoss()const{
       double mse=0;for(int i=0;i<N;i++){
         double e=double(R1[i])-double(R0[i]);mse+=e*e;
       }return mse/double(N);
    }
    void execute(uint64_t seed,uint32_t stride,int feedbackIterations){
       generateSeed(R0,seed);encode();permute(stride);feedback(feedbackIterations);
       matrixPropagate();decode();
    }
    void printChecksum()const{double sum=0;for(int i=0;i<32;i++)sum+=R0[i];std::cout<<"Checksum: "<<sum<<std::endl;}
};
int main(){AutoEncoding3DEngine engine;engine.execute(0xDEADBEEF12345678ULL,73,8);
   engine.printChecksum();std::cout<<"Reconstruction Loss: "<<engine.reconstructionLoss()<<std::endl;return 0;}
