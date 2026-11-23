// Riemannian momentum with a scalar second moment on the unit sphere S^2.
// Independent synthetic control objective; NOT a natural-language C++ optimiser.
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
struct Vec {
    double x, y, z;
    Vec operator+(Vec o) const {return {x+o.x,y+o.y,z+o.z};}
    Vec operator-(Vec o) const {return {x-o.x,y-o.y,z-o.z};}
    Vec operator*(double s) const {return {x*s,y*s,z*s};}
};
static double dot(Vec a,Vec b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
static double norm(Vec a){return std::sqrt(dot(a,a));}
static Vec normalize(Vec a){const double n=norm(a); if(!(std::isfinite(n)&&n>0))throw std::invalid_argument("invalid vector");return a*(1/n);}
static Vec project(Vec v,Vec z){return v-z*dot(v,z);}
static Vec exponential(Vec z,Vec u){
    const double a=norm(u);
    if(a<1e-14) return normalize(z+u);
    return normalize(z*std::cos(a)+u*(std::sin(a)/a));
}
// Exact parallel transport on the shortest great-circle arc (excluding antipodes).
static Vec transport(Vec z,Vec y,Vec tangent){
    const double denominator=1+dot(z,y);
    if(denominator<1e-10)throw std::runtime_error("transport near antipodes");
    return project(tangent-(z+y)*(dot(tangent,y)/denominator),y);
}
static unsigned parse_steps(const char* arg){
    const std::string s(arg);
    if(s.empty()||s.size()>7||s.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument("steps must be decimal in [1, 1000000]");
    const auto u=std::stoul(s);
    if(u<1 || u>1000000)throw std::invalid_argument("steps out of bounds");
    return static_cast<unsigned>(u);
}
int main(int argc,char**argv){
 try{
    const unsigned K=argc==2?parse_steps(argv[1]):1000000;
    if(argc>2)throw std::invalid_argument("usage: sphere_optimizer [steps]");
    const Vec target=normalize({0.31,0.41,0.86});
    Vec z={1.0,0.0,0.0};
    Vec m={0,0,0};
    double v=0, beta1_pow=1, beta2_pow=1;
    const double f0=1-dot(z,target);
    double max_norm_drift=0;
    for(unsigned k=1;k<=K;k++){
        // Riemannian gradient of f(z)=1-<z,target> on S².
        Vec grad=project(target*(-1.0),z);
        m=m*0.9+grad*0.1;
        v=0.999*v+0.001*dot(grad,grad);
        beta1_pow*=0.9; beta2_pow*=0.999;
        const double m_divisor=1-beta1_pow, v_divisor=1-beta2_pow;
        const double eta=0.02/(1.0+0.0001*k);
        const Vec step=project(m,z)*(-eta/(m_divisor*(std::sqrt(v/v_divisor)+1e-8)));
        const Vec next=exponential(z,step);
        m=transport(z,next,m);
        z=next;
        const double drift=std::abs(norm(z)-1.0);
        if(drift>max_norm_drift)max_norm_drift=drift;
        if(!std::isfinite(v)||!std::isfinite(dot(z,target)))throw std::runtime_error("nonfinite state");
    }
    const double f1=1-dot(z,target);
    if(f1>1e-6||max_norm_drift>1e-11||f1>f0)throw std::runtime_error("geometric verification failed");
    std::cout<<std::setprecision(17)
       <<"{\"algorithm\":\"S2_parallel_transport_scalar_adaptive_moment\",\"iterations\":"<<K
       <<",\"initial_loss\":"<<f0<<",\"final_loss\":"<<f1
       <<",\"max_norm_drift\":"<<max_norm_drift
       <<",\"state\":["<<z.x<<","<<z.y<<","<<z.z<<"],\"verified\":true}\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
