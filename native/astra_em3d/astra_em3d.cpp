// Astra EM3D: bounded sampled toroidal codec coupled to periodic Maxwell FDTD.
// Standalone C++17; build: c++ -O2 -std=c++17 -fopenmp astra_em3d.cpp -o astra_em3d
// No actual 10^(6,000,000) agents or RF hardware are executed.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace astra {
constexpr int HIGH=12288, LATENT=128;
constexpr double PI=3.14159265358979323846;
constexpr double R=10.0, r=2.0, BOX=28.0;
using Vec3=std::array<double,3>;
static std::uint64_t hash64(std::uint64_t a) {
 a+=0x9e3779b97f4a7c15ULL;
 a=(a^(a>>30))*0xbf58476d1ce4e5b9ULL;
 a=(a^(a>>27))*0x94d049bb133111ebULL;
 return a^(a>>31);
}
static double unit(std::uint64_t x) {return ((hash64(x)>>11)+0.5)*0x1.0p-53;}
static Vec3 torus(double theta,double phi) {
 return {(R+r*std::cos(phi))*std::cos(theta),
         (R+r*std::cos(phi))*std::sin(theta),r*std::sin(phi)};
}
static void project(double* z) {
 const double theta=std::atan2(z[1],z[0]);
 const double phi=std::atan2(z[2],std::hypot(z[0],z[1])-R);
 const Vec3 p=torus(theta,phi);
 for(int k=0;k<3;++k) z[k]=p[k];
}
static double torus_defect(const double* p) {
 return std::abs(std::pow(std::hypot(p[0],p[1])-R,2)+p[2]*p[2]-r*r);
}
struct Config {
 int samples=256, steps=8, grid=24, threads=0;
 double alpha=0.7, dt=0.08, sigma=0.12, source=8.0, coupling=0.05;
 std::uint64_t seed=1337;
 std::string receipt="artifacts/astra-em3d.json", trace="artifacts/astra-em3d.csv";
 bool self_test=false;
 void validate() const {
  if(samples<1||samples>100000) throw std::runtime_error("samples must be 1..100000");
  if(steps<1||steps>10000) throw std::runtime_error("steps must be 1..10000");
  if(grid<6||grid>256) throw std::runtime_error("grid must be 6..256");
  if(!(alpha>0&&alpha<=1) || !(sigma>=0&&sigma<=1) || !(source>=0&&source<=1e4)
   || !(coupling>=0&&coupling<=10) || !std::isfinite(dt)
   || !std::isfinite(alpha)||!std::isfinite(sigma)||!std::isfinite(source)||!std::isfinite(coupling))
   throw std::runtime_error("invalid nonfinite or out-of-range coupling/codec parameters");
  if(!(dt>0 && dt<=BOX/grid/std::sqrt(3.0))) throw std::runtime_error("time step violates 3D Maxwell CFL limit");
  if(threads<0||threads>256) throw std::runtime_error("threads must be 0..256");
 }
};

// Spatially staggered periodic Yee curl pair: curl+ maps edge E to face B;
// curl- maps face B to edge E. Divergences use the same oriented differences.
// This normalized solver uses eps=mu=c=1 and zero charge. Source J=curl- M,
// so div- J=0 to floating point accuracy by commuting periodic differences.
class Maxwell3D {
 int n_; double h_;
 std::vector<Vec3> e_,b_,m_,j_;
 int wrap(int p) const {return p<0?p+n_:(p>=n_? p-n_:p);}
 int idx(int x,int y,int z) const {return (wrap(z)*n_+wrap(y))*n_+wrap(x);}
 double dplus(const std::vector<Vec3>& f,int x,int y,int z,int axis,int component) const {
  int a[3]={x,y,z}; a[axis]++;return (f[idx(a[0],a[1],a[2])][component]-f[idx(x,y,z)][component])/h_;
 }
 double dminus(const std::vector<Vec3>& f,int x,int y,int z,int axis,int component) const {
  int a[3]={x,y,z}; a[axis]--;return (f[idx(x,y,z)][component]-f[idx(a[0],a[1],a[2])][component])/h_;
 }
 Vec3 curl(const std::vector<Vec3>& f,int x,int y,int z,bool plus) const {
  auto d=[&](int axis,int comp){return plus?dplus(f,x,y,z,axis,comp):dminus(f,x,y,z,axis,comp);};
  return {d(1,2)-d(2,1), d(2,0)-d(0,2), d(0,1)-d(1,0)};
 }
public:
 explicit Maxwell3D(int n): n_(n),h_(BOX/n),e_(std::size_t(n)*n*n),b_(e_.size()),m_(e_.size()),j_(e_.size()) {}
 int size()const{return n_;} double spacing()const{return h_;}
 void clear_magnetization(){std::fill(m_.begin(),m_.end(),Vec3{});}
 void deposit(Vec3 position,Vec3 moment) {
  // Periodic cloud-in-cell deposition. Magnetization lives on face samples;
  // this is a numerical source model, not a resolved physical coil.
  double v[3];int cell[3];double frac[3];
  for(int k=0;k<3;++k){v[k]=(position[k]+BOX/2)/h_;cell[k]=int(std::floor(v[k]));frac[k]=v[k]-cell[k];}
  for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
   const double w=(dx?frac[0]:1-frac[0])*(dy?frac[1]:1-frac[1])*(dz?frac[2]:1-frac[2]);
   auto &a=m_[idx(cell[0]+dx,cell[1]+dy,cell[2]+dz)];
   for(int k=0;k<3;++k) a[k]+=w*moment[k];
  }
 }
 void finish_currents() {
  for(int z=0;z<n_;++z)for(int y=0;y<n_;++y)for(int x=0;x<n_;++x)
   j_[idx(x,y,z)]=curl(m_,x,y,z,false);
 }
 double max_divergence(const std::vector<Vec3>& field,bool plus) const {
  double maxdiv=0;
  for(int z=0;z<n_;++z)for(int y=0;y<n_;++y)for(int x=0;x<n_;++x){
   double sum=0;
   for(int k=0;k<3;++k)sum+=plus?dplus(field,x,y,z,k,k):dminus(field,x,y,z,k,k);
   maxdiv=std::max(maxdiv,std::abs(sum));
  }
  return maxdiv;
 }
 double div_e() const {return max_divergence(e_,false);}
 double div_b() const {return max_divergence(b_,true);}
 double div_j() const {return max_divergence(j_,false);}
 void step(double dt) {
  // Leapfrog magnetic half-step then electric full-step.
  for(int z=0;z<n_;++z)for(int y=0;y<n_;++y)for(int x=0;x<n_;++x){
   const int i=idx(x,y,z); auto c=curl(e_,x,y,z,true);
   for(int k=0;k<3;++k)b_[i][k]-=dt*c[k];
  }
  for(int z=0;z<n_;++z)for(int y=0;y<n_;++y)for(int x=0;x<n_;++x){
   const int i=idx(x,y,z);auto c=curl(b_,x,y,z,false);
   for(int k=0;k<3;++k)e_[i][k]+=dt*(c[k]-j_[i][k]);
  }
 }
 Vec3 sample(const std::vector<Vec3>& f,Vec3 p) const {
  int cell[3];double frac[3];
  for(int k=0;k<3;++k){double a=(p[k]+BOX/2)/h_;cell[k]=int(std::floor(a));frac[k]=a-cell[k];}
  Vec3 ans{};
  for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
   double w=(dx?frac[0]:1-frac[0])*(dy?frac[1]:1-frac[1])*(dz?frac[2]:1-frac[2]);
   auto v=f[idx(cell[0]+dx,cell[1]+dy,cell[2]+dz)];
   for(int k=0;k<3;++k)ans[k]+=w*v[k];
  }
  return ans;
 }
 Vec3 sample_e(Vec3 p)const{return sample(e_,p);}
 Vec3 sample_b(Vec3 p)const{return sample(b_,p);}
 double energy()const{
  double sum=0;
  for(size_t i=0;i<e_.size();++i) {
   for(int k=0;k<3;++k) sum+=0.5*(e_[i][k]*e_[i][k]+b_[i][k]*b_[i][k]);
  }
  return sum*h_*h_*h_;
 }
 void initialize_plane_wave() {
  // E_y(x)=B_z(x), source-free transverse initial wave.
  for(int z=0;z<n_;++z)for(int y=0;y<n_;++y)for(int x=0;x<n_;++x){
   const double a=std::sin(2*PI*x/n_);e_[idx(x,y,z)][1]=a;b_[idx(x,y,z)][2]=a;
  }
 }
};

struct Metrics {double mse=0, residual=0, defect=0, em_energy=0, div_e=0,div_b=0,div_j=0,field_latent_rms=0;};
class Astra {
 Config c_; Maxwell3D em_;
 std::vector<double> basis_,truth_,x_,z_;
 static std::size_t ix(int i,int d,int stride){return std::size_t(i)*stride+d;}
 double truth_x(int i,int d) const {return basis_[d]*truth_[ix(i,d%LATENT,LATENT)];}
 void encode(int i) {
  double* zp=&z_[ix(i,0,LATENT)];std::fill(zp,zp+LATENT,0.0);
  for(int d=0;d<HIGH;++d)zp[d%LATENT]+=basis_[d]*x_[ix(i,d,HIGH)];
  project(zp);
 }
public:
 explicit Astra(Config c):c_(std::move(c)),em_(c_.grid),basis_(HIGH),truth_(std::size_t(c_.samples)*LATENT),
   x_(std::size_t(c_.samples)*HIGH),z_(std::size_t(c_.samples)*LATENT) {
  c_.validate();
  for(int d=0;d<HIGH;++d)basis_[d]=(hash64(c_.seed+d)&1?1.0:-1.0)/std::sqrt(double(HIGH/LATENT));
  for(int i=0;i<c_.samples;++i){
   const double theta=2*PI*unit(c_.seed^((std::uint64_t(i)+1)*73ULL));
   const double phi=2*PI*unit(c_.seed^((std::uint64_t(i)+1)*151ULL));
   auto p=torus(theta,phi);
   for(int k=0;k<3;++k)truth_[ix(i,k,LATENT)]=p[k];
   for(int d=3;d<LATENT;++d) truth_[ix(i,d,LATENT)]=0.4*std::sin((1+d%5)*theta+(1+d%7)*phi);
   for(int d=0;d<HIGH;++d){
    double noise=c_.sigma*std::sqrt(12.0)*(unit(c_.seed^((std::uint64_t(i)*HIGH+d+1)*0x94d049bb133111ebULL))-0.5);
    x_[ix(i,d,HIGH)]=truth_x(i,d)+noise;
   }
  }
 }
 double truth_mse() const {
  double err=0;for(int i=0;i<c_.samples;++i)for(int d=0;d<HIGH;++d){
   double e=x_[ix(i,d,HIGH)]-truth_x(i,d);err+=e*e;
  }return err/(double(c_.samples)*HIGH);
 }
 Metrics pulse(){
  // Encoder is independent of truth; truth is used only for synthetic evaluation.
  #pragma omp parallel for schedule(static)
  for(int i=0;i<c_.samples;++i)encode(i);
  em_.clear_magnetization();
  for(int i=0;i<c_.samples;++i){
   double* z=&z_[ix(i,0,LATENT)];const double rho=std::hypot(z[0],z[1]);
   Vec3 p={z[0],z[1],z[2]};
   Vec3 m={-z[1]/rho,z[0]/rho,0.25*z[2]/r};
   for(double& a:m) a*=c_.source/c_.samples;
   em_.deposit(p,m);
  }
  em_.finish_currents();em_.step(c_.dt);
  double residual=0, truth_error=0, defect=0, field_shift_sq=0;
  for(int i=0;i<c_.samples;++i){
   double* z=&z_[ix(i,0,LATENT)];const Vec3 p={z[0],z[1],z[2]};
   auto E=em_.sample_e(p), B=em_.sample_b(p);
   // An explicitly specified, untrained field-to-latent feature injection.
   // Spatial coordinates 0..2 remain on the torus.
   for(int k=0;k<3;++k){
    const double de=c_.coupling*E[k],db=c_.coupling*B[k];
    z[3+k]+=de;z[6+k]+=db;field_shift_sq+=de*de+db*db;
   }
   defect=std::max(defect,torus_defect(z));
   for(int d=0;d<HIGH;++d){
    double &xi=x_[ix(i,d,HIGH)];const double decoded=basis_[d]*z[d%LATENT];
    const double delta=decoded-xi;residual+=delta*delta;
    xi+=c_.alpha*delta;
    double e=xi-truth_x(i,d);truth_error+=e*e;
   }
  }
  Metrics m;const double denom=double(c_.samples)*HIGH;
  m.mse=truth_error/denom;m.residual=residual/denom;m.defect=defect;
  m.em_energy=em_.energy();m.div_e=em_.div_e();m.div_b=em_.div_b();m.div_j=em_.div_j();
  m.field_latent_rms=std::sqrt(field_shift_sq/(double(c_.samples)*6));
  for(double v:{m.mse,m.residual,m.defect,m.em_energy,m.div_e,m.div_b,m.div_j,m.field_latent_rms})
   if(!std::isfinite(v)) throw std::runtime_error("nonfinite simulation state");
  return m;
 }
 void run(){
  std::ofstream trace,receipt;
  if(c_.trace!="none") {std::filesystem::path p(c_.trace);if(p.has_parent_path())std::filesystem::create_directories(p.parent_path());
   trace.open(c_.trace);if(!trace)throw std::runtime_error("cannot open trace");
   trace<<"step,agent,x,y,z,Ex,Ey,Ez,Bx,By,Bz\n"<<std::setprecision(12);
  }
  const double initial=truth_mse();std::vector<Metrics> history;
  for(int step=1;step<=c_.steps;++step){
   Metrics m=pulse();history.push_back(m);
   if(trace){for(int i=0;i<std::min(c_.samples,128);++i){
    const double* z=&z_[ix(i,0,LATENT)];Vec3 p={z[0],z[1],z[2]};
    auto E=em_.sample_e(p),B=em_.sample_b(p);
    trace<<step<<","<<i<<","<<p[0]<<","<<p[1]<<","<<p[2];
    for(double v:E) trace<<","<<v;
    for(double v:B) trace<<","<<v;
    trace<<"\n";
   }}
   std::cout<<std::setprecision(9)<<"step="<<step<<" truth_mse="<<m.mse
    <<" residual_mse="<<m.residual<<" em_energy="<<m.em_energy
    <<" divE="<<m.div_e<<" divB="<<m.div_b<<" divJ="<<m.div_j
    <<" latent_field_rms="<<m.field_latent_rms<<"\n";
   if(m.div_e>1e-8||m.div_b>1e-8||m.div_j>1e-8||m.defect>1e-7)
    throw std::runtime_error("CTR invariant failed: Maxwell divergence or torus defect");
  }
  if(c_.receipt!="none"){
   std::filesystem::path p(c_.receipt);if(p.has_parent_path())std::filesystem::create_directories(p.parent_path());
   receipt.open(c_.receipt);if(!receipt)throw std::runtime_error("cannot open receipt");
   receipt<<std::setprecision(16);
   const auto& m=history.back();
   receipt<<"{\n  \"provenance\": \"synthetic-numerical-simulation\",\n"
    <<"  \"model\": \"Astra sampled toroidal codec + periodic neutral-current Maxwell FDTD\",\n"
    <<"  \"symbolic_population_executed\": false,\n"
    <<"  \"trained_neural_network\": false,\n"
    <<"  \"samples\": "<<c_.samples<<", \"grid\": "<<c_.grid<<", \"steps\": "<<c_.steps<<",\n"
    <<"  \"dt\": "<<c_.dt<<", \"alpha\": "<<c_.alpha<<", \"coupling\": "<<c_.coupling<<",\n"
    <<"  \"initial_truth_mse\": "<<initial<<", \"final_truth_mse\": "<<m.mse<<",\n"
    <<"  \"max_div_e\": "<<m.div_e<<", \"max_div_b\": "<<m.div_b<<", \"max_div_j\": "<<m.div_j<<",\n"
    <<"  \"max_torus_defect\": "<<m.defect<<", \"em_energy\": "<<m.em_energy<<",\n"
    <<"  \"latent_field_rms\": "<<m.field_latent_rms<<",\n"
    <<"  \"verification\": {\"divergence_constraints\": true, \"torus_constraint\": true, \"finite_metrics\": true}\n} \n";
  }
  std::cout<<"ASTRA EM3D PASS: simulated agents="<<c_.samples<<" grid="<<c_.grid<<"^3, final MSE="<<history.back().mse<<"\n";
 }
};

static void self_test() {
 // Vacuum condition and divergence identities.
 Maxwell3D blank(12);blank.finish_currents();blank.step(0.1);
 if(blank.energy()!=0||blank.div_b()!=0||blank.div_e()!=0)throw std::runtime_error("vacuum failed");
 // Current constructed as curl- M must be solenoidal. Fields must preserve Gauss.
 Maxwell3D source(12);source.deposit({3.1,-1.2,0.6},{1.0,0.2,-0.3});source.finish_currents();
 if(source.div_j()>1e-12)throw std::runtime_error("source current not solenoidal");
 for(int i=0;i<8;++i)source.step(0.1);
 if(!(source.energy()>1e-12)||source.div_e()>1e-11||source.div_b()>1e-11)
  throw std::runtime_error("Maxwell field emission or divergence failure");
 Maxwell3D wave(16);wave.initialize_plane_wave();double initial=wave.energy();
 for(int i=0;i<20;++i)wave.step(0.1);
 if(wave.div_e()>1e-11||wave.div_b()>1e-11||std::abs(wave.energy()/initial-1)>0.05)
  throw std::runtime_error("plane-wave conservation/propagation failed");
 Config c;c.samples=16;c.grid=12;c.steps=3;c.trace="none";c.receipt="none";
 Astra swarm(c);double before=swarm.truth_mse();auto p=swarm.pulse();
 if(!(p.mse<before)||!(p.field_latent_rms>0)||p.div_j>1e-10||p.div_e>1e-10||p.div_b>1e-10)
  throw std::runtime_error("codec/field coupling failed");
 c.dt=100;bool rejected=false;try{c.validate();}catch(const std::exception&){rejected=true;}
 if(!rejected)throw std::runtime_error("CFL guard failed");
 std::cout<<"SELF-TEST PASS: vacuum, solenoidal current, Gauss constraints, transverse wave, codec, CFL\n";
}
static Config parse(int argc,char**argv){
 Config c;
 for(int i=1;i<argc;++i){std::string a=argv[i];
  if(a=="--self-test"){c.self_test=true;continue;}
  if(a=="--help"){std::cout<<"--samples N --grid N --steps N --alpha A --dt DT --sigma S --source S --coupling C --seed N --threads N --receipt FILE|none --trace FILE|none --self-test\n";std::exit(0);}
  if(i+1==argc)throw std::runtime_error("missing value for "+a);
  std::string v=argv[++i];
  if(a=="--samples")c.samples=std::stoi(v);else if(a=="--steps")c.steps=std::stoi(v);
  else if(a=="--grid")c.grid=std::stoi(v);else if(a=="--alpha")c.alpha=std::stod(v);
  else if(a=="--sigma")c.sigma=std::stod(v);else if(a=="--dt")c.dt=std::stod(v);
  else if(a=="--source")c.source=std::stod(v);else if(a=="--coupling")c.coupling=std::stod(v);
  else if(a=="--threads")c.threads=std::stoi(v);else if(a=="--seed")c.seed=std::stoull(v);
  else if(a=="--receipt")c.receipt=v;else if(a=="--trace")c.trace=v;
  else throw std::runtime_error("unknown option: "+a);
 }
 return c;
}
} // namespace astra
#ifndef ASTRA_EM3D_NO_MAIN
int main(int argc,char**argv){try{
 auto c=astra::parse(argc,argv);
 if(c.self_test){astra::self_test();return 0;}
 c.validate();
 #ifdef _OPENMP
 if(c.threads>0)omp_set_num_threads(c.threads);
 #endif
 astra::Astra(c).run();return 0;
 }catch(const std::exception&e){std::cerr<<"ASTRA EM3D ERROR: "<<e.what()<<"\n";return 2;}}
#endif