#include "qsol_reverberation3d.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace qsol3d;
    static_assert(kVoxels == 4096 && kObserverUnits == 256, "lattice geometry");
    Field x{}, input{};
    for (int i=0;i<kVoxels;++i) input.cells[i] = ((i%7)<3)? kOne : -kOne;
    const Field original=x;
    const auto snapshot=observe(x,input,0);
    assert(x.cells == original.cells);
    assert(snapshot.units.size()==256);
    assert(observe(x,input,0).render_hash == snapshot.render_hash);
    assert(observe(x,input,1).render_hash != snapshot.render_hash);
    assert(index(-1,0,0)==index(15,0,0));
    assert(index(16,0,0)==index(0,0,0));
    assert(excitation_from_rgb(255,0)==kOne);
    assert(excitation_from_rgb(0,255)==-kOne);
    for (int n=0;n<256;++n) x=advance(x,input);
    // Bounded-input l_infinity stability: exact for this fixed-point contract.
    for (auto q : x.cells) assert(q>=-kOne && q<=kOne);
    Field no_input{};
    auto cur=x;
    for (int n=0;n<256;++n) cur=advance(cur,no_input);
    for (auto q : cur.cells) assert(q>=-2 && q<=2);
    Field invalid=x;
    invalid.cells[0] = kOne+1;
    bool rejected=false;
    try { (void)advance(invalid,input); }
    catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected);
    // Constant-field exact recurrence check: A=15/16, B=1/16.
    Field uniform{}, drive{};
    drive.cells.fill(kOne);
    uniform=advance(uniform,drive);
    for(auto q:uniform.cells) assert(q==kOne/16);
    std::cout << "QSOL PASS voxels="<<kVoxels<<" observed="<<kObserverUnits
              <<" hash="<<snapshot.render_hash<<"\n";
}
