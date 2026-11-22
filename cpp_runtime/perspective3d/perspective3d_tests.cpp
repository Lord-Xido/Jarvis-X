#include "perspective3d.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

using namespace jarvisx::perspective3d;

static int checks = 0;
static void require(bool value, const std::string& name) {
    ++checks;
    if (!value) throw std::runtime_error(name);
}
static bool near(double a, double b) { return std::abs(a-b) <= 1e-12; }

int main() {
    try {
        const auto rotations = proper_cube_rotations();
        require(rotations.size() == 24, "24 proper cube rotations");
        std::set<std::array<int,9>> unique;
        for (const Rotation& r : rotations) {
            unique.insert(r.m);
            require(determinant(r) == 1, "proper determinant");
        }
        require(unique.size() == 24, "unique group elements");
        for (const auto& a : rotations) for (const auto& b : rotations)
            require(unique.count(compose(a,b).m) == 1, "rotation group closure");

        Cube original(7);
        for (std::size_t z=0; z<original.n; ++z)
            for (std::size_t y=0; y<original.n; ++y)
                for (std::size_t x=0; x<original.n; ++x)
                    if (((17*x + 31*y + 43*z) % 13) < 4)
                        original.at(x,y,z) = static_cast<std::uint8_t>(1+((x+5*y+9*z) % 250));
        const auto h = histogram(original);
        const Vec3 centroid = centred_weighted_centroid(original);
        for (const Rotation& r : rotations) {
            const Cube turned = rotate(original,r);
            require(histogram(turned)==h, "invariant byte histogram");
            require(rotate(turned,inverse(r)) == original, "exact inverse 3d voxel rotation");
            const Vec3 measured = centred_weighted_centroid(turned);
            const Vec3 predicted = apply(r,centroid);
            require(near(measured.x,predicted.x) && near(measured.y,predicted.y)
                    && near(measured.z,predicted.z), "equivariant centroid");
        }
        Cube singleton(1);
        singleton.at(0,0,0)=217;
        for (const auto& r : rotations) require(rotate(singleton,r)==singleton,"degenerate cube");
        bool rejected = false;
        try { Cube invalid(0); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid cube rejected");
        rejected = false;
        try { Cube invalid(129); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"oversize reference cube rejected");

        const Image2D photo(4,3,{1,2,3,4,5,6,7,8,9,10,11,12});
        const Image2D clockwise = clockwise_90(photo);
        require(clockwise.width==3 && clockwise.height==4, "2d dimensions swap");
        require(clockwise.pixels==std::vector<std::uint8_t>({9,5,1,10,6,2,11,7,3,12,8,4}),
                "90deg clockwise is exact permutation");
        require(clockwise_90(clockwise_90(clockwise_90(clockwise)))==photo, "four turns restore photo");
        rejected=false;
        try { Image2D invalid(2,3,{1,2}); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid image rejected");

        Cube near_volume(5), far_volume(5);
        near_volume.at(1,2,0)=1;
        far_volume.at(1,2,4)=1;
        const auto view_a=project_xy_occupancy(near_volume);
        const auto view_b=project_xy_occupancy(far_volume);
        require(view_a==view_b,"distinct 3d states cast identical single projections");
        require(exact_cube_identity(near_volume,far_volume)==Verdict::refuted,"3d truth distinguishes");
        require(projection_identity(view_a,view_b)==Verdict::underdetermined,
                "identical projection not identity proof");
        bool discriminating_new_view=false;
        for (const Rotation& r : rotations) {
            const auto a=project_xy_occupancy(rotate(near_volume,r));
            const auto b=project_xy_occupancy(rotate(far_volume,r));
            if (a!=b) {
                discriminating_new_view=true;
                require(projection_identity(a,b)==Verdict::refuted,"different views refute equality");
            }
        }
        require(discriminating_new_view,"new perspective provides new distinguishing measurement");
        require(projection_identity(view_a,view_a)==Verdict::underdetermined,
                "same projection never proves complete identity");
        require(exact_cube_identity(original,original)==Verdict::verified,"bounded exact equality");
        Cube altered=original;
        altered.at(0,0,0) ^= 1;
        require(exact_cube_identity(original,altered)==Verdict::refuted,"corrupted state rejected");

        std::cout << "{\"schema\":\"dm-ptvp-conformance.v0.1\",\"status\":\"PASS\","
                  << "\"reference\":\"bounded-exact-cpp17\",\"proper_rotations\":24,"
                  << "\"checks\":" << checks << ","
                  << "\"single_projection_identity\":\"UNDERDETERMINED\","
                  << "\"claim_boundary\":\"finite discrete grid and specified transforms; no semantic or physical completeness\"}"
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"schema\":\"dm-ptvp-conformance.v0.1\",\"status\":\"FAIL\","
                  << "\"checks_completed\":" << checks << ",\"reason\":\"" << error.what() << "\"}\n";
        return 1;
    }
}
