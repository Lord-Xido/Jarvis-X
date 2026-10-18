// MB4C mode 3: sampled-field certified bounded-error sparse four-way tree.
// Included after mandelbulb4_adaptive.inc and archive_header(), in namespace mb4.
// Encoder inspects every source byte in every region; guarantees L-infinity
// reconstruction error <= tolerance for the sampled source, NOT the continuous
// Mandelbulb geometry between sampling locations.
struct BoundedNode {
    std::uint8_t tag = 1; // 0 exact constant; 1 raw; 2 split; 3 bounded leaf
    std::uint8_t value = 0, minimum = 0, maximum = 0;
    std::size_t bytes = 0;
    std::array<std::unique_ptr<BoundedNode>, 4> children{};
};
struct BoundedStats {
    std::size_t nodes = 0, splits = 0, constants = 0, literals = 0, approximations = 0;
    std::size_t covered_approx_voxels = 0;
    unsigned measured_max_error = 0;
};

std::unique_ptr<BoundedNode> build_bounded(const Bytes& src, Shape whole,
                                           const AdaptiveRegion& region,
                                           unsigned level, unsigned tolerance) {
    auto node = std::make_unique<BoundedNode>();
    const std::size_t n = count(region.shape);
    unsigned low = 255u, high = 0u;
    for (int z = 0; z < region.shape.z; ++z)
        for (int y = 0; y < region.shape.y; ++y)
            for (int x = 0; x < region.shape.x; ++x) {
                const unsigned v = src[at(whole, region.x+x, region.y+y, region.z+z)];
                low = std::min(low, v);
                high = std::max(high, v);
            }
    if (low == high) {
        node->tag = 0;
        node->value = static_cast<std::uint8_t>(low);
        node->bytes = 2;
        return node;
    }
    node->tag = 1;
    node->bytes = 1u + n;
    const unsigned rep = (low+high)/2u;
    const unsigned error = std::max(rep-low, high-rep);
    if (error <= tolerance && 4u < node->bytes) {
        node->tag = 3;
        node->value = static_cast<std::uint8_t>(rep);
        node->minimum = static_cast<std::uint8_t>(low);
        node->maximum = static_cast<std::uint8_t>(high);
        node->bytes = 4;
    }
    if (split_possible(region.shape, level)) {
        std::array<std::unique_ptr<BoundedNode>,4> candidates;
        std::size_t sum = 1;
        for (unsigned j = 0; j < 4; ++j) {
            candidates[j] = build_bounded(src, whole, child_region(region, level, j),
                                           level+1u, tolerance);
            sum += candidates[j]->bytes;
        }
        if (sum < node->bytes) {
            node->tag = 2;
            node->bytes = sum;
            node->children = std::move(candidates);
        }
    }
    return node;
}

void bounded_stats(const BoundedNode& node, const AdaptiveRegion& region,
                   unsigned level, BoundedStats& stats) {
    ++stats.nodes;
    switch (node.tag) {
    case 0: ++stats.constants; return;
    case 1: ++stats.literals; return;
    case 3:
        ++stats.approximations;
        stats.covered_approx_voxels += count(region.shape);
        return;
    case 2:
        ++stats.splits;
        for (unsigned j = 0; j < 4; ++j) {
            require(static_cast<bool>(node.children[j]), "missing bounded branch");
            bounded_stats(*node.children[j], child_region(region, level, j), level+1u, stats);
        }
        return;
    default: throw std::runtime_error("invalid bounded plan tag");
    }
}
void write_bounded_node(const BoundedNode& node, const Bytes& src, Shape whole,
                        const AdaptiveRegion& region, unsigned level, Bytes& out) {
    out.push_back(node.tag);
    switch (node.tag) {
    case 0: out.push_back(node.value); return;
    case 1:
        for (int z = 0; z < region.shape.z; ++z)
            for (int y = 0; y < region.shape.y; ++y)
                for (int x = 0; x < region.shape.x; ++x)
                    out.push_back(src[at(whole,region.x+x,region.y+y,region.z+z)]);
        return;
    case 2:
        for (unsigned j = 0; j < 4; ++j)
            write_bounded_node(*node.children[j], src, whole,
                               child_region(region,level,j),level+1u,out);
        return;
    case 3:
        out.push_back(node.value);
        out.push_back(node.minimum);
        out.push_back(node.maximum);
        return;
    default: throw std::runtime_error("invalid bounded node tag");
    }
}
void decode_bounded_node(const Bytes& in, std::size_t& pos, Bytes& output, Shape whole,
                         const AdaptiveRegion& region, unsigned level, unsigned tolerance) {
    const std::uint8_t tag = get_byte(in,pos);
    if (tag == 2) {
        require(split_possible(region.shape,level), "invalid bounded split dimensions");
        for (unsigned j = 0; j < 4; ++j)
            decode_bounded_node(in,pos,output,whole,
                                child_region(region,level,j),level+1u,tolerance);
        return;
    }
    require(tag == 0 || tag == 1 || tag == 3, "unknown bounded tree tag");
    unsigned value = 0;
    if (tag == 0) {
        value = get_byte(in,pos);
    } else if (tag == 3) {
        const unsigned representative = get_byte(in,pos);
        const unsigned minimum = get_byte(in,pos);
        const unsigned maximum = get_byte(in,pos);
        require(minimum < maximum && minimum <= representative
                && representative <= maximum, "malformed bounded envelope");
        require(std::max(representative-minimum, maximum-representative) <= tolerance,
                "bounded envelope exceeds declared tolerance");
        value = representative;
    } else {
        require(count(region.shape) <= in.size()-pos, "truncated bounded literal");
    }
    for (int z = 0; z < region.shape.z; ++z)
        for (int y = 0; y < region.shape.y; ++y)
            for (int x = 0; x < region.shape.x; ++x)
                output[at(whole, region.x+x,region.y+y,region.z+z)] =
                    static_cast<std::uint8_t>(tag == 1 ? get_byte(in,pos) : value);
}
// Hashes mode-3 payload metadata and tree, excluding the 8-byte hash field.
// FNV-1a detects accidental corruption; it provides no adversarial authentication.
std::uint64_t bounded_wire_hash(const Bytes& in) {
    constexpr std::size_t metadata_end = header_bytes + 9u;
    constexpr std::size_t hash_end = metadata_end + 8u;
    require(in.size() >= hash_end, "truncated bounded metadata hash");
    Bytes covered(in.begin() + static_cast<std::ptrdiff_t>(header_bytes),
                  in.begin() + static_cast<std::ptrdiff_t>(metadata_end));
    covered.insert(covered.end(),
                   in.begin() + static_cast<std::ptrdiff_t>(hash_end), in.end());
    return checksum(covered);
}
Bytes decode_bounded(const Bytes& in, std::size_t& pos, unsigned side) {
    const unsigned tolerance = get_byte(in,pos);
    require(tolerance <= 64u, "invalid stored tolerance");
    const std::uint64_t source_hash = get_u64(in,pos);
    (void)source_hash; // For provenance comparisons only; unavailable source cannot be checked.
    const std::uint64_t stored_wire_hash = get_u64(in,pos);
    require(stored_wire_hash == bounded_wire_hash(in),
            "bounded payload checksum mismatch");
    const Shape whole{static_cast<int>(side),static_cast<int>(side),static_cast<int>(side)};
    Bytes decoded(count(whole),0);
    decode_bounded_node(in,pos,decoded,whole,AdaptiveRegion{0,0,0,whole},0,tolerance);
    return decoded;
}
Bytes encode_bounded(const Bytes& source, unsigned side, unsigned tolerance,
                     BoundedStats* report = nullptr) {
    require(tolerance <= 64u, "bounded tolerance must be 0..64");
    const unsigned depth = levels_for(side);
    const Shape whole{static_cast<int>(side),static_cast<int>(side),static_cast<int>(side)};
    require(source.size() == count(whole), "bounded source dimension mismatch");
    const AdaptiveRegion root{0,0,0,whole};
    const auto plan = build_bounded(source,whole,root,0,tolerance);
    Bytes archive = archive_header(side,3,depth,source);
    archive.reserve(header_bytes+1u+8u+8u+plan->bytes);
    archive.push_back(static_cast<std::uint8_t>(tolerance));
    put_u64(archive,checksum(source));
    put_u64(archive,0); // placeholder for full metadata + payload checksum
    write_bounded_node(*plan,source,whole,root,0,archive);
    require(archive.size() == header_bytes+17u+plan->bytes,"bounded byte count mismatch");
    const std::uint64_t wire_hash = bounded_wire_hash(archive);
    for (unsigned i=0; i<8; ++i)
        archive[header_bytes+9u+i] =
            static_cast<std::uint8_t>((wire_hash>>(8u*i)) & 255u);
    std::size_t cursor = header_bytes;
    const Bytes reconstructed = decode_bounded(archive,cursor,side);
    require(cursor == archive.size(), "bounded encoder trailing data");
    unsigned actual_error = 0;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const unsigned a = source[i], b = reconstructed[i];
        actual_error = std::max(actual_error, a >= b ? a-b : b-a);
    }
    require(actual_error <= tolerance, "sampled error bound not met");
    const std::uint64_t decoded_hash = checksum(reconstructed);
    for (unsigned i = 0; i < 8; ++i)
        archive[9u+i] = static_cast<std::uint8_t>((decoded_hash >> (8u*i)) & 255u);
    if (report) {
        *report = BoundedStats{};
        bounded_stats(*plan,root,0,*report);
        report->measured_max_error = actual_error;
    }
    return archive;
}
