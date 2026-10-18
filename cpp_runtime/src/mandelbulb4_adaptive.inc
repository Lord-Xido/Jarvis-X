// Adaptive MB4C mode 2: cost-gated four-way spatial partitioning.
// Included inside namespace mb4 after archive_header() in mandelbulb4_main.cpp.
// No distance-estimator guarantees: decisions use actual byte-field variation.
struct AdaptiveRegion {
    int x, y, z;
    Shape shape;
};
struct AdaptiveNode {
    std::uint8_t tag = 0; // 0 constant, 1 literal voxels, 2 split four ways
    std::uint8_t value = 0;
    std::size_t bytes = 0;
    std::array<std::unique_ptr<AdaptiveNode>, 4> children{};
};
struct AdaptiveStats {
    std::size_t nodes = 0, split_nodes = 0, constant_leaves = 0, raw_leaves = 0;
    std::size_t deepest_split = 0;
};

AdaptiveRegion child_region(const AdaptiveRegion& parent, unsigned level, unsigned j) {
    require(j < 4, "invalid adaptive child index");
    const auto axes = split_axes(level);
    const Shape next = reduced(parent.shape, level);
    std::array<int,3> pos{{parent.x, parent.y, parent.z}};
    pos[static_cast<std::size_t>(axes[0])] += ((j & 1u) != 0u)
        ? axis_value(next, axes[0]) : 0;
    pos[static_cast<std::size_t>(axes[1])] += ((j & 2u) != 0u)
        ? axis_value(next, axes[1]) : 0;
    return AdaptiveRegion{pos[0], pos[1], pos[2], next};
}

std::unique_ptr<AdaptiveNode> build_adaptive(
    const Bytes& source, Shape whole, const AdaptiveRegion& region, unsigned level) {
    auto node = std::make_unique<AdaptiveNode>();
    const auto n = count(region.shape);
    const std::uint8_t initial = source[at(whole, region.x, region.y, region.z)];
    bool constant = true;
    for (int z = 0; z < region.shape.z && constant; ++z)
        for (int y = 0; y < region.shape.y && constant; ++y)
            for (int x = 0; x < region.shape.x; ++x)
                if (source[at(whole, region.x + x, region.y + y, region.z + z)] != initial) {
                    constant = false;
                    break;
                }
    if (constant) {
        node->tag = 0;
        node->value = initial;
        node->bytes = 2;
        return node;
    }
    node->tag = 1;
    node->bytes = 1u + n;
    if (!split_possible(region.shape, level)) return node;

    std::array<std::unique_ptr<AdaptiveNode>, 4> choices;
    std::size_t split_bytes = 1;
    for (unsigned j = 0; j < 4; ++j) {
        choices[j] = build_adaptive(source, whole, child_region(region, level, j), level + 1);
        split_bytes += choices[j]->bytes;
    }
    // Split only when it is strictly smaller than the uncompressed local literal.
    // Uniform Mandelbulb interiors remain one node; complicated boundaries may
    // remain literal where subdivision gives no actual storage savings.
    if (split_bytes < node->bytes) {
        node->tag = 2;
        node->bytes = split_bytes;
        node->children = std::move(choices);
    }
    return node;
}

void collect_adaptive_stats(const AdaptiveNode& node, unsigned level, AdaptiveStats& stats) {
    ++stats.nodes;
    if (node.tag == 0) {
        ++stats.constant_leaves;
    } else if (node.tag == 1) {
        ++stats.raw_leaves;
    } else {
        require(node.tag == 2, "invalid adaptive plan");
        ++stats.split_nodes;
        stats.deepest_split = std::max(stats.deepest_split, static_cast<std::size_t>(level + 1u));
        for (const auto& child : node.children) {
            require(static_cast<bool>(child), "null adaptive child");
            collect_adaptive_stats(*child, level + 1, stats);
        }
    }
}

void write_adaptive_node(const AdaptiveNode& node, const Bytes& source, Shape whole,
                         const AdaptiveRegion& region, unsigned level, Bytes& out) {
    out.push_back(node.tag);
    if (node.tag == 0) {
        out.push_back(node.value);
    } else if (node.tag == 1) {
        for (int z = 0; z < region.shape.z; ++z)
            for (int y = 0; y < region.shape.y; ++y)
                for (int x = 0; x < region.shape.x; ++x)
                    out.push_back(source[at(whole, region.x + x, region.y + y, region.z + z)]);
    } else if (node.tag == 2) {
        for (unsigned j = 0; j < 4; ++j)
            write_adaptive_node(*node.children[j], source, whole,
                                child_region(region, level, j), level + 1, out);
    } else {
        throw std::runtime_error("unrecognized adaptive plan tag");
    }
}
void decode_adaptive_node(const Bytes& in, std::size_t& pos, Bytes& output, Shape whole,
                          const AdaptiveRegion& region, unsigned level) {
    const std::uint8_t tag = get_byte(in, pos);
    if (tag == 2) {
        require(split_possible(region.shape, level), "invalid adaptive split geometry");
        for (unsigned j = 0; j < 4; ++j)
            decode_adaptive_node(in, pos, output, whole,
                                 child_region(region, level, j), level + 1);
        return;
    }
    require(tag == 0 || tag == 1, "invalid adaptive node tag");
    const std::uint8_t value = tag == 0 ? get_byte(in, pos) : 0;
    require(tag != 1 || count(region.shape) <= in.size() - pos,
            "truncated adaptive raw node");
    for (int z = 0; z < region.shape.z; ++z)
        for (int y = 0; y < region.shape.y; ++y)
            for (int x = 0; x < region.shape.x; ++x)
                output[at(whole, region.x + x, region.y + y, region.z + z)] =
                    tag == 0 ? value : get_byte(in, pos);
}
Bytes encode_adaptive(const Bytes& source, unsigned side, AdaptiveStats* report = nullptr) {
    const unsigned depth = levels_for(side);
    const Shape whole{static_cast<int>(side), static_cast<int>(side), static_cast<int>(side)};
    require(source.size() == count(whole), "wrong adaptive voxel count");
    const AdaptiveRegion root{0, 0, 0, whole};
    const auto plan = build_adaptive(source, whole, root, 0);
    Bytes archive = archive_header(side, 2, depth, source);
    archive.reserve(header_bytes + plan->bytes);
    write_adaptive_node(*plan, source, whole, root, 0, archive);
    require(archive.size() == header_bytes + plan->bytes, "adaptive cost accounting mismatch");
    if (report) collect_adaptive_stats(*plan, 0, *report);
    return archive;
}
Bytes decode_adaptive(const Bytes& in, std::size_t& pos, unsigned side) {
    const Shape whole{static_cast<int>(side), static_cast<int>(side), static_cast<int>(side)};
    Bytes output(count(whole), 0);
    decode_adaptive_node(in, pos, output, whole, AdaptiveRegion{0, 0, 0, whole}, 0);
    return output;
}
