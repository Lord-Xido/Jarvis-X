#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gcain {

// -----------------------------------------------------------------------------
// Geometry: X=Information, Y=Cognition, Z=Action.
// Each event in the system occupies a point in this 3D state-space.
// -----------------------------------------------------------------------------
struct Vec3 {
    double x{0.0}, y{0.0}, z{0.0};
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
};

static double dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static double norm(const Vec3& v) { return std::sqrt(dot(v,v)); }
static Vec3 normalized(const Vec3& v) {
    const double n = norm(v);
    return n > 1e-12 ? v * (1.0 / n) : Vec3{};
}
static double cosine(const Vec3& a, const Vec3& b) {
    const double d = norm(a) * norm(b);
    return d > 1e-12 ? dot(a,b)/d : 0.0;
}

// -----------------------------------------------------------------------------
// Operational 3D kinetics on the semantic state manifold.
// Position r=(Information,Cognition,Action), velocity v=dr/dt,
// acceleration a=dv/dt. Dynamics are a damped spring toward a semantic target:
//      m r'' + gamma r' + k(r-q) = 0
// integrated with semi-implicit Euler.
// -----------------------------------------------------------------------------
struct KineticState3D {
    Vec3 r{0.50, 0.50, 0.25};
    Vec3 v{0.0, 0.0, 0.0};
    Vec3 a{0.0, 0.0, 0.0};
    double kinetic_energy{0.0};
    double potential_energy{0.0};
    uint64_t steps{0};
};

struct KineticDiagnostics {
    double max_kinematic_residual{0.0};
    double max_dynamic_residual{0.0};
    double max_energy{0.0};
    bool finite{true};
    bool bounded{true};
    uint64_t projected_steps{0};
};

class Kinetics3D {
public:
    Kinetics3D(double mass=1.0, double stiffness=2.4, double damping=1.8, double dt=0.05)
        : m_(mass), k_(stiffness), gamma_(damping), dt_(dt) {}

    void integrate(KineticState3D& s, const Vec3& target, int substeps, KineticDiagnostics& d) const {
        for (int i=0; i<substeps; ++i) {
            const Vec3 r0 = s.r;
            const Vec3 v0 = s.v;
            const Vec3 spring = (target - s.r) * k_;
            const Vec3 drag = s.v * (-gamma_);
            const Vec3 force = spring + drag;
            const Vec3 a0 = force * (1.0 / m_);

            Vec3 v1 = s.v + a0 * dt_;
            Vec3 r1 = s.r + v1 * dt_;

            bool projected = false;
            auto project_axis = [&](double& r, double& v) {
                if (r < 0.0) { r = 0.0; v = 0.0; projected = true; }
                if (r > 1.0) { r = 1.0; v = 0.0; projected = true; }
            };
            project_axis(r1.x, v1.x);
            project_axis(r1.y, v1.y);
            project_axis(r1.z, v1.z);

            if (!projected) {
                const Vec3 kin_res = (r1 - r0) * (1.0/dt_) - v1;
                const Vec3 dyn_res = (v1 - v0) * (1.0/dt_) - a0;
                d.max_kinematic_residual = std::max(d.max_kinematic_residual, norm(kin_res));
                d.max_dynamic_residual = std::max(d.max_dynamic_residual, norm(dyn_res));
            } else {
                ++d.projected_steps;
            }

            s.r = r1;
            s.v = v1;
            s.a = a0;
            s.kinetic_energy = 0.5 * m_ * dot(s.v, s.v);
            const Vec3 e = target - s.r;
            s.potential_energy = 0.5 * k_ * dot(e,e);
            ++s.steps;

            d.max_energy = std::max(d.max_energy, s.kinetic_energy + s.potential_energy);
            d.finite = d.finite && std::isfinite(s.r.x) && std::isfinite(s.r.y) && std::isfinite(s.r.z)
                                && std::isfinite(s.v.x) && std::isfinite(s.v.y) && std::isfinite(s.v.z)
                                && std::isfinite(s.a.x) && std::isfinite(s.a.y) && std::isfinite(s.a.z);
            d.bounded = d.bounded && s.r.x >= 0.0 && s.r.x <= 1.0
                                  && s.r.y >= 0.0 && s.r.y <= 1.0
                                  && s.r.z >= 0.0 && s.r.z <= 1.0;
        }
    }

    bool contraction_self_test(double& start_error, double& end_error,
                               double& energy0, double& energyN) const {
        KineticState3D s;
        s.r = {0.12, 0.18, 0.08};
        const Vec3 q{0.84, 0.80, 0.76};
        start_error = norm(q - s.r);
        const Vec3 e0 = q - s.r;
        energy0 = 0.5*k_*dot(e0,e0);
        KineticDiagnostics d;
        integrate(s, q, 240, d);
        end_error = norm(q - s.r);
        energyN = s.kinetic_energy + s.potential_energy;
        return d.finite && d.bounded && end_error < 1e-3 && energyN < energy0
            && d.max_kinematic_residual < 1e-12 && d.max_dynamic_residual < 1e-12;
    }

    double dt() const { return dt_; }

private:
    double m_, k_, gamma_, dt_;
};

// -----------------------------------------------------------------------------
// Stable hash for provenance / event IDs (FNV-1a 64-bit).
// -----------------------------------------------------------------------------
static uint64_t fnv1a64(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

struct Provenance {
    std::string origin;
    uint64_t content_hash{0};
    bool verified{false};
};

struct Content {
    uint64_t id{0};
    uint64_t author_id{0};
    std::string text;
    Vec3 topic;
    double quality{0.5};
    double novelty{0.5};
    double trust{0.5};
    Provenance provenance;
};

struct Human {
    uint64_t id{0};
    std::string name;
    Vec3 interests;
    double attention_budget{1.0};
};

struct Agent {
    uint64_t id{0};
    uint64_t owner_human{0};
    std::string name;
    Vec3 policy;          // what this agent considers relevant
    double autonomy{0.5};
    double trust_threshold{0.5};
};

struct WorldState {
    // A compact environmental state: cost, congestion, knowledge, risk.
    double resource_cost{1.0};
    double congestion{0.3};
    double knowledge{0.2};
    double risk{0.4};

    std::string str() const {
        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << "{cost=" << resource_cost
           << ", congestion=" << congestion
           << ", knowledge=" << knowledge
           << ", risk=" << risk << "}";
        return os.str();
    }
};

enum class Stage {
    Chatroom = 1,
    Forum = 2,
    Web = 3,
    SocialGraph = 4,
    GlobalMultimedia = 5,
    AIMediator = 6,
    AgentNetwork = 7,
    AmbientIntelligence = 8
};

static const char* stage_name(Stage s) {
    switch (s) {
        case Stage::Chatroom: return "Chatroom";
        case Stage::Forum: return "Forum";
        case Stage::Web: return "World Wide Web";
        case Stage::SocialGraph: return "Social Graph";
        case Stage::GlobalMultimedia: return "Global Multimedia";
        case Stage::AIMediator: return "AI Mediator";
        case Stage::AgentNetwork: return "Agent Network";
        case Stage::AmbientIntelligence: return "Ambient Intelligence";
    }
    return "Unknown";
}

struct Event {
    Stage stage{Stage::Chatroom};
    uint64_t actor_id{0};
    std::string description;
    Vec3 state_point;
    uint64_t hash{0};
};

// -----------------------------------------------------------------------------
// Social graph: humans and agents are vertices; typed edges are kept as strings.
// -----------------------------------------------------------------------------
class Graph {
public:
    void connect(uint64_t a, uint64_t b, std::string type) {
        edges_[a].push_back({b, std::move(type)});
    }

    const std::vector<std::pair<uint64_t,std::string>>& neighbors(uint64_t id) const {
        static const std::vector<std::pair<uint64_t,std::string>> empty;
        auto it = edges_.find(id);
        return it == edges_.end() ? empty : it->second;
    }

    size_t edge_count() const {
        size_t n = 0;
        for (const auto& [_, v] : edges_) n += v.size();
        return n;
    }

private:
    std::unordered_map<uint64_t, std::vector<std::pair<uint64_t,std::string>>> edges_;
};

// -----------------------------------------------------------------------------
// Recommendation model: simple, auditable scoring rather than opaque ML.
// score = semantic relevance + trust + quality + novelty + social affinity.
// -----------------------------------------------------------------------------
class Recommender {
public:
    explicit Recommender(const Graph& graph) : graph_(graph) {}

    double score(const Human& h, const Content& c) const {
        const double rel = 0.5 * (cosine(h.interests, c.topic) + 1.0); // [0,1]
        const double social = social_affinity(h.id, c.author_id);
        return 0.34*rel + 0.24*c.trust + 0.18*c.quality + 0.14*c.novelty + 0.10*social;
    }

    std::vector<const Content*> rank(const Human& h,
                                     const std::vector<Content>& corpus,
                                     size_t top_k) const {
        std::vector<const Content*> items;
        items.reserve(corpus.size());
        for (const auto& c : corpus) items.push_back(&c);
        std::stable_sort(items.begin(), items.end(), [&](const Content* a, const Content* b) {
            return score(h, *a) > score(h, *b);
        });
        if (items.size() > top_k) items.resize(top_k);
        return items;
    }

private:
    double social_affinity(uint64_t human, uint64_t author) const {
        if (human == author) return 1.0;
        for (const auto& [n, type] : graph_.neighbors(human)) {
            if (n == author && (type == "friend" || type == "follows")) return 0.85;
        }
        return 0.25;
    }

    const Graph& graph_;
};

// -----------------------------------------------------------------------------
// AI mediator: compresses a ranked information set into a trusted situation
// model. In a production system this module can be replaced by an LLM.
// -----------------------------------------------------------------------------
struct SituationModel {
    Vec3 centroid{};
    double confidence{0.0};
    double urgency{0.0};
    std::string summary;
};

class Mediator {
public:
    SituationModel synthesize(const Human& h,
                              const std::vector<const Content*>& ranked) const {
        SituationModel out;
        if (ranked.empty()) {
            out.summary = "No trusted evidence available.";
            return out;
        }

        double weight_sum = 0.0;
        double trusted_sum = 0.0;
        std::ostringstream ss;
        ss << "For " << h.name << ": ";

        for (size_t i = 0; i < ranked.size(); ++i) {
            const auto& c = *ranked[i];
            const double w = std::max(0.05, c.quality*c.trust);
            out.centroid = out.centroid + c.topic*w;
            weight_sum += w;
            trusted_sum += c.trust;
            if (i < 3) {
                if (i) ss << " | ";
                ss << c.text;
            }
        }

        out.centroid = out.centroid * (1.0 / weight_sum);
        out.confidence = trusted_sum / static_cast<double>(ranked.size());
        out.urgency = std::clamp(0.5*out.centroid.z + 0.5*(1.0-out.confidence), 0.0, 1.0);
        out.summary = ss.str();
        return out;
    }
};

// -----------------------------------------------------------------------------
// Agent action layer: converts a situation model into a bounded action.
// -----------------------------------------------------------------------------
struct Action {
    uint64_t agent_id{0};
    std::string verb;
    double magnitude{0.0};
    bool approved{false};
};

class PolicyEngine {
public:
    Action decide(const Agent& a, const SituationModel& s, const WorldState& w) const {
        const double alignment = 0.5*(cosine(a.policy, s.centroid)+1.0);
        const double safety = std::clamp(1.0 - w.risk, 0.0, 1.0);
        const double confidence_gate = std::clamp((s.confidence - a.trust_threshold + 1.0), 0.0, 1.0);
        const double utility = alignment * safety * confidence_gate * a.autonomy;

        Action act;
        act.agent_id = a.id;
        act.magnitude = utility;
        act.approved = utility > 0.20;
        act.verb = act.approved ? "allocate_learning_resource" : "request_human_review";
        return act;
    }
};

class WorldActuator {
public:
    void apply(const Action& a, WorldState& w) const {
        if (!a.approved) {
            w.risk = std::clamp(w.risk - 0.01, 0.0, 1.0);
            return;
        }
        // A safe simulated action: spend some resource, increase knowledge,
        // slightly reduce uncertainty/risk, and add small network congestion.
        w.resource_cost += 0.08 * a.magnitude;
        w.knowledge = std::clamp(w.knowledge + 0.18 * a.magnitude, 0.0, 1.0);
        w.risk = std::clamp(w.risk - 0.07 * a.magnitude, 0.0, 1.0);
        w.congestion = std::clamp(w.congestion + 0.02 * a.magnitude, 0.0, 1.0);
    }
};

// -----------------------------------------------------------------------------
// End-to-end engine.
// -----------------------------------------------------------------------------
class CognitiveActionNetwork {
public:
    CognitiveActionNetwork() : recommender_(graph_) {}

    void seed() {
        humans_ = {
            {1, "Ada",    normalized({0.9,0.7,0.5}), 1.0},
            {2, "Turing", normalized({0.8,0.9,0.7}), 1.0},
            {3, "Maya",   normalized({0.4,0.8,0.9}), 1.0}
        };

        agents_ = {
            {101, 1, "Ada.Agent",    normalized({0.8,0.8,0.6}), 0.75, 0.55},
            {102, 2, "Turing.Agent", normalized({0.7,0.9,0.8}), 0.70, 0.60},
            {103, 3, "Maya.Agent",   normalized({0.5,0.7,1.0}), 0.80, 0.50}
        };

        kinetic_[101].r = {0.45, 0.45, 0.20};
        kinetic_[102].r = {0.48, 0.50, 0.22};
        kinetic_[103].r = {0.42, 0.52, 0.24};

        graph_.connect(1,2,"friend"); graph_.connect(2,1,"friend");
        graph_.connect(2,3,"follows"); graph_.connect(3,2,"follows");
        graph_.connect(1,101,"owns"); graph_.connect(2,102,"owns"); graph_.connect(3,103,"owns");
        graph_.connect(101,102,"agent-peer"); graph_.connect(102,103,"agent-peer"); graph_.connect(103,101,"agent-peer");

        add_content(1, "Real-time chat: compute is becoming social.", {0.55,0.35,0.15}, 0.72, 0.60, true);
        add_content(2, "Forum thread: persistent discussion outlives presence.", {0.65,0.45,0.20}, 0.82, 0.65, true);
        add_content(3, "Web graph: documents become globally addressable and linked.", {0.75,0.52,0.25}, 0.88, 0.70, true);
        add_content(1, "Social graph: identity and relationships organize discovery.", {0.78,0.62,0.35}, 0.83, 0.72, true);
        add_content(3, "Multimedia feeds create information abundance and attention scarcity.", {0.88,0.68,0.55}, 0.86, 0.80, true);
        add_content(2, "AI mediators compress streams into personal situation models.", {0.90,0.88,0.62}, 0.92, 0.90, true);
        add_content(3, "Agents transform intent into bounded, verifiable actions.", {0.86,0.92,0.88}, 0.95, 0.94, true);
        add_content(1, "Ambient intelligence closes the loop between digital state and the physical world.", {0.92,0.95,0.98}, 0.96, 0.96, true);
    }

    bool run(size_t steps = 3) {
        std::cout << "=== Global Cognitive-Action Network / C++17 runtime ===\n";
        std::cout << "State-space axes: X=Information, Y=Cognition, Z=Action\n\n";

        emit_historical_stages();

        for (size_t t = 0; t < steps; ++t) {
            std::cout << "\n--- Ambient loop t=" << t << " ---\n";
            std::cout << "World before: " << world_.str() << "\n";

            for (const auto& h : humans_) {
                const auto ranked = recommender_.rank(h, corpus_, 5);
                const auto situation = mediator_.synthesize(h, ranked);
                const Agent* agent = find_agent_for(h.id);
                if (!agent) continue;

                const auto action = policy_.decide(*agent, situation, world_);

                const Vec3 kinetic_target{
                    std::clamp(situation.centroid.x, 0.0, 1.0),
                    std::clamp(situation.centroid.y, 0.0, 1.0),
                    std::clamp(action.magnitude, 0.0, 1.0)
                };
                auto& ks = kinetic_[agent->id];
                kinetics_.integrate(ks, kinetic_target, 20, kinetic_diag_);

                actuator_.apply(action, world_);

                log(Stage::AIMediator, h.id,
                    "synthesized situation model; confidence=" + f3(situation.confidence),
                    {situation.centroid.x, situation.centroid.y, 0.62});

                log(Stage::AgentNetwork, agent->id,
                    action.verb + "; utility=" + f3(action.magnitude),
                    ks.r);

                log(Stage::AmbientIntelligence, agent->id,
                    "world-state feedback applied",
                    ks.r);

                std::cout << h.name << " <- " << situation.summary << "\n";
                std::cout << "  agent=" << agent->name
                          << " action=" << action.verb
                          << " magnitude=" << f3(action.magnitude)
                          << " approved=" << (action.approved ? "yes" : "no")
                          << " kinetic_r=(" << f3(ks.r.x) << "," << f3(ks.r.y) << "," << f3(ks.r.z) << ")"
                          << " |v|=" << f3(norm(ks.v))
                          << " |a|=" << f3(norm(ks.a)) << "\n";
            }

            feedback_update();
            std::cout << "World after : " << world_.str() << "\n";
        }

        const bool verified = verify_invariants();
        print_summary();
        return verified;
    }

private:
    static std::string f3(double x) {
        std::ostringstream os; os << std::fixed << std::setprecision(3) << x; return os.str();
    }

    void add_content(uint64_t author, std::string text, Vec3 topic,
                     double quality, double trust, bool verified) {
        const uint64_t id = corpus_.size() + 1;
        Provenance p{"local-demo://seed/" + std::to_string(id), fnv1a64(text), verified};
        corpus_.push_back({id, author, std::move(text), normalized(topic), quality, 0.65, trust, p});
    }

    const Agent* find_agent_for(uint64_t human_id) const {
        for (const auto& a : agents_) if (a.owner_human == human_id) return &a;
        return nullptr;
    }

    void log(Stage stage, uint64_t actor, std::string description, Vec3 p) {
        const uint64_t sequence = static_cast<uint64_t>(events_.size());
        const std::string material = std::to_string(sequence) + ":" +
                                     std::to_string(static_cast<int>(stage)) + ":" +
                                     std::to_string(actor) + ":" + description + ":" +
                                     f3(p.x) + ":" + f3(p.y) + ":" + f3(p.z);
        events_.push_back({stage, actor, std::move(description), p, fnv1a64(material)});
    }

    void emit_historical_stages() {
        struct SeedStage { Stage s; Vec3 p; const char* msg; };
        const std::array<SeedStage,8> stages{{
            {Stage::Chatroom,            {0.18,0.10,0.05}, "real-time messages in a shared digital place"},
            {Stage::Forum,               {0.28,0.16,0.07}, "persistent topic-thread-reply structure"},
            {Stage::Web,                 {0.42,0.22,0.10}, "globally linked addressable documents"},
            {Stage::SocialGraph,         {0.55,0.35,0.18}, "persistent identities and relationship graph"},
            {Stage::GlobalMultimedia,    {0.72,0.48,0.25}, "continuous text-image-audio-video streams"},
            {Stage::AIMediator,          {0.82,0.70,0.45}, "personal compression, synthesis, reasoning"},
            {Stage::AgentNetwork,        {0.90,0.83,0.75}, "human-agent-agent coordination and action"},
            {Stage::AmbientIntelligence, {1.00,0.95,1.00}, "continuous digital-physical closed loop"}
        }};

        std::cout << "Evolutionary geometry:\n";
        for (const auto& s : stages) {
            log(s.s, 0, s.msg, s.p);
            std::cout << "  " << std::setw(20) << std::left << stage_name(s.s)
                      << " -> (" << f3(s.p.x) << ", " << f3(s.p.y) << ", " << f3(s.p.z) << ")\n";
        }
    }

    void feedback_update() {
        // Closed-loop refinement: as collective knowledge increases, improve the
        // trust of verified content and slightly reduce novelty of repeated items.
        for (auto& c : corpus_) {
            if (c.provenance.verified) {
                c.trust = std::clamp(c.trust + 0.015 * world_.knowledge, 0.0, 1.0);
            }
            c.novelty = std::clamp(c.novelty * 0.985, 0.0, 1.0);
        }

        // World slowly relaxes between cycles.
        world_.congestion = std::clamp(world_.congestion * 0.97, 0.0, 1.0);
    }

    bool verify_invariants() const {
        bool ok = true;
        std::unordered_set<uint64_t> seen;
        for (const auto& e : events_) {
            ok = ok && seen.insert(e.hash).second;
            ok = ok && e.state_point.x >= 0.0 && e.state_point.x <= 1.0;
            ok = ok && e.state_point.y >= 0.0 && e.state_point.y <= 1.0;
            ok = ok && e.state_point.z >= 0.0 && e.state_point.z <= 1.0;
        }
        for (const auto& c : corpus_) {
            ok = ok && (fnv1a64(c.text) == c.provenance.content_hash);
        }
        double start_err=0.0, end_err=0.0, e0=0.0, eN=0.0;
        const bool kinetic_self_test = kinetics_.contraction_self_test(start_err, end_err, e0, eN);
        const bool kinetic_residuals = kinetic_diag_.finite && kinetic_diag_.bounded
            && kinetic_diag_.max_kinematic_residual < 1e-12
            && kinetic_diag_.max_dynamic_residual < 1e-12;
        ok = ok && kinetic_self_test && kinetic_residuals;

        std::cout << "\nVerification: " << (ok ? "PASS" : "FAIL")
                  << " (geometry + provenance + operational 3D kinetics)\n";
        std::cout << "Kinetics: r=(I,C,A), v=dr/dt, a=dv/dt, m*a=k(q-r)-gamma*v\n";
        std::cout << "  dt=" << f3(kinetics_.dt())
                  << " max kinematic residual=" << std::scientific << kinetic_diag_.max_kinematic_residual
                  << " max dynamic residual=" << kinetic_diag_.max_dynamic_residual << std::fixed << "\n";
        std::cout << "  bounded=" << (kinetic_diag_.bounded ? "yes" : "no")
                  << " finite=" << (kinetic_diag_.finite ? "yes" : "no")
                  << " projected_steps=" << kinetic_diag_.projected_steps << "\n";
        std::cout << "  damped-contraction self-test: error " << f3(start_err) << " -> " << f3(end_err)
                  << ", energy " << f3(e0) << " -> " << f3(eN)
                  << " => " << (kinetic_self_test ? "PASS" : "FAIL") << "\n";
        return ok;
    }

    void print_summary() const {
        std::array<size_t,9> counts{};
        Vec3 centroid{};
        for (const auto& e : events_) {
            counts[static_cast<int>(e.stage)]++;
            centroid = centroid + e.state_point;
        }
        if (!events_.empty()) centroid = centroid * (1.0 / static_cast<double>(events_.size()));

        std::cout << "\n=== System summary ===\n";
        std::cout << "Humans  : " << humans_.size() << "\n";
        std::cout << "Agents  : " << agents_.size() << "\n";
        std::cout << "Content : " << corpus_.size() << "\n";
        std::cout << "Edges   : " << graph_.edge_count() << "\n";
        std::cout << "Events  : " << events_.size() << "\n";
        std::cout << "Event centroid = (" << f3(centroid.x) << ", " << f3(centroid.y) << ", " << f3(centroid.z) << ")\n";
        std::cout << "Final world state: " << world_.str() << "\n";
        std::cout << "\nPer-stage event counts:\n";
        for (int i = 1; i <= 8; ++i) {
            std::cout << "  " << std::setw(20) << std::left << stage_name(static_cast<Stage>(i))
                      << " : " << counts[i] << "\n";
        }
    }

    std::vector<Human> humans_;
    std::vector<Agent> agents_;
    std::vector<Content> corpus_;
    Graph graph_;
    Recommender recommender_;
    Mediator mediator_;
    PolicyEngine policy_;
    WorldActuator actuator_;
    Kinetics3D kinetics_;
    std::unordered_map<uint64_t, KineticState3D> kinetic_;
    mutable KineticDiagnostics kinetic_diag_;
    WorldState world_;
    std::vector<Event> events_;
};

} // namespace gcain

int main(int argc, char** argv) {
    size_t steps = 3;
    if (argc > 1) {
        try { steps = static_cast<size_t>(std::stoul(argv[1])); }
        catch (...) {
            std::cerr << "Usage: " << argv[0] << " [steps]\n";
            return 2;
        }
    }

    gcain::CognitiveActionNetwork engine;
    engine.seed();
    return engine.run(steps) ? 0 : 1;
}