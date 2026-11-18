// DM-UGRE Omega3 PUBLIC INTERFACE DEMONSTRATOR ONLY.
// This file does NOT contain the confidential v1.0/v1.1 engine, policy optimizer,
// private search adapters, invention disclosure, or private execution traces.
// Standard-library-only C++17 deterministic connector + evidence-gate demonstration.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ugrefacade {
struct Address {
    std::uint32_t phase, source, evidence;
    std::uint64_t morton() const {
        if (phase >= (1U << 21) || source >= (1U << 21) || evidence >= (1U << 21))
            throw std::out_of_range("Morton coordinate exceeds 21 bits");
        std::uint64_t out = 0;
        for (unsigned bit = 0; bit < 21; ++bit) {
            out |= (std::uint64_t((phase >> bit) & 1U) << (3U * bit));
            out |= (std::uint64_t((source >> bit) & 1U) << (3U * bit + 1U));
            out |= (std::uint64_t((evidence >> bit) & 1U) << (3U * bit + 2U));
        }
        return out;
    }
};
struct Request { std::string principal; std::string query; std::size_t source_budget = 4; };
struct Record {
    std::string source, id, title, owner;
    bool metadata_attested = false;
};
struct Hit { std::string source, id, title; Address location; };
struct Observation { std::string event; Address location; };
struct Outcome { std::vector<Hit> hits; std::vector<Observation> events; std::size_t sources_called = 0; };

static std::string normalized(std::string text) {
    for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return text;
}
class Connector {
public:
    virtual ~Connector() = default;
    virtual std::string source_id() const = 0;
    virtual bool authorized(const std::string& principal) const = 0;
    virtual std::vector<Record> lookup(const Request& request) const = 0;
};
class SyntheticConnector final : public Connector {
    std::string id_;
    std::set<std::string> allowed_;
    std::vector<Record> fixture_;
public:
    SyntheticConnector(std::string id, std::set<std::string> allowed, std::vector<Record> fixture)
        : id_(std::move(id)), allowed_(std::move(allowed)), fixture_(std::move(fixture)) {}
    std::string source_id() const override { return id_; }
    bool authorized(const std::string& principal) const override {
        return allowed_.count(principal) != 0U;
    }
    std::vector<Record> lookup(const Request& request) const override {
        std::vector<Record> matches;
        for (const auto& r : fixture_) {
            if (normalized(r.title).find(normalized(request.query)) != std::string::npos)
                matches.push_back(r);
        }
        return matches;
    }
};
class Facade {
    std::vector<std::unique_ptr<Connector>> connectors_;
public:
    void add(std::unique_ptr<Connector> connector) {
        if (!connector) throw std::invalid_argument("Null connector");
        connectors_.push_back(std::move(connector));
    }
    Outcome search(const Request& req) const {
        Outcome out;
        if (req.principal.empty() || req.query.empty() || req.source_budget == 0U) {
            out.events.push_back({"INVALID_REQUEST", {0, 0, 0}});
            return out;
        }
        std::set<std::pair<std::string,std::string>> seen;
        for (std::size_t i = 0; i < connectors_.size(); ++i) {
            if (out.sources_called >= req.source_budget) {
                out.events.push_back({"BUDGET_STOP", {1, static_cast<std::uint32_t>(i), 0}});
                break;
            }
            const auto& c = *connectors_[i];
            const Address stage{1, static_cast<std::uint32_t>(i), 0};
            if (!c.authorized(req.principal)) {
                out.events.push_back({"SOURCE_DENIED", stage});
                continue;
            }
            ++out.sources_called;
            out.events.push_back({"SOURCE_QUERIED", stage});
            for (const auto& record : c.lookup(req)) {
                // Authorization and evidence gate BEFORE result publication.
                if (record.source != c.source_id() || record.owner != req.principal ||
                    record.id.empty() || !record.metadata_attested) {
                    out.events.push_back({"RECORD_REJECTED", {2, static_cast<std::uint32_t>(i), 0}});
                    continue;
                }
                if (seen.insert({record.source, record.id}).second) {
                    out.hits.push_back({record.source, record.id, record.title,
                                        {3, static_cast<std::uint32_t>(i), 1}});
                    out.events.push_back({"EVIDENCE_ACCEPTED", {3, static_cast<std::uint32_t>(i), 1}});
                }
            }
        }
        std::sort(out.hits.begin(), out.hits.end(), [](const Hit& a, const Hit& b) {
            return std::tie(a.source, a.id) < std::tie(b.source, b.id);
        });
        out.events.push_back({"STOP", {4, 0, 1}});
        return out;
    }
};
static void insist(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static std::unique_ptr<SyntheticConnector> sample() {
    return std::make_unique<SyntheticConnector>("fixture", std::set<std::string>{"tester"},
        std::vector<Record>{
            {"fixture","r1","Synthetic report","tester",true},
            {"fixture","r1","Synthetic report duplicate","tester",true},
            {"fixture","r2","Synthetic report private","other",true},
            {"fixture","r3","Synthetic report unattested","tester",false},
            {"spoofed","r4","Synthetic report wrong-source","tester",true}
        });
}
static int self_test() {
    {
        Facade f; f.add(sample());
        auto out = f.search({"tester","Synthetic report",4});
        insist(out.hits.size()==1 && out.hits[0].id=="r1", "evidence gate / dedup failure");
        insist(out.sources_called==1, "source-call count failure");
    }
    {
        Facade f; f.add(sample());
        const auto denied = f.search({"guest","Synthetic report",4});
        insist(denied.hits.empty() && denied.sources_called==0, "unauthorized source accessed");
    }
    {
        Facade f; f.add(sample());
        insist(f.search({"tester","",4}).hits.empty(), "empty query accepted");
    }
    {
        Facade f; f.add(sample());
        const auto out = f.search({"tester","Synthetic report",0});
        insist(out.hits.empty() && out.sources_called==0, "zero-budget search executed");
    }
    {
        Facade f; f.add(sample()); f.add(sample());
        const auto out = f.search({"tester","Synthetic report",1});
        insist(out.sources_called==1, "budget exceeded");
        insist(out.events.back().event=="STOP", "trace lacks stop");
    }
    {
        insist(Address{1,2,3}.morton()!=Address{1,3,2}.morton(), "Morton collision");
        bool threw=false;
        try { (void)Address{1U << 21,0,0}.morton(); }
        catch (const std::out_of_range&) { threw=true; }
        insist(threw, "out-of-range coordinate accepted");
    }
    std::cout << "SELF_TEST_PASS 6/6\n";
    return 0;
}
static int demo() {
    Facade facade; facade.add(sample());
    const auto outcome = facade.search({"tester","Synthetic report",4});
    for (const auto& event: outcome.events)
        std::cout << "TRACE " << event.event << " morton=" << event.location.morton() << "\n";
    for (const auto& hit: outcome.hits)
        std::cout << "RESULT " << hit.source << '/' << hit.id << " " << hit.title << "\n";
    return 0;
}
}  // namespace ugrefacade
int main(int argc, char** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="--self-test") return ugrefacade::self_test();
        if (argc==2 && std::string(argv[1])=="--demo") return ugrefacade::demo();
        std::cerr << "Usage: dm_ugre_facade --self-test | --demo\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
