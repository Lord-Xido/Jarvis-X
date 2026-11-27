# Nano3D: mathematical and operational model

## Kinetic state

For a 3D context key `c=(x,y,z)` with each axis in `[0,31]`, offline software trains a 3-input/8-output linear-softmax network, then stores `(argmax_class, quantized_class_probability)` in a 32³ array. Every coordinate is mapped to a unique 15-bit Morton index. A code-template generator converts the selected class into a syntactically valid C++ function definition.

Offline:

- `f_i=(2*c_i+1)/32 - 1`.
- `a_k = w_k dot f + b_k`.
- `p(k|c)=exp(a_k)/sum_j exp(a_j)`.
- `P[Morton3(c)]=(argmax_k p(k|c), round(255*max p(k|c)))`.

Online:

- `template := P[Morton3(context)]`.
- `candidate := emit(template, unique_symbol_id)`.
- `check := external C++ compiler -fsyntax-only`.
- `commit` only if all files in a release pass validation; otherwise leave latest release unchanged.

The synthetic task's correct class is the octant of `c` and can be learned by a linear softmax classifier. **This does not demonstrate prediction of useful C++ algorithms from a natural-language prompt.** Semantic symbol tables, AST context and correctness requirements are not represented by these 3D coordinates.

## Capacity

- Target `1e6 LOC/s` = one LOC every `1000 ns` *in aggregate*.
- At 8.2 output tokens/LOC, target output token flux = `8.2e6 tokens/s` and the average system-wide period is `121.9512 ns/token`.
- At 30% full-file reuse, required newly generated tokens = `5.74e6 tokens/s`, provided cache retrieval capacity is sufficient.
- With an independent cache service of capacity `C_hit` LOC/s and inference service of capacity `C_miss` LOC/s, sustainable output must satisfy `R <= C_hit/h` and `R <= C_miss/(1-h)`, where `h` is cache hit probability; it is also limited by planning, compilation and emission capacities.
- Queue conservation: `dQ_i/dt=a_i(t)-d_i(t)`; any enduring `a_i > d_i` causes a growing backlog.

## Measurements and interpretation

In a previous locally executed C++17 prototype experiment (historical observations; not CI performance criteria):

- synthetic grid classification: 32,768 / 32,768 correct;
- precomputed table: 65,536 bytes;
- amortized hot lookup benchmark: around 0.54 ns per query;
- amortized online ANN class prediction: around 73.8 ns per query;
- template source emission: 108,800 LOC in around 0.0206 s (approximately 5.29M simple one-line-template LOC/s, buffered writes, not durable disk or compiler throughput);
- syntax-checked and committed publication: 108,800 LOC across 128 files in around 2.55 s (roughly 42.6k syntax-accepted LOC/s).

The first timing is *throughput divided by query count*, not individual dependent cache-access latency. The test uses a synthetic 3D oracle with preplanned task IDs, very repetitive templates, a hot resident table, a single CPU process and short generated compilation units. None of these local numbers proves general-purpose 1M correct C++ LOC/s.

For online ANN computation time `t_ann` and cached lookup amortized time `t_hit`, with additional precompute time `T_pre`, the break-even number of reused queries in a sequential simplified model is `N_break_even=T_pre/(t_ann-t_hit)`. This compares precomputation against reuse *after the same ANN has already been trained*. Training costs should be added only if the baseline requires separate training too.

## Release semantics

`verify_commit.py` creates all source candidates in a temporary directory, checks full file/line counts, uses an independent C++ syntax checker, writes SHA-256 source receipts and atomically renames the release directory. It then atomically replaces a latest-pointer JSON file. On failure it removes candidate files and leaves the latest pointer unchanged. This is bounded prototype publication, not cryptographic signing, multi-writer isolation, power-failure durability, full C++ link testing, unit-test validation or any security certification.