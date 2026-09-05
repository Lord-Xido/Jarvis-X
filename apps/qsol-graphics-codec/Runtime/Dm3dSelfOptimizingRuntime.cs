using System.Numerics;

namespace QSol.GraphicsCodec.Runtime;

internal readonly record struct Dm3dRuntimeConfig(
    double Alpha,
    int TopK,
    int QuantizationBits,
    int CacheShards,
    int TileSize);

internal readonly record struct Dm3dMetrics(
    double Quality,
    double Latency,
    double Memory,
    double EnergyProxy,
    double Objective);

internal readonly record struct Dm3dInwardStep(
    int Iteration,
    double Radius,
    int TopK,
    int ResolvedBitsPerAxis,
    int FreeBitsPerAxis,
    long CandidateCells);

internal readonly record struct Dm3dEvidence(
    ulong[] BinaryCode,
    byte Authority,
    bool GraphValid,
    bool ContradictionFree);

internal sealed record Dm3dOptimizationResult(
    Dm3dRuntimeConfig Baseline,
    Dm3dRuntimeConfig Selected,
    Dm3dMetrics BaselineMetrics,
    Dm3dMetrics SelectedMetrics,
    IReadOnlyList<(Dm3dRuntimeConfig Config, Dm3dMetrics Metrics, bool Committed)> Trials);

internal static class Dm3dSelfOptimizingRuntime
{
    public const int AxisCells = 1_024_000;
    public const int BitsPerAxis = 20;
    public const int LatentBits = 256;
    public const double MinimumVerificationQuality = 0.95;

    public static IReadOnlyList<Dm3dInwardStep> BuildInwardSchedule(
        Dm3dRuntimeConfig config,
        int maxIterations = 8)
    {
        ValidateConfig(config);
        if (maxIterations is < 1 or > 64)
            throw new ArgumentOutOfRangeException(nameof(maxIterations));

        var radius = 16.0;
        var topK = config.TopK;
        var resolvedBits = 16;
        var steps = new List<Dm3dInwardStep>(maxIterations);

        for (var iteration = 0; iteration < maxIterations; iteration++)
        {
            var freeBits = Math.Max(0, BitsPerAxis - resolvedBits);
            var candidateCells = CandidateCells(freeBits);
            steps.Add(new Dm3dInwardStep(
                iteration,
                radius,
                topK,
                resolvedBits,
                freeBits,
                candidateCells));

            if (topK <= 8 || freeBits == 0)
                break;

            radius *= config.Alpha;
            topK = Math.Max(8, topK >> 3);
            resolvedBits = Math.Min(BitsPerAxis, resolvedBits + 1);
        }

        return steps;
    }

    public static ulong Morton3D20(uint x, uint y, uint z)
    {
        if (x >= AxisCells || y >= AxisCells || z >= AxisCells)
            throw new ArgumentOutOfRangeException(nameof(x), "Coordinates must fit the virtual 1,024,000-cell axis.");

        ulong morton = 0;
        for (var bit = 0; bit < BitsPerAxis; bit++)
        {
            morton |= ((ulong)(x >> bit) & 1UL) << (3 * bit);
            morton |= ((ulong)(y >> bit) & 1UL) << (3 * bit + 1);
            morton |= ((ulong)(z >> bit) & 1UL) << (3 * bit + 2);
        }

        return morton;
    }

    public static int XnorPopCount256(ReadOnlySpan<ulong> left, ReadOnlySpan<ulong> right)
    {
        if (left.Length < 4 || right.Length < 4)
            throw new ArgumentException("A 256-bit code requires four UInt64 words.");

        var matches = 0;
        for (var i = 0; i < 4; i++)
            matches += BitOperations.PopCount(~(left[i] ^ right[i]));
        return matches;
    }

    public static double VerifyEvidence(
        ReadOnlySpan<ulong> queryCode,
        IReadOnlyList<Dm3dEvidence> evidence)
    {
        if (queryCode.Length < 4)
            throw new ArgumentException("A 256-bit query code requires four UInt64 words.", nameof(queryCode));
        if (evidence.Count == 0)
            return 0;

        var best = 0.0;
        foreach (var item in evidence)
        {
            if (!item.GraphValid || !item.ContradictionFree)
                continue;
            if (item.BinaryCode.Length < 4)
                throw new InvalidOperationException("Evidence codes must contain four UInt64 words.");

            var semantic = XnorPopCount256(queryCode, item.BinaryCode) / 256.0;
            var authority = item.Authority / 255.0;
            best = Math.Max(best, semantic * authority);
        }

        return best;
    }

    public static Dm3dOptimizationResult OptimizeRuntime()
    {
        var baseline = new Dm3dRuntimeConfig(
            Alpha: 0.50,
            TopK: 4096,
            QuantizationBits: 8,
            CacheShards: 0,
            TileSize: 32);

        var candidates = new[]
        {
            new Dm3dRuntimeConfig(0.50, 512, 8, 16, 32),
            new Dm3dRuntimeConfig(0.45, 256, 6, 32, 48),
            new Dm3dRuntimeConfig(0.40, 128, 6, 48, 48),
            new Dm3dRuntimeConfig(0.35, 64, 4, 64, 64)
        };

        var selected = baseline;
        var selectedMetrics = Measure(baseline);
        var baselineMetrics = selectedMetrics;
        var trials = new List<(Dm3dRuntimeConfig Config, Dm3dMetrics Metrics, bool Committed)>();

        foreach (var candidate in candidates)
        {
            var metrics = Measure(candidate);
            var passesGuardrail = metrics.Quality >= MinimumVerificationQuality;
            var better = passesGuardrail && metrics.Objective > selectedMetrics.Objective;
            trials.Add((candidate, metrics, better));

            if (!better)
                continue;

            selected = candidate;
            selectedMetrics = metrics;
        }

        return new Dm3dOptimizationResult(
            baseline,
            selected,
            baselineMetrics,
            selectedMetrics,
            trials);
    }

    public static Dm3dMetrics Measure(Dm3dRuntimeConfig config)
    {
        ValidateConfig(config);

        var quality = 0.88
            + 0.025 * (config.QuantizationBits - 4)
            + 0.000012 * Math.Min(config.TopK, 4096)
            - Math.Abs(config.Alpha - 0.50) * 0.08
            - Math.Max(0, config.TileSize - 48) * 0.0008;
        quality = Math.Clamp(quality, 0.0, 0.999);

        var latency = (config.TopK / 4096.0)
            * (config.QuantizationBits / 8.0)
            * (1.0 / (1.0 + config.CacheShards / 24.0))
            * (32.0 / Math.Max(8, config.TileSize))
            * (0.65 + config.Alpha);

        var memory = 0.25 * (config.TopK / 4096.0)
            + 0.45 * (config.QuantizationBits / 8.0)
            + 0.30 * (config.CacheShards / 64.0);

        var energy = 0.70 * latency + 0.30 * memory;
        var objective = 0.55 * quality - 0.20 * latency - 0.15 * memory - 0.10 * energy;
        return new Dm3dMetrics(quality, latency, memory, energy, objective);
    }

    public static int SelfTest()
    {
        var baseline = new Dm3dRuntimeConfig(0.50, 4096, 8, 0, 32);
        var schedule = BuildInwardSchedule(baseline);
        var expectedVolumes = new long[] { 4096, 512, 64, 8 };

        if (schedule.Count < expectedVolumes.Length)
            throw new InvalidOperationException("DM3D inward schedule terminated too early.");

        for (var i = 0; i < expectedVolumes.Length; i++)
        {
            if (schedule[i].CandidateCells != expectedVolumes[i])
                throw new InvalidOperationException(
                    $"DM3D volume contraction mismatch at t={i}: {schedule[i].CandidateCells} != {expectedVolumes[i]}.");
        }

        if (Morton3D20(1, 0, 0) != 1 || Morton3D20(0, 1, 0) != 2 || Morton3D20(0, 0, 1) != 4)
            throw new InvalidOperationException("DM3D Morton XYZ bit interleave invariant failed.");

        var code = new[]
        {
            0x0123_4567_89AB_CDEFUL,
            0xFEDC_BA98_7654_3210UL,
            0x0F0F_F0F0_AA55_55AAUL,
            0x1357_9BDF_2468_ACE0UL
        };
        var complement = code.Select(static word => ~word).ToArray();

        if (XnorPopCount256(code, code) != 256)
            throw new InvalidOperationException("DM3D XNOR/POPCOUNT identity score failed.");
        if (XnorPopCount256(code, complement) != 0)
            throw new InvalidOperationException("DM3D XNOR/POPCOUNT complement score failed.");

        var evidence = new[]
        {
            new Dm3dEvidence(code, 255, GraphValid: true, ContradictionFree: true),
            new Dm3dEvidence(complement, 255, GraphValid: false, ContradictionFree: false)
        };
        if (VerifyEvidence(code, evidence) < 0.999)
            throw new InvalidOperationException("DM3D verified evidence gate failed.");

        var optimization = OptimizeRuntime();
        if (optimization.SelectedMetrics.Quality < MinimumVerificationQuality)
            throw new InvalidOperationException("DM3D optimizer committed a configuration below the verification guardrail.");
        if (optimization.SelectedMetrics.Objective <= optimization.BaselineMetrics.Objective)
            throw new InvalidOperationException("DM3D optimizer failed to improve the bounded objective.");

        var romA = Dm3dRomImage.Build();
        var romB = Dm3dRomImage.Build();
        if (romA.Length != Dm3dRomImage.RomSize || !romA.AsSpan().SequenceEqual(romB))
            throw new InvalidOperationException("DM3D ROM generation is not deterministic.");

        Console.WriteLine("DM3D inward evidence/graphics runtime self-test PASS");
        foreach (var step in schedule)
        {
            Console.WriteLine(FormattableString.Invariant(
                $"t={step.Iteration} radius={step.Radius:G6} topK={step.TopK} resolved={step.ResolvedBitsPerAxis}/axis candidateCells={step.CandidateCells}"));
        }

        Console.WriteLine(FormattableString.Invariant(
            $"optimizer baselineObjective={optimization.BaselineMetrics.Objective:G6} selectedObjective={optimization.SelectedMetrics.Objective:G6} selectedQuality={optimization.SelectedMetrics.Quality:G6}"));
        Console.WriteLine(
            $"selected alpha={optimization.Selected.Alpha:G3} topK={optimization.Selected.TopK} qbits={optimization.Selected.QuantizationBits} cache={optimization.Selected.CacheShards} tile={optimization.Selected.TileSize}");

        return 0;
    }

    private static long CandidateCells(int freeBitsPerAxis)
    {
        var totalFreeBits = 3 * freeBitsPerAxis;
        if (totalFreeBits >= 63)
            throw new InvalidOperationException("Candidate-cell count exceeds Int64 range.");
        return 1L << totalFreeBits;
    }

    private static void ValidateConfig(Dm3dRuntimeConfig config)
    {
        if (config.Alpha is < 0.35 or > 0.75)
            throw new ArgumentOutOfRangeException(nameof(config), "Alpha must be in [0.35, 0.75].");
        if (config.TopK is < 8 or > 4096)
            throw new ArgumentOutOfRangeException(nameof(config), "TopK must be in [8, 4096].");
        if (config.QuantizationBits is < 4 or > 8)
            throw new ArgumentOutOfRangeException(nameof(config), "Quantization bits must be in [4, 8].");
        if (config.CacheShards is < 0 or > 64)
            throw new ArgumentOutOfRangeException(nameof(config), "Cache shards must be in [0, 64].");
        if (config.TileSize is < 8 or > 64)
            throw new ArgumentOutOfRangeException(nameof(config), "Tile size must be in [8, 64].");
    }
}
