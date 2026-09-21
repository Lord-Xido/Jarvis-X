using System.Buffers.Binary;
using System.Security.Cryptography;

namespace QSol.GraphicsCodec.Runtime;

internal enum Dm3dOpcode : byte
{
    Nop = 0x00,
    CfgAxis = 0x01,
    CfgBits = 0x02,
    CfgLatent = 0x03,
    CfgTopK = 0x04,
    CfgAlphaQ16 = 0x05,
    CfgMaxIters = 0x06,
    CfgQBits = 0x07,

    SelfHashCheck = 0x08,
    ProfileBegin = 0x09,
    ProfileEnd = 0x0A,
    MeasureLatency = 0x0B,
    MeasureMemory = 0x0C,
    MeasureVerify = 0x0D,
    ObjectiveScore = 0x0E,
    SaveCheckpoint = 0x0F,

    HostSearch = 0x10,
    LoadInput = 0x11,
    TokenizeHash = 0x12,
    EncodeInt8 = 0x13,
    Binarize256 = 0x14,
    FuseModalities = 0x15,

    Project3D = 0x20,
    Morton3D20 = 0x21,
    PrefixMask = 0x22,
    OctreeLocalize = 0x23,
    CacheLookup = 0x24,
    CachePromote = 0x25,

    AnnXnorPopcount = 0x30,
    AuthorityWeight = 0x31,
    TopK = 0x32,
    DecodeClaims = 0x33,
    GraphValidate = 0x34,
    EvidenceMask = 0x35,
    ContradictionCheck = 0x36,
    VerifyScore = 0x37,

    IfConverged = 0x40,
    UpdateLatent = 0x41,
    ContractRadius = 0x42,
    ResolvePrefix = 0x43,
    DecTopK = 0x44,
    Jump = 0x45,

    ParamDecode = 0x50,
    RenderVector = 0x51,
    InverseRender = 0x52,
    VerifyGraphics = 0x53,
    SpatialErrorField = 0x54,
    FreezeLowError = 0x55,
    RefineHighError = 0x56,

    OptBegin = 0x60,
    TuneAlpha = 0x61,
    TuneTopK = 0x62,
    TuneQBits = 0x63,
    TuneCache = 0x64,
    TuneTile = 0x65,
    GuardrailCheck = 0x66,
    IfBetter = 0x67,
    CommitConfig = 0x68,
    RollbackConfig = 0x69,
    OptEnd = 0x6A,

    EmitOutput = 0x70,
    Halt = 0xFF
}

internal readonly record struct Dm3dInstruction(
    Dm3dOpcode Opcode,
    byte Flags = 0,
    byte Destination = 0,
    byte Source = 0,
    uint Immediate32 = 0,
    ulong Immediate64 = 0);

internal static class Dm3dRomImage
{
    public const int RomSize = 128 * 1024;
    public const int HeaderSize = 256;
    public const int InstructionSize = 16;

    private const ushort VersionMajor = 2;
    private const ushort VersionMinor = 0;

    public static byte[] Build()
    {
        var instructions = BuildProgram();
        var program = SerializeProgram(instructions);
        var rom = new byte[RomSize];
        Array.Fill(rom, (byte)0xFF);

        WriteHeader(rom, program, instructions);
        program.CopyTo(rom.AsSpan(HeaderSize));

        var trailerHash = SHA256.HashData(rom.AsSpan(0, rom.Length - 32));
        trailerHash.CopyTo(rom.AsSpan(rom.Length - 32));
        return rom;
    }

    public static void Write(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        File.WriteAllBytes(path, Build());
    }

    public static string Sha256Hex(byte[] rom)
    {
        ArgumentNullException.ThrowIfNull(rom);
        return Convert.ToHexString(SHA256.HashData(rom)).ToLowerInvariant();
    }

    private static List<Dm3dInstruction> BuildProgram()
    {
        var p = new List<Dm3dInstruction>
        {
            new(Dm3dOpcode.CfgAxis, Destination: 0, Immediate64: Dm3dSelfOptimizingRuntime.AxisCells),
            new(Dm3dOpcode.CfgAxis, Destination: 1, Immediate64: Dm3dSelfOptimizingRuntime.AxisCells),
            new(Dm3dOpcode.CfgAxis, Destination: 2, Immediate64: Dm3dSelfOptimizingRuntime.AxisCells),
            new(Dm3dOpcode.CfgBits, Immediate32: Dm3dSelfOptimizingRuntime.BitsPerAxis),
            new(Dm3dOpcode.CfgLatent, Immediate32: Dm3dSelfOptimizingRuntime.LatentBits),
            new(Dm3dOpcode.CfgTopK, Immediate32: 4096),
            new(Dm3dOpcode.CfgAlphaQ16, Immediate32: Q16(0.50)),
            new(Dm3dOpcode.CfgMaxIters, Immediate32: 8),
            new(Dm3dOpcode.CfgQBits, Immediate32: 8),
            new(Dm3dOpcode.SelfHashCheck, Flags: 1),
            new(Dm3dOpcode.SaveCheckpoint, Flags: 1),
            new(Dm3dOpcode.ProfileBegin),
            new(Dm3dOpcode.HostSearch, Flags: 1, Immediate32: 32),
            new(Dm3dOpcode.LoadInput),
            new(Dm3dOpcode.TokenizeHash, Immediate32: 65_536),
            new(Dm3dOpcode.EncodeInt8, Immediate32: 128),
            new(Dm3dOpcode.Binarize256, Immediate32: 256),
            new(Dm3dOpcode.FuseModalities, Immediate32: 3),
            new(Dm3dOpcode.Project3D, Immediate32: 3),
            new(Dm3dOpcode.Morton3D20, Immediate32: 20),
            new(Dm3dOpcode.PrefixMask, Immediate32: 16),
            new(Dm3dOpcode.OctreeLocalize, Immediate32: 4),
            new(Dm3dOpcode.CacheLookup)
        };

        var innerLoopPc = p.Count;
        p.Add(new(Dm3dOpcode.AnnXnorPopcount, Immediate32: 256));
        p.Add(new(Dm3dOpcode.AuthorityWeight, Immediate32: 8));
        p.Add(new(Dm3dOpcode.TopK));
        p.Add(new(Dm3dOpcode.DecodeClaims, Immediate32: 64));
        p.Add(new(Dm3dOpcode.GraphValidate, Flags: 1));
        p.Add(new(Dm3dOpcode.EvidenceMask));
        p.Add(new(Dm3dOpcode.ContradictionCheck, Flags: 1));
        p.Add(new(Dm3dOpcode.VerifyScore, Immediate32: Q16(0.90)));

        var convergencePc = p.Count;
        p.Add(new(Dm3dOpcode.IfConverged));
        p.Add(new(Dm3dOpcode.UpdateLatent, Immediate32: Q16(0.125)));
        p.Add(new(Dm3dOpcode.ContractRadius, Immediate32: Q16(0.50)));
        p.Add(new(Dm3dOpcode.ResolvePrefix, Immediate32: 1));
        p.Add(new(Dm3dOpcode.DecTopK, Immediate32: 3));
        p.Add(new(Dm3dOpcode.Project3D, Immediate32: 3));
        p.Add(new(Dm3dOpcode.Morton3D20, Immediate32: 20));
        p.Add(new(Dm3dOpcode.OctreeLocalize, Immediate32: 1));
        p.Add(new(Dm3dOpcode.CacheLookup));
        p.Add(new(Dm3dOpcode.Jump, Immediate32: (uint)innerLoopPc));

        var graphicsPc = p.Count;
        p[convergencePc] = p[convergencePc] with { Immediate32 = (uint)graphicsPc };

        p.Add(new(Dm3dOpcode.ParamDecode, Immediate32: 256));
        p.Add(new(Dm3dOpcode.RenderVector, Flags: 1));
        p.Add(new(Dm3dOpcode.InverseRender));
        p.Add(new(Dm3dOpcode.VerifyGraphics, Immediate32: Q16(0.95)));
        p.Add(new(Dm3dOpcode.SpatialErrorField, Immediate32: 32));
        p.Add(new(Dm3dOpcode.FreezeLowError, Immediate32: Q16(0.02)));
        p.Add(new(Dm3dOpcode.RefineHighError, Immediate32: Q16(0.10)));
        p.Add(new(Dm3dOpcode.CachePromote, Immediate32: 16));
        p.Add(new(Dm3dOpcode.ProfileEnd));
        p.Add(new(Dm3dOpcode.MeasureLatency));
        p.Add(new(Dm3dOpcode.MeasureMemory));
        p.Add(new(Dm3dOpcode.MeasureVerify));
        p.Add(new(Dm3dOpcode.ObjectiveScore, Immediate32: 0x0001_0001));
        p.Add(new(Dm3dOpcode.OptBegin));
        p.Add(new(Dm3dOpcode.TuneAlpha, Immediate32: PackU16(Q16Short(0.35), Q16Short(0.75))));
        p.Add(new(Dm3dOpcode.TuneTopK, Immediate32: 4096, Immediate64: 8));
        p.Add(new(Dm3dOpcode.TuneQBits, Immediate32: PackU16(4, 8)));
        p.Add(new(Dm3dOpcode.TuneCache, Immediate32: 64));
        p.Add(new(Dm3dOpcode.TuneTile, Immediate32: PackU16(8, 64)));
        p.Add(new(Dm3dOpcode.GuardrailCheck, Flags: 1, Immediate32: Q16(0.95)));

        var ifBetterPc = p.Count;
        p.Add(new(Dm3dOpcode.IfBetter));
        p.Add(new(Dm3dOpcode.RollbackConfig, Flags: 1));
        var jumpEmitPc = p.Count;
        p.Add(new(Dm3dOpcode.Jump));
        var commitPc = p.Count;
        p.Add(new(Dm3dOpcode.CommitConfig, Flags: 1));
        p.Add(new(Dm3dOpcode.OptEnd));
        var emitPc = p.Count;
        p.Add(new(Dm3dOpcode.EmitOutput, Flags: 1));
        p.Add(new(Dm3dOpcode.Halt));

        p[ifBetterPc] = p[ifBetterPc] with { Immediate32 = (uint)commitPc };
        p[jumpEmitPc] = p[jumpEmitPc] with { Immediate32 = (uint)emitPc };
        return p;
    }

    private static byte[] SerializeProgram(IReadOnlyList<Dm3dInstruction> instructions)
    {
        var program = new byte[instructions.Count * InstructionSize];
        for (var i = 0; i < instructions.Count; i++)
        {
            var offset = i * InstructionSize;
            var instruction = instructions[i];
            program[offset] = (byte)instruction.Opcode;
            program[offset + 1] = instruction.Flags;
            program[offset + 2] = instruction.Destination;
            program[offset + 3] = instruction.Source;
            BinaryPrimitives.WriteUInt32LittleEndian(program.AsSpan(offset + 4, 4), instruction.Immediate32);
            BinaryPrimitives.WriteUInt64LittleEndian(program.AsSpan(offset + 8, 8), instruction.Immediate64);
        }
        return program;
    }

    private static void WriteHeader(
        Span<byte> rom,
        ReadOnlySpan<byte> program,
        IReadOnlyList<Dm3dInstruction> instructions)
    {
        "DM3DSO2\0"u8.CopyTo(rom);
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(8, 2), VersionMajor);
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(10, 2), VersionMinor);
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(12, 2), HeaderSize);
        BinaryPrimitives.WriteUInt64LittleEndian(rom.Slice(16, 8), Dm3dSelfOptimizingRuntime.AxisCells);
        rom[24] = Dm3dSelfOptimizingRuntime.BitsPerAxis;
        rom[25] = Dm3dSelfOptimizingRuntime.LatentBits / 8;
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(26, 2), 8);

        var virtualCells = checked((ulong)Dm3dSelfOptimizingRuntime.AxisCells
            * (ulong)Dm3dSelfOptimizingRuntime.AxisCells
            * (ulong)Dm3dSelfOptimizingRuntime.AxisCells);
        BinaryPrimitives.WriteUInt64LittleEndian(rom.Slice(32, 8), virtualCells);
        BinaryPrimitives.WriteUInt32LittleEndian(rom.Slice(40, 4), (uint)program.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(rom.Slice(44, 4), (uint)instructions.Count);
        BinaryPrimitives.WriteUInt32LittleEndian(rom.Slice(48, 4), Crc32(program));

        var programHash = SHA256.HashData(program);
        programHash.CopyTo(rom.Slice(64, 32));

        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(104, 2), Q16Short(0.55));
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(106, 2), Q16Short(0.20));
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(108, 2), Q16Short(0.15));
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(110, 2), Q16Short(0.10));
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(120, 2), Q16Short(0.95));
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(122, 2), Q16Short(0.35));
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(124, 2), Q16Short(0.75));
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(126, 2), 4);
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(128, 2), 8);
        BinaryPrimitives.WriteUInt16LittleEndian(rom.Slice(130, 2), 64);
    }

    private static uint Q16(double value) => (uint)Math.Round(Math.Clamp(value, 0, 1) * 65_535.0);

    private static ushort Q16Short(double value) => (ushort)Q16(value);

    private static uint PackU16(ushort high, ushort low) => ((uint)high << 16) | low;

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        var crc = 0xFFFF_FFFFu;
        foreach (var value in data)
        {
            crc ^= value;
            for (var bit = 0; bit < 8; bit++)
                crc = (crc >> 1) ^ (0xEDB8_8320u & (uint)-(int)(crc & 1));
        }
        return ~crc;
    }
}
