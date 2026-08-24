using System.Numerics;

namespace QSol.GraphicsCodec.Dmkb1;

public static class DmkbSelfTest
{
    public static void Run()
    {
        TestBitStream();
        TestInstructions();
        TestMeshlet();
        TestLatentVector();
        Console.WriteLine("DMKB-1 kinetic bytecode self-test PASS");
    }

    private static void TestBitStream()
    {
        Span<byte> bytes = stackalloc byte[8];
        var writer = new BitWriter(bytes);
        writer.WriteBits(0b10_1101u, 6);
        writer.WriteBits(0xDEADBEEFu, 32);
        writer.WriteSignedInt(-17, 8);

        var reader = new BitReader(bytes[..writer.ByteLength]);
        Require(reader.ReadBits(6) == 0b10_1101u, "Variable-width bit prefix did not round-trip.");
        Require(reader.ReadBits(32) == 0xDEADBEEFu, "32-bit field did not round-trip across byte boundaries.");
        Require(reader.ReadSignedInt(8) == -17, "Signed ZigZag value did not round-trip.");
        Require(reader.BitPosition == writer.BitLength, "Bit reader/writer positions diverged.");

        ExpectThrows<InvalidDataException>(() =>
        {
            var shortReader = new BitReader(new byte[1]);
            _ = shortReader.ReadBits(16);
        });
    }

    private static void TestInstructions()
    {
        var instruction = new CompactInstruction(GraphicsOpcode.RotAxis, 1, 2, 3, 7, 42);
        var unpacked = CompactInstruction.Unpack32(instruction.Pack32());
        Require(unpacked == instruction, "Compact 32-bit graphics instruction lost a field during round-trip.");

        var program = new[]
        {
            instruction,
            new CompactInstruction(GraphicsOpcode.BindMeshlet, 4, 0, 0, 0, 11),
            new CompactInstruction(GraphicsOpcode.DrawIndexed, 0, 4, 0, 1, 63)
        };
        var encoded = InstructionPacket.Encode(program);
        var decoded = InstructionPacket.Decode(encoded);
        Require(program.SequenceEqual(decoded), "DMKB-1 instruction stream did not round-trip exactly.");
        ExpectThrows<ArgumentOutOfRangeException>(() =>
            _ = new CompactInstruction(GraphicsOpcode.Nop, 32, 0, 0, 0, 0));
    }

    private static void TestMeshlet()
    {
        const int bits = 10;
        var meshlet = new MeshletPacketData
        {
            Anchor = new Vector3(100f, 50f, -25f),
            Vertices =
            [
                new Vector3(131.2f, 42.4f, -4.1f),
                new Vector3(89.1f, 78.3f, -56.5f),
                new Vector3(104.6f, 61.0f, -19.75f)
            ],
            Indices = [0, 1, 2, 2, 1, 0]
        };

        var encoded = MeshletPacket.Encode(meshlet, bits);
        var decoded = MeshletPacket.Decode(encoded);
        Require(meshlet.Indices.SequenceEqual(decoded.Indices), "DMKB-1 meshlet topology changed during decode.");
        Require(decoded.Anchor == meshlet.Anchor, "DMKB-1 meshlet anchor was not preserved exactly.");

        var actualError = MaxVertexError(meshlet.Vertices, decoded.Vertices);
        var bound = MeshletPacket.TheoreticalMaxVertexError(meshlet, bits);
        Require(actualError <= bound + 1e-5f, $"Meshlet error {actualError} exceeded deterministic bound {bound}.");

        var corrupt = encoded.ToArray();
        corrupt[Math.Min(20, corrupt.Length - 33)] ^= 0x40;
        ExpectThrows<InvalidDataException>(() => _ = MeshletPacket.Decode(corrupt));
        ExpectThrows<InvalidDataException>(() => _ = MeshletPacket.Decode(encoded.AsSpan(0, encoded.Length - 1)));

        Console.WriteLine(FormattableString.Invariant(
            $"DMKB meshlet: bytes={encoded.Length} vertices={meshlet.Vertices.Length} indices={meshlet.Indices.Length} maxError={actualError:G6} bound={bound:G6}"));
    }

    private static void TestLatentVector()
    {
        const int quantBits = 4;
        var random = new Random(42);
        var latent = new float[128];
        for (var i = 0; i < latent.Length; i++)
            latent[i] = (float)((random.NextDouble() * 2.0) - 1.0);

        var encoded = LatentPacket.Encode(latent, quantBits);
        var decoded = LatentPacket.Decode(encoded);
        var metrics = LatentPacket.Measure(latent, decoded, encoded.Length);
        var bound = LatentPacket.TheoreticalMaxAbsError(latent, quantBits);
        Require(metrics.MaxAbsError <= bound + 1e-5f, $"Latent max error {metrics.MaxAbsError} exceeded bound {bound}.");

        var constant = Enumerable.Repeat(0.5f, 17).ToArray();
        var constantDecoded = LatentPacket.Decode(LatentPacket.Encode(constant, 4));
        Require(constant.SequenceEqual(constantDecoded), "Constant latent range did not round-trip exactly.");

        var corrupt = encoded.ToArray();
        corrupt[^1] ^= 0x01;
        ExpectThrows<InvalidDataException>(() => _ = LatentPacket.Decode(corrupt));
        ExpectThrows<ArgumentOutOfRangeException>(() => _ = LatentPacket.Encode(latent, 0));
        ExpectThrows<ArgumentOutOfRangeException>(() => _ = LatentPacket.Encode(latent, 25));

        Console.WriteLine(FormattableString.Invariant(
            $"DMKB latent: raw={metrics.RawBytes}B encoded={metrics.EncodedBytes}B ratio={metrics.CompressionRatio:F2}x mse={metrics.Mse:G6} maxError={metrics.MaxAbsError:G6} bound={bound:G6}"));
    }

    private static float MaxVertexError(Vector3[] a, Vector3[] b)
    {
        Require(a.Length == b.Length, "Vertex arrays have different lengths.");
        var max = 0f;
        for (var i = 0; i < a.Length; i++)
            max = MathF.Max(max, Vector3.Distance(a[i], b[i]));
        return max;
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
            throw new InvalidOperationException(message);
    }

    private static void ExpectThrows<T>(Action action) where T : Exception
    {
        try
        {
            action();
        }
        catch (T)
        {
            return;
        }
        throw new InvalidOperationException($"Expected {typeof(T).Name} was not thrown.");
    }
}
