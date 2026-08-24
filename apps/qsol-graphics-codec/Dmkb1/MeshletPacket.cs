using System.Buffers.Binary;
using System.Numerics;

namespace QSol.GraphicsCodec.Dmkb1;

public sealed class MeshletPacketData
{
    public required Vector3 Anchor { get; init; }
    public required Vector3[] Vertices { get; init; }
    public required ushort[] Indices { get; init; }
}

public static class MeshletPacket
{
    private const int HeaderSize = 48;

    public static byte[] Encode(MeshletPacketData meshlet, int bitsPerComponent = 10)
    {
        ArgumentNullException.ThrowIfNull(meshlet);
        _ = Quantization.MaxCode(bitsPerComponent);
        ValidateVector(meshlet.Anchor, nameof(meshlet.Anchor));

        if (meshlet.Vertices.Length is < 1 or > ushort.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(meshlet), "DMKB-1 meshlets require 1..65535 vertices.");
        if (meshlet.Indices.Length > ushort.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(meshlet), "DMKB-1 meshlets support at most 65535 local indices.");

        var minDelta = new Vector3(float.PositiveInfinity);
        var maxDelta = new Vector3(float.NegativeInfinity);
        foreach (var vertex in meshlet.Vertices)
        {
            ValidateVector(vertex, nameof(meshlet.Vertices));
            var delta = vertex - meshlet.Anchor;
            minDelta = Vector3.Min(minDelta, delta);
            maxDelta = Vector3.Max(maxDelta, delta);
        }

        var indexBits = IndexBitWidth(meshlet.Vertices.Length);
        foreach (var index in meshlet.Indices)
        {
            if (index >= meshlet.Vertices.Length)
                throw new InvalidDataException($"Meshlet index {index} is outside the local vertex range.");
        }

        var bitLengthLong = checked(
            ((long)meshlet.Vertices.Length * 3L * bitsPerComponent) +
            ((long)meshlet.Indices.Length * indexBits));
        if (bitLengthLong > int.MaxValue)
            throw new InvalidDataException("DMKB-1 meshlet bit payload exceeds the implementation limit.");

        var bitLength = (int)bitLengthLong;
        var bitBytes = (bitLength + 7) >> 3;
        var bitPayload = new byte[bitBytes];
        var writer = new BitWriter(bitPayload);

        foreach (var vertex in meshlet.Vertices)
        {
            var delta = vertex - meshlet.Anchor;
            writer.WriteQuantizedFloat(delta.X, minDelta.X, maxDelta.X, bitsPerComponent);
            writer.WriteQuantizedFloat(delta.Y, minDelta.Y, maxDelta.Y, bitsPerComponent);
            writer.WriteQuantizedFloat(delta.Z, minDelta.Z, maxDelta.Z, bitsPerComponent);
        }
        foreach (var index in meshlet.Indices)
            writer.WriteBits(index, indexBits);

        if (writer.BitLength != bitLength)
            throw new InvalidOperationException("DMKB-1 meshlet bit accounting mismatch.");

        var payload = new byte[checked(HeaderSize + bitBytes)];
        payload[0] = checked((byte)bitsPerComponent);
        payload[1] = checked((byte)indexBits);
        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(2, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(4, 2), checked((ushort)meshlet.Vertices.Length));
        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(6, 2), checked((ushort)meshlet.Indices.Length));
        WriteVector3(payload.AsSpan(8, 12), meshlet.Anchor);
        WriteVector3(payload.AsSpan(20, 12), minDelta);
        WriteVector3(payload.AsSpan(32, 12), maxDelta);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(44, 4), checked((uint)bitLength));
        bitPayload.CopyTo(payload.AsSpan(HeaderSize));

        return DmkbContainer.Wrap(DmkbPacketKind.Meshlet, payload);
    }

    public static MeshletPacketData Decode(ReadOnlySpan<byte> encoded)
    {
        var envelope = DmkbContainer.Unwrap(encoded, DmkbPacketKind.Meshlet);
        var payload = envelope.Payload.AsSpan();
        if (payload.Length < HeaderSize)
            throw new InvalidDataException("Truncated DMKB-1 meshlet header.");

        var quantBits = payload[0];
        _ = Quantization.MaxCode(quantBits);
        var indexBits = payload[1];
        if (indexBits is < 1 or > 16)
            throw new InvalidDataException("DMKB-1 meshlet index width must be 1..16 bits.");
        if (BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice(2, 2)) != 0)
            throw new InvalidDataException("DMKB-1 meshlet reserved header bits must be zero.");

        var vertexCount = BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice(4, 2));
        var indexCount = BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice(6, 2));
        if (vertexCount == 0)
            throw new InvalidDataException("DMKB-1 meshlet must contain at least one vertex.");
        if (indexBits != IndexBitWidth(vertexCount))
            throw new InvalidDataException("DMKB-1 meshlet index width is not canonical for its vertex count.");

        var anchor = ReadVector3(payload.Slice(8, 12));
        var minDelta = ReadVector3(payload.Slice(20, 12));
        var maxDelta = ReadVector3(payload.Slice(32, 12));
        ValidateVector(anchor, nameof(anchor));
        ValidateVector(minDelta, nameof(minDelta));
        ValidateVector(maxDelta, nameof(maxDelta));
        if (maxDelta.X < minDelta.X || maxDelta.Y < minDelta.Y || maxDelta.Z < minDelta.Z)
            throw new InvalidDataException("DMKB-1 meshlet delta bounds are invalid.");

        var declaredBits = BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(44, 4));
        var expectedBits = checked(
            ((long)vertexCount * 3L * quantBits) +
            ((long)indexCount * indexBits));
        if (declaredBits != expectedBits)
            throw new InvalidDataException("DMKB-1 meshlet declared bit length does not match its header counts.");
        if (declaredBits > int.MaxValue)
            throw new InvalidDataException("DMKB-1 meshlet bit payload exceeds the implementation limit.");

        var bitBytes = (checked((int)declaredBits) + 7) >> 3;
        if (payload.Length != HeaderSize + bitBytes)
            throw new InvalidDataException("DMKB-1 meshlet payload byte length is not canonical.");

        var bitSpan = payload.Slice(HeaderSize, bitBytes);
        ValidateZeroPadding(bitSpan, checked((int)declaredBits));
        var reader = new BitReader(bitSpan);
        var vertices = new Vector3[vertexCount];
        for (var i = 0; i < vertices.Length; i++)
        {
            var delta = new Vector3(
                reader.ReadQuantizedFloat(minDelta.X, maxDelta.X, quantBits),
                reader.ReadQuantizedFloat(minDelta.Y, maxDelta.Y, quantBits),
                reader.ReadQuantizedFloat(minDelta.Z, maxDelta.Z, quantBits));
            vertices[i] = anchor + delta;
        }

        var indices = new ushort[indexCount];
        for (var i = 0; i < indices.Length; i++)
        {
            var index = reader.ReadBits(indexBits);
            if (index >= vertexCount)
                throw new InvalidDataException("Decoded DMKB-1 meshlet index is outside the local vertex range.");
            indices[i] = checked((ushort)index);
        }

        if (reader.BitPosition != declaredBits)
            throw new InvalidDataException("DMKB-1 meshlet decoder did not consume the declared bit payload exactly.");

        return new MeshletPacketData
        {
            Anchor = anchor,
            Vertices = vertices,
            Indices = indices
        };
    }

    public static float TheoreticalMaxVertexError(MeshletPacketData meshlet, int bitsPerComponent)
    {
        ArgumentNullException.ThrowIfNull(meshlet);
        if (meshlet.Vertices.Length == 0)
            return 0f;

        var minDelta = new Vector3(float.PositiveInfinity);
        var maxDelta = new Vector3(float.NegativeInfinity);
        foreach (var vertex in meshlet.Vertices)
        {
            var delta = vertex - meshlet.Anchor;
            minDelta = Vector3.Min(minDelta, delta);
            maxDelta = Vector3.Max(maxDelta, delta);
        }

        var halfStep = new Vector3(
            Quantization.Step(minDelta.X, maxDelta.X, bitsPerComponent) * 0.5f,
            Quantization.Step(minDelta.Y, maxDelta.Y, bitsPerComponent) * 0.5f,
            Quantization.Step(minDelta.Z, maxDelta.Z, bitsPerComponent) * 0.5f);
        return halfStep.Length();
    }

    private static int IndexBitWidth(int vertexCount)
    {
        if (vertexCount <= 1)
            return 1;
        return BitOperations.Log2(checked((uint)(vertexCount - 1))) + 1;
    }

    private static void ValidateVector(Vector3 value, string name)
    {
        if (!float.IsFinite(value.X) || !float.IsFinite(value.Y) || !float.IsFinite(value.Z))
            throw new InvalidDataException($"{name} contains NaN or Infinity.");
    }

    private static void WriteVector3(Span<byte> target, Vector3 value)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(target.Slice(0, 4), BitConverter.SingleToUInt32Bits(value.X));
        BinaryPrimitives.WriteUInt32LittleEndian(target.Slice(4, 4), BitConverter.SingleToUInt32Bits(value.Y));
        BinaryPrimitives.WriteUInt32LittleEndian(target.Slice(8, 4), BitConverter.SingleToUInt32Bits(value.Z));
    }

    private static Vector3 ReadVector3(ReadOnlySpan<byte> source) => new(
        BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(0, 4))),
        BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(4, 4))),
        BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(8, 4))));

    private static void ValidateZeroPadding(ReadOnlySpan<byte> payload, int bitLength)
    {
        if (payload.Length == 0 || (bitLength & 7) == 0)
            return;
        var used = bitLength & 7;
        var unusedMask = (byte)~((1 << used) - 1);
        if ((payload[^1] & unusedMask) != 0)
            throw new InvalidDataException("DMKB-1 meshlet has non-zero padding bits.");
    }
}
