using System.Buffers.Binary;

namespace QSol.GraphicsCodec.Dmkb1;

public readonly record struct LatentPacketMetrics(
    int EncodedBytes,
    int RawBytes,
    float Mse,
    float MaxAbsError,
    double CompressionRatio);

public static class LatentPacket
{
    private const int HeaderSize = 20;
    private const int MaxElements = 1_000_000;

    public static byte[] Encode(ReadOnlySpan<float> values, int quantizationBits = 4)
    {
        _ = Quantization.MaxCode(quantizationBits);
        if (values.Length is < 1 or > MaxElements)
            throw new ArgumentOutOfRangeException(nameof(values), $"DMKB-1 latent vectors require 1..{MaxElements} elements.");

        var min = float.PositiveInfinity;
        var max = float.NegativeInfinity;
        foreach (var value in values)
        {
            if (!float.IsFinite(value))
                throw new InvalidDataException("DMKB-1 latent vector contains NaN or Infinity.");
            min = MathF.Min(min, value);
            max = MathF.Max(max, value);
        }

        var bitLengthLong = checked((long)values.Length * quantizationBits);
        if (bitLengthLong > int.MaxValue)
            throw new InvalidDataException("DMKB-1 latent bit payload exceeds the implementation limit.");
        var bitLength = (int)bitLengthLong;
        var bitBytes = (bitLength + 7) >> 3;
        var bitPayload = new byte[bitBytes];
        var writer = new BitWriter(bitPayload);
        foreach (var value in values)
            writer.WriteQuantizedFloat(value, min, max, quantizationBits);
        if (writer.BitLength != bitLength)
            throw new InvalidOperationException("DMKB-1 latent bit accounting mismatch.");

        var payload = new byte[checked(HeaderSize + bitBytes)];
        payload[0] = checked((byte)quantizationBits);
        payload[1] = 0;
        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(2, 2), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(4, 4), checked((uint)values.Length));
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(8, 4), BitConverter.SingleToUInt32Bits(min));
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(12, 4), BitConverter.SingleToUInt32Bits(max));
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(16, 4), checked((uint)bitLength));
        bitPayload.CopyTo(payload.AsSpan(HeaderSize));
        return DmkbContainer.Wrap(DmkbPacketKind.LatentVector, payload);
    }

    public static float[] Decode(ReadOnlySpan<byte> encoded)
    {
        var envelope = DmkbContainer.Unwrap(encoded, DmkbPacketKind.LatentVector);
        var payload = envelope.Payload.AsSpan();
        if (payload.Length < HeaderSize)
            throw new InvalidDataException("Truncated DMKB-1 latent header.");

        var quantBits = payload[0];
        _ = Quantization.MaxCode(quantBits);
        if (payload[1] != 0 || BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice(2, 2)) != 0)
            throw new InvalidDataException("DMKB-1 latent reserved header bits must be zero.");

        var length = BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(4, 4));
        if (length is 0 or > MaxElements)
            throw new InvalidDataException("DMKB-1 latent element count is outside the implementation limit.");

        var min = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(8, 4)));
        var max = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(12, 4)));
        Quantization.ValidateRange(min, max, quantBits);

        var declaredBits = BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(16, 4));
        var expectedBits = checked((long)length * quantBits);
        if (declaredBits != expectedBits)
            throw new InvalidDataException("DMKB-1 latent declared bit length does not match its element count.");
        if (declaredBits > int.MaxValue)
            throw new InvalidDataException("DMKB-1 latent bit payload exceeds the implementation limit.");

        var bitLength = checked((int)declaredBits);
        var bitBytes = (bitLength + 7) >> 3;
        if (payload.Length != HeaderSize + bitBytes)
            throw new InvalidDataException("DMKB-1 latent payload byte length is not canonical.");

        var bitSpan = payload.Slice(HeaderSize, bitBytes);
        ValidateZeroPadding(bitSpan, bitLength);
        var reader = new BitReader(bitSpan);
        var decoded = new float[checked((int)length)];
        for (var i = 0; i < decoded.Length; i++)
            decoded[i] = reader.ReadQuantizedFloat(min, max, quantBits);
        if (reader.BitPosition != declaredBits)
            throw new InvalidDataException("DMKB-1 latent decoder did not consume the declared bit payload exactly.");
        return decoded;
    }

    public static LatentPacketMetrics Measure(ReadOnlySpan<float> original, ReadOnlySpan<float> decoded, int encodedBytes)
    {
        if (original.Length != decoded.Length || original.Length == 0)
            throw new ArgumentException("Latent metric inputs must be non-empty and have matching lengths.");
        if (encodedBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(encodedBytes));

        double squared = 0;
        var maxError = 0f;
        for (var i = 0; i < original.Length; i++)
        {
            var error = MathF.Abs(original[i] - decoded[i]);
            squared += (double)error * error;
            maxError = MathF.Max(maxError, error);
        }

        var rawBytes = checked(original.Length * sizeof(float));
        return new LatentPacketMetrics(
            encodedBytes,
            rawBytes,
            (float)(squared / original.Length),
            maxError,
            (double)rawBytes / encodedBytes);
    }

    public static float TheoreticalMaxAbsError(ReadOnlySpan<float> values, int quantizationBits)
    {
        if (values.Length == 0)
            return 0f;
        var min = float.PositiveInfinity;
        var max = float.NegativeInfinity;
        foreach (var value in values)
        {
            if (!float.IsFinite(value))
                throw new InvalidDataException("Latent vector contains NaN or Infinity.");
            min = MathF.Min(min, value);
            max = MathF.Max(max, value);
        }
        return Quantization.Step(min, max, quantizationBits) * 0.5f;
    }

    private static void ValidateZeroPadding(ReadOnlySpan<byte> payload, int bitLength)
    {
        if (payload.Length == 0 || (bitLength & 7) == 0)
            return;
        var used = bitLength & 7;
        var unusedMask = (byte)~((1 << used) - 1);
        if ((payload[^1] & unusedMask) != 0)
            throw new InvalidDataException("DMKB-1 latent packet has non-zero padding bits.");
    }
}
