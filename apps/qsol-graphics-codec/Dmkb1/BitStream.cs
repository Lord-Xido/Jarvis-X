using System.Runtime.CompilerServices;

namespace QSol.GraphicsCodec.Dmkb1;

public ref struct BitWriter
{
    private Span<byte> _buffer;
    private int _bitPosition;

    public BitWriter(Span<byte> buffer)
    {
        _buffer = buffer;
        _buffer.Clear();
        _bitPosition = 0;
    }

    public readonly int BitLength => _bitPosition;
    public readonly int ByteLength => (_bitPosition + 7) >> 3;
    public readonly int CapacityBits => checked(_buffer.Length * 8);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void WriteBits(uint value, int bitCount)
    {
        ValidateBitCount(bitCount);
        EnsureWritable(bitCount);

        var consumed = 0;
        while (consumed < bitCount)
        {
            var byteIndex = _bitPosition >> 3;
            var bitOffset = _bitPosition & 7;
            var take = Math.Min(8 - bitOffset, bitCount - consumed);
            var mask = take == 8 ? 0xFFu : (1u << take) - 1u;
            var chunk = (byte)(((value >> consumed) & mask) << bitOffset);
            _buffer[byteIndex] |= chunk;
            _bitPosition += take;
            consumed += take;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void WriteFloat32(float value)
    {
        if (!float.IsFinite(value))
            throw new InvalidDataException("DMKB-1 does not permit NaN or Infinity in float payloads.");
        WriteBits(BitConverter.SingleToUInt32Bits(value), 32);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void WriteQuantizedFloat(float value, float min, float max, int bits)
    {
        Quantization.Validate(value, min, max, bits);
        if (min == max)
        {
            WriteBits(0, bits);
            return;
        }

        var clamped = Math.Clamp(value, min, max);
        var normalized = (clamped - min) / (max - min);
        var maxQuant = Quantization.MaxCode(bits);
        var quantized = (uint)Math.Round(normalized * maxQuant, MidpointRounding.AwayFromZero);
        if (quantized > maxQuant)
            quantized = maxQuant;
        WriteBits(quantized, bits);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void WriteSignedInt(int value, int bits)
    {
        ValidateBitCount(bits);
        var zigZag = unchecked((uint)((value << 1) ^ (value >> 31)));
        if (bits < 32 && zigZag > ((1u << bits) - 1u))
            throw new ArgumentOutOfRangeException(nameof(value), "Signed value does not fit the declared DMKB-1 bit width.");
        WriteBits(zigZag, bits);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private readonly void EnsureWritable(int bitCount)
    {
        if (_bitPosition > CapacityBits - bitCount)
            throw new InvalidDataException("DMKB-1 bitstream capacity exceeded.");
    }

    private static void ValidateBitCount(int bitCount)
    {
        if (bitCount is < 1 or > 32)
            throw new ArgumentOutOfRangeException(nameof(bitCount), "Bit count must be in the range 1..32.");
    }
}

public ref struct BitReader
{
    private readonly ReadOnlySpan<byte> _buffer;
    private int _bitPosition;

    public BitReader(ReadOnlySpan<byte> buffer)
    {
        _buffer = buffer;
        _bitPosition = 0;
    }

    public readonly int BitPosition => _bitPosition;
    public readonly int CapacityBits => checked(_buffer.Length * 8);
    public readonly int RemainingBits => CapacityBits - _bitPosition;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public uint ReadBits(int bitCount)
    {
        ValidateBitCount(bitCount);
        EnsureReadable(bitCount);

        uint value = 0;
        var produced = 0;
        while (produced < bitCount)
        {
            var byteIndex = _bitPosition >> 3;
            var bitOffset = _bitPosition & 7;
            var take = Math.Min(8 - bitOffset, bitCount - produced);
            var mask = take == 8 ? 0xFFu : (1u << take) - 1u;
            var chunk = ((uint)_buffer[byteIndex] >> bitOffset) & mask;
            value |= chunk << produced;
            _bitPosition += take;
            produced += take;
        }
        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float ReadFloat32()
    {
        var value = BitConverter.UInt32BitsToSingle(ReadBits(32));
        if (!float.IsFinite(value))
            throw new InvalidDataException("DMKB-1 float payload contains NaN or Infinity.");
        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float ReadQuantizedFloat(float min, float max, int bits)
    {
        Quantization.ValidateRange(min, max, bits);
        var quantized = ReadBits(bits);
        if (min == max)
            return min;
        var normalized = (float)quantized / Quantization.MaxCode(bits);
        return min + (normalized * (max - min));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int ReadSignedInt(int bits)
    {
        var zigZag = ReadBits(bits);
        return (int)(zigZag >> 1) ^ -(int)(zigZag & 1u);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private readonly void EnsureReadable(int bitCount)
    {
        if (_bitPosition > CapacityBits - bitCount)
            throw new InvalidDataException("Truncated DMKB-1 bitstream.");
    }

    private static void ValidateBitCount(int bitCount)
    {
        if (bitCount is < 1 or > 32)
            throw new ArgumentOutOfRangeException(nameof(bitCount), "Bit count must be in the range 1..32.");
    }
}

internal static class Quantization
{
    public const int MinBits = 1;
    public const int MaxFloatBits = 24;

    public static uint MaxCode(int bits)
    {
        if (bits is < MinBits or > MaxFloatBits)
            throw new ArgumentOutOfRangeException(nameof(bits), $"Float quantization bits must be in the range {MinBits}..{MaxFloatBits}.");
        return (1u << bits) - 1u;
    }

    public static void Validate(float value, float min, float max, int bits)
    {
        ValidateRange(min, max, bits);
        if (!float.IsFinite(value))
            throw new ArgumentOutOfRangeException(nameof(value), "Quantized values must be finite.");
    }

    public static void ValidateRange(float min, float max, int bits)
    {
        _ = MaxCode(bits);
        if (!float.IsFinite(min) || !float.IsFinite(max) || max < min)
            throw new ArgumentOutOfRangeException(nameof(max), "Quantization range must be finite and max >= min.");
    }

    public static float Step(float min, float max, int bits)
    {
        ValidateRange(min, max, bits);
        return min == max ? 0f : (max - min) / MaxCode(bits);
    }
}
