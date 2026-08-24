namespace QSol.GraphicsCodec.Codec;

internal sealed class BitWriter
{
    private readonly Stream _stream;
    private ulong _buffer;
    private int _bitCount;

    public BitWriter(Stream stream) => _stream = stream;

    public void Write(uint value, int bits)
    {
        if (bits is < 1 or > 32) throw new ArgumentOutOfRangeException(nameof(bits));
        var mask = bits == 32 ? uint.MaxValue : (1u << bits) - 1u;
        _buffer |= ((ulong)(value & mask)) << _bitCount;
        _bitCount += bits;

        while (_bitCount >= 8)
        {
            _stream.WriteByte((byte)_buffer);
            _buffer >>= 8;
            _bitCount -= 8;
        }
    }

    public void AlignToByte()
    {
        if (_bitCount <= 0) return;
        _stream.WriteByte((byte)_buffer);
        _buffer = 0;
        _bitCount = 0;
    }
}

internal sealed class BitReader
{
    private readonly Stream _stream;
    private ulong _buffer;
    private int _bitCount;

    public BitReader(Stream stream) => _stream = stream;

    public uint Read(int bits)
    {
        if (bits is < 1 or > 32) throw new ArgumentOutOfRangeException(nameof(bits));
        while (_bitCount < bits)
        {
            var next = _stream.ReadByte();
            if (next < 0) throw new EndOfStreamException();
            _buffer |= ((ulong)(byte)next) << _bitCount;
            _bitCount += 8;
        }

        var mask = bits == 32 ? uint.MaxValue : (1u << bits) - 1u;
        var value = (uint)_buffer & mask;
        _buffer >>= bits;
        _bitCount -= bits;
        return value;
    }

    public void AlignToByte()
    {
        _buffer = 0;
        _bitCount = 0;
    }
}

internal static class VarInt
{
    public static void WriteUInt(Stream stream, uint value)
    {
        while (value >= 0x80)
        {
            stream.WriteByte((byte)(value | 0x80));
            value >>= 7;
        }
        stream.WriteByte((byte)value);
    }

    public static uint ReadUInt(Stream stream)
    {
        uint value = 0;
        var shift = 0;
        while (shift < 35)
        {
            var next = stream.ReadByte();
            if (next < 0) throw new EndOfStreamException();
            value |= (uint)(next & 0x7F) << shift;
            if ((next & 0x80) == 0) return value;
            shift += 7;
        }
        throw new InvalidDataException("Invalid varint.");
    }

    public static uint ZigZagEncode(int value) => unchecked((uint)((value << 1) ^ (value >> 31)));
    public static int ZigZagDecode(uint value) => (int)(value >> 1) ^ -((int)value & 1);
}
