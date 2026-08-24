using System.Buffers.Binary;
using System.Security.Cryptography;

namespace QSol.GraphicsCodec.Dmkb1;

public enum DmkbPacketKind : byte
{
    Meshlet = 1,
    LatentVector = 2,
    InstructionStream = 3
}

public readonly record struct DmkbEnvelope(DmkbPacketKind Kind, ushort Flags, byte[] Payload);

public static class DmkbContainer
{
    private static ReadOnlySpan<byte> Magic => "DMKB"u8;
    public const byte Version = 1;
    public const int HeaderSize = 12;
    public const int DigestSize = 32;
    public const int MaxPayloadBytes = 256 * 1024 * 1024;

    public static byte[] Wrap(DmkbPacketKind kind, ReadOnlySpan<byte> payload, ushort flags = 0)
    {
        if (!Enum.IsDefined(kind))
            throw new ArgumentOutOfRangeException(nameof(kind));
        if (payload.Length > MaxPayloadBytes)
            throw new ArgumentOutOfRangeException(nameof(payload), "DMKB-1 payload exceeds the bounded container limit.");

        var total = checked(HeaderSize + payload.Length + DigestSize);
        var output = new byte[total];
        Magic.CopyTo(output);
        output[4] = Version;
        output[5] = (byte)kind;
        BinaryPrimitives.WriteUInt16LittleEndian(output.AsSpan(6, 2), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(output.AsSpan(8, 4), (uint)payload.Length);
        payload.CopyTo(output.AsSpan(HeaderSize));

        var authenticatedLength = HeaderSize + payload.Length;
        SHA256.HashData(output.AsSpan(0, authenticatedLength), output.AsSpan(authenticatedLength, DigestSize));
        return output;
    }

    public static DmkbEnvelope Unwrap(ReadOnlySpan<byte> encoded, DmkbPacketKind? expectedKind = null)
    {
        if (encoded.Length < HeaderSize + DigestSize)
            throw new InvalidDataException("Truncated DMKB-1 container.");
        if (!encoded[..4].SequenceEqual(Magic))
            throw new InvalidDataException("Invalid DMKB-1 magic.");
        if (encoded[4] != Version)
            throw new InvalidDataException($"Unsupported DMKB version {encoded[4]}.");

        var kind = (DmkbPacketKind)encoded[5];
        if (!Enum.IsDefined(kind))
            throw new InvalidDataException("Unknown DMKB-1 packet kind.");
        if (expectedKind is not null && kind != expectedKind.Value)
            throw new InvalidDataException($"Expected {expectedKind.Value} packet, received {kind}.");

        var flags = BinaryPrimitives.ReadUInt16LittleEndian(encoded.Slice(6, 2));
        var payloadLength = BinaryPrimitives.ReadUInt32LittleEndian(encoded.Slice(8, 4));
        if (payloadLength > MaxPayloadBytes)
            throw new InvalidDataException("DMKB-1 declared payload exceeds the bounded container limit.");

        var expectedLength = checked(HeaderSize + (int)payloadLength + DigestSize);
        if (encoded.Length != expectedLength)
            throw new InvalidDataException("DMKB-1 container length does not match its declared payload length.");

        var authenticatedLength = HeaderSize + (int)payloadLength;
        Span<byte> digest = stackalloc byte[DigestSize];
        SHA256.HashData(encoded[..authenticatedLength], digest);
        if (!CryptographicOperations.FixedTimeEquals(digest, encoded.Slice(authenticatedLength, DigestSize)))
            throw new InvalidDataException("DMKB-1 SHA-256 integrity verification failed.");

        return new DmkbEnvelope(kind, flags, encoded.Slice(HeaderSize, (int)payloadLength).ToArray());
    }
}
