using System.Buffers.Binary;

namespace QSol.GraphicsCodec.Dmkb1;

public static class InstructionPacket
{
    private const int MaxInstructions = 1_000_000;

    public static byte[] Encode(ReadOnlySpan<CompactInstruction> instructions)
    {
        if (instructions.Length > MaxInstructions)
            throw new ArgumentOutOfRangeException(nameof(instructions), $"DMKB-1 instruction streams support at most {MaxInstructions} words.");

        var payload = new byte[checked(4 + (instructions.Length * 4))];
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0, 4), checked((uint)instructions.Length));
        for (var i = 0; i < instructions.Length; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(4 + (i * 4), 4), instructions[i].Pack32());
        return DmkbContainer.Wrap(DmkbPacketKind.InstructionStream, payload);
    }

    public static CompactInstruction[] Decode(ReadOnlySpan<byte> encoded)
    {
        var envelope = DmkbContainer.Unwrap(encoded, DmkbPacketKind.InstructionStream);
        var payload = envelope.Payload.AsSpan();
        if (payload.Length < 4)
            throw new InvalidDataException("Truncated DMKB-1 instruction stream header.");

        var count = BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0, 4));
        if (count > MaxInstructions)
            throw new InvalidDataException("DMKB-1 instruction count exceeds the implementation limit.");
        var expected = checked(4 + (checked((int)count) * 4));
        if (payload.Length != expected)
            throw new InvalidDataException("DMKB-1 instruction payload length is not canonical.");

        var instructions = new CompactInstruction[checked((int)count)];
        for (var i = 0; i < instructions.Length; i++)
        {
            var word = BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(4 + (i * 4), 4));
            instructions[i] = CompactInstruction.Unpack32(word);
        }
        return instructions;
    }
}
