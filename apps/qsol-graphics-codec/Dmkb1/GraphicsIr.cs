namespace QSol.GraphicsCodec.Dmkb1;

public enum GraphicsOpcode : byte
{
    Nop = 0x00,
    LoadVec3 = 0x01,
    LoadVec4 = 0x02,
    StoreVec3 = 0x03,
    AddVec3 = 0x04,
    MulScalar = 0x05,
    RotAxis = 0x06,
    BindMeshlet = 0x07,
    BindShader = 0x08,
    DrawIndexed = 0x09,
    ReflectEval = 0x0A
}

public readonly record struct CompactInstruction
{
    public GraphicsOpcode Opcode { get; }
    public byte Dst { get; }
    public byte Src1 { get; }
    public byte Src2 { get; }
    public byte Flags { get; }
    public byte Immediate { get; }

    public CompactInstruction(
        GraphicsOpcode opcode,
        byte dst,
        byte src1,
        byte src2,
        byte flags,
        byte immediate)
    {
        if (!Enum.IsDefined(opcode))
            throw new ArgumentOutOfRangeException(nameof(opcode), "Unknown DMKB-1 graphics opcode.");
        if (dst > 31) throw new ArgumentOutOfRangeException(nameof(dst), "Destination register must be 0..31.");
        if (src1 > 31) throw new ArgumentOutOfRangeException(nameof(src1), "Source register 1 must be 0..31.");
        if (src2 > 31) throw new ArgumentOutOfRangeException(nameof(src2), "Source register 2 must be 0..31.");
        if (flags > 7) throw new ArgumentOutOfRangeException(nameof(flags), "Flags must be 0..7.");
        if (immediate > 63) throw new ArgumentOutOfRangeException(nameof(immediate), "Immediate must be 0..63.");

        Opcode = opcode;
        Dst = dst;
        Src1 = src1;
        Src2 = src2;
        Flags = flags;
        Immediate = immediate;
    }

    public uint Pack32()
    {
        uint packed = 0;
        packed |= (uint)Opcode;
        packed |= (uint)Dst << 8;
        packed |= (uint)Src1 << 13;
        packed |= (uint)Src2 << 18;
        packed |= (uint)Flags << 23;
        packed |= (uint)Immediate << 26;
        return packed;
    }

    public static CompactInstruction Unpack32(uint packed)
    {
        var opcode = (GraphicsOpcode)(packed & 0xFFu);
        return new CompactInstruction(
            opcode,
            (byte)((packed >> 8) & 0x1Fu),
            (byte)((packed >> 13) & 0x1Fu),
            (byte)((packed >> 18) & 0x1Fu),
            (byte)((packed >> 23) & 0x07u),
            (byte)((packed >> 26) & 0x3Fu));
    }

    public override string ToString() =>
        $"[{Opcode}] Dst:r{Dst}, Src1:r{Src1}, Src2:r{Src2}, Flags:{Flags}, Imm:{Immediate}";
}
