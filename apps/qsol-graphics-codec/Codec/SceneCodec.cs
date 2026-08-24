using System.IO.Compression;
using System.Numerics;
using System.Text;
using QSol.GraphicsCodec.Core;

namespace QSol.GraphicsCodec.Codec;

public sealed record EncodedScene(byte[] Data, int QuantizationBits, int RawBytes);
public sealed record CodecCandidate(int Bits, int EncodedBytes, int RawBytes, float MaxVertexError, bool MeetsTolerance);
public sealed record AutoEncodeResult(EncodedScene Encoded, Scene3D Decoded, float MaxVertexError, IReadOnlyList<CodecCandidate> Candidates);

public static class SceneCodec
{
    private static readonly byte[] EnvelopeMagic = "QSC1"u8.ToArray();
    private static readonly byte[] PayloadMagic = "Q3D1"u8.ToArray();
    private const byte EnvelopeVersion = 1;
    private const byte PayloadVersion = 1;

    public static AutoEncodeResult AutoEncode(Scene3D scene, float targetMaxVertexError = 0.0025f, bool compress = true)
    {
        if (targetMaxVertexError <= 0f) throw new ArgumentOutOfRangeException(nameof(targetMaxVertexError));

        var candidates = new List<CodecCandidate>();
        EncodedScene? best = null;
        Scene3D? bestDecoded = null;
        var bestError = float.PositiveInfinity;

        foreach (var bits in new[] { 8, 10, 12, 14, 16 })
        {
            var encoded = Encode(scene, bits, compress);
            var decoded = Decode(encoded.Data);
            var error = SceneMetrics.MaxVertexError(scene, decoded);
            var meets = error <= targetMaxVertexError;
            candidates.Add(new CodecCandidate(bits, encoded.Data.Length, encoded.RawBytes, error, meets));

            if (meets && (best is null || encoded.Data.Length < best.Data.Length))
            {
                best = encoded;
                bestDecoded = decoded;
                bestError = error;
            }
        }

        if (best is null)
        {
            best = Encode(scene, 16, compress);
            bestDecoded = Decode(best.Data);
            bestError = SceneMetrics.MaxVertexError(scene, bestDecoded);
        }

        return new AutoEncodeResult(best, bestDecoded!, bestError, candidates);
    }

    public static EncodedScene Encode(Scene3D scene, int quantizationBits = 14, bool compress = true)
    {
        if (quantizationBits is < 8 or > 16)
            throw new ArgumentOutOfRangeException(nameof(quantizationBits), "Quantization must be 8..16 bits per axis.");

        var raw = EncodePayload(scene, quantizationBits);
        using var output = new MemoryStream();
        using var writer = new BinaryWriter(output, Encoding.UTF8, leaveOpen: true);
        writer.Write(EnvelopeMagic);
        writer.Write(EnvelopeVersion);
        writer.Write((byte)(compress ? 1 : 0));
        writer.Write(raw.Length);
        writer.Flush();

        if (compress)
        {
            using var brotli = new BrotliStream(output, CompressionLevel.SmallestSize, leaveOpen: true);
            brotli.Write(raw);
        }
        else
        {
            output.Write(raw);
        }

        return new EncodedScene(output.ToArray(), quantizationBits, raw.Length);
    }

    public static Scene3D Decode(ReadOnlySpan<byte> encoded)
    {
        using var input = new MemoryStream(encoded.ToArray(), writable: false);
        using var reader = new BinaryReader(input, Encoding.UTF8, leaveOpen: true);

        RequireMagic(reader.ReadBytes(4), EnvelopeMagic, "QSC envelope");
        if (reader.ReadByte() != EnvelopeVersion) throw new InvalidDataException("Unsupported QSC envelope version.");
        var compressed = reader.ReadByte() != 0;
        var rawLength = reader.ReadInt32();
        if (rawLength is < 0 or > 1_073_741_824) throw new InvalidDataException("Invalid payload length.");

        byte[] raw;
        if (compressed)
        {
            using var brotli = new BrotliStream(input, CompressionMode.Decompress, leaveOpen: true);
            using var rawStream = new MemoryStream(rawLength > 0 ? rawLength : 0);
            brotli.CopyTo(rawStream);
            raw = rawStream.ToArray();
        }
        else
        {
            raw = reader.ReadBytes(rawLength);
        }

        if (raw.Length != rawLength) throw new InvalidDataException("Payload length mismatch.");
        return DecodePayload(raw);
    }

    private static byte[] EncodePayload(Scene3D scene, int bits)
    {
        var bounds = scene.ComputeLocalGeometryBounds();
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(PayloadMagic);
        writer.Write(PayloadVersion);
        writer.Write((byte)bits);
        WriteVector3(writer, bounds.Min);
        WriteVector3(writer, bounds.Max);
        writer.Flush();
        VarInt.WriteUInt(stream, checked((uint)scene.Entities.Count));

        foreach (var entity in scene.Entities)
        {
            writer.Write(entity.Name ?? string.Empty);
            WriteVector3(writer, entity.Transform.Position);
            WriteQuaternion(writer, entity.Transform.Rotation);
            WriteVector3(writer, entity.Transform.Scale);
            WriteVector3(writer, entity.LinearVelocity);
            WriteVector3(writer, entity.AngularVelocity);
            WriteVector4(writer, entity.Material.BaseColor);
            writer.Write(entity.Material.Metallic);
            writer.Write(entity.Material.Roughness);
            writer.Flush();

            VarInt.WriteUInt(stream, checked((uint)entity.Mesh.Vertices.Length));
            VarInt.WriteUInt(stream, checked((uint)entity.Mesh.Indices.Length));

            var bitWriter = new BitWriter(stream);
            foreach (var vertex in entity.Mesh.Vertices)
            {
                bitWriter.Write(Quantize(vertex.X, bounds.Min.X, bounds.Max.X, bits), bits);
                bitWriter.Write(Quantize(vertex.Y, bounds.Min.Y, bounds.Max.Y, bits), bits);
                bitWriter.Write(Quantize(vertex.Z, bounds.Min.Z, bounds.Max.Z, bits), bits);
            }
            bitWriter.AlignToByte();

            var previous = 0;
            foreach (var index in entity.Mesh.Indices)
            {
                if ((uint)index >= (uint)entity.Mesh.Vertices.Length)
                    throw new InvalidDataException($"Entity '{entity.Name}' contains an out-of-range mesh index.");
                var delta = index - previous;
                VarInt.WriteUInt(stream, VarInt.ZigZagEncode(delta));
                previous = index;
            }
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static Scene3D DecodePayload(byte[] raw)
    {
        using var stream = new MemoryStream(raw, writable: false);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        RequireMagic(reader.ReadBytes(4), PayloadMagic, "Q3D payload");
        if (reader.ReadByte() != PayloadVersion) throw new InvalidDataException("Unsupported Q3D payload version.");
        var bits = reader.ReadByte();
        if (bits is < 8 or > 16) throw new InvalidDataException("Invalid quantization depth.");

        var min = ReadVector3(reader);
        var max = ReadVector3(reader);
        var entityCount = checked((int)VarInt.ReadUInt(stream));
        if (entityCount > 1_000_000) throw new InvalidDataException("Entity count exceeds decoder limit.");

        var scene = new Scene3D();
        for (var e = 0; e < entityCount; e++)
        {
            var name = reader.ReadString();
            var transform = new Transform3D(ReadVector3(reader), ReadQuaternion(reader), ReadVector3(reader));
            var linearVelocity = ReadVector3(reader);
            var angularVelocity = ReadVector3(reader);
            var material = new MaterialPbr(ReadVector4(reader), reader.ReadSingle(), reader.ReadSingle());
            var vertexCount = checked((int)VarInt.ReadUInt(stream));
            var indexCount = checked((int)VarInt.ReadUInt(stream));
            if (vertexCount > 50_000_000 || indexCount > 150_000_000)
                throw new InvalidDataException("Mesh exceeds decoder limits.");

            var vertices = new Vector3[vertexCount];
            var bitReader = new BitReader(stream);
            for (var i = 0; i < vertexCount; i++)
            {
                vertices[i] = new Vector3(
                    Dequantize(bitReader.Read(bits), min.X, max.X, bits),
                    Dequantize(bitReader.Read(bits), min.Y, max.Y, bits),
                    Dequantize(bitReader.Read(bits), min.Z, max.Z, bits));
            }
            bitReader.AlignToByte();

            var indices = new int[indexCount];
            var previous = 0;
            for (var i = 0; i < indexCount; i++)
            {
                var delta = VarInt.ZigZagDecode(VarInt.ReadUInt(stream));
                var index = checked(previous + delta);
                if ((uint)index >= (uint)vertexCount) throw new InvalidDataException("Decoded mesh index is out of range.");
                indices[i] = index;
                previous = index;
            }

            scene.Entities.Add(new Entity3D
            {
                Name = name,
                Mesh = new MeshData { Vertices = vertices, Indices = indices },
                Material = material,
                Transform = transform,
                LinearVelocity = linearVelocity,
                AngularVelocity = angularVelocity
            });
        }

        return scene;
    }

    private static uint Quantize(float value, float min, float max, int bits)
    {
        var levels = (1u << bits) - 1u;
        var t = Math.Clamp((value - min) / (max - min), 0f, 1f);
        return (uint)MathF.Round(t * levels);
    }

    private static float Dequantize(uint value, float min, float max, int bits)
    {
        var levels = (1u << bits) - 1u;
        return min + (value / (float)levels) * (max - min);
    }

    private static void WriteVector3(BinaryWriter w, Vector3 v) { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); }
    private static Vector3 ReadVector3(BinaryReader r) => new(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
    private static void WriteVector4(BinaryWriter w, Vector4 v) { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); w.Write(v.W); }
    private static Vector4 ReadVector4(BinaryReader r) => new(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
    private static void WriteQuaternion(BinaryWriter w, Quaternion q) { w.Write(q.X); w.Write(q.Y); w.Write(q.Z); w.Write(q.W); }
    private static Quaternion ReadQuaternion(BinaryReader r) => new(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle());

    private static void RequireMagic(byte[] actual, byte[] expected, string label)
    {
        if (!actual.AsSpan().SequenceEqual(expected)) throw new InvalidDataException($"Invalid {label} magic.");
    }
}
