using System.Buffers.Binary;
using System.Text;

namespace Magika.Onnx;

internal sealed class OnnxAttr
{
    public bool HasInt { get; set; }
    public long Int { get; set; }
    public bool HasFloat { get; set; }
    public float Float { get; set; }
    public List<long>? Ints { get; set; }
    public List<float>? Floats { get; set; }
    public OnnxTensor? Tensor { get; set; }
}

internal sealed class OnnxNode
{
    public required string Op { get; init; }
    public required string Name { get; init; }
    public required string[] Inputs { get; init; }
    public required string[] Outputs { get; init; }
    public required Dictionary<string, OnnxAttr> Attributes { get; init; }

    public long Int(string name, long fallback = 0) =>
        Attributes.TryGetValue(name, out OnnxAttr? attr) && attr.HasInt ? attr.Int : fallback;

    public int[] Ints(string name)
    {
        if (!Attributes.TryGetValue(name, out OnnxAttr? attr) || attr.Ints is null)
        {
            return [];
        }

        int[] values = new int[attr.Ints.Count];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = checked((int)attr.Ints[i]);
        }

        return values;
    }
}

internal sealed class OnnxModel
{
    public required string InputName { get; init; }
    public required string OutputName { get; init; }
    public required OnnxNode[] Nodes { get; init; }
    public required Dictionary<string, OnnxTensor> Initializers { get; init; }
    public required bool EmbeddingFused { get; init; }

    public static OnnxModel Load(ReadOnlySpan<byte> model)
    {
        if (!BitConverter.IsLittleEndian)
        {
            throw new NotSupportedException("The managed ONNX loader requires a little-endian runtime.");
        }

        Dictionary<string, OnnxTensor> initializers = new(StringComparer.Ordinal);
        List<OnnxNode> nodes = [];
        List<string> inputs = [];
        List<string> outputs = [];
        ProtoReader reader = new(model);
        while (reader.TryReadTag(out int field, out int wire))
        {
            if (field == 7 && wire == 2)
            {
                ParseGraph(reader.ReadBytes(), nodes, initializers, inputs, outputs);
            }
            else
            {
                reader.Skip(wire);
            }
        }

        if (nodes.Count == 0 || outputs.Count == 0)
        {
            throw new InvalidDataException("ONNX model has no graph.");
        }

        string inputName = "";
        foreach (string name in inputs)
        {
            if (!initializers.ContainsKey(name))
            {
                inputName = name;
                break;
            }
        }

        if (inputName.Length == 0)
        {
            throw new InvalidDataException("ONNX model has no runtime input.");
        }

        bool fused = FuseEmbedding(nodes, initializers);
        return new OnnxModel
        {
            InputName = inputName,
            OutputName = outputs[0],
            Nodes = nodes.ToArray(),
            Initializers = initializers,
            EmbeddingFused = fused,
        };
    }

    private static void ParseGraph(
        ReadOnlySpan<byte> bytes,
        List<OnnxNode> nodes,
        Dictionary<string, OnnxTensor> initializers,
        List<string> inputs,
        List<string> outputs)
    {
        ProtoReader reader = new(bytes);
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2:
                    nodes.Add(ParseNode(reader.ReadBytes()));
                    break;
                case 5 when wire == 2:
                    OnnxTensor tensor = ParseTensor(reader.ReadBytes(), out string name);
                    initializers[name] = tensor;
                    break;
                case 11 when wire == 2:
                    inputs.Add(ParseValueName(reader.ReadBytes()));
                    break;
                case 12 when wire == 2:
                    outputs.Add(ParseValueName(reader.ReadBytes()));
                    break;
                default:
                    reader.Skip(wire);
                    break;
            }
        }
    }

    private static OnnxNode ParseNode(ReadOnlySpan<byte> bytes)
    {
        List<string> inputs = [];
        List<string> outputs = [];
        string name = "";
        string op = "";
        Dictionary<string, OnnxAttr> attributes = new(StringComparer.Ordinal);
        ProtoReader reader = new(bytes);
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2:
                    inputs.Add(ReadString(reader.ReadBytes()));
                    break;
                case 2 when wire == 2:
                    outputs.Add(ReadString(reader.ReadBytes()));
                    break;
                case 3 when wire == 2:
                    name = ReadString(reader.ReadBytes());
                    break;
                case 4 when wire == 2:
                    op = ReadString(reader.ReadBytes());
                    break;
                case 5 when wire == 2:
                    (string attrName, OnnxAttr attr) = ParseAttribute(reader.ReadBytes());
                    attributes[attrName] = attr;
                    break;
                default:
                    reader.Skip(wire);
                    break;
            }
        }

        return new OnnxNode
        {
            Op = op,
            Name = name,
            Inputs = inputs.ToArray(),
            Outputs = outputs.ToArray(),
            Attributes = attributes,
        };
    }

    private static (string Name, OnnxAttr Attr) ParseAttribute(ReadOnlySpan<byte> bytes)
    {
        string name = "";
        OnnxAttr attr = new();
        ProtoReader reader = new(bytes);
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2:
                    name = ReadString(reader.ReadBytes());
                    break;
                case 2 when wire == 5:
                    attr.HasFloat = true;
                    attr.Float = BitConverter.Int32BitsToSingle((int)reader.ReadFixed32());
                    break;
                case 3 when wire == 0:
                    attr.HasInt = true;
                    attr.Int = unchecked((long)reader.ReadVarint());
                    break;
                case 5 when wire == 2:
                    attr.Tensor = ParseTensor(reader.ReadBytes(), out _);
                    break;
                case 7:
                    attr.Floats ??= [];
                    if (wire == 5)
                    {
                        attr.Floats.Add(BitConverter.Int32BitsToSingle((int)reader.ReadFixed32()));
                    }
                    else if (wire == 2)
                    {
                        ReadPackedFloats(reader.ReadBytes(), attr.Floats);
                    }
                    else
                    {
                        reader.Skip(wire);
                    }

                    break;
                case 8:
                    attr.Ints ??= [];
                    if (wire == 0)
                    {
                        attr.Ints.Add(unchecked((long)reader.ReadVarint()));
                    }
                    else if (wire == 2)
                    {
                        ReadPackedVarints(reader.ReadBytes(), attr.Ints);
                    }
                    else
                    {
                        reader.Skip(wire);
                    }

                    break;
                default:
                    reader.Skip(wire);
                    break;
            }
        }

        return (name, attr);
    }

    private static string ParseValueName(ReadOnlySpan<byte> bytes)
    {
        ProtoReader reader = new(bytes);
        string name = "";
        while (reader.TryReadTag(out int field, out int wire))
        {
            if (field == 1 && wire == 2)
            {
                name = ReadString(reader.ReadBytes());
            }
            else
            {
                reader.Skip(wire);
            }
        }

        return name;
    }

    private static OnnxTensor ParseTensor(ReadOnlySpan<byte> bytes, out string name)
    {
        name = "";
        List<long> dims = [];
        int dataType = 0;
        List<float>? floats = null;
        List<int>? ints = null;
        List<long>? longs = null;
        byte[]? raw = null;
        int location = 0;
        ProtoReader reader = new(bytes);
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1:
                    if (wire == 0)
                    {
                        dims.Add(unchecked((long)reader.ReadVarint()));
                    }
                    else if (wire == 2)
                    {
                        ReadPackedVarints(reader.ReadBytes(), dims);
                    }
                    else
                    {
                        reader.Skip(wire);
                    }

                    break;
                case 2 when wire == 0:
                    dataType = (int)reader.ReadVarint();
                    break;
                case 4:
                    floats ??= [];
                    if (wire == 5)
                    {
                        floats.Add(BitConverter.Int32BitsToSingle((int)reader.ReadFixed32()));
                    }
                    else if (wire == 2)
                    {
                        ReadPackedFloats(reader.ReadBytes(), floats);
                    }
                    else
                    {
                        reader.Skip(wire);
                    }

                    break;
                case 5:
                    ints ??= [];
                    if (wire == 0)
                    {
                        ints.Add(unchecked((int)reader.ReadVarint()));
                    }
                    else if (wire == 2)
                    {
                        ReadPackedInt32(reader.ReadBytes(), ints);
                    }
                    else
                    {
                        reader.Skip(wire);
                    }

                    break;
                case 7:
                    longs ??= [];
                    if (wire == 0)
                    {
                        longs.Add(unchecked((long)reader.ReadVarint()));
                    }
                    else if (wire == 2)
                    {
                        ReadPackedVarints(reader.ReadBytes(), longs);
                    }
                    else
                    {
                        reader.Skip(wire);
                    }

                    break;
                case 8 when wire == 2:
                    name = ReadString(reader.ReadBytes());
                    break;
                case 9 when wire == 2:
                    raw = reader.ReadBytes().ToArray();
                    break;
                case 13 when wire == 0:
                    location = (int)reader.ReadVarint();
                    break;
                default:
                    reader.Skip(wire);
                    break;
            }
        }

        if (location != 0)
        {
            throw new NotSupportedException($"External ONNX tensor data is not supported ({name}).");
        }

        int[] shape = new int[dims.Count];
        for (int i = 0; i < dims.Count; i++)
        {
            shape[i] = checked((int)dims[i]);
        }

        int count = OnnxTensor.ElementCount(shape);
        if (raw is { Length: > 0 })
        {
            return TensorFromRaw(name, dataType, shape, count, raw);
        }

        return dataType switch
        {
            1 => OnnxTensor.Float(shape, Pad(floats, count, name)),
            6 => OnnxTensor.Int32(shape, Pad(ints, count, name)),
            7 => OnnxTensor.Int64(shape, Pad(longs, count, name)),
            9 => OnnxTensor.Bool(shape, new byte[count]),
            _ => throw new NotSupportedException($"ONNX tensor '{name}' has element type {dataType}."),
        };
    }

    private static OnnxTensor TensorFromRaw(string name, int dataType, int[] shape, int count, byte[] raw)
    {
        switch (dataType)
        {
            case 1:
                float[] floats = new float[count];
                if (raw.Length != count * 4)
                {
                    throw new InvalidDataException($"ONNX tensor '{name}' raw length {raw.Length} is not {count} floats.");
                }

                Buffer.BlockCopy(raw, 0, floats, 0, raw.Length);
                return OnnxTensor.Float(shape, floats);
            case 6:
                int[] ints = new int[count];
                if (raw.Length != count * 4)
                {
                    throw new InvalidDataException($"ONNX tensor '{name}' raw length {raw.Length} is not {count} int32s.");
                }

                Buffer.BlockCopy(raw, 0, ints, 0, raw.Length);
                return OnnxTensor.Int32(shape, ints);
            case 7:
                long[] longs = new long[count];
                if (raw.Length != count * 8)
                {
                    throw new InvalidDataException($"ONNX tensor '{name}' raw length {raw.Length} is not {count} int64s.");
                }

                Buffer.BlockCopy(raw, 0, longs, 0, raw.Length);
                return OnnxTensor.Int64(shape, longs);
            case 9:
                if (raw.Length != count)
                {
                    throw new InvalidDataException($"ONNX tensor '{name}' raw length {raw.Length} is not {count} bools.");
                }

                return OnnxTensor.Bool(shape, raw);
            default:
                throw new NotSupportedException($"ONNX tensor '{name}' has element type {dataType}.");
        }
    }

    private static bool FuseEmbedding(List<OnnxNode> nodes, Dictionary<string, OnnxTensor> initializers)
    {
        Dictionary<string, int> producer = new(StringComparer.Ordinal);
        Dictionary<string, int> consumers = new(StringComparer.Ordinal);
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (string output in nodes[i].Outputs)
            {
                producer[output] = i;
            }

            foreach (string input in nodes[i].Inputs)
            {
                if (input.Length == 0)
                {
                    continue;
                }

                consumers.TryGetValue(input, out int count);
                consumers[input] = count + 1;
            }
        }

        bool fused = false;
        for (int i = 0; i < nodes.Count; i++)
        {
            OnnxNode matmul = nodes[i];
            if (matmul.Op != "MatMul" || matmul.Inputs.Length != 2 || matmul.Outputs.Length != 1)
            {
                continue;
            }

            if (!initializers.TryGetValue(matmul.Inputs[1], out OnnxTensor? weight)
                || weight.DType != OnnxDType.Float
                || weight.Rank != 2)
            {
                continue;
            }

            if (!producer.TryGetValue(matmul.Inputs[0], out int castIndex))
            {
                continue;
            }

            OnnxNode cast = nodes[castIndex];
            if (cast.Op != "Cast" || cast.Inputs.Length != 1 || cast.Int("to") != 1)
            {
                continue;
            }

            if (!producer.TryGetValue(cast.Inputs[0], out int equalIndex))
            {
                continue;
            }

            OnnxNode equal = nodes[equalIndex];
            if (equal.Op != "Equal" || equal.Inputs.Length != 2 || equal.Outputs.Length != 1 || cast.Outputs.Length != 1)
            {
                continue;
            }

            string codesName;
            string tokenName;
            if (initializers.ContainsKey(equal.Inputs[1]) && !initializers.ContainsKey(equal.Inputs[0]))
            {
                codesName = equal.Inputs[1];
                tokenName = equal.Inputs[0];
            }
            else if (initializers.ContainsKey(equal.Inputs[0]) && !initializers.ContainsKey(equal.Inputs[1]))
            {
                codesName = equal.Inputs[0];
                tokenName = equal.Inputs[1];
            }
            else
            {
                continue;
            }

            if (!IsArange(initializers[codesName], weight.Shape[0]))
            {
                continue;
            }

            if (consumers.GetValueOrDefault(equal.Outputs[0]) != 1
                || consumers.GetValueOrDefault(cast.Outputs[0]) != 1)
            {
                continue;
            }

            nodes[i] = new OnnxNode
            {
                Op = "Embedding",
                Name = matmul.Name,
                Inputs = [tokenName, matmul.Inputs[1]],
                Outputs = matmul.Outputs,
                Attributes = new Dictionary<string, OnnxAttr>(StringComparer.Ordinal),
            };
            nodes[equalIndex] = Skip(equal);
            nodes[castIndex] = Skip(cast);
            fused = true;
        }

        if (fused)
        {
            nodes.RemoveAll(static node => node.Op == "Skip");
        }

        return fused;
    }

    private static OnnxNode Skip(OnnxNode node) => new()
    {
        Op = "Skip",
        Name = node.Name,
        Inputs = [],
        Outputs = [],
        Attributes = node.Attributes,
    };

    private static bool IsArange(OnnxTensor tensor, int length)
    {
        if (tensor.Length != length || tensor.DType is not (OnnxDType.Int32 or OnnxDType.Int64))
        {
            return false;
        }

        for (int i = 0; i < length; i++)
        {
            if (tensor.GetInt64(i) != i)
            {
                return false;
            }
        }

        return true;
    }

    private static string ReadString(ReadOnlySpan<byte> bytes) => Encoding.UTF8.GetString(bytes);

    private static void ReadPackedVarints(ReadOnlySpan<byte> bytes, List<long> dest)
    {
        ProtoReader reader = new(bytes);
        while (!reader.Eof)
        {
            dest.Add(unchecked((long)reader.ReadVarint()));
        }
    }

    private static void ReadPackedInt32(ReadOnlySpan<byte> bytes, List<int> dest)
    {
        ProtoReader reader = new(bytes);
        while (!reader.Eof)
        {
            dest.Add(unchecked((int)reader.ReadVarint()));
        }
    }

    private static void ReadPackedFloats(ReadOnlySpan<byte> bytes, List<float> dest)
    {
        if (bytes.Length % 4 != 0)
        {
            throw new InvalidDataException("Packed float field is not a multiple of 4 bytes.");
        }

        for (int i = 0; i < bytes.Length; i += 4)
        {
            dest.Add(BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.Slice(i, 4))));
        }
    }

    private static T[] Pad<T>(List<T>? values, int count, string name)
    {
        if (values is null || values.Count != count)
        {
            throw new InvalidDataException($"ONNX tensor '{name}' does not contain {count} values.");
        }

        return values.ToArray();
    }
}
