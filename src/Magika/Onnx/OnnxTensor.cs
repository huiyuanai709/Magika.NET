namespace Magika.Onnx;

internal enum OnnxDType
{
    Float,
    Int32,
    Int64,
    Bool,
}

internal sealed class OnnxTensor
{
    private OnnxTensor(int[] shape, OnnxDType dtype, float[]? floats, int[]? ints, long[]? longs, byte[]? bools)
    {
        Shape = shape;
        DType = dtype;
        Floats = floats;
        Ints = ints;
        Longs = longs;
        Bools = bools;
        Length = ElementCount(shape);
        int stored = floats?.Length ?? ints?.Length ?? longs?.Length ?? bools?.Length ?? 0;
        if (stored != Length)
        {
            throw new InvalidDataException($"Tensor storage length {stored} does not match shape size {Length}.");
        }
    }

    public int[] Shape { get; }
    public OnnxDType DType { get; }
    public float[]? Floats { get; }
    public int[]? Ints { get; }
    public long[]? Longs { get; }
    public byte[]? Bools { get; }
    public int Length { get; }
    public int Rank => Shape.Length;

    public static OnnxTensor Float(int[] shape, float[] data) => new(shape, OnnxDType.Float, data, null, null, null);
    public static OnnxTensor Int32(int[] shape, int[] data) => new(shape, OnnxDType.Int32, null, data, null, null);
    public static OnnxTensor Int64(int[] shape, long[] data) => new(shape, OnnxDType.Int64, null, null, data, null);
    public static OnnxTensor Bool(int[] shape, byte[] data) => new(shape, OnnxDType.Bool, null, null, null, data);

    public static int ElementCount(ReadOnlySpan<int> shape)
    {
        long count = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] < 0)
            {
                throw new InvalidDataException($"Negative tensor dimension {shape[i]}.");
            }

            count *= shape[i];
            if (count > int.MaxValue)
            {
                throw new InvalidDataException("Tensor is too large.");
            }
        }

        return (int)count;
    }

    public long GetInt64(int index) => DType switch
    {
        OnnxDType.Int64 => Longs![index],
        OnnxDType.Int32 => Ints![index],
        _ => throw new InvalidDataException($"Tensor is {DType}, expected an integer."),
    };

    public long[] AsInt64Array()
    {
        if (Longs is not null)
        {
            return Longs;
        }

        if (Ints is null)
        {
            throw new InvalidDataException($"Tensor is {DType}, expected an integer.");
        }

        long[] values = new long[Ints.Length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Ints[i];
        }

        return values;
    }

    public int GetToken(int index) => DType switch
    {
        OnnxDType.Int32 => Ints![index],
        OnnxDType.Int64 => checked((int)Longs![index]),
        _ => throw new InvalidDataException($"Embedding tokens are {DType}, expected an integer."),
    };
}
