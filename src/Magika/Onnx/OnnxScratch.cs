namespace Magika.Onnx;

/// <summary>
/// Exact-sized activation buffers reused across runs on one session.
/// <see cref="Enter"/> binds the scratch to the calling thread for the duration of <see cref="OnnxSession.Run"/>.
/// </summary>
internal sealed class OnnxScratch
{
    [ThreadStatic]
    private static OnnxScratch? t_current;

    private string _slot = "";
    private readonly Dictionary<SlotKey, float[]> _floats = new();
    private readonly Dictionary<SlotKey, int[]> _ints = new();
    private readonly Dictionary<SlotKey, long[]> _longs = new();
    private readonly Dictionary<SlotKey, byte[]> _bools = new();
    private readonly Dictionary<SlotKey, OnnxTensor> _tensors = new();

    public readonly ref struct Scope
    {
        public void Dispose() => t_current = null;
    }

    public Scope Enter()
    {
        if (t_current is not null)
        {
            throw new InvalidOperationException("An ONNX scratch buffer is already active on this thread.");
        }

        t_current = this;
        return default;
    }

    public void SetSlot(string slot) => _slot = slot;

    public static float[] RentFloat(int length) =>
        Current.Rent(Current._floats, length, static n => GC.AllocateUninitializedArray<float>(n));

    public static int[] RentInt(int length) =>
        Current.Rent(Current._ints, length, static n => GC.AllocateUninitializedArray<int>(n));

    public static long[] RentLong(int length) =>
        Current.Rent(Current._longs, length, static n => GC.AllocateUninitializedArray<long>(n));

    public static byte[] RentBool(int length) =>
        Current.Rent(Current._bools, length, static n => GC.AllocateUninitializedArray<byte>(n));

    public static OnnxTensor Float(int[] shape, float[] data) => Current.Wrap(shape, data, OnnxTensor.Float);

    public static OnnxTensor Int32(int[] shape, int[] data) => Current.Wrap(shape, data, OnnxTensor.Int32);

    public static OnnxTensor Int64(int[] shape, long[] data) => Current.Wrap(shape, data, OnnxTensor.Int64);

    public static OnnxTensor Bool(int[] shape, byte[] data) => Current.Wrap(shape, data, OnnxTensor.Bool);

    public static T[] Rent<T>(int length)
    {
        if (typeof(T) == typeof(float))
        {
            return (T[])(object)RentFloat(length);
        }

        if (typeof(T) == typeof(int))
        {
            return (T[])(object)RentInt(length);
        }

        if (typeof(T) == typeof(long))
        {
            return (T[])(object)RentLong(length);
        }

        if (typeof(T) == typeof(byte))
        {
            return (T[])(object)RentBool(length);
        }

        throw new NotSupportedException($"Cannot rent a {typeof(T)} activation buffer.");
    }

    private static OnnxScratch Current =>
        t_current ?? throw new InvalidOperationException("ONNX scratch is not active.");

    private T[] Rent<T>(Dictionary<SlotKey, T[]> cache, int length, Func<int, T[]> allocate)
    {
        var key = new SlotKey(_slot, length);
        if (cache.TryGetValue(key, out T[]? existing))
        {
            return existing;
        }

        T[] created = allocate(length);
        cache[key] = created;
        return created;
    }

    private OnnxTensor Wrap<T>(int[] shape, T[] data, Func<int[], T[], OnnxTensor> factory)
    {
        var key = new SlotKey(_slot, data.Length);
        if (_tensors.TryGetValue(key, out OnnxTensor? existing)
            && SameStorage(existing, data)
            && ShapeEquals(existing.Shape, shape))
        {
            return existing;
        }

        int[] owned = new int[shape.Length];
        shape.AsSpan().CopyTo(owned);
        OnnxTensor created = factory(owned, data);
        _tensors[key] = created;
        return created;
    }

    private static bool SameStorage<T>(OnnxTensor tensor, T[] data) => data switch
    {
        float[] floats => ReferenceEquals(tensor.Floats, floats),
        int[] ints => ReferenceEquals(tensor.Ints, ints),
        long[] longs => ReferenceEquals(tensor.Longs, longs),
        byte[] bools => ReferenceEquals(tensor.Bools, bools),
        _ => false,
    };

    private static bool ShapeEquals(int[] left, int[] right)
    {
        if (left.Length != right.Length)
        {
            return false;
        }

        for (int i = 0; i < left.Length; i++)
        {
            if (left[i] != right[i])
            {
                return false;
            }
        }

        return true;
    }

    private readonly record struct SlotKey(string Slot, int Length);
}
