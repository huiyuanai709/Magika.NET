using System.Buffers;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Magika.Onnx;

internal static class OnnxOps
{
    internal static OnnxTensor Reshape(OnnxTensor data, ReadOnlySpan<long> requested)
    {
        int[] shape = new int[requested.Length];
        int infer = -1;
        long known = 1;
        for (int i = 0; i < requested.Length; i++)
        {
            long dim = requested[i];
            if (dim == -1)
            {
                if (infer >= 0)
                {
                    throw new InvalidDataException("Reshape has more than one inferred dimension.");
                }

                infer = i;
                continue;
            }

            if (dim == 0)
            {
                if (i >= data.Rank)
                {
                    throw new InvalidDataException("Reshape cannot copy a dimension past the input rank.");
                }

                dim = data.Shape[i];
            }

            if (dim < 0)
            {
                throw new InvalidDataException($"Invalid reshape dimension {dim}.");
            }

            shape[i] = (int)dim;
            known *= dim;
        }

        if (infer >= 0)
        {
            if (known == 0 || data.Length % known != 0)
            {
                throw new InvalidDataException("Reshape cannot infer a matching dimension.");
            }

            shape[infer] = (int)(data.Length / known);
        }
        else if (OnnxTensor.ElementCount(shape) != data.Length)
        {
            throw new InvalidDataException("Reshape element count does not match.");
        }

        return Copy(data, shape);
    }

    internal static OnnxTensor ShapeOf(OnnxTensor data)
    {
        long[] dims = new long[data.Rank];
        for (int i = 0; i < dims.Length; i++)
        {
            dims[i] = data.Shape[i];
        }

        return OnnxTensor.Int64([dims.Length], dims);
    }

    internal static OnnxTensor Cast(OnnxTensor data, int to)
    {
        OnnxDType dtype = to switch
        {
            1 => OnnxDType.Float,
            6 => OnnxDType.Int32,
            7 => OnnxDType.Int64,
            9 => OnnxDType.Bool,
            _ => throw new NotSupportedException($"Cast to ONNX element type {to} is not supported."),
        };
        if (data.DType == dtype)
        {
            return Copy(data, data.Shape.ToArray());
        }

        int n = data.Length;
        switch (dtype)
        {
            case OnnxDType.Float:
            {
                float[] dest = new float[n];
                for (int i = 0; i < n; i++)
                {
                    dest[i] = data.DType switch
                    {
                        OnnxDType.Bool => data.Bools![i] == 0 ? 0f : 1f,
                        OnnxDType.Int32 => data.Ints![i],
                        OnnxDType.Int64 => data.Longs![i],
                        _ => data.Floats![i],
                    };
                }

                return OnnxTensor.Float(data.Shape.ToArray(), dest);
            }
            case OnnxDType.Int32:
            {
                int[] dest = new int[n];
                for (int i = 0; i < n; i++)
                {
                    dest[i] = data.DType switch
                    {
                        OnnxDType.Int64 => checked((int)data.Longs![i]),
                        OnnxDType.Float => checked((int)data.Floats![i]),
                        OnnxDType.Bool => data.Bools![i],
                        _ => data.Ints![i],
                    };
                }

                return OnnxTensor.Int32(data.Shape.ToArray(), dest);
            }
            case OnnxDType.Int64:
            {
                long[] dest = new long[n];
                for (int i = 0; i < n; i++)
                {
                    dest[i] = data.DType switch
                    {
                        OnnxDType.Int32 => data.Ints![i],
                        OnnxDType.Float => checked((long)data.Floats![i]),
                        OnnxDType.Bool => data.Bools![i],
                        _ => data.Longs![i],
                    };
                }

                return OnnxTensor.Int64(data.Shape.ToArray(), dest);
            }
            default:
                throw new NotSupportedException($"Cast from {data.DType} to {dtype} is not supported.");
        }
    }

    internal static OnnxTensor Concat(ReadOnlySpan<OnnxTensor> inputs, int axis)
    {
        if (inputs.Length == 0)
        {
            throw new InvalidDataException("Concat requires at least one input.");
        }

        OnnxTensor first = inputs[0];
        int rank = first.Rank;
        axis = axis < 0 ? axis + rank : axis;
        if ((uint)axis >= (uint)rank)
        {
            throw new InvalidDataException($"Concat axis {axis} is out of range.");
        }

        int[] shape = first.Shape.ToArray();
        int total = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            OnnxTensor tensor = inputs[i];
            if (tensor.DType != first.DType || tensor.Rank != rank)
            {
                throw new InvalidDataException("Concat inputs differ in type or rank.");
            }

            for (int d = 0; d < rank; d++)
            {
                if (d != axis && tensor.Shape[d] != shape[d])
                {
                    throw new InvalidDataException("Concat inputs differ outside the concat axis.");
                }
            }

            total += tensor.Shape[axis];
        }

        shape[axis] = total;
        int outer = 1;
        for (int d = 0; d < axis; d++)
        {
            outer *= shape[d];
        }

        int inner = 1;
        for (int d = axis + 1; d < rank; d++)
        {
            inner *= shape[d];
        }

        return first.DType switch
        {
            OnnxDType.Float => ConcatFloat(inputs, shape, axis, outer, inner),
            OnnxDType.Int32 => ConcatInt32(inputs, shape, axis, outer, inner),
            OnnxDType.Int64 => ConcatInt64(inputs, shape, axis, outer, inner),
            _ => throw new NotSupportedException($"Concat of {first.DType} is not supported."),
        };
    }

    internal static OnnxTensor Slice(OnnxTensor data, long[] starts, long[] ends, long[]? axes, long[]? steps)
    {
        int rank = data.Rank;
        long[] start = new long[rank];
        long[] end = new long[rank];
        long[] step = new long[rank];
        for (int i = 0; i < rank; i++)
        {
            start[i] = 0;
            end[i] = data.Shape[i];
            step[i] = 1;
        }

        int count = starts.Length;
        if (ends.Length != count || (steps is not null && steps.Length != count))
        {
            throw new InvalidDataException("Slice starts, ends, and steps differ in length.");
        }

        if (axes is null)
        {
            if (count != rank)
            {
                throw new InvalidDataException("Slice without axes must cover every dimension.");
            }

            for (int i = 0; i < count; i++)
            {
                start[i] = starts[i];
                end[i] = ends[i];
                if (steps is not null)
                {
                    step[i] = steps[i];
                }
            }
        }
        else
        {
            if (axes.Length != count)
            {
                throw new InvalidDataException("Slice axes and starts differ in length.");
            }

            for (int i = 0; i < count; i++)
            {
                int axis = (int)axes[i];
                if (axis < 0)
                {
                    axis += rank;
                }

                start[axis] = starts[i];
                end[axis] = ends[i];
                if (steps is not null)
                {
                    step[axis] = steps[i];
                }
            }
        }

        int[] outShape = new int[rank];
        int[] normStart = new int[rank];
        int[] normStep = new int[rank];
        for (int axis = 0; axis < rank; axis++)
        {
            if (step[axis] <= 0)
            {
                throw new NotSupportedException("Negative or zero Slice steps are not supported.");
            }

            int dim = data.Shape[axis];
            long s = start[axis] < 0 ? start[axis] + dim : start[axis];
            long e = end[axis] < 0 ? end[axis] + dim : end[axis];
            s = Math.Clamp(s, 0, dim);
            e = Math.Clamp(e, 0, dim);
            normStart[axis] = (int)s;
            normStep[axis] = (int)step[axis];
            long span = Math.Max(0, e - s);
            outShape[axis] = (int)((span + step[axis] - 1) / step[axis]);
        }

        int n = OnnxTensor.ElementCount(outShape);
        bool identity = n == data.Length && outShape.AsSpan().SequenceEqual(data.Shape);
        if (identity)
        {
            for (int axis = 0; axis < rank; axis++)
            {
                if (normStart[axis] != 0 || normStep[axis] != 1)
                {
                    identity = false;
                    break;
                }
            }
        }

        if (identity)
        {
            return data;
        }

        int[] inStrides = Strides(data.Shape);
        int[] outStrides = Strides(outShape);
        return data.DType switch
        {
            OnnxDType.Float => Gather(data.Floats!, n, outShape, outStrides, inStrides, normStart, normStep, OnnxTensor.Float),
            OnnxDType.Int32 => Gather(data.Ints!, n, outShape, outStrides, inStrides, normStart, normStep, OnnxTensor.Int32),
            OnnxDType.Int64 => Gather(data.Longs!, n, outShape, outStrides, inStrides, normStart, normStep, OnnxTensor.Int64),
            _ => throw new NotSupportedException($"Slice of {data.DType} is not supported."),
        };
    }

    internal static OnnxTensor Expand(OnnxTensor data, int[] shape)
    {
        int[] input = LeftPad(data.Shape, shape.Length);
        for (int i = 0; i < shape.Length; i++)
        {
            if (input[i] != shape[i] && input[i] != 1)
            {
                throw new InvalidDataException($"Cannot expand dimension {input[i]} to {shape[i]}.");
            }
        }

        if (data.Shape.AsSpan().SequenceEqual(shape))
        {
            return data;
        }

        return data.DType switch
        {
            OnnxDType.Float => OnnxTensor.Float(shape, BroadcastCopy(data.Floats!, data.Shape, shape)),
            OnnxDType.Int32 => OnnxTensor.Int32(shape, BroadcastCopy(data.Ints!, data.Shape, shape)),
            OnnxDType.Int64 => OnnxTensor.Int64(shape, BroadcastCopy(data.Longs!, data.Shape, shape)),
            OnnxDType.Bool => OnnxTensor.Bool(shape, BroadcastCopy(data.Bools!, data.Shape, shape)),
            _ => throw new NotSupportedException($"Expand of {data.DType} is not supported."),
        };
    }

    internal static OnnxTensor Equal(OnnxTensor a, OnnxTensor b)
    {
        int[] shape = BroadcastShape(a.Shape, b.Shape);
        int n = OnnxTensor.ElementCount(shape);
        byte[] dest = new byte[n];
        int[] strideA = BroadcastStrides(a.Shape, shape);
        int[] strideB = BroadcastStrides(b.Shape, shape);
        Span<int> coord = shape.Length <= 8 ? stackalloc int[shape.Length] : new int[shape.Length];
        for (int i = 0; i < n; i++)
        {
            int ia = Offset(coord, strideA);
            int ib = Offset(coord, strideB);
            dest[i] = ValuesEqual(a, ia, b, ib) ? (byte)1 : (byte)0;
            Increment(coord, shape);
        }

        return OnnxTensor.Bool(shape, dest);
    }

    internal static OnnxTensor Binary(OnnxTensor a, OnnxTensor b, FloatOp op)
    {
        if (a.DType != OnnxDType.Float || b.DType != OnnxDType.Float)
        {
            throw new NotSupportedException($"Binary op requires float tensors, got {a.DType} and {b.DType}.");
        }

        int[] shape = BroadcastShape(a.Shape, b.Shape);
        int n = OnnxTensor.ElementCount(shape);
        float[] dest = GC.AllocateUninitializedArray<float>(n);
        int[] strideA = BroadcastStrides(a.Shape, shape);
        int[] strideB = BroadcastStrides(b.Shape, shape);
        if (TryContiguousBinary(a, b, dest, op) ||
            (shape.Length > 0 && n > 0 && TryVectorBinary(a.Floats!, b.Floats!, dest, shape, strideA, strideB, op)))
        {
            return OnnxTensor.Float(shape, dest);
        }

        Span<int> coord = shape.Length <= 8 ? stackalloc int[shape.Length] : new int[shape.Length];
        float[] left = a.Floats!;
        float[] right = b.Floats!;
        for (int i = 0; i < n; i++)
        {
            dest[i] = Apply(left[Offset(coord, strideA)], right[Offset(coord, strideB)], op);
            Increment(coord, shape);
        }

        return OnnxTensor.Float(shape, dest);
    }

    internal static OnnxTensor Tanh(OnnxTensor data)
    {
        if (data.DType != OnnxDType.Float)
        {
            throw new NotSupportedException($"Unary op requires float, got {data.DType}.");
        }

        float[] source = data.Floats!;
        float[] dest = GC.AllocateUninitializedArray<float>(source.Length);
        SimdMath.Tanh(source, dest);
        return OnnxTensor.Float(data.Shape.ToArray(), dest);
    }

    internal static OnnxTensor UnaryFloat(OnnxTensor data, FloatUnary op)
    {
        if (data.DType != OnnxDType.Float)
        {
            throw new NotSupportedException($"Unary op requires float, got {data.DType}.");
        }

        float[] source = data.Floats!;
        float[] dest = GC.AllocateUninitializedArray<float>(source.Length);
        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = op(source[i]);
        }

        return OnnxTensor.Float(data.Shape.ToArray(), dest);
    }

    internal static OnnxTensor Reduce(OnnxTensor data, int[] axes, bool keepDims, bool max, bool noopEmpty)
    {
        if (data.DType != OnnxDType.Float)
        {
            throw new NotSupportedException($"Reduce requires float, got {data.DType}.");
        }

        if (axes.Length == 0 && noopEmpty)
        {
            return data;
        }

        int rank = data.Rank;
        bool[] reduced = new bool[rank];
        if (axes.Length == 0)
        {
            Array.Fill(reduced, true);
        }
        else
        {
            foreach (int axis in axes)
            {
                int normalized = axis < 0 ? axis + rank : axis;
                if ((uint)normalized >= (uint)rank)
                {
                    throw new InvalidDataException($"Reduce axis {axis} is out of range.");
                }

                reduced[normalized] = true;
            }
        }

        List<int> shape = new(rank);
        for (int i = 0; i < rank; i++)
        {
            if (!reduced[i])
            {
                shape.Add(data.Shape[i]);
            }
            else if (keepDims)
            {
                shape.Add(1);
            }
        }

        int[] outShape = shape.ToArray();
        if (data.Floats is not null && TryFastReduce(data.Floats, data.Shape, reduced, keepDims, max, out float[]? fast))
        {
            return OnnxTensor.Float(outShape, fast);
        }

        int n = OnnxTensor.ElementCount(outShape);
        float[] acc = new float[n];
        if (max)
        {
            Array.Fill(acc, float.NegativeInfinity);
        }

        int[] inStrides = Strides(data.Shape);
        int[] outStrides = Strides(outShape);
        float[] source = data.Floats!;
        Span<int> coord = rank <= 8 ? stackalloc int[rank] : new int[rank];
        for (int i = 0; i < data.Length; i++)
        {
            int outIndex = 0;
            int outAxis = 0;
            for (int axis = 0; axis < rank; axis++)
            {
                if (!reduced[axis])
                {
                    outIndex += coord[axis] * outStrides[outAxis];
                    outAxis++;
                }
                else if (keepDims)
                {
                    outAxis++;
                }
            }

            float value = source[Offset(coord, inStrides)];
            acc[outIndex] = max ? MathF.Max(acc[outIndex], value) : acc[outIndex] + value;
            Increment(coord, data.Shape);
        }

        return OnnxTensor.Float(outShape, acc);
    }

    internal static OnnxTensor GlobalMaxPool(OnnxTensor data)
    {
        if (data.Rank < 2)
        {
            throw new InvalidDataException("GlobalMaxPool expects rank >= 2.");
        }

        int[] axes = new int[data.Rank - 2];
        for (int i = 0; i < axes.Length; i++)
        {
            axes[i] = i + 2;
        }

        return Reduce(data, axes, keepDims: true, max: true, noopEmpty: false);
    }

    internal static OnnxTensor Squeeze(OnnxTensor data, int[] axes)
    {
        bool[] drop = new bool[data.Rank];
        foreach (int axis in axes)
        {
            int normalized = axis < 0 ? axis + data.Rank : axis;
            if ((uint)normalized >= (uint)data.Rank || data.Shape[normalized] != 1)
            {
                throw new InvalidDataException($"Cannot squeeze axis {axis}.");
            }

            drop[normalized] = true;
        }

        List<int> shape = new(data.Rank);
        for (int i = 0; i < data.Rank; i++)
        {
            if (!drop[i])
            {
                shape.Add(data.Shape[i]);
            }
        }

        return Copy(data, shape.ToArray());
    }

    internal static OnnxTensor Unsqueeze(OnnxTensor data, int[] axes)
    {
        int outRank = data.Rank + axes.Length;
        bool[] insert = new bool[outRank];
        foreach (int axis in axes)
        {
            int normalized = axis < 0 ? axis + outRank : axis;
            if ((uint)normalized >= (uint)outRank || insert[normalized])
            {
                throw new InvalidDataException($"Cannot unsqueeze axis {axis}.");
            }

            insert[normalized] = true;
        }

        int[] shape = new int[outRank];
        int source = 0;
        for (int i = 0; i < outRank; i++)
        {
            shape[i] = insert[i] ? 1 : data.Shape[source++];
        }

        return Copy(data, shape);
    }

    internal static OnnxTensor Transpose(OnnxTensor data, int[] perm)
    {
        if (perm.Length != data.Rank)
        {
            throw new InvalidDataException(
                $"Transpose perm [{string.Join(',', perm)}] does not match rank {data.Rank} shape [{string.Join('x', data.Shape)}].");
        }

        if (data.DType != OnnxDType.Float)
        {
            throw new NotSupportedException($"Transpose of {data.DType} is not supported.");
        }

        int[] shape = new int[perm.Length];
        for (int i = 0; i < perm.Length; i++)
        {
            shape[i] = data.Shape[perm[i]];
        }

        int n = data.Length;
        int[] inStrides = Strides(data.Shape);
        int[] outStrides = Strides(shape);
        float[] source = data.Floats!;
        float[] dest = GC.AllocateUninitializedArray<float>(n);
        if (TryTransposeLastSingleton(data.Shape, perm, source, dest))
        {
            return OnnxTensor.Float(shape, dest);
        }

        int rank = shape.Length;
        Span<int> coord = rank <= 8 ? stackalloc int[rank] : new int[rank];
        for (int i = 0; i < n; i++)
        {
            int inIndex = 0;
            for (int axis = 0; axis < rank; axis++)
            {
                inIndex += coord[axis] * inStrides[perm[axis]];
            }

            dest[Offset(coord, outStrides)] = source[inIndex];
            Increment(coord, shape);
        }

        return OnnxTensor.Float(shape, dest);
    }

    internal static OnnxTensor MatMul(OnnxTensor a, OnnxTensor b)
    {
        if (a.DType != OnnxDType.Float || b.DType != OnnxDType.Float)
        {
            throw new NotSupportedException("MatMul requires float tensors.");
        }

        if (a.Rank < 2 || b.Rank != 2 || a.Shape[^1] != b.Shape[0])
        {
            throw new NotSupportedException(
                $"MatMul shape [{string.Join('x', a.Shape)}] x [{string.Join('x', b.Shape)}] is not supported.");
        }

        int k = b.Shape[0];
        int n = b.Shape[1];
        int rows = a.Length / k;
        int[] shape = a.Shape.ToArray();
        shape[^1] = n;
        float[] dest = GC.AllocateUninitializedArray<float>(rows * n);
        SimdGemm.Multiply(a.Floats!, b.Floats!, dest, rows, k, n);
        return OnnxTensor.Float(shape, dest);
    }

    internal static OnnxTensor Embedding(OnnxTensor tokens, OnnxTensor weight)
    {
        if (weight.DType != OnnxDType.Float || weight.Rank != 2)
        {
            throw new InvalidDataException("Embedding weight must be a float matrix.");
        }

        int vocab = weight.Shape[0];
        int dim = weight.Shape[1];
        int[] seq = tokens.Rank >= 2 && tokens.Shape[^1] == 1
            ? tokens.Shape[..^1]
            : tokens.Shape.ToArray();
        int positions = OnnxTensor.ElementCount(seq);
        if (tokens.Length != positions)
        {
            throw new InvalidDataException("Embedding token tensor does not match the sequence shape.");
        }

        int[] shape = new int[seq.Length + 1];
        seq.CopyTo(shape, 0);
        shape[^1] = dim;
        float[] dest = GC.AllocateUninitializedArray<float>(positions * dim);
        float[] table = weight.Floats!;
        for (int i = 0; i < positions; i++)
        {
            int id = tokens.GetToken(i);
            if ((uint)id >= (uint)vocab)
            {
                throw new InvalidDataException($"Embedding index {id} is outside 0..{vocab - 1}.");
            }

            table.AsSpan(id * dim, dim).CopyTo(dest.AsSpan(i * dim, dim));
        }

        return OnnxTensor.Float(shape, dest);
    }

    internal static OnnxTensor Conv(OnnxTensor input, OnnxTensor weight, int[] strides, int[] pads, int[] dilations, int group)
    {
        if (input.DType != OnnxDType.Float || weight.DType != OnnxDType.Float)
        {
            throw new NotSupportedException("Conv requires float tensors.");
        }

        if (input.Rank != 4 || weight.Rank != 4 || group != 1)
        {
            throw new NotSupportedException(
                $"Conv input [{string.Join('x', input.Shape)}] weight [{string.Join('x', weight.Shape)}] group {group} is not supported.");
        }

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inH = input.Shape[2];
        int inW = input.Shape[3];
        int filters = weight.Shape[0];
        int kC = weight.Shape[1];
        int kH = weight.Shape[2];
        int kW = weight.Shape[3];
        if (kC != channels)
        {
            throw new InvalidDataException("Conv weight channel count does not match the input.");
        }

        int strideH = strides.Length > 0 ? strides[0] : 1;
        int strideW = strides.Length > 1 ? strides[1] : strideH;
        int dilH = dilations.Length > 0 ? dilations[0] : 1;
        int dilW = dilations.Length > 1 ? dilations[1] : dilH;
        int padTop = pads.Length > 0 ? pads[0] : 0;
        int padLeft = pads.Length > 1 ? pads[1] : 0;
        int padBottom = pads.Length > 2 ? pads[2] : padTop;
        int padRight = pads.Length > 3 ? pads[3] : padLeft;
        int outH = (inH + padTop + padBottom - dilH * (kH - 1) - 1) / strideH + 1;
        int outW = (inW + padLeft + padRight - dilW * (kW - 1) - 1) / strideW + 1;
        if (outH <= 0 || outW <= 0)
        {
            throw new InvalidDataException("Conv output spatial size is empty.");
        }

        int spatial = outH * outW;
        int depth = channels * kH * kW;
        float[] dest = GC.AllocateUninitializedArray<float>(batch * filters * spatial);
        float[] source = input.Floats!;
        int colCount = depth * spatial;
        float[] column = ArrayPool<float>.Shared.Rent(colCount);
        try
        {
            for (int n = 0; n < batch; n++)
            {
                Im2Col(source, column, n, channels, inH, inW, kH, kW, outH, outW, strideH, strideW, dilH, dilW, padTop, padLeft);
                SimdGemm.Multiply(weight.Floats!, column, dest.AsSpan(n * filters * spatial, filters * spatial), filters, depth, spatial);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(column);
        }

        return OnnxTensor.Float([batch, filters, outH, outW], dest);
    }

    private static void Im2Col(
        float[] input,
        float[] column,
        int batch,
        int channels,
        int inH,
        int inW,
        int kH,
        int kW,
        int outH,
        int outW,
        int strideH,
        int strideW,
        int dilH,
        int dilW,
        int padTop,
        int padLeft)
    {
        int spatial = outH * outW;
        int row = 0;
        int batchBase = batch * channels * inH * inW;
        bool fast = inW == 1 && outW == 1 && kW == 1 && padLeft == 0 && strideW == 1 && dilW == 1
            && padTop == 0 && strideH == 1 && dilH == 1;
        if (fast)
        {
            for (int ic = 0; ic < channels; ic++)
            {
                int channelBase = batchBase + ic * inH;
                for (int kh = 0; kh < kH; kh++)
                {
                    int dest = row * spatial;
                    int src = channelBase + kh;
                    input.AsSpan(src, outH).CopyTo(column.AsSpan(dest, outH));

                    row++;
                }
            }

            return;
        }

        for (int ic = 0; ic < channels; ic++)
        {
            for (int kh = 0; kh < kH; kh++)
            {
                for (int kw = 0; kw < kW; kw++)
                {
                    int dest = row * spatial;
                    int index = 0;
                    for (int oh = 0; oh < outH; oh++)
                    {
                        int ih = oh * strideH + kh * dilH - padTop;
                        for (int ow = 0; ow < outW; ow++)
                        {
                            int iw = ow * strideW + kw * dilW - padLeft;
                            float value = 0;
                            if ((uint)ih < (uint)inH && (uint)iw < (uint)inW)
                            {
                                value = input[batchBase + ((ic * inH + ih) * inW) + iw];
                            }

                            column[dest + index] = value;
                            index++;
                        }
                    }

                    row++;
                }
            }
        }
    }

    private static OnnxTensor ConcatFloat(ReadOnlySpan<OnnxTensor> inputs, int[] shape, int axis, int outer, int inner)
    {
        float[] dest = new float[OnnxTensor.ElementCount(shape)];
        CopyConcat(inputs, dest, axis, outer, inner, static (t, i) => t.Floats![i], static (d, i, v) => d[i] = v);
        return OnnxTensor.Float(shape, dest);
    }

    private static OnnxTensor ConcatInt32(ReadOnlySpan<OnnxTensor> inputs, int[] shape, int axis, int outer, int inner)
    {
        int[] dest = new int[OnnxTensor.ElementCount(shape)];
        CopyConcat(inputs, dest, axis, outer, inner, static (t, i) => t.Ints![i], static (d, i, v) => d[i] = v);
        return OnnxTensor.Int32(shape, dest);
    }

    private static OnnxTensor ConcatInt64(ReadOnlySpan<OnnxTensor> inputs, int[] shape, int axis, int outer, int inner)
    {
        long[] dest = new long[OnnxTensor.ElementCount(shape)];
        CopyConcat(inputs, dest, axis, outer, inner, static (t, i) => t.Longs![i], static (d, i, v) => d[i] = v);
        return OnnxTensor.Int64(shape, dest);
    }

    private static void CopyConcat<T>(
        ReadOnlySpan<OnnxTensor> inputs,
        T[] dest,
        int axis,
        int outer,
        int inner,
        Func<OnnxTensor, int, T> read,
        Action<T[], int, T> write)
    {
        for (int o = 0; o < outer; o++)
        {
            int cursor = 0;
            for (int t = 0; t < inputs.Length; t++)
            {
                int block = inputs[t].Shape[axis] * inner;
                int src = o * block;
                int dst = (o * dest.Length / outer) + cursor;
                for (int i = 0; i < block; i++)
                {
                    write(dest, dst + i, read(inputs[t], src + i));
                }

                cursor += block;
            }
        }

        _ = axis;
    }

    private delegate OnnxTensor TensorFactory<T>(int[] shape, T[] data);

    private static OnnxTensor Gather<T>(
        T[] source,
        int n,
        int[] outShape,
        int[] outStrides,
        int[] inStrides,
        int[] start,
        int[] step,
        TensorFactory<T> factory)
    {
        T[] dest = new T[n];
        int rank = outShape.Length;
        Span<int> coord = rank <= 8 ? stackalloc int[rank] : new int[rank];
        for (int i = 0; i < n; i++)
        {
            int inIndex = 0;
            for (int axis = 0; axis < rank; axis++)
            {
                inIndex += (start[axis] + coord[axis] * step[axis]) * inStrides[axis];
            }

            dest[Offset(coord, outStrides)] = source[inIndex];
            Increment(coord, outShape);
        }

        return factory(outShape, dest);
    }

    private static OnnxTensor Copy(OnnxTensor data, int[] shape)
    {
        return data.DType switch
        {
            OnnxDType.Float => OnnxTensor.Float(shape, (float[])data.Floats!.Clone()),
            OnnxDType.Int32 => OnnxTensor.Int32(shape, (int[])data.Ints!.Clone()),
            OnnxDType.Int64 => OnnxTensor.Int64(shape, (long[])data.Longs!.Clone()),
            OnnxDType.Bool => OnnxTensor.Bool(shape, (byte[])data.Bools!.Clone()),
            _ => throw new NotSupportedException($"Cannot copy {data.DType}."),
        };
    }

    private static T[] BroadcastCopy<T>(T[] source, int[] inputShape, int[] outShape)
    {
        int n = OnnxTensor.ElementCount(outShape);
        T[] dest = new T[n];
        int[] strides = BroadcastStrides(inputShape, outShape);
        Span<int> coord = outShape.Length <= 8 ? stackalloc int[outShape.Length] : new int[outShape.Length];
        for (int i = 0; i < n; i++)
        {
            dest[i] = source[Offset(coord, strides)];
            Increment(coord, outShape);
        }

        return dest;
    }

    internal static int[] BroadcastShape(int[] a, int[] b)
    {
        int rank = Math.Max(a.Length, b.Length);
        int[] shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int da = DimFromRight(a, rank, i);
            int db = DimFromRight(b, rank, i);
            if (da == db || db == 1)
            {
                shape[i] = da;
            }
            else if (da == 1)
            {
                shape[i] = db;
            }
            else
            {
                throw new InvalidDataException($"Cannot broadcast {da} and {db}.");
            }
        }

        return shape;
    }

    internal static int[] Strides(int[] shape)
    {
        int[] strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    private static int[] BroadcastStrides(int[] inputShape, int[] outShape)
    {
        int[] result = new int[outShape.Length];
        int[] inputStrides = Strides(LeftPad(inputShape, outShape.Length));
        int[] padded = LeftPad(inputShape, outShape.Length);
        for (int i = 0; i < outShape.Length; i++)
        {
            result[i] = padded[i] == outShape[i] ? inputStrides[i] : 0;
        }

        return result;
    }

    private static int[] LeftPad(int[] shape, int rank)
    {
        if (shape.Length == rank)
        {
            return shape;
        }

        int[] padded = new int[rank];
        Array.Fill(padded, 1);
        shape.CopyTo(padded, rank - shape.Length);
        return padded;
    }

    private static int DimFromRight(int[] shape, int rank, int index)
    {
        int local = index - (rank - shape.Length);
        return local < 0 ? 1 : shape[local];
    }

    private static int Offset(ReadOnlySpan<int> coord, int[] strides)
    {
        int index = 0;
        for (int i = 0; i < strides.Length; i++)
        {
            index += coord[i] * strides[i];
        }

        return index;
    }

    private static void Increment(Span<int> coord, int[] shape)
    {
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            coord[i]++;
            if (coord[i] < shape[i])
            {
                return;
            }

            coord[i] = 0;
        }
    }

    private static bool TryContiguousBinary(OnnxTensor a, OnnxTensor b, float[] dest, FloatOp op)
    {
        float[] left = a.Floats!;
        float[] right = b.Floats!;
        bool same = a.Shape.AsSpan().SequenceEqual(b.Shape);
        if (same && left.Length == dest.Length && right.Length == dest.Length)
        {
            MapContiguous(left, right, dest, op);
            return true;
        }

        if (left.Length == 1 && right.Length == dest.Length)
        {
            MapScalar(right, left[0], dest, op, scalarOnRight: false);
            return true;
        }

        if (right.Length == 1 && left.Length == dest.Length)
        {
            MapScalar(left, right[0], dest, op, scalarOnRight: true);
            return true;
        }

        return false;
    }

    private static void MapContiguous(float[] left, float[] right, float[] dest, FloatOp op)
    {
        int i = 0;
        if (Avx512F.IsSupported)
        {
            for (; i + 16 <= dest.Length; i += 16)
            {
                Vector512<float> value = Apply(
                    Vector512.LoadUnsafe(ref left[0], (nuint)i),
                    Vector512.LoadUnsafe(ref right[0], (nuint)i),
                    op);
                value.StoreUnsafe(ref dest[0], (nuint)i);
            }
        }

        if (Avx2.IsSupported)
        {
            for (; i + 8 <= dest.Length; i += 8)
            {
                Vector256<float> value = Apply(
                    Vector256.LoadUnsafe(ref left[0], (nuint)i),
                    Vector256.LoadUnsafe(ref right[0], (nuint)i),
                    op);
                value.StoreUnsafe(ref dest[0], (nuint)i);
            }
        }

        for (; i < dest.Length; i++)
        {
            dest[i] = Apply(left[i], right[i], op);
        }
    }

    private static void MapScalar(float[] vector, float scalar, float[] dest, FloatOp op, bool scalarOnRight)
    {
        int i = 0;
        if (Avx512F.IsSupported)
        {
            Vector512<float> broadcast = Vector512.Create(scalar);
            for (; i + 16 <= dest.Length; i += 16)
            {
                Vector512<float> value = Vector512.LoadUnsafe(ref vector[0], (nuint)i);
                Vector512<float> result = scalarOnRight ? Apply(value, broadcast, op) : Apply(broadcast, value, op);
                result.StoreUnsafe(ref dest[0], (nuint)i);
            }
        }

        if (Avx2.IsSupported)
        {
            Vector256<float> broadcast = Vector256.Create(scalar);
            for (; i + 8 <= dest.Length; i += 8)
            {
                Vector256<float> value = Vector256.LoadUnsafe(ref vector[0], (nuint)i);
                Vector256<float> result = scalarOnRight ? Apply(value, broadcast, op) : Apply(broadcast, value, op);
                result.StoreUnsafe(ref dest[0], (nuint)i);
            }
        }

        for (; i < dest.Length; i++)
        {
            dest[i] = scalarOnRight ? Apply(vector[i], scalar, op) : Apply(scalar, vector[i], op);
        }
    }

    private static Vector512<float> Apply(Vector512<float> left, Vector512<float> right, FloatOp op) => op switch
    {
        FloatOp.Add => Avx512F.Add(left, right),
        FloatOp.Sub => Avx512F.Subtract(left, right),
        FloatOp.Mul => Avx512F.Multiply(left, right),
        FloatOp.Div => Avx512F.Divide(left, right),
        FloatOp.Max => Avx512F.Max(left, right),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private static Vector256<float> Apply(Vector256<float> left, Vector256<float> right, FloatOp op) => op switch
    {
        FloatOp.Add => Avx.Add(left, right),
        FloatOp.Sub => Avx.Subtract(left, right),
        FloatOp.Mul => Avx.Multiply(left, right),
        FloatOp.Div => Avx.Divide(left, right),
        FloatOp.Max => Avx.Max(left, right),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    /// <summary>
    /// Magika's pre-conv transpose is perm [0,2,1,3] on a tensor whose last dimension is 1,
    /// which is a matrix transpose of the two middle axes.
    /// </summary>
    private static bool TryTransposeLastSingleton(int[] shape, int[] perm, float[] source, float[] dest)
    {
        if (shape.Length != 4 || perm.Length != 4 || shape[3] != 1)
        {
            return false;
        }

        if (perm[0] != 0 || perm[1] != 2 || perm[2] != 1 || perm[3] != 3)
        {
            return false;
        }

        int batch = shape[0];
        int rows = shape[1];
        int cols = shape[2];
        const int Tile = 16;
        for (int b = 0; b < batch; b++)
        {
            int srcBatch = b * rows * cols;
            int dstBatch = srcBatch;
            for (int i0 = 0; i0 < rows; i0 += Tile)
            {
                int iLimit = Math.Min(i0 + Tile, rows);
                for (int j0 = 0; j0 < cols; j0 += Tile)
                {
                    int jLimit = Math.Min(j0 + Tile, cols);
                    for (int i = i0; i < iLimit; i++)
                    {
                        int src = srcBatch + i * cols + j0;
                        for (int j = j0; j < jLimit; j++)
                        {
                            dest[dstBatch + j * rows + i] = source[src + (j - j0)];
                        }
                    }
                }
            }
        }

        return true;
    }

    private static bool TryVectorBinary(
        float[] left,
        float[] right,
        float[] dest,
        int[] shape,
        int[] strideA,
        int[] strideB,
        FloatOp op)
    {
        int last = shape[^1];
        int lastA = strideA[^1];
        int lastB = strideB[^1];
        if (last < Vector<float>.Count || (lastA != 0 && lastA != 1) || (lastB != 0 && lastB != 1) || (lastA == 0 && lastB == 0))
        {
            return false;
        }

        int rows = dest.Length / last;
        int lead = shape.Length - 1;
        Span<int> coord = lead <= 8 ? stackalloc int[lead] : new int[lead];
        int[] leadShape = shape[..^1];
        int[] leadA = strideA[..^1];
        int[] leadB = strideB[..^1];
        for (int row = 0; row < rows; row++)
        {
            int baseA = Offset(coord, leadA);
            int baseB = Offset(coord, leadB);
            Span<float> target = dest.AsSpan(row * last, last);
            if (lastA == 1 && lastB == 1)
            {
                VectorSpan(left.AsSpan(baseA, last), right.AsSpan(baseB, last), target, op);
            }
            else if (lastB == 0)
            {
                VectorScalar(left.AsSpan(baseA, last), right[baseB], target, op, scalarOnRight: true);
            }
            else
            {
                VectorScalar(right.AsSpan(baseB, last), left[baseA], target, op, scalarOnRight: false);
            }

            if (lead > 0)
            {
                Increment(coord, leadShape);
            }
        }

        return true;
    }

    private static void VectorSpan(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> dest, FloatOp op)
    {
        int width = Vector<float>.Count;
        int i = 0;
        for (; i + width <= dest.Length; i += width)
        {
            var a = new Vector<float>(left.Slice(i, width));
            var b = new Vector<float>(right.Slice(i, width));
            Apply(a, b, op).CopyTo(dest.Slice(i, width));
        }

        for (; i < dest.Length; i++)
        {
            dest[i] = Apply(left[i], right[i], op);
        }
    }

    private static void VectorScalar(ReadOnlySpan<float> vector, float scalar, Span<float> dest, FloatOp op, bool scalarOnRight)
    {
        int width = Vector<float>.Count;
        var broadcast = new Vector<float>(scalar);
        int i = 0;
        for (; i + width <= dest.Length; i += width)
        {
            var value = new Vector<float>(vector.Slice(i, width));
            Vector<float> result = scalarOnRight ? Apply(value, broadcast, op) : Apply(broadcast, value, op);
            result.CopyTo(dest.Slice(i, width));
        }

        for (; i < dest.Length; i++)
        {
            dest[i] = scalarOnRight ? Apply(vector[i], scalar, op) : Apply(scalar, vector[i], op);
        }
    }

    private static float Apply(float left, float right, FloatOp op) => op switch
    {
        FloatOp.Add => left + right,
        FloatOp.Sub => left - right,
        FloatOp.Mul => left * right,
        FloatOp.Div => left / right,
        FloatOp.Max => MathF.Max(left, right),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private static Vector<float> Apply(Vector<float> left, Vector<float> right, FloatOp op) => op switch
    {
        FloatOp.Add => left + right,
        FloatOp.Sub => left - right,
        FloatOp.Mul => left * right,
        FloatOp.Div => left / right,
        FloatOp.Max => Vector.Max(left, right),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private static bool TryFastReduce(float[] source, int[] shape, bool[] reduced, bool keepDims, bool max, out float[] dest)
    {
        dest = [];
        int axis = -1;
        for (int i = 0; i < reduced.Length; i++)
        {
            if (!reduced[i])
            {
                continue;
            }

            if (axis >= 0)
            {
                return false;
            }

            axis = i;
        }

        if (axis < 0)
        {
            return false;
        }

        int outer = 1;
        for (int i = 0; i < axis; i++)
        {
            outer *= shape[i];
        }

        int reduce = shape[axis];
        int inner = 1;
        for (int i = axis + 1; i < shape.Length; i++)
        {
            inner *= shape[i];
        }

        dest = new float[Math.Max(1, outer * inner)];
        if (inner == 1)
        {
            for (int o = 0; o < outer; o++)
            {
                int start = o * reduce;
                float acc = max ? float.NegativeInfinity : 0f;
                for (int r = 0; r < reduce; r++)
                {
                    float value = source[start + r];
                    acc = max ? MathF.Max(acc, value) : acc + value;
                }

                dest[o] = acc;
            }
        }
        else
        {
            for (int o = 0; o < outer; o++)
            {
                for (int i = 0; i < inner; i++)
                {
                    float acc = max ? float.NegativeInfinity : 0f;
                    int start = (o * reduce * inner) + i;
                    for (int r = 0; r < reduce; r++)
                    {
                        float value = source[start + (r * inner)];
                        acc = max ? MathF.Max(acc, value) : acc + value;
                    }

                    dest[(o * inner) + i] = acc;
                }
            }
        }

        _ = keepDims;
        return true;
    }

    private static bool ValuesEqual(OnnxTensor a, int ia, OnnxTensor b, int ib)
    {
        if (a.DType != b.DType)
        {
            throw new InvalidDataException($"Equal types differ: {a.DType} vs {b.DType}.");
        }

        return a.DType switch
        {
            OnnxDType.Int32 => a.Ints![ia] == b.Ints![ib],
            OnnxDType.Int64 => a.Longs![ia] == b.Longs![ib],
            OnnxDType.Float => a.Floats![ia] == b.Floats![ib],
            OnnxDType.Bool => a.Bools![ia] == b.Bools![ib],
            _ => false,
        };
    }
}

internal enum FloatOp
{
    Add,
    Sub,
    Mul,
    Div,
    Max,
}

internal delegate float FloatUnary(float value);
