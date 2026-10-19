namespace Magika.Onnx;

/// <summary>
/// One managed ONNX evaluation state. <see cref="Run"/> is not thread-safe; the public session serializes it.
/// Weights are shared with sessions created by <see cref="Fork"/>. Activation scratch is not.
/// </summary>
internal sealed class OnnxSession
{
    private readonly OnnxModel _model;
    private readonly Dictionary<string, OnnxTensor> _values;
    private readonly string[] _restoreInitializers;
    private readonly OnnxScratch _scratch = new();
    private readonly Dictionary<int, int[]> _tokenBuffers = new();
    private readonly Dictionary<int, OnnxTensor> _inputs = new();

    private OnnxSession(OnnxModel model)
    {
        _model = model;
        _values = new Dictionary<string, OnnxTensor>(model.Initializers, StringComparer.Ordinal);
        List<string>? overwritten = null;
        foreach (OnnxNode node in model.Nodes)
        {
            foreach (string output in node.Outputs)
            {
                if (output.Length > 0 && model.Initializers.ContainsKey(output))
                {
                    overwritten ??= [];
                    overwritten.Add(output);
                }
            }
        }

        _restoreInitializers = overwritten?.ToArray() ?? [];
    }

    public bool EmbeddingFused => _model.EmbeddingFused;

    public int OutputWidth { get; private set; }

    public static OnnxSession Load(ReadOnlySpan<byte> model) => new(OnnxModel.Load(model));

    /// <summary>Another evaluation state over the same immutable weights.</summary>
    public OnnxSession Fork() => new(_model);

    public float[] Run(ReadOnlySpan<int> tokens, int batch, int sequence)
    {
        if (batch <= 0 || sequence <= 0 || tokens.Length != batch * sequence)
        {
            throw new ArgumentException("Token buffer does not match batch and sequence.", nameof(tokens));
        }

        foreach (string name in _restoreInitializers)
        {
            _values[name] = _model.Initializers[name];
        }

        BindInput(tokens, batch, sequence);
        using (_scratch.Enter())
        {
            foreach (OnnxNode node in _model.Nodes)
            {
                try
                {
                    _scratch.SetSlot(node.Outputs.Length == 0 ? node.Name : node.Outputs[0]);
                    OnnxTensor output = Execute(node, _values);
                    if (node.Outputs.Length == 0)
                    {
                        throw new InvalidDataException($"ONNX node '{node.Name}' ({node.Op}) has no output.");
                    }

                    _values[node.Outputs[0]] = output;
                }
                catch (Exception ex) when (ex is not InvalidDataException || ex.Message.StartsWith("ONNX node", StringComparison.Ordinal) == false)
                {
                    throw new InvalidDataException($"ONNX node '{node.Name}' ({node.Op}) failed: {ex.Message}", ex);
                }
            }
        }

        if (!_values.TryGetValue(_model.OutputName, out OnnxTensor? result) || result.Floats is null)
        {
            throw new InvalidDataException($"ONNX output '{_model.OutputName}' is missing.");
        }

        OutputWidth = result.Shape[^1];
        if (result.Length != batch * OutputWidth)
        {
            throw new InvalidDataException(
                $"ONNX output length {result.Length} does not match batch {batch} by {OutputWidth}.");
        }

        // The activation buffer is reused on the next run. Callers keep this copy.
        float[] owned = GC.AllocateUninitializedArray<float>(result.Length);
        result.Floats.AsSpan(0, result.Length).CopyTo(owned);
        return owned;
    }

    private void BindInput(ReadOnlySpan<int> tokens, int batch, int sequence)
    {
        int length = tokens.Length;
        if (!_tokenBuffers.TryGetValue(length, out int[]? buffer))
        {
            buffer = new int[length];
            _tokenBuffers[length] = buffer;
        }

        tokens.CopyTo(buffer);
        if (!_inputs.TryGetValue(length, out OnnxTensor? input)
            || input.Shape.Length != 2
            || input.Shape[0] != batch
            || input.Shape[1] != sequence
            || !ReferenceEquals(input.Ints, buffer))
        {
            input = OnnxTensor.Int32([batch, sequence], buffer);
            _inputs[length] = input;
        }

        _values[_model.InputName] = input;
    }

    private static OnnxTensor Execute(OnnxNode node, Dictionary<string, OnnxTensor> values)
    {
        OnnxTensor Input(int index)
        {
            if ((uint)index >= (uint)node.Inputs.Length || node.Inputs[index].Length == 0)
            {
                throw new InvalidDataException($"Node '{node.Name}' is missing input {index}.");
            }

            if (!values.TryGetValue(node.Inputs[index], out OnnxTensor? tensor))
            {
                throw new InvalidDataException($"Node '{node.Name}' references missing tensor '{node.Inputs[index]}'.");
            }

            return tensor;
        }

        bool Has(int index) => (uint)index < (uint)node.Inputs.Length && node.Inputs[index].Length > 0;

        switch (node.Op)
        {
            case "Add":
                return OnnxOps.Binary(Input(0), Input(1), FloatOp.Add);
            case "Sub":
                return OnnxOps.Binary(Input(0), Input(1), FloatOp.Sub);
            case "Mul":
                return OnnxOps.Binary(Input(0), Input(1), FloatOp.Mul);
            case "Div":
                return OnnxOps.Binary(Input(0), Input(1), FloatOp.Div);
            case "Max":
                return OnnxOps.Binary(Input(0), Input(1), FloatOp.Max);
            case "Tanh":
                return OnnxOps.Tanh(Input(0));
            case "Sqrt":
                return OnnxOps.UnaryFloat(Input(0), MathF.Sqrt);
            case "Reciprocal":
                return OnnxOps.UnaryFloat(Input(0), static x => 1f / x);
            case "Exp":
                return OnnxOps.UnaryFloat(Input(0), MathF.Exp);
            case "Relu":
                return OnnxOps.UnaryFloat(Input(0), static x => MathF.Max(x, 0f));
            case "Reshape":
                return OnnxOps.Reshape(Input(0), Input(1).AsInt64Array());
            case "Shape":
                return OnnxOps.ShapeOf(Input(0));
            case "Cast":
                return OnnxOps.Cast(Input(0), checked((int)node.Int("to")));
            case "Concat":
            {
                OnnxTensor[] parts = new OnnxTensor[node.Inputs.Length];
                for (int i = 0; i < parts.Length; i++)
                {
                    parts[i] = Input(i);
                }

                return OnnxOps.Concat(parts, checked((int)node.Int("axis")));
            }
            case "Slice":
                return OnnxOps.Slice(
                    Input(0),
                    Input(1).AsInt64Array(),
                    Input(2).AsInt64Array(),
                    Has(3) ? Input(3).AsInt64Array() : null,
                    Has(4) ? Input(4).AsInt64Array() : null);
            case "Expand":
                return OnnxOps.Expand(Input(0), IntShape(Input(1)));
            case "Equal":
                return OnnxOps.Equal(Input(0), Input(1));
            case "MatMul":
                return OnnxOps.MatMul(Input(0), Input(1));
            case "Embedding":
                return OnnxOps.Embedding(Input(0), Input(1));
            case "ReduceSum":
                return OnnxOps.Reduce(Input(0), ReduceAxes(node, values), node.Int("keepdims") != 0, max: false, node.Int("noop_with_empty_axes") != 0);
            case "ReduceMax":
                return OnnxOps.Reduce(Input(0), ReduceAxes(node, values), node.Int("keepdims") != 0, max: true, node.Int("noop_with_empty_axes") != 0);
            case "GlobalMaxPool":
                return OnnxOps.GlobalMaxPool(Input(0));
            case "Squeeze":
                return OnnxOps.Squeeze(Input(0), Has(1) ? IntShape(Input(1)) : node.Ints("axes"));
            case "Unsqueeze":
                return OnnxOps.Unsqueeze(Input(0), Has(1) ? IntShape(Input(1)) : node.Ints("axes"));
            case "Transpose":
                return OnnxOps.Transpose(Input(0), node.Ints("perm"));
            case "Conv":
                return OnnxOps.Conv(
                    Input(0),
                    Input(1),
                    node.Ints("strides"),
                    node.Ints("pads"),
                    node.Ints("dilations"),
                    (int)node.Int("group", 1));
            default:
                throw new NotSupportedException($"ONNX operator '{node.Op}' is not implemented.");
        }
    }

    private static int[] ReduceAxes(OnnxNode node, Dictionary<string, OnnxTensor> values)
    {
        if (node.Inputs.Length > 1 && node.Inputs[1].Length > 0 && values.TryGetValue(node.Inputs[1], out OnnxTensor? axes))
        {
            return IntShape(axes);
        }

        return node.Ints("axes");
    }

    private static int[] IntShape(OnnxTensor tensor)
    {
        long[] values = tensor.AsInt64Array();
        int[] shape = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            shape[i] = checked((int)values[i]);
        }

        return shape;
    }
}
