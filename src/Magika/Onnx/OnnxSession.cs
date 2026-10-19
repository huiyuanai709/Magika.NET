namespace Magika.Onnx;

internal sealed class OnnxSession
{
    private readonly OnnxModel _model;

    private OnnxSession(OnnxModel model) => _model = model;

    public bool EmbeddingFused => _model.EmbeddingFused;

    public int OutputWidth { get; private set; }

    public static OnnxSession Load(ReadOnlySpan<byte> model) => new(OnnxModel.Load(model));

    public float[] Run(ReadOnlySpan<int> tokens, int batch, int sequence)
    {
        if (batch <= 0 || sequence <= 0 || tokens.Length != batch * sequence)
        {
            throw new ArgumentException("Token buffer does not match batch and sequence.", nameof(tokens));
        }

        Dictionary<string, OnnxTensor> values = new(_model.Initializers, StringComparer.Ordinal);
        int[] input = new int[tokens.Length];
        tokens.CopyTo(input);
        values[_model.InputName] = OnnxTensor.Int32([batch, sequence], input);

        foreach (OnnxNode node in _model.Nodes)
        {
            try
            {
                OnnxTensor output = Execute(node, values);
                if (node.Outputs.Length == 0)
                {
                    throw new InvalidDataException($"ONNX node '{node.Name}' ({node.Op}) has no output.");
                }

                values[node.Outputs[0]] = output;
            }
            catch (Exception ex) when (ex is not InvalidDataException || ex.Message.StartsWith("ONNX node", StringComparison.Ordinal) == false)
            {
                throw new InvalidDataException($"ONNX node '{node.Name}' ({node.Op}) failed: {ex.Message}", ex);
            }
        }

        if (!values.TryGetValue(_model.OutputName, out OnnxTensor? result) || result.Floats is null)
        {
            throw new InvalidDataException($"ONNX output '{_model.OutputName}' is missing.");
        }

        OutputWidth = result.Shape[^1];
        if (result.Length != batch * OutputWidth)
        {
            throw new InvalidDataException(
                $"ONNX output length {result.Length} does not match batch {batch} by {OutputWidth}.");
        }

        return result.Floats;
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
                return OnnxOps.UnaryFloat(Input(0), MathF.Tanh);
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
