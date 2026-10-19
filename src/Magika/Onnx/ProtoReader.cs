using System.Buffers.Binary;

namespace Magika.Onnx;

internal ref struct ProtoReader
{
    private readonly ReadOnlySpan<byte> _data;
    private int _pos;
    private readonly int _end;

    public ProtoReader(ReadOnlySpan<byte> data)
    {
        _data = data;
        _pos = 0;
        _end = data.Length;
    }

    public bool Eof => _pos >= _end;

    public bool TryReadTag(out int field, out int wire)
    {
        if (Eof)
        {
            field = 0;
            wire = 0;
            return false;
        }

        ulong key = ReadVarint();
        field = (int)(key >> 3);
        wire = (int)(key & 7);
        return true;
    }

    public ulong ReadVarint()
    {
        ulong result = 0;
        int shift = 0;
        while (true)
        {
            if (_pos >= _end)
            {
                throw new InvalidDataException("Truncated ONNX protobuf varint.");
            }

            byte value = _data[_pos++];
            result |= (ulong)(value & 0x7F) << shift;
            if ((value & 0x80) == 0)
            {
                return result;
            }

            shift += 7;
            if (shift > 63)
            {
                throw new InvalidDataException("ONNX protobuf varint is too long.");
            }
        }
    }

    public uint ReadFixed32()
    {
        if (_pos + 4 > _end)
        {
            throw new InvalidDataException("Truncated ONNX protobuf fixed32.");
        }

        uint value = BinaryPrimitives.ReadUInt32LittleEndian(_data.Slice(_pos, 4));
        _pos += 4;
        return value;
    }

    public ulong ReadFixed64()
    {
        if (_pos + 8 > _end)
        {
            throw new InvalidDataException("Truncated ONNX protobuf fixed64.");
        }

        ulong value = BinaryPrimitives.ReadUInt64LittleEndian(_data.Slice(_pos, 8));
        _pos += 8;
        return value;
    }

    public ReadOnlySpan<byte> ReadBytes()
    {
        int length = checked((int)ReadVarint());
        if ((uint)(_pos + length) > (uint)_end)
        {
            throw new InvalidDataException("Truncated ONNX protobuf length-delimited field.");
        }

        ReadOnlySpan<byte> slice = _data.Slice(_pos, length);
        _pos += length;
        return slice;
    }

    public void Skip(int wire)
    {
        switch (wire)
        {
            case 0:
                ReadVarint();
                break;
            case 1:
                _ = ReadFixed64();
                break;
            case 2:
                _ = ReadBytes();
                break;
            case 5:
                _ = ReadFixed32();
                break;
            default:
                throw new InvalidDataException($"Unsupported protobuf wire type {wire}.");
        }
    }
}
