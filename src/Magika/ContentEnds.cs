using System.Buffers;
using Microsoft.Win32.SafeHandles;

namespace Magika;

/// <summary>
/// Reads the prefix and suffix blocks Magika features need into <see cref="ArrayPool{T}"/> buffers.
/// </summary>
internal static class ContentEnds
{
    internal static Window ReadHandle(SafeFileHandle handle, long length, int blockSize) =>
        ReadRandom(length, blockSize, new HandleReader(handle));

    internal static Window ReadSeekable(Stream stream, long length, int blockSize) =>
        ReadRandom(length, blockSize, new StreamReader(stream));

    internal static Window ReadForward(Stream stream, int blockSize, out long length)
    {
        int window = checked(blockSize * 2);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(window);
        int filled;
        try
        {
            filled = ReadUpTo(stream, buffer.AsSpan(0, window));
        }
        catch
        {
            ArrayPool<byte>.Shared.Return(buffer);
            throw;
        }

        if (filled < window)
        {
            length = filled;
            if (filled == 0)
            {
                ArrayPool<byte>.Shared.Return(buffer);
                return default;
            }

            return Window.Single(buffer, filled, Math.Min(blockSize, filled));
        }

        byte[] chunk = ArrayPool<byte>.Shared.Rent(blockSize);
        int extra;
        try
        {
            extra = stream.Read(chunk.AsSpan(0, blockSize));
        }
        catch
        {
            ArrayPool<byte>.Shared.Return(buffer);
            ArrayPool<byte>.Shared.Return(chunk);
            throw;
        }

        if (extra == 0)
        {
            ArrayPool<byte>.Shared.Return(chunk);
            length = window;
            return Window.Single(buffer, window, blockSize);
        }

        byte[] prefix = ArrayPool<byte>.Shared.Rent(blockSize);
        byte[] suffix = ArrayPool<byte>.Shared.Rent(blockSize);
        buffer.AsSpan(0, blockSize).CopyTo(prefix);
        buffer.AsSpan(blockSize, blockSize).CopyTo(suffix);
        ArrayPool<byte>.Shared.Return(buffer);
        try
        {
            AppendTail(suffix, blockSize, chunk.AsSpan(0, extra));
            length = (long)window + extra;
            while (true)
            {
                int n = stream.Read(chunk.AsSpan(0, blockSize));
                if (n == 0)
                {
                    break;
                }

                AppendTail(suffix, blockSize, chunk.AsSpan(0, n));
                length = checked(length + n);
            }
        }
        catch
        {
            ArrayPool<byte>.Shared.Return(prefix);
            ArrayPool<byte>.Shared.Return(suffix);
            ArrayPool<byte>.Shared.Return(chunk);
            throw;
        }

        ArrayPool<byte>.Shared.Return(chunk);
        return Window.Split(prefix, suffix, blockSize);
    }

    private static Window ReadRandom<TReader>(long length, int blockSize, TReader reader)
        where TReader : IRandomReader
    {
        int block = (int)Math.Min(blockSize, length);
        if (length <= (long)blockSize * 2)
        {
            int count = (int)length;
            byte[] buffer = ArrayPool<byte>.Shared.Rent(count);
            try
            {
                reader.Read(buffer.AsSpan(0, count), offset: 0);
            }
            catch
            {
                ArrayPool<byte>.Shared.Return(buffer);
                throw;
            }

            return Window.Single(buffer, count, block);
        }

        byte[] prefix = ArrayPool<byte>.Shared.Rent(block);
        byte[] suffix = ArrayPool<byte>.Shared.Rent(block);
        try
        {
            reader.Read(prefix.AsSpan(0, block), offset: 0);
            reader.Read(suffix.AsSpan(0, block), length - block);
        }
        catch
        {
            ArrayPool<byte>.Shared.Return(prefix);
            ArrayPool<byte>.Shared.Return(suffix);
            throw;
        }

        return Window.Split(prefix, suffix, block);
    }

    private static int ReadUpTo(Stream stream, Span<byte> buffer)
    {
        int read = 0;
        while (read < buffer.Length)
        {
            int n = stream.Read(buffer[read..]);
            if (n == 0)
            {
                return read;
            }

            if ((uint)n > (uint)(buffer.Length - read))
            {
                throw new IOException("Stream read returned an invalid count.");
            }

            read += n;
        }

        return read;
    }

    private static void ReadExact(Stream stream, Span<byte> buffer)
    {
        if (ReadUpTo(stream, buffer) != buffer.Length)
        {
            throw new EndOfStreamException("Stream ended before the requested bytes were read.");
        }
    }

    private static void ReadExactAt(SafeFileHandle handle, Span<byte> buffer, long offset)
    {
        int read = 0;
        while (read < buffer.Length)
        {
            int n = RandomAccess.Read(handle, buffer[read..], offset + read);
            if (n == 0)
            {
                throw new EndOfStreamException("Stream ended before the requested bytes were read.");
            }

            if ((uint)n > (uint)(buffer.Length - read))
            {
                throw new IOException("File read returned an invalid count.");
            }

            read += n;
        }
    }

    private static void AppendTail(byte[] suffix, int block, ReadOnlySpan<byte> incoming)
    {
        if (incoming.Length >= block)
        {
            incoming[^block..].CopyTo(suffix);
            return;
        }

        int keep = block - incoming.Length;
        suffix.AsSpan(incoming.Length, keep).CopyTo(suffix);
        incoming.CopyTo(suffix.AsSpan(keep));
    }

    private interface IRandomReader
    {
        void Read(Span<byte> destination, long offset);
    }

    private readonly struct HandleReader(SafeFileHandle handle) : IRandomReader
    {
        public void Read(Span<byte> destination, long offset) => ReadExactAt(handle, destination, offset);
    }

    private readonly struct StreamReader(Stream stream) : IRandomReader
    {
        public void Read(Span<byte> destination, long offset)
        {
            stream.Position = offset;
            ReadExact(stream, destination);
        }
    }

    /// <summary>Pooled prefix and suffix. Dispose returns the arrays to <see cref="ArrayPool{T}.Shared"/>.</summary>
    internal ref struct Window
    {
        private byte[]? _head;
        private byte[]? _tail;
        private int _headCount;
        private int _tailCount;
        private int _tailOffset;

        public ReadOnlySpan<byte> Prefix => _head is null ? [] : _head.AsSpan(0, _headCount);

        public ReadOnlySpan<byte> Suffix =>
            _tail is not null ? _tail.AsSpan(0, _tailCount) :
            _head is null ? [] :
            _head.AsSpan(_tailOffset, _tailCount);

        public void Dispose()
        {
            byte[]? head = _head;
            byte[]? tail = _tail;
            _head = null;
            _tail = null;
            if (head is not null)
            {
                ArrayPool<byte>.Shared.Return(head);
            }

            if (tail is not null)
            {
                ArrayPool<byte>.Shared.Return(tail);
            }
        }

        public static Window Single(byte[] buffer, int count, int block) =>
            new()
            {
                _head = buffer,
                _headCount = block,
                _tailCount = block,
                _tailOffset = count - block,
            };

        public static Window Split(byte[] prefix, byte[] suffix, int block) =>
            new()
            {
                _head = prefix,
                _tail = suffix,
                _headCount = block,
                _tailCount = block,
            };
    }
}
