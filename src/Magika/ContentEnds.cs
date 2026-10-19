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

    internal static ValueTask<Lease> ReadHandleAsync(SafeFileHandle handle, long length, int blockSize, CancellationToken cancellationToken) =>
        ReadRandomAsync(length, blockSize, new HandleReaderAsync(handle), cancellationToken);

    internal static ValueTask<Lease> ReadSeekableAsync(Stream stream, long length, int blockSize, CancellationToken cancellationToken) =>
        ReadRandomAsync(length, blockSize, new StreamReaderAsync(stream), cancellationToken);

    internal static async ValueTask<Lease> ReadForwardAsync(Stream stream, int blockSize, CancellationToken cancellationToken)
    {
        int window = checked(blockSize * 2);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(window);
        int filled;
        try
        {
            filled = await ReadUpToAsync(stream, buffer.AsMemory(0, window), cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            ArrayPool<byte>.Shared.Return(buffer);
            throw;
        }

        if (filled < window)
        {
            if (filled == 0)
            {
                ArrayPool<byte>.Shared.Return(buffer);
                return Lease.Empty();
            }

            return Lease.Single(buffer, filled, Math.Min(blockSize, filled), filled);
        }

        byte[] chunk = ArrayPool<byte>.Shared.Rent(blockSize);
        int extra;
        try
        {
            extra = await stream.ReadAsync(chunk.AsMemory(0, blockSize), cancellationToken).ConfigureAwait(false);
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
            return Lease.Single(buffer, window, blockSize, window);
        }

        if ((uint)extra > (uint)blockSize)
        {
            ArrayPool<byte>.Shared.Return(buffer);
            ArrayPool<byte>.Shared.Return(chunk);
            throw new IOException("Stream read returned an invalid count.");
        }

        byte[] prefix = ArrayPool<byte>.Shared.Rent(blockSize);
        byte[] suffix = ArrayPool<byte>.Shared.Rent(blockSize);
        Array.Copy(buffer, 0, prefix, 0, blockSize);
        Array.Copy(buffer, blockSize, suffix, 0, blockSize);
        ArrayPool<byte>.Shared.Return(buffer);
        long length;
        try
        {
            AppendTail(suffix, blockSize, chunk, extra);
            length = (long)window + extra;
            while (true)
            {
                int n = await stream.ReadAsync(chunk.AsMemory(0, blockSize), cancellationToken).ConfigureAwait(false);
                if (n == 0)
                {
                    break;
                }

                if ((uint)n > (uint)blockSize)
                {
                    throw new IOException("Stream read returned an invalid count.");
                }

                AppendTail(suffix, blockSize, chunk, n);
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
        return Lease.Split(prefix, suffix, blockSize, length);
    }

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

    private static async ValueTask<Lease> ReadRandomAsync<TReader>(long length, int blockSize, TReader reader, CancellationToken cancellationToken)
        where TReader : IAsyncRandomReader
    {
        int block = (int)Math.Min(blockSize, length);
        if (length <= (long)blockSize * 2)
        {
            int count = (int)length;
            byte[] buffer = ArrayPool<byte>.Shared.Rent(count);
            try
            {
                await reader.ReadAsync(buffer.AsMemory(0, count), offset: 0, cancellationToken).ConfigureAwait(false);
            }
            catch
            {
                ArrayPool<byte>.Shared.Return(buffer);
                throw;
            }

            return Lease.Single(buffer, count, block, length);
        }

        byte[] prefix = ArrayPool<byte>.Shared.Rent(block);
        byte[] suffix = ArrayPool<byte>.Shared.Rent(block);
        try
        {
            await reader.ReadAsync(prefix.AsMemory(0, block), offset: 0, cancellationToken).ConfigureAwait(false);
            await reader.ReadAsync(suffix.AsMemory(0, block), length - block, cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            ArrayPool<byte>.Shared.Return(prefix);
            ArrayPool<byte>.Shared.Return(suffix);
            throw;
        }

        return Lease.Split(prefix, suffix, block, length);
    }

    private static async ValueTask<int> ReadUpToAsync(Stream stream, Memory<byte> buffer, CancellationToken cancellationToken)
    {
        int read = 0;
        while (read < buffer.Length)
        {
            int n = await stream.ReadAsync(buffer[read..], cancellationToken).ConfigureAwait(false);
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

    private static async ValueTask ReadExactAsync(Stream stream, Memory<byte> buffer, CancellationToken cancellationToken)
    {
        if (await ReadUpToAsync(stream, buffer, cancellationToken).ConfigureAwait(false) != buffer.Length)
        {
            throw new EndOfStreamException("Stream ended before the requested bytes were read.");
        }
    }

    private static async ValueTask ReadExactAtAsync(SafeFileHandle handle, Memory<byte> buffer, long offset, CancellationToken cancellationToken)
    {
        int read = 0;
        while (read < buffer.Length)
        {
            int n = await RandomAccess.ReadAsync(handle, buffer[read..], offset + read, cancellationToken).ConfigureAwait(false);
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

    private static void AppendTail(byte[] suffix, int block, byte[] incoming, int count) =>
        AppendTail(suffix, block, incoming.AsSpan(0, count));

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

    private interface IAsyncRandomReader
    {
        ValueTask ReadAsync(Memory<byte> destination, long offset, CancellationToken cancellationToken);
    }

    private readonly struct HandleReaderAsync(SafeFileHandle handle) : IAsyncRandomReader
    {
        public ValueTask ReadAsync(Memory<byte> destination, long offset, CancellationToken cancellationToken) =>
            ReadExactAtAsync(handle, destination, offset, cancellationToken);
    }

    private readonly struct StreamReaderAsync(Stream stream) : IAsyncRandomReader
    {
        public ValueTask ReadAsync(Memory<byte> destination, long offset, CancellationToken cancellationToken)
        {
            stream.Position = offset;
            return ReadExactAsync(stream, destination, cancellationToken);
        }
    }

    /// <summary>Pooled prefix and suffix owned by an async read. Dispose returns the arrays to <see cref="ArrayPool{T}.Shared"/>.</summary>
    internal sealed class Lease : IDisposable
    {
        private byte[]? _head;
        private byte[]? _tail;
        private int _headCount;
        private int _tailCount;
        private int _tailOffset;

        public long Length { get; private set; }

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

        public static Lease Empty() => new();

        public static Lease Single(byte[] buffer, int count, int block, long length) =>
            new()
            {
                _head = buffer,
                _headCount = block,
                _tailCount = block,
                _tailOffset = count - block,
                Length = length,
            };

        public static Lease Split(byte[] prefix, byte[] suffix, int block, long length) =>
            new()
            {
                _head = prefix,
                _tail = suffix,
                _headCount = block,
                _tailCount = block,
                Length = length,
            };
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
