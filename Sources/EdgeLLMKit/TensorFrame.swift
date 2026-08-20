import Foundation

/// Wire format for tensors and control messages exchanged between devices.
///
/// Layout (all integers little-endian):
///
/// ```
/// offset  size  field
/// 0       4     magic "ELC1"
/// 4       1     version (currently 1)
/// 5       1     kind (0 = hiddenState, 1 = token, 2 = control)
/// 6       1     dtype (0 = float16, 1 = float32, 2 = bfloat16)
/// 7       1     rank (number of dimensions, max 8)
/// 8       4     seqPos     sequence position of the first element
/// 12      2     layerStart producing shard's first layer
/// 14      2     layerEnd   producing shard's end layer (exclusive)
/// 16      4*r   shape dims (Int32 each)
/// 16+4r   4     payload byte count
/// ...           payload bytes
/// ```
public struct TensorFrame: Equatable, Sendable {
    public static let magic: [UInt8] = Array("ELC1".utf8)
    public static let version: UInt8 = 1
    public static let maxRank = 8

    public enum Kind: UInt8, Sendable {
        case hiddenState = 0
        case token = 1
        case control = 2
    }

    public enum DType: UInt8, Sendable {
        case float16 = 0
        case float32 = 1
        case bfloat16 = 2

        public var bytesPerElement: Int {
            switch self {
            case .float16, .bfloat16: return 2
            case .float32: return 4
            }
        }
    }

    public enum DecodeError: Error, Equatable {
        case truncated
        case badMagic
        case unsupportedVersion(UInt8)
        case invalidField(String)
        case payloadSizeMismatch(expected: Int, actual: Int)
    }

    public var kind: Kind
    public var dtype: DType
    public var seqPos: Int
    public var layerStart: Int
    public var layerEnd: Int
    public var shape: [Int]
    public var payload: Data

    public init(
        kind: Kind, dtype: DType, seqPos: Int,
        layerStart: Int, layerEnd: Int, shape: [Int], payload: Data
    ) {
        precondition(shape.count <= Self.maxRank, "rank exceeds \(Self.maxRank)")
        self.kind = kind
        self.dtype = dtype
        self.seqPos = seqPos
        self.layerStart = layerStart
        self.layerEnd = layerEnd
        self.shape = shape
        self.payload = payload
    }

    public var elementCount: Int { shape.reduce(1, *) }

    // MARK: - Encoding

    public func encoded() -> Data {
        var data = Data(capacity: 20 + 4 * shape.count + payload.count)
        data.append(contentsOf: Self.magic)
        data.append(Self.version)
        data.append(kind.rawValue)
        data.append(dtype.rawValue)
        data.append(UInt8(shape.count))
        data.appendLE(UInt32(seqPos))
        data.appendLE(UInt16(layerStart))
        data.appendLE(UInt16(layerEnd))
        for dim in shape {
            data.appendLE(Int32(dim))
        }
        data.appendLE(UInt32(payload.count))
        data.append(payload)
        return data
    }

    // MARK: - Decoding

    public static func decode(_ data: Data) throws -> TensorFrame {
        var reader = DataReader(data)

        guard let magicBytes = reader.read(count: 4) else { throw DecodeError.truncated }
        guard Array(magicBytes) == magic else { throw DecodeError.badMagic }

        guard let version = reader.readByte() else { throw DecodeError.truncated }
        guard version == Self.version else { throw DecodeError.unsupportedVersion(version) }

        guard let kindRaw = reader.readByte(), let dtypeRaw = reader.readByte(),
              let rank = reader.readByte()
        else { throw DecodeError.truncated }

        guard let kind = Kind(rawValue: kindRaw) else {
            throw DecodeError.invalidField("kind \(kindRaw)")
        }
        guard let dtype = DType(rawValue: dtypeRaw) else {
            throw DecodeError.invalidField("dtype \(dtypeRaw)")
        }
        guard rank <= maxRank else { throw DecodeError.invalidField("rank \(rank)") }

        guard let seqPos: UInt32 = reader.readLE(),
              let layerStart: UInt16 = reader.readLE(),
              let layerEnd: UInt16 = reader.readLE()
        else { throw DecodeError.truncated }

        var shape: [Int] = []
        for _ in 0 ..< rank {
            guard let dim: Int32 = reader.readLE() else { throw DecodeError.truncated }
            guard dim > 0 else { throw DecodeError.invalidField("dim \(dim)") }
            shape.append(Int(dim))
        }

        guard let payloadCount: UInt32 = reader.readLE() else { throw DecodeError.truncated }
        guard let payload = reader.read(count: Int(payloadCount)) else {
            throw DecodeError.truncated
        }

        let expected = shape.reduce(1, *) * dtype.bytesPerElement
        if kind == .hiddenState, expected != payload.count {
            throw DecodeError.payloadSizeMismatch(expected: expected, actual: payload.count)
        }

        return TensorFrame(
            kind: kind, dtype: dtype, seqPos: Int(seqPos),
            layerStart: Int(layerStart), layerEnd: Int(layerEnd),
            shape: shape, payload: Data(payload))
    }
}

// MARK: - Little-endian helpers

extension Data {
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}

struct DataReader {
    private let data: Data
    private var offset: Int

    init(_ data: Data) {
        self.data = data
        self.offset = data.startIndex
    }

    mutating func readByte() -> UInt8? {
        guard offset < data.endIndex else { return nil }
        defer { offset += 1 }
        return data[offset]
    }

    mutating func read(count: Int) -> Data? {
        guard count >= 0, offset + count <= data.endIndex else { return nil }
        defer { offset += count }
        return data.subdata(in: offset ..< offset + count)
    }

    mutating func readLE<T: FixedWidthInteger>() -> T? {
        guard let bytes = read(count: MemoryLayout<T>.size) else { return nil }
        var value: T = 0
        _ = withUnsafeMutableBytes(of: &value) { bytes.copyBytes(to: $0) }
        return T(littleEndian: value)
    }
}
