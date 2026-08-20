import Foundation

/// A contiguous range of transformer layers assigned to one device.
///
/// The first shard also owns the token embedding; the last shard owns the
/// final norm and the LM head (for tied-embedding models the last shard keeps
/// its own copy of the embedding weights to project hidden states to logits).
public struct ShardSpec: Equatable, Hashable, Codable, Sendable {
    /// First layer index owned by this shard (inclusive).
    public let start: Int
    /// One past the last layer index owned by this shard (exclusive).
    public let end: Int
    /// Total number of transformer layers in the model.
    public let totalLayers: Int

    public init(start: Int, end: Int, totalLayers: Int) {
        precondition(totalLayers > 0, "totalLayers must be positive")
        precondition(start >= 0 && end <= totalLayers && start < end,
                     "invalid shard range \(start)..<\(end) for \(totalLayers) layers")
        self.start = start
        self.end = end
        self.totalLayers = totalLayers
    }

    /// A single shard covering the whole model.
    public static func full(_ totalLayers: Int) -> ShardSpec {
        ShardSpec(start: 0, end: totalLayers, totalLayers: totalLayers)
    }

    public var isFirst: Bool { start == 0 }
    public var isLast: Bool { end == totalLayers }
    public var layerCount: Int { end - start }

    /// Split `totalLayers` into contiguous shards proportional to `weights`
    /// (e.g. per-device usable memory in bytes). Every shard receives at
    /// least one layer; rounding remainders go to the largest weights first.
    public static func split(totalLayers: Int, weights: [Double]) -> [ShardSpec] {
        precondition(!weights.isEmpty, "weights must not be empty")
        precondition(weights.allSatisfy { $0 > 0 }, "weights must be positive")
        precondition(totalLayers >= weights.count,
                     "need at least one layer per shard")

        let totalWeight = weights.reduce(0, +)
        let exact = weights.map { Double(totalLayers) * $0 / totalWeight }

        // Floor with a minimum of one layer each, then hand out the
        // remaining layers by largest fractional remainder.
        var counts = exact.map { max(1, Int($0.rounded(.down))) }
        var assigned = counts.reduce(0, +)

        if assigned > totalLayers {
            // The min-1 guarantee overshot: take layers back from the
            // largest allocations until we fit.
            while assigned > totalLayers {
                let i = counts.indices.max { counts[$0] < counts[$1] }!
                precondition(counts[i] > 1, "cannot satisfy one layer per shard")
                counts[i] -= 1
                assigned -= 1
            }
        } else {
            let remainders = exact.enumerated()
                .map { (index: $0.offset, frac: $0.element - $0.element.rounded(.down)) }
                .sorted { $0.frac > $1.frac }
            var r = 0
            while assigned < totalLayers {
                counts[remainders[r % remainders.count].index] += 1
                assigned += 1
                r += 1
            }
        }

        var shards: [ShardSpec] = []
        var cursor = 0
        for count in counts {
            shards.append(ShardSpec(start: cursor, end: cursor + count, totalLayers: totalLayers))
            cursor += count
        }
        return shards
    }
}

extension ShardSpec: CustomStringConvertible {
    public var description: String {
        "layers \(start)..<\(end) of \(totalLayers)" +
            (isFirst ? " +embed" : "") + (isLast ? " +head" : "")
    }
}
