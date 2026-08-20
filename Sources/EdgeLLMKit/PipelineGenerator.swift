import Foundation
import MLX
import MLXLMCommon

/// Greedy token generation across a chain of pipeline shards.
///
/// Each shard keeps its own KV caches; only hidden states cross shard
/// boundaries. The `link` hook intercepts every hidden state handed from one
/// shard to the next — the single-process proof uses it to round-trip tensors
/// through the wire codec, and the networked runtime replaces it with an
/// actual socket.
public enum PipelineGenerator {

    public struct Metrics {
        public var promptTokenCount = 0
        public var prefillSeconds = 0.0
        public var decodeSeconds = 0.0
        public var linkBytes = 0

        public var tokensPerSecond: Double {
            decodeSeconds > 0 ? Double(generatedCount) / decodeSeconds : 0
        }
        public internal(set) var generatedCount = 0
    }

    public struct Output {
        public let tokens: [Int]
        public let metrics: Metrics
    }

    /// Hand-off hook between consecutive shards. Receives the hidden state,
    /// the producing shard's spec, and the current sequence position; returns
    /// the tensor to feed the next shard and the bytes it moved (0 for a
    /// direct in-memory hand-off).
    public typealias Link = (MLXArray, ShardSpec, Int) throws -> (MLXArray, Int)

    /// In-memory hand-off: no serialization, no copies counted.
    public static let directLink: Link = { hidden, _, _ in (hidden, 0) }

    /// Serialize + deserialize through `TensorFrame`, exactly as the network
    /// transport will. Proves the wire format loses nothing.
    public static let codecLink: Link = { hidden, spec, seqPos in
        let frame = TensorFrame(hiddenState: hidden, seqPos: seqPos, producedBy: spec)
        let wire = frame.encoded()
        let decoded = try TensorFrame.decode(wire)
        return (try decoded.hiddenState(), wire.count)
    }

    public static func generate(
        shards: [Qwen2Shard],
        promptTokens: [Int],
        maxTokens: Int,
        eosTokenIds: Set<Int>,
        link: Link = directLink
    ) throws -> Output {
        precondition(!shards.isEmpty)
        precondition(shards.first!.spec.isFirst && shards.last!.spec.isLast,
                     "shard chain must start with the embedding shard and end with the head shard")

        let caches = shards.map { $0.newCaches() }
        var metrics = Metrics()
        metrics.promptTokenCount = promptTokens.count

        var tokens: [Int] = []
        var seqPos = 0
        var nextInput = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, promptTokens.count)

        for step in 0 ..< maxTokens {
            let stepStart = Date()

            var current: Qwen2Shard.Input = .tokens(nextInput)
            for (i, shard) in shards.enumerated() {
                var out = shard(current, caches: caches[i])
                if i < shards.count - 1 {
                    eval(out)
                    let (linked, bytes) = try link(out, shard.spec, seqPos)
                    metrics.linkBytes += bytes
                    out = linked
                }
                current = .hidden(out)
            }

            guard case .hidden(let logits) = current else { fatalError("unreachable") }
            let last = logits[0, logits.dim(1) - 1]
            let tokenArray = argMax(last).asType(.int32)
            eval(tokenArray)
            let token = Int(tokenArray.item(Int32.self))

            let elapsed = Date().timeIntervalSince(stepStart)
            if step == 0 {
                metrics.prefillSeconds = elapsed
                seqPos += promptTokens.count
            } else {
                metrics.decodeSeconds += elapsed
                seqPos += 1
            }

            if eosTokenIds.contains(token) { break }
            tokens.append(token)
            metrics.generatedCount = tokens.count
            nextInput = MLXArray([Int32(token)]).reshaped(1, 1)
        }

        return Output(tokens: tokens, metrics: metrics)
    }
}
