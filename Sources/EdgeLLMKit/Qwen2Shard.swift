import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Model architecture adapted from mlx-swift-lm's Qwen2 implementation
// (https://github.com/ml-explore/mlx-swift-lm, MIT license), restructured to
// own an arbitrary contiguous layer range instead of the whole stack.

/// The subset of `config.json` needed to build a Qwen2-family shard.
public struct Qwen2ShardConfig: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var ropeTheta: Float
    public var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decode(String.self, forKey: .modelType)
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        self.rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

final class Qwen2ShardAttention: Module {
    let heads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ config: Qwen2ShardConfig) {
        let dim = config.hiddenSize
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        let headDim = dim / heads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

final class Qwen2ShardMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class Qwen2ShardBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen2ShardAttention
    let mlp: Qwen2ShardMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: Qwen2ShardConfig) {
        _attention.wrappedValue = Qwen2ShardAttention(config)
        self.mlp = Qwen2ShardMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

/// One pipeline stage of a Qwen2-family model.
///
/// - The first shard maps token ids to hidden states (embedding) before its
///   layer range.
/// - Middle shards transform hidden states only.
/// - The last shard applies the final norm and LM head, producing logits.
///   For tied-embedding checkpoints the last shard holds a copy of the
///   embedding weights and uses them as the output projection.
public final class Qwen2Shard: Module {
    public let spec: ShardSpec
    public let config: Qwen2ShardConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    let layers: [Qwen2ShardBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm?
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public enum Input {
        /// Token ids, shape `[batch, length]`. Only valid for the first shard.
        case tokens(MLXArray)
        /// Hidden states from the previous shard, shape `[batch, length, hidden]`.
        case hidden(MLXArray)
    }

    public init(config: Qwen2ShardConfig, spec: ShardSpec) {
        precondition(spec.totalLayers == config.hiddenLayers,
                     "shard spec does not match model layer count")
        self.spec = spec
        self.config = config

        let needsEmbedding = spec.isFirst || (spec.isLast && config.tieWordEmbeddings)
        if needsEmbedding {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        }

        self.layers = (spec.start ..< spec.end).map { _ in Qwen2ShardBlock(config) }

        if spec.isLast {
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false)
            }
        }
    }

    /// One fresh KV cache per owned layer. Caches never leave the device.
    public func newCaches() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }

    /// Run the shard. Returns hidden states, or logits if this is the last shard.
    public func callAsFunction(_ input: Input, caches: [KVCache]?) -> MLXArray {
        var h: MLXArray
        switch input {
        case .tokens(let tokens):
            guard let embedTokens else {
                fatalError("shard \(spec) received token input but has no embedding")
            }
            precondition(spec.isFirst, "token input is only valid for the first shard")
            h = embedTokens(tokens)
        case .hidden(let hidden):
            precondition(!spec.isFirst || hidden.ndim == 3,
                         "hidden input must be [batch, length, hidden]")
            h = hidden
        }

        let mask = createAttentionMask(h: h, cache: caches?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: caches?[i])
        }

        if spec.isLast {
            h = norm!(h)
            if let lmHead {
                h = lmHead(h)
            } else {
                h = embedTokens!.asLinear(h)
            }
        }

        return h
    }
}
