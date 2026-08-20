import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Loads a `Qwen2Shard` from a local model snapshot (the directory produced
/// by a Hugging Face download: `config.json` + `*.safetensors`).
///
/// Checkpoint weight keys are global (`model.layers.17.…`); each shard's
/// module tree is local (`layers.0.…` when the shard starts at layer 17), so
/// keys are filtered to the shard's slice and rebased before loading. Weights
/// outside the shard are never materialized on this device.
public enum ShardLoader {

    public enum Error: Swift.Error, CustomStringConvertible {
        case unsupportedModelType(String)

        public var description: String {
            switch self {
            case .unsupportedModelType(let type):
                return "unsupported model_type '\(type)': only the qwen2 family is supported for now"
            }
        }
    }

    /// Read and decode `config.json` from a model snapshot directory.
    public static func readConfig(modelDirectory: URL) throws -> Qwen2ShardConfig {
        let url = modelDirectory.appending(component: "config.json")
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(Qwen2ShardConfig.self, from: data)
        guard config.modelType == "qwen2" else {
            throw Error.unsupportedModelType(config.modelType)
        }
        return config
    }

    /// Build a shard and load exactly the weights it owns.
    public static func loadShard(modelDirectory: URL, spec: ShardSpec) throws -> Qwen2Shard {
        let config = try readConfig(modelDirectory: modelDirectory)

        let configData = try Data(
            contentsOf: modelDirectory.appending(component: "config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)

        let shard = Qwen2Shard(config: config, spec: spec)

        // Collect all tensors from every safetensors file in the snapshot.
        var allWeights = [String: MLXArray]()
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator where url.pathExtension == "safetensors" {
            for (key, value) in try loadArrays(url: url) {
                allWeights[key] = value
            }
        }

        // Keep only this shard's slice, rebasing global keys to local paths.
        var shardWeights = [String: MLXArray]()
        var originalKeys = [String: String]()
        for (key, value) in allWeights {
            guard let localKey = localKey(for: key, spec: spec, config: config) else {
                continue
            }
            shardWeights[localKey] = value
            originalKeys[localKey] = key
        }

        // Quantize modules whose checkpoint carries scales (4-bit etc.).
        // The per-layer table is keyed by the checkpoint's global module
        // paths, so look up via the original key, not the rebased one.
        if let perLayer = base.perLayerQuantization {
            quantize(model: shard) { path, _ in
                guard shardWeights["\(path).scales"] != nil,
                      let original = originalKeys["\(path).weight"]
                else { return nil }
                let originalModulePath = String(original.dropLast(".weight".count))
                return perLayer.quantization(layer: originalModulePath)?.asTuple
            }
        }

        try shard.update(
            parameters: ModuleParameters.unflattened(shardWeights), verify: [.all])
        eval(shard)
        return shard
    }

    /// Map a checkpoint key to this shard's local module path, or nil when
    /// the tensor belongs to a different shard.
    static func localKey(for key: String, spec: ShardSpec, config: Qwen2ShardConfig) -> String? {
        // Precomputed rotary tables are rebuilt at runtime; never load them.
        if key.contains("self_attn.rotary_emb.inv_freq") { return nil }

        let needsEmbedding = spec.isFirst || (spec.isLast && config.tieWordEmbeddings)

        if key.hasPrefix("model.embed_tokens.") {
            return needsEmbedding
                ? "embed_tokens." + key.dropFirst("model.embed_tokens.".count) : nil
        }

        if key.hasPrefix("model.layers.") {
            let rest = key.dropFirst("model.layers.".count)
            guard let dot = rest.firstIndex(of: "."),
                  let layer = Int(rest[..<dot])
            else { return nil }
            guard layer >= spec.start && layer < spec.end else { return nil }
            return "layers.\(layer - spec.start)" + rest[dot...]
        }

        if key.hasPrefix("model.norm.") {
            return spec.isLast ? "norm." + key.dropFirst("model.norm.".count) : nil
        }

        if key.hasPrefix("lm_head.") {
            return (spec.isLast && !config.tieWordEmbeddings) ? key : nil
        }

        return nil
    }
}
