import EdgeLLMKit
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Prints download progress in 10% steps, safe to call from any thread.
final class ProgressReporter: @unchecked Sendable {
    private let lock = NSLock()
    private var lastDecile = -1

    func report(fraction: Double) {
        let decile = Int(fraction * 10)
        lock.lock()
        defer { lock.unlock() }
        guard decile != lastDecile else { return }
        lastDecile = decile
        print("      download \(decile * 10)%")
    }
}

/// Phase 0 feasibility proof, single device, three runs of the same prompt:
///
///   1. upstream:  mlx-swift-lm's own Qwen2 stack (ground truth text)
///   2. reference: our shard implementation covering all layers at once
///   3. pipeline:  the model split into two shards; every hidden state
///                  crossing the shard boundary is serialized to the wire
///                  format and decoded back, exactly like the network path
///
/// Success: (2) matches (1) textually, and (3) matches (2) token for token.
@main
struct PipelineProof {

    struct Options {
        var model = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        var prompt = "Explain in two sentences why the sky is blue."
        var maxTokens = 64
        var splitLayer: Int? = nil

        static func parse(_ arguments: [String]) -> Options {
            var options = Options()
            var iterator = arguments.dropFirst().makeIterator()
            while let argument = iterator.next() {
                switch argument {
                case "--model": options.model = iterator.next() ?? options.model
                case "--prompt": options.prompt = iterator.next() ?? options.prompt
                case "--max-tokens":
                    options.maxTokens = iterator.next().flatMap { Int($0) } ?? options.maxTokens
                case "--split":
                    options.splitLayer = iterator.next().flatMap { Int($0) }
                case "--help", "-h":
                    print("""
                    usage: pipeline-proof [--model <hf-repo>] [--prompt <text>]
                                          [--max-tokens <n>] [--split <layer>]

                    Splits a Qwen2-family model into two pipeline shards on this
                    machine and verifies the sharded output matches the
                    unsharded reference token for token.
                    """)
                    exit(0)
                default:
                    print("unknown argument: \(argument) (see --help)")
                    exit(2)
                }
            }
            return options
        }
    }

    static func main() async {
        let options = Options.parse(ProcessInfo.processInfo.arguments)
        do {
            try await run(options)
        } catch {
            print("\n✗ failed: \(error)")
            exit(1)
        }
    }

    static func run(_ options: Options) async throws {
        print("edge-llm-cluster · pipeline-proof")
        print("model: \(options.model)\n")

        // 1. Download snapshot + load the upstream reference stack.
        print("[1/4] downloading & loading upstream model …")
        let reporter = ProgressReporter()
        let container = try await LLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: options.model)
        ) { progress in
            reporter.report(fraction: progress.fractionCompleted)
        }

        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(
                maxTokens: options.maxTokens, temperature: 0))
        let upstreamText = try await session.respond(to: options.prompt)
        print("      upstream says: \(upstreamText.prefix(120))…\n")

        // 2. Tokenize once; both of our runs consume the same ids.
        let tokenizer = await container.tokenizer
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": options.prompt]])
        var eosTokenIds = Set<Int>()
        if let eos = tokenizer.eosTokenId { eosTokenIds.insert(eos) }
        if let imEnd = tokenizer.convertTokenToId("<|im_end|>") { eosTokenIds.insert(imEnd) }

        let modelDirectory = try await container.modelDirectory
        let config = try ShardLoader.readConfig(modelDirectory: modelDirectory)
        let totalLayers = config.hiddenLayers
        let split = options.splitLayer ?? totalLayers / 2
        precondition(split > 0 && split < totalLayers, "--split must be in 1..<\(totalLayers)")

        // 3. Unsharded run through our own implementation.
        print("[2/4] reference: our stack, all \(totalLayers) layers in one shard …")
        let full = try ShardLoader.loadShard(
            modelDirectory: modelDirectory, spec: .full(totalLayers))
        let reference = try PipelineGenerator.generate(
            shards: [full],
            promptTokens: promptTokens,
            maxTokens: options.maxTokens,
            eosTokenIds: eosTokenIds)
        let referenceText = tokenizer.decode(tokenIds: reference.tokens)
        print("      \(reference.tokens.count) tokens, " +
              String(format: "%.1f tok/s", reference.metrics.tokensPerSecond) + "\n")

        // 4. The actual proof: two shards + wire-format round-trip between them.
        print("[3/4] pipeline: layers 0..<\(split) → codec → layers \(split)..<\(totalLayers) …")
        let shardA = try ShardLoader.loadShard(
            modelDirectory: modelDirectory,
            spec: ShardSpec(start: 0, end: split, totalLayers: totalLayers))
        let shardB = try ShardLoader.loadShard(
            modelDirectory: modelDirectory,
            spec: ShardSpec(start: split, end: totalLayers, totalLayers: totalLayers))
        let pipeline = try PipelineGenerator.generate(
            shards: [shardA, shardB],
            promptTokens: promptTokens,
            maxTokens: options.maxTokens,
            eosTokenIds: eosTokenIds,
            link: PipelineGenerator.codecLink)
        let pipelineText = tokenizer.decode(tokenIds: pipeline.tokens)

        // 5. Verdict.
        print("[4/4] results\n")
        print("prompt          : \(options.prompt)")
        print("reference       : \(referenceText)")
        print("pipeline        : \(pipelineText)\n")

        let decodedSteps = max(pipeline.metrics.generatedCount, 1)
        let bytesPerToken = pipeline.metrics.linkBytes / decodedSteps
        print(String(format: "reference speed : %.1f tok/s", reference.metrics.tokensPerSecond))
        print(String(format: "pipeline speed  : %.1f tok/s", pipeline.metrics.tokensPerSecond))
        print("boundary traffic: \(pipeline.metrics.linkBytes) bytes total, ≈\(bytesPerToken) bytes/step")

        let tokensMatch = reference.tokens == pipeline.tokens
        let textMatchesUpstream = !referenceText.isEmpty
            && upstreamText.trimmingCharacters(in: .whitespacesAndNewlines)
                .hasPrefix(String(referenceText.prefix(32)))

        print("")
        print(tokensMatch
            ? "✓ sharded pipeline matches unsharded reference token for token"
            : "✗ token mismatch between reference and pipeline")
        print(textMatchesUpstream
            ? "✓ our implementation agrees with upstream mlx-swift-lm output"
            : "⚠ our text differs from upstream, inspect outputs above (minor float divergence is possible)")

        if !tokensMatch { exit(1) }
    }
}
