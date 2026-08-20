import EdgeLLMKit
import Foundation
import MLXLMCommon

/// The coordinator's chat loop: assigns layer ranges to the connected
/// workers, relays frames through the pipeline for every token, and streams
/// the decoded answer back to the caller.
///
/// Version 1 topology is a star: each frame goes worker by worker through
/// the coordinator, and the worker holding the last shard replies with the
/// greedily sampled token id. Conversation history is kept here and
/// re-prefilled on every turn after a `reset`, so worker caches never span
/// generations.
public final class ChatPipeline {

    public enum PipelineError: Error, CustomStringConvertible {
        case noWorkers
        case workerNotIntroduced
        case notReadyInTime([String])
        case unexpectedReply

        public var description: String {
            switch self {
            case .noWorkers:
                return "no workers connected"
            case .workerNotIntroduced:
                return "a worker connected but never sent hello"
            case .notReadyInTime(let names):
                return "workers not ready in time: \(names.joined(separator: ", "))"
            case .unexpectedReply:
                return "the last shard did not reply with a token frame"
            }
        }
    }

    private let workers: [WorkerHandle]
    private let tokenizer: any Tokenizer
    private let state: ClusterState
    private let model: String
    private let totalLayers: Int
    private let eosTokenIds: Set<Int>

    private var messages: [[String: any Sendable]] = []
    private var sessionTokens = 0

    public init(
        workers: [WorkerHandle], tokenizer: any Tokenizer,
        state: ClusterState, model: String, totalLayers: Int
    ) throws {
        guard !workers.isEmpty else { throw PipelineError.noWorkers }
        self.workers = workers
        self.tokenizer = tokenizer
        self.state = state
        self.model = model
        self.totalLayers = totalLayers

        var eos = Set<Int>()
        if let id = tokenizer.eosTokenId { eos.insert(id) }
        if let id = tokenizer.convertTokenToId("<|im_end|>") { eos.insert(id) }
        self.eosTokenIds = eos
    }

    // MARK: - Setup

    /// Compute the memory proportional layer plan for a worker list.
    /// Exposed for tests; `assignShards` applies it to the live cluster.
    public static func plan(
        model: String, totalLayers: Int, infos: [ControlMessage.DeviceInfo]
    ) -> [(spec: ShardSpec, assignment: ControlMessage.ShardAssignment)] {
        let specs = ShardSpec.split(
            totalLayers: totalLayers,
            weights: infos.map { Double($0.usableMemory) })
        return specs.enumerated().map { index, spec in
            let next = index + 1 < infos.count ? infos[index + 1].id : ""
            let assignment = ControlMessage.ShardAssignment(
                model: model, layerStart: spec.start, layerEnd: spec.end,
                totalLayers: totalLayers, nextWorker: next)
            return (spec, assignment)
        }
    }

    /// Send every worker its layer range and record it in the cluster state.
    public func assignShards() throws {
        let infos = try workers.map { worker in
            guard let info = worker.info else { throw PipelineError.workerNotIntroduced }
            return info
        }
        state.setModel(model, totalLayers: totalLayers)
        for (index, step) in Self.plan(
            model: model, totalLayers: totalLayers, infos: infos
        ).enumerated() {
            state.assign(deviceID: infos[index].id, spec: step.spec)
            try workers[index].sendControl(.assignment(step.assignment))
        }
    }

    /// Wait until every assigned worker has reported ready. Workers may be
    /// downloading weights, so the default timeout is generous.
    public func waitUntilReady(timeout: TimeInterval = 900) async throws {
        let ids = Set(workers.compactMap { $0.info?.id })
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            let devices = state.snapshot().devices
            let ready = Set(devices.filter(\.ready).map(\.info.id))
            if ids.subtracting(ready).isEmpty { return }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        let missing = state.snapshot().devices
            .filter { !$0.ready && ids.contains($0.info.id) }
            .map(\.info.name)
        throw PipelineError.notReadyInTime(missing)
    }

    // MARK: - Generation

    /// Run one chat turn. The prompt is appended to the running conversation,
    /// decoded text is streamed through `onToken`, and the full reply is
    /// returned and remembered for the next turn.
    public func respond(
        to prompt: String, maxTokens: Int = 256,
        onToken: (String) -> Void = { _ in }
    ) async throws -> String {
        messages.append(["role": "user", "content": prompt])
        let promptTokens = try tokenizer.applyChatTemplate(messages: messages)

        for worker in workers {
            try worker.sendControl(.reset)
        }

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var produced: [Int] = []
        var input = TensorFrame(tokens: promptTokens.map { Int32($0) }, seqPos: 0)
        let started = Date()

        for _ in 0 ..< maxTokens {
            var frame = input
            for worker in workers {
                frame = try await worker.request(frame)
            }
            guard let token = frame.tokenIDs()?.first.map(Int.init) else {
                throw PipelineError.unexpectedReply
            }
            if eosTokenIds.contains(token) { break }

            produced.append(token)
            sessionTokens += 1
            detokenizer.append(token: token)
            if let segment = detokenizer.next() {
                onToken(segment)
            }

            let elapsed = Date().timeIntervalSince(started)
            if elapsed > 0 {
                state.setThroughput(
                    tokensPerSecond: Double(produced.count) / elapsed,
                    generatedTokens: sessionTokens)
            }

            input = TensorFrame(tokens: [Int32(token)], seqPos: frame.seqPos)
        }

        let text = tokenizer.decode(tokenIds: produced)
        messages.append(["role": "assistant", "content": text])
        return text
    }
}
