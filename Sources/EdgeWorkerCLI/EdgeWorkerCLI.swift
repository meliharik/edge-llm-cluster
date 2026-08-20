import EdgeLLMKit
import EdgeLLMTransport
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon

/// Runs one pipeline worker: discovers the coordinator on the local network,
/// announces itself, loads the layer range it is assigned, and then serves
/// inference frames until disconnected.
///
///   edge-worker [--name <name>] [--id <stable id>] [--memory-gb <quota>]
///               [--model-dir <path>] [--coordinator <name>]
///
/// Weights: with --model-dir the worker loads from a local snapshot; without
/// it, the model named in the assignment is downloaded from Hugging Face on
/// first use.
@main
struct EdgeWorkerCLI {

    static func main() {
        var name = Host.current().localizedName ?? "worker"
        var id = ""
        var memoryGB: Double? = nil
        var modelDir: URL? = nil
        var coordinatorFilter: String? = nil

        var iterator = ProcessInfo.processInfo.arguments.dropFirst().makeIterator()
        while let argument = iterator.next() {
            switch argument {
            case "--name": name = iterator.next() ?? name
            case "--id": id = iterator.next() ?? id
            case "--memory-gb": memoryGB = iterator.next().flatMap { Double($0) }
            case "--model-dir":
                modelDir = iterator.next().map { URL(filePath: $0) }
            case "--coordinator": coordinatorFilter = iterator.next()
            case "--help", "-h":
                print("""
                usage: edge-worker [--name <name>] [--id <stable id>]
                                   [--memory-gb <quota>] [--model-dir <path>]
                                   [--coordinator <name>]
                """)
                return
            default:
                print("unknown argument: \(argument) (see --help)")
                exit(2)
            }
        }

        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let usableMemory = memoryGB.map { UInt64($0 * 1_073_741_824) }
            ?? totalMemory * 6 / 10
        if id.isEmpty {
            id = name.lowercased().replacingOccurrences(of: " ", with: "-")
                + "-" + String(UInt32.random(in: 0x1000 ... 0xffff), radix: 16)
        }

        let info = ControlMessage.DeviceInfo(
            id: id, name: name, platform: "macos", runtime: "mlx",
            totalMemory: totalMemory, usableMemory: usableMemory)

        let worker = Worker(
            info: info, modelDirOverride: modelDir,
            coordinatorFilter: coordinatorFilter)
        worker.start()
        RunLoop.main.run()
    }
}

/// Connection lifecycle and inference loop for a single worker process.
final class Worker {

    private let info: ControlMessage.DeviceInfo
    private let modelDirOverride: URL?
    private let coordinatorFilter: String?

    private let browser = PeerBrowser()
    private var connection: FrameConnection?
    private var connectedPeer: String?

    private var shard: Qwen2Shard?
    private var caches: [KVCache]?
    private var millisecondsPerToken = 0.0

    init(info: ControlMessage.DeviceInfo, modelDirOverride: URL?, coordinatorFilter: String?) {
        self.info = info
        self.modelDirOverride = modelDirOverride
        self.coordinatorFilter = coordinatorFilter
    }

    func start() {
        print("worker \(info.id): \(info.name), quota \(gb(info.usableMemory)) of \(gb(info.totalMemory))")
        print("searching for a coordinator (\(edgeLLMServiceType)) …")

        browser.peersChangedHandler = { [weak self] peers in
            self?.peersChanged(peers)
        }
        browser.start()
        startMetricsTimer()
    }

    // MARK: - Discovery and connection

    private func peersChanged(_ peers: [PeerBrowser.Peer]) {
        guard connection == nil else { return }
        let candidates = peers.filter { peer in
            coordinatorFilter.map { peer.name == $0 } ?? true
        }
        guard let peer = candidates.first else { return }

        print("connecting to \(peer.name) …")
        connectedPeer = peer.name
        let connection = browser.connect(to: peer)
        self.connection = connection

        connection.stateHandler = { [weak self] state in
            if case .failed = state { self?.dropConnection() }
        }

        do {
            let hello = try TensorFrame.control(.hello(info))
            connection.send(hello) { [weak self] error in
                if error != nil { self?.dropConnection() }
            }
        } catch {
            dropConnection()
            return
        }
        receiveNext()
    }

    private func dropConnection() {
        guard connection != nil else { return }
        print("connection to \(connectedPeer ?? "coordinator") lost, searching again …")
        connection?.cancel()
        connection = nil
        connectedPeer = nil
        caches = nil
    }

    private func receiveNext() {
        connection?.receiveFrame { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure:
                self.dropConnection()
            case .success(let frame):
                self.handle(frame)
                self.receiveNext()
            }
        }
    }

    // MARK: - Frame handling

    private func handle(_ frame: TensorFrame) {
        if frame.kind == .control {
            guard let message = try? frame.controlMessage() else { return }
            handleControl(message)
            return
        }
        compute(frame)
    }

    private func handleControl(_ message: ControlMessage) {
        switch message {
        case .assignment(let assignment):
            load(assignment)
        case .reset:
            caches = shard?.newCaches()
        case .goodbye:
            dropConnection()
        case .hello, .metrics:
            break
        }
    }

    private func load(_ assignment: ControlMessage.ShardAssignment) {
        let spec = ShardSpec(
            start: assignment.layerStart, end: assignment.layerEnd,
            totalLayers: assignment.totalLayers)
        print("assigned \(spec) of \(assignment.model)")

        Task {
            do {
                let directory: URL
                if let modelDirOverride {
                    directory = modelDirOverride
                } else {
                    print("fetching weights for \(assignment.model) …")
                    directory = try await #hubDownloader().download(
                        id: assignment.model, revision: nil,
                        matching: ["*.safetensors", "*.json"],
                        useLatest: false, progressHandler: { _ in })
                }
                let shard = try ShardLoader.loadShard(modelDirectory: directory, spec: spec)
                self.shard = shard
                self.caches = shard.newCaches()
                print("shard loaded, \(gb(UInt64(Memory.activeMemory))) resident, ready")
            } catch {
                print("could not load shard: \(error)")
                self.shard = nil
            }
        }
    }

    private func compute(_ frame: TensorFrame) {
        guard let shard, let caches else {
            print("dropping \(frame.kind) frame: no shard loaded yet")
            return
        }

        do {
            let input: Qwen2Shard.Input
            let length: Int
            switch frame.kind {
            case .token:
                guard shard.spec.isFirst, let ids = frame.tokenIDs() else { return }
                length = ids.count
                input = .tokens(MLXArray(ids).reshaped(1, length))
            case .hiddenState:
                let hidden = try frame.hiddenState()
                length = hidden.dim(1)
                input = .hidden(hidden)
            case .control:
                return
            }

            let started = Date()
            let output = shard(input, caches: caches)

            let reply: TensorFrame
            if shard.spec.isLast {
                let last = output[0, output.dim(1) - 1]
                let token = argMax(last).asType(.int32).item(Int32.self)
                reply = TensorFrame(tokens: [token], seqPos: frame.seqPos + length)
            } else {
                eval(output)
                reply = TensorFrame(
                    hiddenState: output, seqPos: frame.seqPos, producedBy: shard.spec)
            }

            let elapsed = Date().timeIntervalSince(started) * 1000
            millisecondsPerToken = millisecondsPerToken == 0
                ? elapsed : millisecondsPerToken * 0.8 + elapsed * 0.2

            connection?.send(reply) { [weak self] error in
                if error != nil { self?.dropConnection() }
            }
        } catch {
            print("compute failed: \(error)")
        }
    }

    // MARK: - Metrics

    private func startMetricsTimer() {
        let timer = Timer(timeInterval: 3, repeats: true) { [weak self] _ in
            self?.sendMetrics()
        }
        RunLoop.main.add(timer, forMode: .common)
    }

    private func sendMetrics() {
        guard let connection else { return }
        let metrics = ControlMessage.DeviceMetrics(
            id: info.id,
            usedMemory: UInt64(max(0, Memory.activeMemory)),
            millisecondsPerToken: millisecondsPerToken)
        if let frame = try? TensorFrame.control(.metrics(metrics)) {
            connection.send(frame) { _ in }
        }
    }
}

private func gb(_ bytes: UInt64) -> String {
    String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
}
