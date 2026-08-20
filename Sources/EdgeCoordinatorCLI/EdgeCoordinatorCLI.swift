import EdgeLLMCoordinator
import EdgeLLMKit
import EdgeLLMTransport
import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Runs the cluster coordinator: advertises the service, accepts workers
/// from any platform, serves the web dashboard, and drives the chat loop
/// across the connected shards.
///
///   edge-coordinator [--model <hf repo>] [--min-workers <n>]
///                    [--max-tokens <n>] [--port 8080] [--name <name>]
///                    [--demo]
///
/// With --demo the dashboard is served with simulated devices and no chat
/// loop runs. Otherwise the coordinator waits for `--min-workers` workers,
/// assigns layer ranges proportional to their memory, waits for every shard
/// to load, and then reads prompts from stdin, streaming answers back.
@main
struct EdgeCoordinatorCLI {

    static func main() async {
        var port: UInt16 = 8080
        var name = Host.current().localizedName ?? "coordinator"
        var demo = false
        var model = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        var minWorkers = 1
        var maxTokens = 256

        var iterator = ProcessInfo.processInfo.arguments.dropFirst().makeIterator()
        while let argument = iterator.next() {
            switch argument {
            case "--port": port = iterator.next().flatMap { UInt16($0) } ?? port
            case "--name": name = iterator.next() ?? name
            case "--demo": demo = true
            case "--model": model = iterator.next() ?? model
            case "--min-workers": minWorkers = iterator.next().flatMap { Int($0) } ?? minWorkers
            case "--max-tokens": maxTokens = iterator.next().flatMap { Int($0) } ?? maxTokens
            case "--help", "-h":
                print("""
                usage: edge-coordinator [--model <hf repo>] [--min-workers <n>]
                                        [--max-tokens <n>] [--port 8080]
                                        [--name <name>] [--demo]
                """)
                return
            default:
                print("unknown argument: \(argument) (see --help)")
                exit(2)
            }
        }

        let state = ClusterState()
        if demo {
            seedDemoDevices(into: state)
        }

        do {
            let server = try StatusServer(state: state, port: port)
            server.start()
            print("dashboard: http://localhost:\(port)  (reachable from any device on this network)")
        } catch {
            print("could not start the dashboard server: \(error)")
            exit(1)
        }

        let registry = WorkerRegistry()
        do {
            let advertiser = try PeerAdvertiser(name: name)
            advertiser.newConnectionHandler = { connection in
                let handle = WorkerHandle(connection: connection, state: state)
                handle.onHello = { handle in
                    guard let info = handle.info else { return }
                    registry.add(handle)
                    print("worker joined: \(info.name) [\(info.platform)/\(info.runtime)]")
                }
                handle.onClose = { handle in
                    registry.remove(handle)
                    if let info = handle.info {
                        print("worker left: \(info.name)")
                    }
                }
                handle.start()
            }
            advertiser.start()
            print("advertising \(edgeLLMServiceType) as \"\(name)\"")
        } catch {
            print("could not start the listener: \(error)")
            exit(1)
        }

        if demo {
            RunLoop.main.run()
            return
        }

        await runChat(
            state: state, registry: registry, model: model,
            minWorkers: minWorkers, maxTokens: maxTokens)
    }

    // MARK: - Chat loop

    static func runChat(
        state: ClusterState, registry: WorkerRegistry,
        model: String, minWorkers: Int, maxTokens: Int
    ) async {
        do {
            print("fetching tokenizer and configuration for \(model) …")
            let directory = try await #hubDownloader().download(
                id: model, revision: nil,
                matching: ["*.json", "*.txt", "*.jinja"],
                useLatest: false, progressHandler: { _ in })
            let config = try ShardLoader.readConfig(modelDirectory: directory)
            let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)

            print("waiting for \(minWorkers) worker(s) to join …")
            while registry.count < minWorkers {
                try await Task.sleep(nanoseconds: 500_000_000)
            }

            let workers = registry.all()
            print("assigning \(config.hiddenLayers) layers across \(workers.count) worker(s)")
            let pipeline = try ChatPipeline(
                workers: workers, tokenizer: tokenizer, state: state,
                model: model, totalLayers: config.hiddenLayers)
            try pipeline.assignShards()

            print("waiting for shards to load (workers may be downloading weights) …")
            try await pipeline.waitUntilReady()
            for device in state.snapshot().devices where device.layerStart != nil {
                print("  \(device.info.name): layers \(device.layerStart!)..<\(device.layerEnd!)")
            }
            print("\ncluster ready. Type a prompt; empty line or ctrl-d quits.\n")

            while true {
                print("> ", terminator: "")
                fflush(stdout)
                guard let line = readLine(),
                      !line.trimmingCharacters(in: .whitespaces).isEmpty
                else { break }

                do {
                    _ = try await pipeline.respond(to: line, maxTokens: maxTokens) { segment in
                        print(segment, terminator: "")
                        fflush(stdout)
                    }
                    let stats = state.snapshot()
                    print(String(format: "\n[%.1f tok/s]\n", stats.tokensPerSecond))
                } catch {
                    print("\ngeneration failed: \(error)")
                    print("if a worker dropped, restart the workers and this coordinator\n")
                }
            }

            for worker in registry.all() {
                try? worker.sendControl(.goodbye)
            }
            print("bye")
        } catch {
            print("setup failed: \(error)")
            exit(1)
        }
    }

    // MARK: - Demo data

    static func seedDemoDevices(into state: ClusterState) {
        state.setModel("mlx-community/Qwen2.5-7B-Instruct-4bit", totalLayers: 28)
        state.setThroughput(tokensPerSecond: 6.2, generatedTokens: 184)

        let gb: UInt64 = 1 << 30
        let devices: [(ControlMessage.DeviceInfo, ShardSpec, UInt64, Double)] = [
            (.init(id: "demo-iphone", name: "iPhone 15 Pro", platform: "ios",
                   runtime: "mlx", totalMemory: 8 * gb, usableMemory: 45 * gb / 10),
             ShardSpec(start: 0, end: 8, totalLayers: 28), 31 * gb / 10, 48.5),
            (.init(id: "demo-pixel", name: "Pixel 8", platform: "android",
                   runtime: "llamacpp", totalMemory: 8 * gb, usableMemory: 4 * gb),
             ShardSpec(start: 8, end: 15, totalLayers: 28), 28 * gb / 10, 61.0),
            (.init(id: "demo-ipad", name: "iPad Air", platform: "ipados",
                   runtime: "mlx", totalMemory: 8 * gb, usableMemory: 45 * gb / 10),
             ShardSpec(start: 15, end: 22, totalLayers: 28), 29 * gb / 10, 44.2),
            (.init(id: "demo-mac", name: "MacBook Pro", platform: "macos",
                   runtime: "mlx", totalMemory: 24 * gb, usableMemory: 14 * gb),
             ShardSpec(start: 22, end: 28, totalLayers: 28), 33 * gb / 10, 12.9),
        ]

        for (info, spec, used, ms) in devices {
            state.apply(.hello(info))
            state.assign(deviceID: info.id, spec: spec)
            state.markReady(deviceID: info.id)
            state.apply(.metrics(.init(
                id: info.id, usedMemory: used, millisecondsPerToken: ms)))
        }

        // Keep the demo devices looking alive.
        let timer = Timer(timeInterval: 3, repeats: true) { _ in
            for (info, _, used, ms) in devices {
                let jitter = Double.random(in: 0.9 ... 1.1)
                state.apply(.metrics(.init(
                    id: info.id, usedMemory: used,
                    millisecondsPerToken: ms * jitter)))
            }
        }
        RunLoop.main.add(timer, forMode: .common)
    }
}

/// Connected workers in join order. Rejoining workers (same id) replace
/// their old entry.
final class WorkerRegistry {

    private let lock = NSLock()
    private var handles: [WorkerHandle] = []

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return handles.count
    }

    func all() -> [WorkerHandle] {
        lock.lock()
        defer { lock.unlock() }
        return handles
    }

    func add(_ handle: WorkerHandle) {
        guard let id = handle.info?.id else { return }
        lock.lock()
        defer { lock.unlock() }
        handles.removeAll { $0.info?.id == id && $0 !== handle }
        if !handles.contains(where: { $0 === handle }) {
            handles.append(handle)
        }
    }

    func remove(_ handle: WorkerHandle) {
        lock.lock()
        defer { lock.unlock() }
        handles.removeAll { $0 === handle }
    }
}
