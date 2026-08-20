import EdgeLLMCoordinator
import EdgeLLMKit
import EdgeLLMTransport
import Foundation

/// Runs the cluster coordinator: advertises the Bonjour service, accepts
/// workers from any platform, tracks their state, and serves the web
/// dashboard.
///
///   edge-coordinator [--port 8080] [--name <cluster name>] [--demo]
///
/// With --demo the state is seeded with simulated devices so the dashboard
/// can be exercised without any workers running.
@main
struct EdgeCoordinatorCLI {

    static func main() {
        var port: UInt16 = 8080
        var name = Host.current().localizedName ?? "coordinator"
        var demo = false

        var iterator = ProcessInfo.processInfo.arguments.dropFirst().makeIterator()
        while let argument = iterator.next() {
            switch argument {
            case "--port": port = iterator.next().flatMap { UInt16($0) } ?? port
            case "--name": name = iterator.next() ?? name
            case "--demo": demo = true
            case "--help", "-h":
                print("usage: edge-coordinator [--port 8080] [--name <name>] [--demo]")
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

        do {
            let advertiser = try PeerAdvertiser(name: name)
            advertiser.newConnectionHandler = { connection in
                WorkerSession(connection: connection, state: state).run()
            }
            advertiser.start()
            print("advertising \(edgeLLMServiceType) as \"\(name)\", waiting for workers")
        } catch {
            print("could not start the Bonjour listener: \(error)")
            exit(1)
        }

        dispatchMain()
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

/// Handles one worker connection: reads control frames and mirrors them into
/// the cluster state until the peer disconnects.
final class WorkerSession {

    private let connection: FrameConnection
    private let state: ClusterState
    private var deviceID: String?

    init(connection: FrameConnection, state: ClusterState) {
        self.connection = connection
        self.state = state
    }

    func run() {
        receiveNext()
    }

    private func receiveNext() {
        connection.receiveFrame { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure:
                self.disconnect()
            case .success(let frame):
                self.handle(frame)
                self.receiveNext()
            }
        }
    }

    private func handle(_ frame: TensorFrame) {
        guard let message = try? frame.controlMessage() else { return }
        switch message {
        case .hello(let info):
            deviceID = info.id
            print("worker joined: \(info.name) [\(info.platform)/\(info.runtime)]")
            state.apply(message)
        case .goodbye:
            disconnect()
        default:
            state.apply(message)
        }
    }

    private func disconnect() {
        if let deviceID {
            print("worker left: \(deviceID)")
            state.remove(deviceID: deviceID)
        }
        connection.cancel()
    }
}
