import EdgeLLMKit
import Foundation

/// Thread safe registry of every worker known to the coordinator.
///
/// Workers register with a `hello`, refresh with `metrics`, and disappear on
/// `goodbye` or after a silence timeout. The dashboard reads snapshots.
public final class ClusterState: @unchecked Sendable {

    public struct Device: Codable, Equatable, Sendable {
        public var info: ControlMessage.DeviceInfo
        public var layerStart: Int?
        public var layerEnd: Int?
        public var usedMemory: UInt64
        public var millisecondsPerToken: Double
        public var lastSeen: Date

        public init(info: ControlMessage.DeviceInfo, now: Date) {
            self.info = info
            self.usedMemory = 0
            self.millisecondsPerToken = 0
            self.lastSeen = now
        }
    }

    public struct Snapshot: Codable, Equatable, Sendable {
        public var model: String
        public var totalLayers: Int
        public var tokensPerSecond: Double
        public var generatedTokens: Int
        public var devices: [Device]
    }

    /// Workers silent for longer than this are reported as offline.
    public static let offlineAfterSeconds: TimeInterval = 10

    private let lock = NSLock()
    private var devices: [String: Device] = [:]
    private var model = ""
    private var totalLayers = 0
    private var tokensPerSecond = 0.0
    private var generatedTokens = 0

    public init() {}

    public func setModel(_ model: String, totalLayers: Int) {
        lock.lock()
        defer { lock.unlock() }
        self.model = model
        self.totalLayers = totalLayers
    }

    public func setThroughput(tokensPerSecond: Double, generatedTokens: Int) {
        lock.lock()
        defer { lock.unlock() }
        self.tokensPerSecond = tokensPerSecond
        self.generatedTokens = generatedTokens
    }

    /// Apply any incoming control message from a worker.
    public func apply(_ message: ControlMessage, now: Date = Date()) {
        lock.lock()
        defer { lock.unlock() }
        switch message {
        case .hello(let info):
            var device = devices[info.id] ?? Device(info: info, now: now)
            device.info = info
            device.lastSeen = now
            devices[info.id] = device
        case .assignment:
            // Assignments flow coordinator to worker; record them via
            // assign(deviceID:spec:) at send time instead.
            break
        case .metrics(let metrics):
            guard var device = devices[metrics.id] else { return }
            device.usedMemory = metrics.usedMemory
            device.millisecondsPerToken = metrics.millisecondsPerToken
            device.lastSeen = now
            devices[metrics.id] = device
        case .goodbye:
            break
        }
    }

    public func assign(deviceID: String, spec: ShardSpec) {
        lock.lock()
        defer { lock.unlock() }
        guard var device = devices[deviceID] else { return }
        device.layerStart = spec.start
        device.layerEnd = spec.end
        devices[deviceID] = device
    }

    public func remove(deviceID: String) {
        lock.lock()
        defer { lock.unlock() }
        devices[deviceID] = nil
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            model: model,
            totalLayers: totalLayers,
            tokensPerSecond: tokensPerSecond,
            generatedTokens: generatedTokens,
            devices: devices.values.sorted { $0.info.name < $1.info.name })
    }
}
