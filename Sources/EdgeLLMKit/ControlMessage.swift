import Foundation

/// Control messages exchanged between the coordinator and workers.
///
/// Encoded as JSON inside a `TensorFrame` of kind `.control`, so any worker
/// implementation on any platform can produce and consume them with a plain
/// JSON library. The full wire contract lives in `docs/PROTOCOL.md`.
public enum ControlMessage: Codable, Equatable, Sendable {

    /// Sent by a worker right after connecting.
    case hello(DeviceInfo)

    /// Sent by the coordinator to give a worker its layer range.
    case assignment(ShardAssignment)

    /// Sent by a worker once its assigned shard is loaded and it can serve
    /// inference frames.
    case ready(Ready)

    /// Periodic worker report used by the coordinator and the dashboard.
    case metrics(DeviceMetrics)

    /// Sent by the coordinator before a new generation: the worker clears
    /// its KV caches and starts the sequence from position zero.
    case reset

    /// Sent by either side before an orderly disconnect.
    case goodbye

    public struct DeviceInfo: Codable, Equatable, Sendable {
        /// Stable identifier chosen by the worker, unique within the cluster.
        public var id: String
        /// Human readable name shown in the dashboard.
        public var name: String
        /// Operating system: "macos", "ios", "ipados", "android", "linux", …
        public var platform: String
        /// Inference runtime: "mlx", "llamacpp", …
        public var runtime: String
        /// Physical memory in bytes.
        public var totalMemory: UInt64
        /// Memory the worker is willing to spend on model weights and cache.
        public var usableMemory: UInt64

        public init(
            id: String, name: String, platform: String, runtime: String,
            totalMemory: UInt64, usableMemory: UInt64
        ) {
            self.id = id
            self.name = name
            self.platform = platform
            self.runtime = runtime
            self.totalMemory = totalMemory
            self.usableMemory = usableMemory
        }
    }

    public struct ShardAssignment: Codable, Equatable, Sendable {
        /// Hugging Face style model identifier every worker must have locally.
        public var model: String
        public var layerStart: Int
        public var layerEnd: Int
        public var totalLayers: Int
        /// Worker id of the next stage, empty for the last shard.
        public var nextWorker: String

        public init(
            model: String, layerStart: Int, layerEnd: Int,
            totalLayers: Int, nextWorker: String
        ) {
            self.model = model
            self.layerStart = layerStart
            self.layerEnd = layerEnd
            self.totalLayers = totalLayers
            self.nextWorker = nextWorker
        }
    }

    public struct Ready: Codable, Equatable, Sendable {
        public var id: String

        public init(id: String) {
            self.id = id
        }
    }

    public struct DeviceMetrics: Codable, Equatable, Sendable {
        public var id: String
        /// Memory currently used by weights plus KV cache, in bytes.
        public var usedMemory: UInt64
        /// Average time this worker spends per token, in milliseconds.
        public var millisecondsPerToken: Double

        public init(id: String, usedMemory: UInt64, millisecondsPerToken: Double) {
            self.id = id
            self.usedMemory = usedMemory
            self.millisecondsPerToken = millisecondsPerToken
        }
    }
}

extension TensorFrame {

    /// Wrap a control message for the wire.
    public static func control(_ message: ControlMessage) throws -> TensorFrame {
        let payload = try JSONEncoder().encode(message)
        return TensorFrame(
            kind: .control, dtype: .float16, seqPos: 0,
            layerStart: 0, layerEnd: 0, shape: [], payload: payload)
    }

    /// Decode the control message carried by this frame, if it is one.
    public func controlMessage() throws -> ControlMessage? {
        guard kind == .control else { return nil }
        return try JSONDecoder().decode(ControlMessage.self, from: payload)
    }
}
