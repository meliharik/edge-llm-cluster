import EdgeLLMKit
import EdgeLLMTransport
import Foundation

/// One connected worker as seen by the coordinator.
///
/// Owns the connection's receive loop and splits traffic in two: control
/// frames (hello, ready, metrics, goodbye) update the cluster state, while
/// tensor frames answer the inference request in flight. The chat loop talks
/// to workers exclusively through `request` and `sendControl`.
public final class WorkerHandle {

    public enum HandleError: Error, CustomStringConvertible {
        case disconnected
        case busy

        public var description: String {
            switch self {
            case .disconnected: return "worker disconnected"
            case .busy: return "worker already has a request in flight"
            }
        }
    }

    private let connection: FrameConnection
    private let state: ClusterState
    private let lock = NSLock()
    private var pending: CheckedContinuation<TensorFrame, Error>?
    private var closed = false

    /// Set once the worker's hello arrives.
    public private(set) var info: ControlMessage.DeviceInfo?

    /// Called after the worker introduces itself.
    public var onHello: ((WorkerHandle) -> Void)?
    /// Called once, when the connection ends for any reason.
    public var onClose: ((WorkerHandle) -> Void)?

    public init(connection: FrameConnection, state: ClusterState) {
        self.connection = connection
        self.state = state
    }

    public func start() {
        receiveNext()
    }

    /// Send an inference frame and await the worker's tensor reply.
    /// One request may be in flight at a time.
    public func request(_ frame: TensorFrame) async throws -> TensorFrame {
        try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            if closed {
                lock.unlock()
                continuation.resume(throwing: HandleError.disconnected)
                return
            }
            guard pending == nil else {
                lock.unlock()
                continuation.resume(throwing: HandleError.busy)
                return
            }
            pending = continuation
            lock.unlock()

            connection.send(frame) { [weak self] error in
                if error != nil { self?.close() }
            }
        }
    }

    public func sendControl(_ message: ControlMessage) throws {
        let frame = try TensorFrame.control(message)
        connection.send(frame) { [weak self] error in
            if error != nil { self?.close() }
        }
    }

    public func close() {
        lock.lock()
        let alreadyClosed = closed
        closed = true
        lock.unlock()
        guard !alreadyClosed else { return }

        takePending()?.resume(throwing: HandleError.disconnected)
        if let info {
            state.remove(deviceID: info.id)
        }
        connection.cancel()
        onClose?(self)
    }

    // MARK: - Receive loop

    private func receiveNext() {
        connection.receiveFrame { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure:
                self.close()
            case .success(let frame):
                self.handle(frame)
                self.receiveNext()
            }
        }
    }

    private func handle(_ frame: TensorFrame) {
        guard frame.kind == .control else {
            takePending()?.resume(returning: frame)
            return
        }
        guard let message = try? frame.controlMessage() else { return }
        switch message {
        case .hello(let deviceInfo):
            info = deviceInfo
            state.apply(message)
            onHello?(self)
        case .ready, .metrics:
            state.apply(message)
        case .goodbye:
            close()
        case .assignment, .reset:
            break
        }
    }

    private func takePending() -> CheckedContinuation<TensorFrame, Error>? {
        lock.lock()
        defer { lock.unlock() }
        let taken = pending
        pending = nil
        return taken
    }
}
