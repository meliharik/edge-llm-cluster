import EdgeLLMKit
import Foundation
import Network

/// The Bonjour service type all cluster peers advertise and browse.
public let edgeLLMServiceType = "_edgellm._tcp"

/// A TCP connection that speaks length-prefixed `TensorFrame`s.
///
/// Wire layout per message: `UInt32` little-endian frame byte count, followed
/// by the encoded frame (see `TensorFrame` for the frame layout).
public final class FrameConnection {

    public enum ConnectionError: Error {
        case cancelled
        case oversizedFrame(UInt32)
    }

    /// Upper bound on a single frame: a full prefill hidden state for long
    /// prompts stays well under this; anything larger is a protocol error.
    public static let maxFrameBytes: UInt32 = 256 * 1024 * 1024

    private let connection: NWConnection
    private let queue: DispatchQueue

    public var stateHandler: ((NWConnection.State) -> Void)? {
        get { connection.stateUpdateHandler }
        set { connection.stateUpdateHandler = newValue }
    }

    /// Wrap an accepted connection (listener side).
    public init(connection: NWConnection, queue: DispatchQueue = .init(label: "edgellm.frame")) {
        self.connection = connection
        self.queue = queue
    }

    /// Dial a discovered peer (browser side).
    public convenience init(
        to endpoint: NWEndpoint, queue: DispatchQueue = .init(label: "edgellm.frame")
    ) {
        self.init(connection: NWConnection(to: endpoint, using: .tcp), queue: queue)
    }

    public func start() {
        connection.start(queue: queue)
    }

    public func cancel() {
        connection.cancel()
    }

    // MARK: - Sending

    public func send(_ frame: TensorFrame, completion: @escaping (Error?) -> Void) {
        let body = frame.encoded()
        var message = Data(capacity: 4 + body.count)
        var length = UInt32(body.count).littleEndian
        withUnsafeBytes(of: &length) { message.append(contentsOf: $0) }
        message.append(body)

        connection.send(
            content: message,
            completion: .contentProcessed { error in
                completion(error)
            })
    }

    public func send(_ frame: TensorFrame) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            send(frame) { error in
                if let error { cont.resume(throwing: error) } else { cont.resume() }
            }
        }
    }

    // MARK: - Receiving

    /// Receive exactly one frame.
    public func receiveFrame(completion: @escaping (Result<TensorFrame, Error>) -> Void) {
        receiveExactly(4) { [weak self] result in
            guard let self else { return completion(.failure(ConnectionError.cancelled)) }
            switch result {
            case .failure(let error):
                completion(.failure(error))
            case .success(let header):
                let length = header.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
                    .littleEndian
                guard length <= Self.maxFrameBytes else {
                    return completion(.failure(ConnectionError.oversizedFrame(length)))
                }
                self.receiveExactly(Int(length)) { result in
                    switch result {
                    case .failure(let error):
                        completion(.failure(error))
                    case .success(let body):
                        do {
                            completion(.success(try TensorFrame.decode(body)))
                        } catch {
                            completion(.failure(error))
                        }
                    }
                }
            }
        }
    }

    public func receiveFrame() async throws -> TensorFrame {
        try await withCheckedThrowingContinuation { cont in
            receiveFrame { cont.resume(with: $0) }
        }
    }

    private func receiveExactly(_ count: Int, completion: @escaping (Result<Data, Error>) -> Void) {
        connection.receive(
            minimumIncompleteLength: count, maximumLength: count
        ) { data, _, isComplete, error in
            if let error {
                completion(.failure(error))
            } else if let data, data.count == count {
                completion(.success(data))
            } else {
                completion(.failure(ConnectionError.cancelled))
            }
        }
    }
}
