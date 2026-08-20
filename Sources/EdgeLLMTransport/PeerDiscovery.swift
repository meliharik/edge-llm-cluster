import Foundation
import Network

/// Advertises this device as a cluster peer over Bonjour and accepts
/// incoming frame connections.
///
/// iOS/macOS app targets must declare the service in `Info.plist`
/// (`NSBonjourServices: ["_edgellm._tcp"]`) and provide an
/// `NSLocalNetworkUsageDescription` string, or discovery is silently blocked.
public final class PeerAdvertiser {

    private let listener: NWListener
    private let queue = DispatchQueue(label: "edgellm.advertiser")

    public var newConnectionHandler: ((FrameConnection) -> Void)?
    public var stateHandler: ((NWListener.State) -> Void)?

    /// - Parameter name: human-readable peer name shown in discovery
    ///   (device name works well, e.g. "Melih's iPhone").
    public init(name: String) throws {
        let parameters = NWParameters.tcp
        parameters.includePeerToPeer = true
        self.listener = try NWListener(using: parameters)
        listener.service = NWListener.Service(name: name, type: edgeLLMServiceType)

        listener.stateUpdateHandler = { [weak self] state in
            self?.stateHandler?(state)
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            let frameConnection = FrameConnection(connection: connection)
            frameConnection.start()
            self.newConnectionHandler?(frameConnection)
        }
    }

    public func start() {
        listener.start(queue: queue)
    }

    public func cancel() {
        listener.cancel()
    }

    /// The port the listener bound to, once ready.
    public var port: UInt16? {
        listener.port?.rawValue
    }
}

/// Browses the local network for cluster peers.
public final class PeerBrowser {

    public struct Peer: Hashable {
        public let name: String
        public let endpoint: NWEndpoint
    }

    private let browser: NWBrowser
    private let queue = DispatchQueue(label: "edgellm.browser")

    public var peersChangedHandler: (([Peer]) -> Void)?
    public var stateHandler: ((NWBrowser.State) -> Void)?

    public init() {
        let parameters = NWParameters()
        parameters.includePeerToPeer = true
        self.browser = NWBrowser(
            for: .bonjour(type: edgeLLMServiceType, domain: nil), using: parameters)

        browser.stateUpdateHandler = { [weak self] state in
            self?.stateHandler?(state)
        }
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            guard let self else { return }
            let peers = results.compactMap { result -> Peer? in
                guard case .service(let name, _, _, _) = result.endpoint else { return nil }
                return Peer(name: name, endpoint: result.endpoint)
            }
            self.peersChangedHandler?(peers.sorted { $0.name < $1.name })
        }
    }

    public func start() {
        browser.start(queue: queue)
    }

    public func cancel() {
        browser.cancel()
    }

    /// Open a frame connection to a discovered peer.
    public func connect(to peer: Peer) -> FrameConnection {
        let connection = FrameConnection(to: peer.endpoint)
        connection.start()
        return connection
    }
}
