import Foundation
import Network

/// Minimal HTTP server that exposes the cluster dashboard.
///
/// Routes:
///   GET /            the dashboard page (static HTML, polls the API)
///   GET /api/status  cluster snapshot as JSON
///
/// Serves one response per connection and closes it, which keeps the server
/// tiny and dependency free. The dashboard polls once per second, so the
/// connection churn is negligible on a LAN.
public final class StatusServer {

    private let listener: NWListener
    private let state: ClusterState
    private let queue = DispatchQueue(label: "edgellm.status")
    private let encoder: JSONEncoder

    public init(state: ClusterState, port: UInt16 = 8080) throws {
        self.state = state
        self.listener = try NWListener(
            using: .tcp, on: NWEndpoint.Port(rawValue: port) ?? 8080)
        self.encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .secondsSince1970

        listener.newConnectionHandler = { [weak self] connection in
            self?.handle(connection)
        }
    }

    public func start() {
        listener.start(queue: queue)
    }

    public func cancel() {
        listener.cancel()
    }

    public var port: UInt16? {
        listener.port?.rawValue
    }

    // MARK: - Request handling

    private func handle(_ connection: NWConnection) {
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16 * 1024) {
            [weak self] data, _, _, error in
            guard let self, error == nil, let data,
                  let request = String(data: data, encoding: .utf8)
            else {
                connection.cancel()
                return
            }
            let response = self.response(for: Self.requestPath(request))
            connection.send(
                content: response,
                completion: .contentProcessed { _ in connection.cancel() })
        }
    }

    /// Extract the path from an HTTP request line like "GET /api/status HTTP/1.1".
    static func requestPath(_ request: String) -> String {
        let firstLine = request.split(separator: "\r\n", maxSplits: 1)[0]
        let parts = firstLine.split(separator: " ")
        guard parts.count >= 2, parts[0] == "GET" else { return "" }
        return String(parts[1])
    }

    private func response(for path: String) -> Data {
        switch path {
        case "/", "/index.html":
            return Self.httpResponse(
                status: "200 OK", contentType: "text/html; charset=utf-8",
                body: Data(DashboardPage.html.utf8))
        case "/api/status":
            let body = (try? encoder.encode(state.snapshot())) ?? Data("{}".utf8)
            return Self.httpResponse(
                status: "200 OK", contentType: "application/json", body: body)
        case "":
            return Self.httpResponse(
                status: "405 Method Not Allowed", contentType: "text/plain",
                body: Data("only GET is supported\n".utf8))
        default:
            return Self.httpResponse(
                status: "404 Not Found", contentType: "text/plain",
                body: Data("not found\n".utf8))
        }
    }

    static func httpResponse(status: String, contentType: String, body: Data) -> Data {
        var head = "HTTP/1.1 \(status)\r\n"
        head += "Content-Type: \(contentType)\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Cache-Control: no-store\r\n"
        head += "Connection: close\r\n\r\n"
        var response = Data(head.utf8)
        response.append(body)
        return response
    }
}
