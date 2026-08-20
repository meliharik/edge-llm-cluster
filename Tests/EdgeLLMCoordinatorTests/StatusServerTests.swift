import XCTest

@testable import EdgeLLMCoordinator

final class StatusServerTests: XCTestCase {

    func testRequestPathParsing() {
        XCTAssertEqual(
            StatusServer.requestPath("GET /api/status HTTP/1.1\r\nHost: x\r\n\r\n"),
            "/api/status")
        XCTAssertEqual(StatusServer.requestPath("GET / HTTP/1.1\r\n\r\n"), "/")
    }

    func testNonGetRequestsYieldEmptyPath() {
        XCTAssertEqual(StatusServer.requestPath("POST /api/status HTTP/1.1\r\n\r\n"), "")
        XCTAssertEqual(StatusServer.requestPath("garbage"), "")
    }

    func testHttpResponseLayout() {
        let body = Data("hello".utf8)
        let response = StatusServer.httpResponse(
            status: "200 OK", contentType: "text/plain", body: body)
        let text = String(data: response, encoding: .utf8)!

        XCTAssertTrue(text.hasPrefix("HTTP/1.1 200 OK\r\n"))
        XCTAssertTrue(text.contains("Content-Length: 5\r\n"))
        XCTAssertTrue(text.hasSuffix("\r\n\r\nhello"))
    }
}
