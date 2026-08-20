import XCTest

@testable import EdgeLLMKit

final class ControlMessageTests: XCTestCase {

    func testHelloRoundTripsThroughFrame() throws {
        let hello = ControlMessage.hello(.init(
            id: "pixel-8", name: "Pixel 8", platform: "android", runtime: "llamacpp",
            totalMemory: 8 << 30, usableMemory: 4 << 30))

        let frame = try TensorFrame.control(hello)
        XCTAssertEqual(frame.kind, .control)

        let decoded = try TensorFrame.decode(frame.encoded())
        XCTAssertEqual(try decoded.controlMessage(), hello)
    }

    func testAssignmentRoundTrips() throws {
        let assignment = ControlMessage.assignment(.init(
            model: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            layerStart: 8, layerEnd: 15, totalLayers: 28, nextWorker: "ipad-1"))
        let frame = try TensorFrame.control(assignment)
        let decoded = try TensorFrame.decode(frame.encoded())
        XCTAssertEqual(try decoded.controlMessage(), assignment)
    }

    func testHiddenStateFrameIsNotAControlMessage() throws {
        let frame = TensorFrame(
            kind: .hiddenState, dtype: .float16, seqPos: 0,
            layerStart: 0, layerEnd: 1, shape: [1, 1, 2], payload: Data(count: 4))
        XCTAssertNil(try frame.controlMessage())
    }
}
