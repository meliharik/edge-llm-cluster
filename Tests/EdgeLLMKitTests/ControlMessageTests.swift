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

    func testResetRoundTrips() throws {
        let frame = try TensorFrame.control(.reset)
        let decoded = try TensorFrame.decode(frame.encoded())
        XCTAssertEqual(try decoded.controlMessage(), .reset)
    }

    func testTokenFrameRoundTrips() throws {
        let frame = TensorFrame(tokens: [151644, 872, 198], seqPos: 0)
        XCTAssertEqual(frame.shape, [1, 3])
        XCTAssertEqual(frame.dtype, .int32)

        let decoded = try TensorFrame.decode(frame.encoded())
        XCTAssertEqual(decoded.tokenIDs(), [151644, 872, 198])
        XCTAssertNil(try decoded.controlMessage())
    }

    func testTokenFrameSizeIsValidated() {
        var frame = TensorFrame(tokens: [1, 2, 3], seqPos: 0)
        frame.shape = [1, 4]
        XCTAssertThrowsError(try TensorFrame.decode(frame.encoded()))
    }

    func testHiddenStateFrameIsNotAControlMessage() throws {
        let frame = TensorFrame(
            kind: .hiddenState, dtype: .float16, seqPos: 0,
            layerStart: 0, layerEnd: 1, shape: [1, 1, 2], payload: Data(count: 4))
        XCTAssertNil(try frame.controlMessage())
    }
}
