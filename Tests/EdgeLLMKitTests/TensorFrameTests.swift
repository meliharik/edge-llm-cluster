import XCTest

@testable import EdgeLLMKit

final class TensorFrameTests: XCTestCase {

    private func makeHiddenFrame() -> TensorFrame {
        // [1, 1, 8] float16 payload: 16 bytes of recognizable content.
        let payload = Data((0 ..< 16).map { UInt8($0) })
        return TensorFrame(
            kind: .hiddenState, dtype: .float16, seqPos: 42,
            layerStart: 0, layerEnd: 12, shape: [1, 1, 8], payload: payload)
    }

    func testRoundTripPreservesEverything() throws {
        let original = makeHiddenFrame()
        let decoded = try TensorFrame.decode(original.encoded())
        XCTAssertEqual(decoded, original)
    }

    func testEncodedSizeMatchesLayout() {
        let frame = makeHiddenFrame()
        // 16 fixed header bytes + 3 dims * 4 + 4 payload-count + 16 payload.
        XCTAssertEqual(frame.encoded().count, 16 + 12 + 4 + 16)
    }

    func testBadMagicIsRejected() {
        var data = makeHiddenFrame().encoded()
        data[0] = UInt8(ascii: "X")
        XCTAssertThrowsError(try TensorFrame.decode(data)) { error in
            XCTAssertEqual(error as? TensorFrame.DecodeError, .badMagic)
        }
    }

    func testUnsupportedVersionIsRejected() {
        var data = makeHiddenFrame().encoded()
        data[4] = 99
        XCTAssertThrowsError(try TensorFrame.decode(data)) { error in
            XCTAssertEqual(error as? TensorFrame.DecodeError, .unsupportedVersion(99))
        }
    }

    func testTruncatedPayloadIsRejected() {
        let data = makeHiddenFrame().encoded().dropLast(4)
        XCTAssertThrowsError(try TensorFrame.decode(Data(data))) { error in
            XCTAssertEqual(error as? TensorFrame.DecodeError, .truncated)
        }
    }

    func testPayloadShapeMismatchIsRejected() {
        // Claims [1, 1, 9] float16 (18 bytes) but carries 16 bytes.
        let payload = Data(count: 16)
        let frame = TensorFrame(
            kind: .hiddenState, dtype: .float16, seqPos: 0,
            layerStart: 0, layerEnd: 1, shape: [1, 1, 9], payload: payload)
        XCTAssertThrowsError(try TensorFrame.decode(frame.encoded())) { error in
            XCTAssertEqual(
                error as? TensorFrame.DecodeError,
                .payloadSizeMismatch(expected: 18, actual: 16))
        }
    }

    func testDecodeSurvivesSlicedData() throws {
        // Data slices keep their parent's indices; the reader must not
        // assume a zero start index.
        let original = makeHiddenFrame()
        var padded = Data([0xAA, 0xBB, 0xCC])
        padded.append(original.encoded())
        let slice = padded.dropFirst(3)
        let decoded = try TensorFrame.decode(Data(slice))
        XCTAssertEqual(decoded, original)

        // Also decode the raw slice without re-wrapping.
        let decodedFromSlice = try TensorFrame.decode(slice)
        XCTAssertEqual(decodedFromSlice, original)
    }
}
