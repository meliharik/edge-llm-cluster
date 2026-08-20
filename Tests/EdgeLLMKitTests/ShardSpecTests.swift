import XCTest

@testable import EdgeLLMKit

final class ShardSpecTests: XCTestCase {

    func testFullShardOwnsEverything() {
        let spec = ShardSpec.full(24)
        XCTAssertTrue(spec.isFirst)
        XCTAssertTrue(spec.isLast)
        XCTAssertEqual(spec.layerCount, 24)
    }

    func testRolesForMiddleShard() {
        let spec = ShardSpec(start: 8, end: 16, totalLayers: 24)
        XCTAssertFalse(spec.isFirst)
        XCTAssertFalse(spec.isLast)
        XCTAssertEqual(spec.layerCount, 8)
    }

    func testEqualSplitCoversAllLayersContiguously() {
        let shards = ShardSpec.split(totalLayers: 24, weights: [1, 1, 1])
        XCTAssertEqual(shards.map(\.layerCount), [8, 8, 8])
        assertContiguous(shards, totalLayers: 24)
    }

    func testProportionalSplitFollowsWeights() {
        // 8 GB phone, 8 GB tablet, 16 GB laptop.
        let shards = ShardSpec.split(totalLayers: 36, weights: [8, 8, 16])
        XCTAssertEqual(shards.map(\.layerCount).reduce(0, +), 36)
        XCTAssertGreaterThan(shards[2].layerCount, shards[0].layerCount)
        assertContiguous(shards, totalLayers: 36)
    }

    func testEveryShardGetsAtLeastOneLayer() {
        let shards = ShardSpec.split(totalLayers: 4, weights: [1000, 1, 1, 1])
        XCTAssertEqual(shards.map(\.layerCount), [1, 1, 1, 1])
        assertContiguous(shards, totalLayers: 4)
    }

    func testUnevenRemainderIsFullyAssigned() {
        let shards = ShardSpec.split(totalLayers: 28, weights: [1, 1, 1])
        XCTAssertEqual(shards.map(\.layerCount).reduce(0, +), 28)
        assertContiguous(shards, totalLayers: 28)
    }

    private func assertContiguous(_ shards: [ShardSpec], totalLayers: Int) {
        XCTAssertEqual(shards.first?.start, 0)
        XCTAssertEqual(shards.last?.end, totalLayers)
        for (previous, next) in zip(shards, shards.dropFirst()) {
            XCTAssertEqual(previous.end, next.start)
        }
    }
}
