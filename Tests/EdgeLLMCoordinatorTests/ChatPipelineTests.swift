import XCTest

import EdgeLLMKit

@testable import EdgeLLMCoordinator

final class ChatPipelineTests: XCTestCase {

    private func info(_ id: String, gb: UInt64) -> ControlMessage.DeviceInfo {
        .init(
            id: id, name: id, platform: "macos", runtime: "mlx",
            totalMemory: gb << 30, usableMemory: gb << 30)
    }

    func testPlanCoversAllLayersContiguously() {
        let plan = ChatPipeline.plan(
            model: "m", totalLayers: 28,
            infos: [info("a", gb: 4), info("b", gb: 4), info("c", gb: 12)])

        XCTAssertEqual(plan.count, 3)
        XCTAssertEqual(plan.first?.spec.start, 0)
        XCTAssertEqual(plan.last?.spec.end, 28)
        for (previous, next) in zip(plan, plan.dropFirst()) {
            XCTAssertEqual(previous.spec.end, next.spec.start)
        }
        XCTAssertGreaterThan(plan[2].spec.layerCount, plan[0].spec.layerCount)
    }

    func testPlanChainsNextWorkerIds() {
        let plan = ChatPipeline.plan(
            model: "m", totalLayers: 12,
            infos: [info("a", gb: 8), info("b", gb: 8), info("c", gb: 8)])

        XCTAssertEqual(plan.map { $0.assignment.nextWorker }, ["b", "c", ""])
        XCTAssertEqual(plan.map { $0.assignment.totalLayers }, [12, 12, 12])
        XCTAssertEqual(plan[0].assignment.layerStart, plan[0].spec.start)
    }

    func testSingleWorkerPlanOwnsEverything() {
        let plan = ChatPipeline.plan(model: "m", totalLayers: 24, infos: [info("solo", gb: 16)])
        XCTAssertEqual(plan.count, 1)
        XCTAssertTrue(plan[0].spec.isFirst)
        XCTAssertTrue(plan[0].spec.isLast)
        XCTAssertEqual(plan[0].assignment.nextWorker, "")
    }
}
