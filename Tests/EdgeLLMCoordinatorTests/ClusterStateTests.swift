import XCTest

import EdgeLLMKit

@testable import EdgeLLMCoordinator

final class ClusterStateTests: XCTestCase {

    private func phone(id: String = "phone-1") -> ControlMessage.DeviceInfo {
        .init(
            id: id, name: "Test Phone", platform: "android", runtime: "llamacpp",
            totalMemory: 8 << 30, usableMemory: 4 << 30)
    }

    func testHelloRegistersDevice() {
        let state = ClusterState()
        state.apply(.hello(phone()))

        let snapshot = state.snapshot()
        XCTAssertEqual(snapshot.devices.count, 1)
        XCTAssertEqual(snapshot.devices[0].info.platform, "android")
        XCTAssertNil(snapshot.devices[0].layerStart)
    }

    func testMetricsUpdateExistingDevice() {
        let state = ClusterState()
        state.apply(.hello(phone()))
        state.apply(.metrics(.init(
            id: "phone-1", usedMemory: 1 << 30, millisecondsPerToken: 42)))

        let device = state.snapshot().devices[0]
        XCTAssertEqual(device.usedMemory, 1 << 30)
        XCTAssertEqual(device.millisecondsPerToken, 42)
    }

    func testMetricsForUnknownDeviceAreIgnored() {
        let state = ClusterState()
        state.apply(.metrics(.init(id: "ghost", usedMemory: 1, millisecondsPerToken: 1)))
        XCTAssertTrue(state.snapshot().devices.isEmpty)
    }

    func testAssignmentIsReflectedInSnapshot() {
        let state = ClusterState()
        state.apply(.hello(phone()))
        state.assign(deviceID: "phone-1", spec: ShardSpec(start: 8, end: 15, totalLayers: 28))

        let device = state.snapshot().devices[0]
        XCTAssertEqual(device.layerStart, 8)
        XCTAssertEqual(device.layerEnd, 15)
    }

    func testRemoveDropsDevice() {
        let state = ClusterState()
        state.apply(.hello(phone()))
        state.remove(deviceID: "phone-1")
        XCTAssertTrue(state.snapshot().devices.isEmpty)
    }

    func testSnapshotSortsByName() {
        let state = ClusterState()
        var b = phone(id: "b")
        b.name = "Zed"
        var a = phone(id: "a")
        a.name = "Alpha"
        state.apply(.hello(b))
        state.apply(.hello(a))
        XCTAssertEqual(state.snapshot().devices.map(\.info.name), ["Alpha", "Zed"])
    }
}
