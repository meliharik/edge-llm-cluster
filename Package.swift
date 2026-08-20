// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "edge-llm-cluster",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "EdgeLLMKit", targets: ["EdgeLLMKit"]),
        .library(name: "EdgeLLMTransport", targets: ["EdgeLLMTransport"]),
        .library(name: "EdgeLLMCoordinator", targets: ["EdgeLLMCoordinator"]),
        .executable(name: "pipeline-proof", targets: ["PipelineProof"]),
        .executable(name: "edge-coordinator", targets: ["EdgeCoordinatorCLI"]),
        .executable(name: "edge-worker", targets: ["EdgeWorkerCLI"]),
    ],
    dependencies: [
        // Pinned exactly: model/shard code mirrors the internals of this release.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.31.4"),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.6")),
        // Downloader + tokenizer implementations consumed via MLXHuggingFace macros.
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "EdgeLLMKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "EdgeLLMTransport",
            dependencies: ["EdgeLLMKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "EdgeLLMCoordinator",
            dependencies: ["EdgeLLMKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "EdgeCoordinatorCLI",
            dependencies: ["EdgeLLMKit", "EdgeLLMTransport", "EdgeLLMCoordinator"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "EdgeWorkerCLI",
            dependencies: [
                "EdgeLLMKit",
                "EdgeLLMTransport",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "PipelineProof",
            dependencies: [
                "EdgeLLMKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "EdgeLLMKitTests",
            dependencies: ["EdgeLLMKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "EdgeLLMCoordinatorTests",
            dependencies: ["EdgeLLMCoordinator", "EdgeLLMKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
