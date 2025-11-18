// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VectorIndex",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .watchOS(.v10),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "VectorIndex",
            targets: ["VectorIndex"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/gifton/VectorCore", from: "0.1.2")
    ],
    targets: [
        .target(
            name: "CAtomicsShim",
            publicHeadersPath: "include"
        ),
        .target(
            name: "CPQEncode",
            publicHeadersPath: "include"
        ),
        .target(
            name: "CS2RNG",
            publicHeadersPath: "include",
            cSettings: [
                .define("S2_ENABLE_TELEMETRY", to: "1")
            ]
        ),
        .target(
            name: "VectorIndex",
            dependencies: [
                "CAtomicsShim",
                "CPQEncode",
                "CS2RNG",
                .product(name: "VectorCore", package: "VectorCore")
            ],
            exclude: [
                // Exclude scratch files relative to Sources/VectorIndex
                // Note: residual kernel docs moved to /docs; no longer under Sources.
                "Kernels/PQTrain.swift.new"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ]
        ),
        .executableTarget(
            name: "L2SqrMicrobench",
            dependencies: [
                "VectorIndex",
                .product(name: "VectorCore", package: "VectorCore")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ]
        ),
        .executableTarget(
            name: "VectorIndexBenchmarks",
            dependencies: [
                "VectorIndex",
                .product(name: "VectorCore", package: "VectorCore")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ]
        ),
        .testTarget(
            name: "VectorIndexTests",
            dependencies: ["VectorIndex"],
            exclude: [
                // Exclude temporary scratch tests (relative to Tests/VectorIndexTests)
                "PQTrainTests.swift.tmp"
            ],
            swiftSettings: [ .enableExperimentalFeature("StrictConcurrency") ]
        ),
    ]
)
