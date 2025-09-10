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
        .package(path: "../VectorCore")
    ],
    targets: [
        .target(
            name: "VectorIndex",
            dependencies: [
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
            swiftSettings: [ .enableExperimentalFeature("StrictConcurrency") ]
        ),
    ]
)
