# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

Cluster coordinator and web dashboard:

- `EdgeLLMCoordinator`: thread safe cluster state and a dependency free HTTP server exposing a live dashboard (connected devices, platform and runtime, memory quota and usage, layer assignment, per token latency) plus a JSON status API
- `edge-coordinator` executable: advertises the cluster service, accepts workers from any platform, serves the dashboard; `--demo` seeds simulated devices for a quick look
- `ControlMessage`: JSON control protocol (hello, assignment, metrics, goodbye) carried in control frames, usable from any language
- `docs/PROTOCOL.md`: full language agnostic wire specification so workers on Android, Linux or anything else can join the cluster

Phase 0, single machine sharding proof:

- `EdgeLLMKit`: shard aware Qwen2 port (`Qwen2Shard`) that builds an arbitrary contiguous layer range, with embedding on the first shard and norm plus LM head on the last (tied embedding checkpoints supported)
- `ShardLoader`: loads exactly one shard's weights from a Hugging Face MLX snapshot, rebasing checkpoint keys and applying group wise quantization
- `TensorFrame`: versioned little endian binary wire format for hidden states, with strict validation and unit tests
- `PipelineGenerator`: greedy decoding across a shard chain with a pluggable shard boundary link (in memory, codec round trip, or a real socket later)
- `pipeline-proof` executable: verifies the two shard pipeline with full serialization matches the unsharded model token for token
- `EdgeLLMTransport`: mDNS peer discovery and length prefixed TCP frame connections
- Project documentation: architecture deep dive, phased roadmap, setup guide, CI on macOS runners
