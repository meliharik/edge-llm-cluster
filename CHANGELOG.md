# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 0: single-machine sharding proof

- `EdgeLLMKit`: shard-aware Qwen2 port (`Qwen2Shard`) that builds an arbitrary contiguous layer range, with embedding on the first shard and norm/LM head on the last (tied-embedding checkpoints supported)
- `ShardLoader`: loads exactly one shard's weights from a Hugging Face MLX snapshot, rebasing checkpoint keys and applying group-wise quantization
- `TensorFrame`: versioned little-endian binary wire format for hidden states, with strict validation and unit tests
- `PipelineGenerator`: greedy decoding across a shard chain with a pluggable shard-boundary link (in-memory, codec round-trip, or — later — a real socket)
- `pipeline-proof` executable: verifies the two-shard pipeline with full serialization matches the unsharded model token for token
- `EdgeLLMTransport`: Bonjour peer discovery (`PeerAdvertiser`, `PeerBrowser`) and length-prefixed TCP frame connections for Phase 1
- Project documentation: architecture deep-dive, phased roadmap, setup guide, CI on macOS runners
