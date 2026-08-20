# Cluster wire protocol

This document specifies the complete wire contract between the coordinator and workers. It is intentionally language and platform agnostic: a worker can be written in Swift with MLX, in Kotlin with llama.cpp, in Rust, or anything else that can compute a contiguous range of transformer layers. If you implement this spec, your device can join the cluster.

Protocol version: **1**. See [Versioning](#versioning) for compatibility rules.

## 1. Discovery

The coordinator advertises a DNS-SD (mDNS) service on the local network:

| Field | Value |
|---|---|
| Service type | `_edgellm._tcp` |
| Instance name | human readable coordinator name |
| Port | the coordinator's TCP listening port |

Workers browse for this service type and connect over plain TCP. Every mainstream platform has an mDNS client: Bonjour on Apple platforms, `NsdManager` on Android, Avahi on Linux.

## 2. Message framing

All traffic in both directions is a sequence of length prefixed frames:

```
uint32 little endian: N = byte length of the frame body
N bytes:              frame body (layout below)
```

An implementation must reject frame bodies larger than 256 MiB and treat any malformed frame as a fatal connection error.

### Frame body layout

All integers are little endian.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 4 | magic | ASCII `ELC1` |
| 4 | 1 | version | currently `1` |
| 5 | 1 | kind | `0` hiddenState, `1` token, `2` control |
| 6 | 1 | dtype | `0` float16, `1` float32, `2` bfloat16 |
| 7 | 1 | rank | number of shape dimensions, at most 8 |
| 8 | 4 | seqPos | sequence position of the first element |
| 12 | 2 | layerStart | producing shard's first layer |
| 14 | 2 | layerEnd | producing shard's end layer, exclusive |
| 16 | 4 x rank | shape | one int32 per dimension |
| 16 + 4 x rank | 4 | payloadLength | byte count of the payload |
| next | payloadLength | payload | see per kind rules |

Per kind rules:

- **hiddenState (0)**: payload is the tensor in row major order using the declared dtype. `payloadLength` must equal `product(shape) * bytesPerElement(dtype)`; receivers must verify this. Shape is `[batch, length, hidden]`, with `batch = 1` in version 1.
- **token (1)**: payload is a single int32 token id. `seqPos` is the position the token was sampled at.
- **control (2)**: payload is a UTF-8 JSON document, defined in the next section. Tensor fields (`dtype`, `shape`, `layerStart`, `layerEnd`, `seqPos`) are ignored on control frames.

## 3. Control messages

Control payloads are JSON objects encoding one of the following messages. The encoding matches Swift's `Codable` representation of an enum with associated values: a single key naming the case, whose value holds the parameters under `_0`.

### hello (worker to coordinator, first message after connect)

```json
{ "hello": { "_0": {
  "id": "pixel-8-a1b2",
  "name": "Pixel 8",
  "platform": "android",
  "runtime": "llamacpp",
  "totalMemory": 8589934592,
  "usableMemory": 4294967296
}}}
```

`id` must be stable across reconnects and unique in the cluster. `platform` is a lowercase OS name (`macos`, `ios`, `ipados`, `android`, `linux`). `runtime` names the inference engine (`mlx`, `llamacpp`). Memory values are bytes; `usableMemory` is the quota the worker is willing to spend on weights plus KV cache.

### assignment (coordinator to worker)

```json
{ "assignment": { "_0": {
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "layerStart": 8,
  "layerEnd": 15,
  "totalLayers": 28,
  "nextWorker": "ipad-1"
}}}
```

The worker must load layers `layerStart` to `layerEnd - 1` of the named model. The first shard (`layerStart == 0`) also owns the token embedding; the last shard (`layerEnd == totalLayers`) owns the final norm and the output head. `nextWorker` is the id of the next pipeline stage, empty for the last shard.

### metrics (worker to coordinator, periodic, suggested every 2 to 5 seconds)

```json
{ "metrics": { "_0": {
  "id": "pixel-8-a1b2",
  "usedMemory": 3006477312,
  "millisecondsPerToken": 61.0
}}}
```

### goodbye (either direction, before an orderly disconnect)

```json
{ "goodbye": {} }
```

A worker that stops sending metrics for 10 seconds is shown as offline in the dashboard; the coordinator may drop its connection.

## 4. Pipeline data flow

Once every worker is loaded and ready:

1. The coordinator tokenizes the prompt and sends the first shard a hiddenState frame containing token embeddings, or, when the coordinator itself runs the first shard, starts locally. Version 1 keeps tokenization and sampling on the coordinator side of the chain.
2. Each worker runs its layers and sends the resulting hidden state to the next stage as a hiddenState frame. KV caches stay on the worker; they are never transmitted.
3. The last shard produces logits; the sampled token id returns to the first stage as a token frame, and the next decode step begins.
4. Prefill is one pass with `length = prompt length`; decode steps have `length = 1`.

Numerics: hidden states are exchanged as float16 in version 1. Workers compute in whatever precision their runtime uses internally, but must accept float16 input and emit float16 output at shard boundaries.

## 5. Model availability

Version 1 assumes every worker already has its model weights locally (the same base model, in a format its runtime can load: safetensors for MLX, GGUF for llama.cpp). The coordinator does not stream weights. Mixed runtime clusters must use checkpoints derived from the same base model so that layer semantics line up; quantization may differ per worker.

## 6. Cross platform workers

The reference worker is Swift plus MLX (this repository). A llama.cpp based worker needs three pieces:

1. mDNS browse for `_edgellm._tcp` (`NsdManager` on Android)
2. this framing and control protocol (a few hundred lines in any language)
3. partial layer execution with hidden state input and output, which llama.cpp exposes through its C API

## 7. Versioning

The frame header carries the protocol version. Receivers must reject frames with an unknown version. Breaking changes to the frame layout or to required control fields bump the version; adding new optional JSON fields to control messages does not.

## 8. Security

Version 1 has no authentication or encryption and targets trusted local networks only. See [SECURITY.md](../SECURITY.md). Planned: pre-shared key handshake, then TLS.
