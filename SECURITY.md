# Security Policy

## Scope & threat model

edge-llm-cluster v0.x is designed for **trusted local networks only**. The transport is intentionally unauthenticated and unencrypted at this stage, so any peer on the LAN advertising the `_edgellm._tcp` service can join a cluster. Do not run it on networks you do not control. Peer authentication (a pre-shared key, then TLS) is on the [roadmap](docs/ROADMAP.md) before any non-demo use.

Reports that "the LAN protocol has no auth" are therefore known-by-design for v0.x. In-scope examples of real issues:

- Memory-safety bugs in `TensorFrame` decoding (malformed frames causing crashes or out-of-bounds reads)
- Path traversal or unexpected file access in `ShardLoader`
- Anything that lets a hostile LAN peer execute code or read files on another peer

## Supported versions

Only the latest commit on `main` is supported.

## Reporting a vulnerability

Please report vulnerabilities privately via [GitHub Security Advisories](https://github.com/meliharik/edge-llm-cluster/security/advisories/new) rather than a public issue. You can expect an initial response within a week. Please include a reproduction (a malformed frame, a crafted snapshot directory, etc.) where possible.
