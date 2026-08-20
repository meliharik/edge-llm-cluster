import Foundation
import MLX

// Bridging between MLXArray hidden states and the wire format. Hidden states
// travel as float16, so for a 3B model that is hidden_dim (2048) x 2 bytes,
// about 4 KB per generated token.

extension TensorFrame {

    public enum TensorError: Error {
        case unsupportedDType(DType)
    }

    /// Wrap a hidden-state tensor produced by `spec` at sequence position `seqPos`.
    public init(hiddenState: MLXArray, seqPos: Int, producedBy spec: ShardSpec) {
        let converted = hiddenState.asType(.float16)
        let elements: [Float16] = converted.asArray(Float16.self)
        let payload = elements.withUnsafeBytes { Data($0) }
        self.init(
            kind: .hiddenState,
            dtype: .float16,
            seqPos: seqPos,
            layerStart: spec.start,
            layerEnd: spec.end,
            shape: converted.shape,
            payload: payload)
    }

    /// Reconstruct the hidden-state tensor carried by this frame.
    public func hiddenState() throws -> MLXArray {
        guard dtype == .float16 else { throw TensorError.unsupportedDType(dtype) }
        let elements = payload.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float16.self))
        }
        return MLXArray(elements).reshaped(shape)
    }
}
