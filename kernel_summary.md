# Your 12 CUDA kernels: a quick guide

Your project makes common model operations more efficient by reducing temporary tensors, GPU memory traffic, and kernel launches. Python provides the API, Rust/PyO3 validates and passes existing GPU pointers, and CUDA performs the computation. These are forward operations; custom backward kernels are not implemented.

## 1. RMSNorm + residual addition

- **What it does:** Adds a residual connection to an activation, then normalizes each row by its root-mean-square magnitude and applies a learned weight. Returns both the normalized result and the residual sum.
- **Formula:** `u = x + residual`; `y = u * rsqrt(mean(u²) + eps) * weight`.
- **Your implementation:** One 256-thread block per row, with FP32 accumulation and warp/shared-memory reductions. It reads inputs twice within one fused launch.
- **Main benefit:** Avoids separate residual and normalization operations and their intermediates. Shared-memory row staging was tried and rejected because it did not improve performance.

## 2. SwiGLU activation

- **What it does:** Applies a smooth activation to the gate branch and multiplies it by the up branch inside a transformer's feed-forward layer.
- **Formula:** `y = SiLU(gate) * up`, where `SiLU(g) = g / (1 + exp(-g))`.
- **Your implementation:** Fuses activation and multiplication, using wide vector loads/stores for aligned data and scalar fallbacks for unaligned inputs or tails.
- **Main benefit:** Reads the two inputs and writes the final output without materializing a separate activation tensor. The surrounding linear projections remain outside this kernel.

## 3. Rotary position embedding (RoPE)

- **What it does:** Rotates pairs of query/key features using position-dependent sine and cosine values, introducing positional information into attention.
- **Formula:** For a pair `(a,b)`, the result is `(a*cos - b*sin, b*cos + a*sin)`.
- **Your implementation:** Rotates both Q and K in one launch using the half-split pairing convention. Supports different query and KV head counts, with vectorized paths and scalar fallbacks.
- **Main benefit:** Combines several rotation operations. An early vectorized version wasted threads; switching its blocks to 32 threads fixed the work mapping. This is a rotation operation, not an attention or KV-cache implementation.

## 4. Linear cross-entropy

- **What it does:** Computes token loss from hidden states and vocabulary projection weights without creating the entire logits matrix at once.
- **Formula:** `loss = logsumexp(logits) - target_logit`, with `logits = hidden @ weight.T`.
- **Your implementation:** PyTorch/cuBLAS computes vocabulary chunks; your CUDA kernel updates a running maximum, exponential sum, and target logit. Rescaling the running sum keeps the computation numerically stable.
- **Main benefit:** Much smaller temporary memory for large vocabularies. It still evaluates the full vocabulary and is **forward-only**, suitable for evaluation loss rather than training through backward.

## 5. Matrix multiplication + bias

- **What it does:** Performs a linear projection and adds bias.
- **Formula:** `y = x @ weight.T + bias`.
- **Your implementation:** Shared-memory tiled GEMM with register blocking: each thread accumulates eight output values. Bias is added before the final write.
- **Main benefit/limitation:** Eliminates a separate bias pass and improves on your initial GEMM. It remains slower than cuBLAS; your implementation does not use tensor-core matrix instructions.

## 6. MoE top-k router

- **What it does:** Selects which experts should process each token and returns their routing weights.
- **Formula:** `probabilities = softmax(logits)`; select the largest `k`, optionally renormalizing their weights to sum to one.
- **Your implementation:** One warp per token, keeping expert values in registers. Warp reductions compute softmax; repeated argmax selects and removes each winner. Supports up to 256 experts.
- **Main benefit:** Avoids a separate full softmax output followed by top-k selection. It handles routing, not expert networks, capacity management, or distributed communication.

## 7. Token gather + weighted combine

- **What it does:** Rearranges token rows for expert processing, then reconstructs original token outputs by combining expert results.
- **Formula:** `gathered[i] = input[indices[i]]`; `out[t] = sum_j weight[t,j] * expert_output[inverse_index[t,j]]`.
- **Your implementation:** Destination-owned gather operations, with vectorized copies and a fused weighted reduction for combine. Each output element has one writer, avoiding atomics.
- **Main benefit:** Combine avoids a large gathered/weighted intermediate. Plain gather performs near PyTorch's `index_select`; the larger opportunity is fusing the combination work.

## 8. Cosine similarity + top-k

- **What it does:** Finds the `k` most similar candidate embeddings for each query through an exhaustive scan.
- **Formula:** `score = dot(query,candidate) / max(norm(query)*norm(candidate), eps)`.
- **Your implementation:** Splits candidates into partitions, computes partial top-k results, then merges them with a second kernel. Supports `k <= 32`.
- **Main benefit/limitation:** Avoids storing the full query-by-candidate score matrix. Partitioning fixed poor GPU utilization for few queries, but the final implementation remains slower than the cuBLAS-backed eager baseline. This is exact search, not an approximate index.

## 9. Squared pairwise distance

- **What it does:** Computes squared Euclidean distance between every row in two embedding matrices.
- **Formula:** `distance²(a,b) = max(norm(a)² + norm(b)² - 2*dot(a,b), 0)`.
- **Your implementation:** Precomputes row norms in PyTorch and uses a tiled CUDA dot-product calculation with a distance epilogue.
- **Main benefit/limitation:** Reuses data across pair comparisons and computes norms once. Performance depends on embedding width; the full distance matrix is still allocated. Clamping negative values does not recover precision lost through cancellation.

## 10. Spatiotemporal graph message passing

- **What it does:** Aggregates weighted neighbor features from the current and previous time steps.
- **Formula:** `out[node] = sum_spatial weight*x_current[source] + sum_temporal weight*x_previous[source]`.
- **Your implementation:** Converts edge lists from COO to destination-indexed CSR. One warp owns a destination node; lanes process feature stripes and accumulate incoming messages.
- **Main benefit:** Each output has one owner, eliminating competing atomic updates. Conversion cost and high-degree nodes still matter. This is an aggregation primitive, not a complete trained GNN.

## 11. Viterbi decoding

- **What it does:** Finds the most likely sequence of hidden states given emissions, transitions, and initial-state probabilities.
- **Formula:** `score_t[s] = emission_t[s] + max_r(score_previous[r] + transition[r,s])`; backpointers reconstruct the path.
- **Your implementation:** One persistent CUDA launch, with one block per sequence and transitions stored in shared memory. Current states run in parallel; timesteps remain sequential inside the kernel.
- **Main benefit:** Removes repeated launches from a Python timestep loop and reuses transition data. It preserves the dynamic-programming complexity; large state counts are constrained by shared memory and block limits.

## 12. Dynamic FP8 quantization

- **What it does:** Converts activations to FP8 and returns scales needed to approximately reconstruct their values.
- **Formula:** `scale = max(max(abs(x))/FP8_MAX, 1e-12)`; `x_fp8 = cast(x/scale)`.
- **Your implementation:** Supports E4M3 and E5M2. Block mode computes one scale per 128×128 tile in one launch; tensor mode uses separate maximum-reduction and casting launches. Packs output bytes for wide stores.
- **Main benefit/limitation:** Smaller element storage and faster fused scaling/casting. It does not implement FP8 matrix multiplication or prove that a quantized classifier preserves quality.

## The easiest way to remember the project

| Group | Kernels | Main idea |
|---|---|---|
| Transformer operations | 1–3 | Fuse normalization, activation, and position rotation |
| Projection and loss | 4–5 | Reduce temporary logits or fuse a GEMM epilogue |
| Mixture of experts | 6–7 | Select experts, rearrange tokens, combine outputs |
| Similarity calculations | 8–9 | Search embeddings or calculate distances |
| Graphs and sequences | 10–11 | Own output reductions or keep iterative work on the GPU |
| Quantization | 12 | Calculate scales and compress values to FP8 |

For the interview, remember three themes: **less memory traffic, fewer intermediate tensors, fewer launches**. Explain both the improvements and the cases where vendor implementations still win.
