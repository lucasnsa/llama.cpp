# TurboQuant KV cache integration

Status: **PENDING USER VALIDATION** for compilation, regression execution, model inference, numerical correctness, benchmarks, and GPU compatibility of the current branch. The user supplied successful RTX 3070 results for the earlier `ef52091d3` build, recorded below; those results do not validate the later TURBO2 or prefill changes. Development performed source inspection and diff checks only. No builds, CMake, tests, project binaries, CUDA code, inference, or source generators were executed.

## Scope and architecture

This is a Bonsai-first integration for a personal PrismML fork. The implementation starts at PrismML `e3de715222b2667c54a267441159a5729f11155e`. It keeps the model loader, Bonsai weight representations, and existing CPU/CUDA weight kernels as the source of truth.

Reference: [jarkevithwlad/turboquant-prismml-cuda](https://github.com/jarkevithwlad/turboquant-prismml-cuda), pinned to `f7d1a4381104ed72cca090f8f0b56633fbebd620`. Relevant history includes `45a2ef747c3b8bb56bd44787ae39af1d59508519` (reference rebase), `1bc46807d98df97d94ce37a7934e70a314c72eac` (normalization and V rotation corrections), and `0bfe33411f7af381964510d083a0d8ee047201ef` (external/internal rotation interaction). These are provenance references, not commits merged into this branch. The reference's attention-sharpening heuristic, AMX types, other head dimensions, FP8 type remapping, and unrelated server changes are excluded.

The 2-bit addition is on local branch `feature/turboquant-2bit-kv-cache`, descended from `feature/turboquant-kv-cache` at `ef52091d3f54cece74b07115cdec9a27742f106c`. It preserves the earlier 3-bit and 4-bit integration. Neither this addition nor its validation commands require merging or synchronizing upstream.

The prefill changes are on local branch `feature/turboquant-prefill-cuda`, descended from the 2-bit branch at `57e1458b35a18be2307fc407bd79f8d007e20868`. The commits are local only. The codecs, type IDs, weight kernels, server API, and compressed cache representation are unchanged by these optimizations.

Cache writes use the existing CUDA SET_ROWS interface and its per-block writer dispatch. TBQ3, TBQ4, and TBQP3 use 256-element blocks with deterministic sign flips, a Walsh-Hadamard transform, and Lloyd-Max scalar quantization. TBQP3 stores a 2-bit centroid index and a 1-bit QJL residual projection. Shared internal codec routines keep the CPU reference and CUDA writer tables and packing consistent.

Attention uses the existing vector Flash Attention kernel for both prefill and decode. It transforms Q once per query, evaluates compressed K directly, accumulates V in the transformed domain, and applies an inverse transform to the result. TBQP3 additionally transforms Q for its QJL correction. Normalization matches the reference: the centroid query uses `scale/256`, the QJL query uses `scale*1.2533/(256*256)`, and the V output uses `1/256`. The scalar TBQP3 decoder reconstructs this same linear estimator.

### Pure 2-bit source and encoding

The pinned PrismML reference contains no pure 2-bit KV type. AmesianX's inspected reference snapshot `570f5e790b12923a2b9a454a7a4f4637239ac772` also has no suitable pure 2-bit type, and its README marks that project discontinued. Official llama.cpp snapshot `fd55cb98fe29d3df0054d3f3e6e6f15958ea72ab` contains no native TurboQuant type. PQ2_0 remains a Bonsai weight format, not a TurboQuant cache alias. Sources: [PrismML types](https://github.com/jarkevithwlad/turboquant-prismml-cuda/blob/f7d1a4381104ed72cca090f8f0b56633fbebd620/ggml/include/ggml.h), [AmesianX](https://github.com/AmesianX/TurboQuant/tree/570f5e790b12923a2b9a454a7a4f4637239ac772), [official llama.cpp types](https://github.com/ggml-org/llama.cpp/blob/fd55cb98fe29d3df0054d3f3e6e6f15958ea72ab/ggml/include/ggml.h).

The usable source is [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant/tree/ca28b4bc0e531635d83fe530c6e6e8d694c4330d), pinned to `ca28b4bc0e531635d83fe530c6e6e8d694c4330d`. That fork was receiving updates at inspection time; this does not establish Bonsai accuracy or GPU compatibility. Its actual `QK_TURBO2=128` structure is 34 bytes, despite stale comments describing an older 32-value/10-byte layout. See [block definitions](https://github.com/TheTom/llama-cpp-turboquant/blob/ca28b4bc0e531635d83fe530c6e6e8d694c4330d/ggml/src/ggml-common.h) and [scalar codecs](https://github.com/TheTom/llama-cpp-turboquant/blob/ca28b4bc0e531635d83fe530c6e6e8d694c4330d/ggml/src/ggml-turbo-quant.c).

TURBO2_0 is a 2-bit PolarQuant codec in that TurboQuant fork. Each group stores one FP16 corrected norm followed by 32 bytes of indices, with four indices per byte and the first index in the lowest two bits. The four centroids are `{-0.133462, -0.039994, 0.039994, 0.133462}`; thresholds are `{-0.086728, 0, 0.086728}` with ties selecting the higher index. Forward rotation is `S2 * H128 * S1 / sqrt(128)`, using the source's fixed signs. The stored norm is the original L2 norm divided by the norm of the selected centroid vector. No residual or QJL bits are present. A zero group has norm zero and index bytes `0xaa`.

For a 256-dimensional head, TURBO2 uses two independent groups. Query rotation follows K's type, while inverse V rotation follows V's type, so mixed 128-group/256-group pairs remain valid. The basic source decoder emits rotated coordinates; this port's scalar `to_float` also applies the inverse rotation to provide spatial F32 references. The compressed bytes and corrected norm retain the source representation. CPU inference remains unsupported. No global rotation-group setting, padding to other head dimensions, InnerQ calibration, adaptive layers, or additional GGML operator is imported.

The source has CUDA K and V support, including a [256-head native instance](https://github.com/TheTom/llama-cpp-turboquant/blob/ca28b4bc0e531635d83fe530c6e6e8d694c4330d/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-turbo2_0.cu). This port reuses PrismML's existing CUDA writer dispatch and vector attention rather than importing the fork's streaming and matrix-attention subsystems. Writer throughput and prefill performance are unmeasured.

External KV rotation is disabled only for TurboQuant caches. The separate Bonsai weight Hadamard transformations are preserved. Existing attention scaling, masking, softmax, and server endpoints are unchanged. TurboQuant is limited to attention KV state; Bonsai's recurrent state is unchanged.

The context checks cache pair, placement, head dimensions, and compiled CUDA capability before allocating cache buffers. The CUDA attention selector and scratch allocator use the same native pair dispatch. Unsupported TurboQuant operations do not fall back to full-cache F16 conversion or CPU attention.

## Tensor types and supported configurations

| Type | ID | Elements/block | Bytes/block | Bits/value |
|---|---:|---:|---:|---:|
| Q1_0 | 41 | unchanged | unchanged | unchanged |
| Q2_0 | 42 | 64 | unchanged | unchanged |
| PQ2_0 | 142 | 128 | 34 | 2.125 |
| PTQ1_0 | 143 | 128 | 28 | 1.75 |
| TBQ3_0 | 144 | 256 | 98 | 3.0625 |
| TBQ4_0 | 145 | 256 | 130 | 4.0625 |
| TBQP3_0 | 146 | 256 | 100 | 3.125 |
| TURBO2_0 | 147 | 128 | 34 | 2.125 |

`GGML_TYPE_COUNT` is 148. No existing GGUF tensor or model file-type identifiers are reassigned. Rebuild the libraries and tools together after changing the enum. Neither reference's type IDs nor Q2_0 loader remapping are used. The model loader and model quantizer reject TurboQuant weight types.

| K argument | V arguments |
|---|---|
| `turbo2` or `turbo2_0` | `turbo2`, `turbo2_0`, `tbq3`, `tbq3_0`, `tbq4`, `tbq4_0` |
| `tbq3` or `tbq3_0` | `turbo2`, `turbo2_0`, `tbq3`, `tbq3_0`, `tbq4`, `tbq4_0` |
| `tbq4` or `tbq4_0` | `turbo2`, `turbo2_0`, `tbq3`, `tbq3_0`, `tbq4`, `tbq4_0` |
| `tbqp3` or `tbqp3_0` | `turbo2`, `turbo2_0`, `tbq3`, `tbq3_0`, `tbq4`, `tbq4_0` |

Requirements: NVIDIA CUDA backend, Flash Attention enabled, 256-dimensional K and V attention heads, and CUDA KV buffers for every attention layer. `-fa auto` enables FA through the existing quantized-V rule; use `-fa on` for validation. CPU codecs are provided for serialization and numerical references, not CPU TurboQuant inference.

Not supported: TBQP3 V, mixed TurboQuant/conventional cache pairs, other head dimensions, MLA cache layouts, non-CUDA KV placement, and context shifting. The normal memory capability API reports shifting unavailable, allowing the server to disable it. Cache reuse without position changes remains available. Existing conventional cache types keep their behavior. TurboQuant state files are specific to this fork's type mapping.

The reference-style vector path can be slower than the existing MMA attention path, especially during prefill. Memory savings, output quality, and speed on the RTX 3070 are unmeasured. A 32K cache fitting in 8 GB cannot be guaranteed before measuring the complete model, cache, and compute buffers.

### Memory accounting and 2-bit accuracy risks

"Pure 2-bit" refers to the index payload, not total storage: TURBO2 uses 2.125 bits/value including its FP16 norm. TBQP3 uses 2-bit indices plus a 1-bit QJL correction and two FP16 norms, totaling 3.125 bits/value. TBQP3 is not a substitute for TURBO2.

| K/V pair | Bytes per 256-dimensional K/V head pair |
|---|---:|
| F16/F16 | 1024 |
| Q4_1/Q4_1 | 320 |
| TBQ4/TBQ4 | 260 |
| TBQ3/TBQ3 | 196 |
| TBQP3/TBQ3 | 198 |
| TURBO2/TBQ3 | 166 |
| TURBO2/TURBO2 | 136 |

Uniform TURBO2 reduces KV payload by 47.7% versus TBQ4/TBQ4, 30.6% versus TBQ3/TBQ3, 57.5% versus Q4_1/Q4_1, and 86.7% versus F16/F16. TURBO2/TBQ3 saves 36.2% versus TBQ4/TBQ4. Multiply the bytes above by cached tokens, KV heads, and attention layers with KV state. The count must come from the actual model. Weights, recurrent state, compute buffers, alignment, and allocator overhead are separate; these are not total VRAM savings or a 32K fit guarantee.

Uniform 2-bit V is experimental. The source fork normally substitutes Q8 V in boundary layers for turbo2-V on sufficiently deep models; see its [layer policy](https://github.com/TheTom/llama-cpp-turboquant/blob/ca28b4bc0e531635d83fe530c6e6e8d694c4330d/src/llama-kv-cache.cpp). At the user's request, this port makes no hidden precision substitutions: selecting `turbo2/turbo2` uses TURBO2 for every attention layer's K and V. Existing runtime cache defaults remain unchanged.

Norm correction preserves reconstructed magnitude, not direction or exact attention logits. TURBO2 K has no QJL residual correction. Lower precision can affect retrieval and attention ordering; V error can alter the information delivered to later layers. Long coding sessions may expose wrong identifiers, lost distant constraints, inconsistent multi-file edits, or repetition. These are risks to evaluate, not measured Bonsai outcomes. Begin with `turbo2/tbq3` and compare `turbo2/turbo2` separately. Keep TBQ3/TBQ4 and the Q4_1 baseline available. A decoded-byte F32 attention match establishes kernel consistency, not model quality.

## Manual compilation on Debian 13

All commands below are for the user's compilation machine only. No dependencies are installed by this guide. Have CUDA 12.8, a CUDA-supported host compiler, CMake, and the dependencies in `docs/build.md` available. These commits are local and have not been pushed; first transfer the branch using your chosen Git workflow.

Run from the repository root in Bash. Adjust only the CUDA installation path if necessary. Two build jobs limit peak compiler memory on the approximately 16 GB RAM machine.

```bash
git switch feature/turboquant-prefill-cuda
git rev-parse HEAD
mkdir -p turboquant-logs
set -o pipefail

cmake -S . -B build-turboquant \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_NCCL=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DGGML_CUDA_FA_QUANTS='f16-f16;bf16-bf16;q4_0-q4_0;q4_1-q4_1;q5_0-q5_0;q8_0-q8_0;tbq3_0-tbq3_0;tbq3_0-tbq4_0;tbq4_0-tbq3_0;tbq4_0-tbq4_0;tbqp3_0-tbq3_0;tbqp3_0-tbq4_0;turbo2_0-turbo2_0;turbo2_0-tbq3_0;turbo2_0-tbq4_0;tbq3_0-turbo2_0;tbq4_0-turbo2_0;tbqp3_0-turbo2_0' \
  2>&1 | tee turboquant-logs/configure.log

cmake --build build-turboquant --parallel 2 \
  --target llama-server llama-bench llama-perplexity \
  test-quantize-fns test-arg-parser test-gguf test-backend-ops \
  2>&1 | tee turboquant-logs/build.log
```

Fresh CUDA builds include all 12 TurboQuant pairs by default. Existing CMake caches keep their previous selection, so the explicit selection above is intentional. `GGML_CUDA_FA_QUANTS=all` includes the existing conventional pairs and the 12 supported TurboQuant pairs; it does not instantiate unsupported cross-products. A selective build may omit pairs, in which case context initialization reports the missing pair. For example, `-DGGML_CUDA_FA_QUANTS='q4_1-q4_1;turbo2_0-tbq3_0;turbo2_0-turbo2_0'` requests only those pairs plus the existing mandatory F16/F16 instance. HIP and MUSA keep their conventional pair set and ignore the CUDA-only TurboQuant entries.

### CUDA 12.8, glibc, and sinpi/cospi

Debian 13's glibc headers can expose an exception-specification mismatch with CUDA 12.8 declarations of `sinpi`, `cospi`, `sinpif`, and `cospif`. NVIDIA documents the issue in [this compiler discussion](https://forums.developer.nvidia.com/t/error-exception-specification-is-incompatible-for-cospi-sinpi-cospif-sinpif-with-glibc-2-41/323591). Related CUDA packages can also expose `rsqrt`/`rsqrtf` declaration mismatches; see [NVIDIA's Debian 13 report](https://forums.developer.nvidia.com/t/cuda-headers-in-crt-math-functions-h-still-broken-in-debian-13-repo/362940).

This branch does not patch CUDA Toolkit or system headers. If configuration fails in a CUDA compiler probe with one of these declarations, retain the full log and the CUDA, glibc, and host-compiler versions. Changing only the GCC executable may not fix a conflict involving the same system glibc headers. A compatible user-managed toolchain/container or vendor-fixed Toolkit package is an environment decision for the compilation machine. Do not interpret a compiler-probe failure as a TurboQuant kernel validation result.

## Manual regression checks

Run these only after a successful user-managed build:

```bash
build-turboquant/bin/test-quantize-fns -v 2>&1 | tee turboquant-logs/quantize.log
build-turboquant/bin/test-arg-parser 2>&1 | tee turboquant-logs/arguments.log
build-turboquant/bin/test-gguf 12345 2>&1 | tee turboquant-logs/gguf.log

build-turboquant/bin/test-backend-ops test -b CUDA0 \
  -o SET_ROWS,FLASH_ATTN_EXT -p 'type_k=(turbo2|tbq|f16)' \
  2>&1 | tee turboquant-logs/turboquant-ops.log

build-turboquant/bin/test-backend-ops test -b CUDA0 \
  -o MUL_MAT,GET_ROWS -p 'ptq1_0|pq2_0' \
  2>&1 | tee turboquant-logs/bonsai-ops.log

build-turboquant/bin/test-backend-ops test -b CUDA0 \
  -o SET_ROWS,FLASH_ATTN_EXT -p 'q4_1|q5_0' \
  2>&1 | tee turboquant-logs/baseline-ops.log
```

The new cases cover codec block layouts, fixed IDs, zero input, golden packed bytes, multiple blocks, chunk offsets, nonfinite scale rejection, QJL equivalence against a direct Walsh matrix calculation, CLI aliases and rejected pairs, and a synthetic PTQ1_0 `output.weight` GGUF round trip with tensor data.

CUDA cases cover I32/I64 row indices, multiple blocks per row, untouched rows, strided source views, sequences, all 12 cache pairs, grouped-query attention, masked prefill, odd query counts, longer-cache decode, and rejected head sizes/types. They compare cache writes to scalar encoding and attention to a separate F32 CPU graph decoded from the same encoded cache bytes, including the TBQP3 correction. The attention comparison retains the NMSE threshold of `5e-4`. The test harness uses a dedicated evaluation override because the production CPU backend deliberately rejects TurboQuant operators. TURBO2 scalar cases independently compute the rotation with a Walsh matrix, verify corrected norms, packed codes and group boundaries, and check spatial versus transformed-domain dot products.

Check that TurboQuant cases actually executed. A green exit with every case marked unsupported is not validation. With the complete pair list above, expect 16 CUDA cache-write cases, 89 CUDA attention comparisons, and 6 expected-rejection cases. Existing Bonsai regressions remain in the suite. Missing-pair selective builds should report unsupported cases and reject the corresponding server configuration. The prefill additions cover 8-row tiles, 17-row tails, GQA ratios 1/2/6/8, contiguous and strided caches, ALiBi, softcap, sinks, a broadcast mask across sequences, and a 1025-query case that exercises mask-bound scanning. Run the forced-split builds below to cover both partition paths deterministically.

## Manual model and endpoint validation

Set the model path on the compilation machine. No weights are downloaded by these commands.

```bash
export BONSAI_MODEL='/absolute/path/Ternary-Bonsai-2-27B-Uncensored-Heretic-PTQ1_0.gguf'
```

Run one server at a time, stopping it before the next configuration. Keep context, offload, batches, and parallel slots identical for comparison. Each command stays in the foreground:

```bash
build-turboquant/bin/llama-server -m "$BONSAI_MODEL" \
  --cache-type-k q4_1 --cache-type-v q4_1 \
  -c 32768 -ngl 99 -fa on -np 1 -b 512 -ub 128 \
  --host 127.0.0.1 --port 8080 \
  2>&1 | tee turboquant-logs/server-q4_1.log

build-turboquant/bin/llama-server -m "$BONSAI_MODEL" \
  --cache-type-k tbq4 --cache-type-v tbq4 \
  -c 32768 -ngl 99 -fa on -np 1 -b 512 -ub 128 \
  --host 127.0.0.1 --port 8080 \
  2>&1 | tee turboquant-logs/server-tbq4.log

build-turboquant/bin/llama-server -m "$BONSAI_MODEL" \
  --cache-type-k tbqp3 --cache-type-v tbq3 \
  -c 32768 -ngl 99 -fa on -np 1 -b 512 -ub 128 \
  --host 127.0.0.1 --port 8080 \
  2>&1 | tee turboquant-logs/server-tbqp3-tbq3.log

build-turboquant/bin/llama-server -m "$BONSAI_MODEL" \
  --cache-type-k turbo2 --cache-type-v tbq3 \
  -c 32768 -ngl 99 -fa on -np 1 -b 512 -ub 128 \
  --host 127.0.0.1 --port 8080 \
  2>&1 | tee turboquant-logs/server-turbo2-tbq3.log

build-turboquant/bin/llama-server -m "$BONSAI_MODEL" \
  --cache-type-k turbo2 --cache-type-v turbo2 \
  -c 32768 -ngl 99 -fa on -np 1 -b 512 -ub 128 \
  --host 127.0.0.1 --port 8080 \
  2>&1 | tee turboquant-logs/server-turbo2-turbo2.log
```

Check loading logs for `ptq1_0` tensors, the selected cache types, CUDA cache buffers, and Flash Attention. TurboQuant runs should report `attn_rot_k = 0` and `attn_rot_v = 0`; Bonsai weight transformations must still load normally. A context-shift-disabled message is expected.

In another terminal, repeat these requests for each server configuration. Record responses under distinct filenames when comparing runs:

```bash
curl --fail-with-body http://127.0.0.1:8080/health
curl --fail-with-body http://127.0.0.1:8080/v1/models

curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain in three sentences why the sky is blue."}],"temperature":0,"seed":12345,"max_tokens":128}'

curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain in three sentences why the sky is blue."},{"role":"assistant","content":"Sunlight scatters in the atmosphere. Shorter wavelengths scatter more strongly. This makes the daytime sky appear blue."},{"role":"user","content":"Why does sunset look red?"}],"temperature":0,"seed":12345,"max_tokens":128}'

curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"List three prime numbers."}],"temperature":0,"seed":12345,"max_tokens":64,"stream":true}'
```

Repeat an identical request to exercise cache reuse, then alternate unrelated and shared-prefix requests. Test longer prompts below the 32K limit. Compare coherent output, request timing, and reuse logs; different quantizers need not produce identical tokens. To validate multiple slots at the server level, repeat with the same `-np 2` setting for each cache configuration and send two requests concurrently, noting the per-slot context in the log.

For long-context coding quality, reuse identical local code and prompts at several lengths below 32K, leaving room for generated output. Place relevant identifiers and requirements near the beginning, middle, and end; ask for exact symbol retrieval, dependency reasoning, and patches that preserve earlier constraints. Record incorrect identifiers, missed requirements, repetitions, and patch correctness alongside timing. Repeat follow-ups sharing the same prefix. Compare Q4_1, TBQ4, TBQP3/TBQ3, TURBO2/TBQ3, and uniform TURBO2 with identical seeds, batches, and generation limits. Do not infer long-context coding quality from the short perplexity check alone.

Expected startup rejections can be checked manually with the same model:

```bash
build-turboquant/bin/llama-server -m "$BONSAI_MODEL" -ngl 99 -c 512 -fa on -ctk tbq4 -ctv tbqp3
build-turboquant/bin/llama-server -m "$BONSAI_MODEL" -ngl 99 -c 512 -fa on -ctk tbq4 -ctv q4_1
build-turboquant/bin/llama-server -m "$BONSAI_MODEL" -ngl 99 -c 512 -fa off -ctk tbq4 -ctv tbq4
build-turboquant/bin/llama-server -m "$BONSAI_MODEL" -ngl 99 -c 512 -fa on --no-kv-offload -ctk tbq4 -ctv tbq4
build-turboquant/bin/llama-server -m "$BONSAI_MODEL" -ngl 0 -c 512 -fa on -ctk tbq4 -ctv tbq4
```

## Manual performance and quality comparison

Stop the server before benchmarks. Run identical settings for each pair and retain both stdout and stderr:

```bash
for pair in q4_1:q4_1 tbq4:tbq4 tbqp3:tbq3 turbo2:tbq3 turbo2:turbo2; do
  cache_k=${pair%:*}
  cache_v=${pair#*:}
  build-turboquant/bin/llama-bench -m "$BONSAI_MODEL" \
    -ngl 99 -fa on -ctk "$cache_k" -ctv "$cache_v" \
    -p 512 -n 128 -r 3 -b 512 -ub 128 \
    2>&1 | tee "turboquant-logs/bench-${cache_k}-${cache_v}.log"
  build-turboquant/bin/llama-bench -m "$BONSAI_MODEL" \
    -ngl 99 -fa on -ctk "$cache_k" -ctv "$cache_v" \
    -p 0 -n 128 -d 16384 -r 3 -b 512 -ub 128 \
    2>&1 | tee "turboquant-logs/bench-16k-${cache_k}-${cache_v}.log"
done
```

Benchmark context allocation differs from the server's fixed 32K allocation; compare server memory separately. While each server or benchmark runs, monitor GPU memory and process RAM in another terminal:

```bash
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 1
ps -C llama-server -C llama-bench -o pid,rss,vsz,%cpu,etime,args
```

For a numerical quality comparison, use an existing local evaluation text; do not download a dataset as part of this task:

```bash
export BONSAI_EVAL_TEXT='/absolute/path/to/local/evaluation.txt'
for pair in q4_1:q4_1 tbq4:tbq4 tbqp3:tbq3 turbo2:tbq3 turbo2:turbo2; do
  cache_k=${pair%:*}
  cache_v=${pair#*:}
  build-turboquant/bin/llama-perplexity -m "$BONSAI_MODEL" \
    -f "$BONSAI_EVAL_TEXT" --chunks 16 -c 512 -b 256 -ub 128 \
    -ngl 99 -fa on -ctk "$cache_k" -ctv "$cache_v" \
    2>&1 | tee "turboquant-logs/perplexity-${cache_k}-${cache_v}.log"
done
```

Perplexity, long-context quality, RAM, VRAM, and tokens/second are all **PENDING USER VALIDATION**. The source implementation makes no quality or performance guarantee.

## Troubleshooting and reporting

- `invalid ggml type 143`: verify the executable came from this branch and its freshly rebuilt libraries. This branch keeps PTQ1_0=143 and COUNT=148. Do not convert or relabel model tensors to work around a stale executable.
- `turbo2` rejected or type 147 unknown: verify the branch and rebuild libraries and tools together. `tbq2` is not an alias; use `turbo2` or `turbo2_0`. TBQP3 is not a pure 2-bit replacement.
- CUDA pair unavailable: inspect the CMake pair list and rebuild with `GGML_CUDA_FA=ON` and the required `_0` pair. Existing build caches do not pick up changed defaults automatically.
- CUDA KV buffers required: keep every attention layer's KV cache on CUDA. Remove `--no-kv-offload`; inspect `-ngl` and actual placement. CPU-resident recurrent layers do not by themselves require TurboQuant CPU support.
- Unsupported head size: this release requires 256 for K and V in attention layers. It does not silently choose another block format or F16.
- Out of memory: retain allocation sizes from the log. Reduce context or microbatch for diagnosis and rerun the conventional baseline with identical settings. Reducing offload can make an attention layer CPU-resident, which is unsupported for TurboQuant.
- Context full: shifting is disabled for TurboQuant. Start a new conversation or reduce the retained prompt instead of relying on position shifts.
- Bad outputs or numerical mismatches: retain the cache pair, prompt, seed, batch settings, complete log, and backend reference results. Avoid simultaneous unrelated tuning changes.
- Compiler failure: retain configure/build logs, the first complete error, commit hash, host compiler version, CUDA Toolkit version, and driver information. No automatic build retry is performed on the development machine.

Report the exact commit, commands, model filename, successful baseline results, regression counts, and relevant logs. Corrections will be limited to the reported source issue and returned for another user-managed build.

## CUDA prefill optimization and manual A/B validation

### User-supplied baseline

The user measured Bonsai 2 Heretic 27B PTQ1_0 on RTX 3070 8 GB, sm_86, CUDA 12.8, build `ef52091d3`, with `-ngl 99 -fa 1 -p 4096 -n 256 -r 3`:

| K/V pair | pp4096 tokens/s | tg256 tokens/s |
|---|---:|---:|
| Q4_1/Q4_1 | 870.36 | 53.45 |
| TBQ3/TBQ3 | 138.37 | 46.32 |

These are user-reported results for the earlier build, not measurements made during development. Prefill was approximately 6.29x slower; decode throughput was approximately 13.3% lower. The Q4_1 result is a comparison baseline, not a promised target. Every optimization below is **PENDING USER VALIDATION**.

### Implemented changes

1. Wider vector query tiles: TurboQuant prefill uses 8 queries per block when at least 8 queries are present. Each decoded V vector is reused across those queries. `GGML_CUDA_TBQ_PREFILL_ROWS=2/4/8` permits independent comparisons; 8 is the initial default, not a measured optimum. Small batches retain the previous 1/2-query path. Batch-invariant mode retains the earlier path. This change does not itself share K decoding between queries or reuse cache reads across GQA heads.
2. Deferred V inverse: for wide prefill with multiple KV partitions, partial numerators stay in V's transformed basis. The combiner applies the same maximum-based softmax rescaling and denominator, then performs the inverse once per output query/head. Linearity makes this equivalent in real arithmetic; floating-point ordering changes still require numerical regression. The single-partition and decode paths retain their previous inverse placement. `GGML_CUDA_TBQ_DEFER_V=OFF` restores the previous placement.
3. Shared Q preparation: wide split-prefill prepares each query/head once, including the two TBQP3 centroid/QJL query projections, rather than rotating Q in every KV partition. It respects source strides and attention scale, including softcap scaling. A temporary F32 query buffer holds 256 values per query/head, or 512 for TBQP3. K/V remain compressed. No query buffer is allocated for decode, small batches, batch-invariant mode, or a single partition. `GGML_CUDA_TBQ_PRE_ROTATE_Q=OFF` disables it independently of V placement.
4. Mask-scanner bounds: the shared mask-bound scanner now ignores rows outside the query count in a partial tile and broadcasts its sequence index consistently with attention. This addresses out-of-bounds reads exposed by odd query counts and sequence-broadcast masks. Valid mask values and attention math are unchanged; the fix also applies to conventional attention using this scanner.
5. Partition sizing: when a split-prefill variant is available, the occupancy query uses that candidate kernel rather than the unprepared variant. Forced single-partition validation uses the ordinary kernel. Actual partition efficiency still needs target profiling.

The new paths support all 12 existing TurboQuant pairs, including mixed TURBO2 128-element and TBQ 256-element rotations. They introduce no new attention approximation, precision mode, weight transform, GGML type, or full-cache F16 conversion. Conventional attention dispatch stays unchanged. The one-query decode kernel remains the earlier specialization with both optimization flags false; measured tg256 still needs comparison after rebuilding.

| Build control | Default | Purpose |
|---|---|---|
| `GGML_CUDA_TBQ_PREFILL_ROWS` | `8` | Select 2, 4, or 8 query rows per block |
| `GGML_CUDA_TBQ_DEFER_V` | `ON` | Apply V inverse after wide split-prefill combination |
| `GGML_CUDA_TBQ_PRE_ROTATE_Q` | `ON` | Share wide split-prefill Q rotations |
| `GGML_CUDA_TBQ_SPLIT_KV` | `0` | Automatic partitions; 1/2/4/8 force a count for validation, clamped to available KV tiles |

Forced partitions are for diagnosis and regression coverage. Restore 0 for performance comparisons. They also affect TurboQuant decode partitioning when explicitly set, so forced-split measurements are not the production decode baseline. None of these controls changes conventional cache partitioning.

### Investigated and deferred

- GQA reuse: query heads that share a KV head currently run in separate vector blocks. A grouped-head tile could reuse the same K/V data, but needs new head-to-query indexing, per-head ALiBi slopes, query strides, output metadata, masking, and combination tests. It is not implemented in this increment. The regression matrix now includes GQA 1/2/6/8 to protect that future work.
- Native MMA: the existing in-place quantized MMA reader supports only Q4_0/Q8_0, a single shared K/V type, and its own block/alignment assumptions. TurboQuant requires separate K/V type parameters, 98/130/100/34-byte-aware tile readers, query rotation, independent V inverse, and explicit TBQP3 QJL products. A future path should decode only the current tile, reuse it across query rows and GQA heads, and retain vector fallback. Any F16 tile rounding must be assessed against the same-byte F32 reference before enabling it. Changing the selector alone would be incorrect. No TurboQuant MMA path is claimed or enabled here.
- Packed-data loading: TBQ3 crosses byte boundaries and its 98-byte blocks are only guaranteed 2-byte alignment. Specialized loads may help, but are deferred until target profiling shows the decoder is material. This increment keeps the reference-compatible packing and readers unchanged; no unmeasured loading rewrite is presented as an improvement.
- Cooperative writers and warp-shuffle WHTs: the current SET_ROWS writer handles an entire codec block in each thread, while attention WHTs use shared-memory barriers. These are profiling candidates, not implemented optimizations.

Larger tiles increase register use and shared memory; 8 rows can lose occupancy or spill, particularly with TBQP3's extra query projection. Shared Q preparation adds a kernel launch, a temporary allocation, and query-buffer traffic; its benefit depends on the actual partition count. Deferred V helps only when multiple partitions are used. None of these effects can be ranked reliably from source inspection alone.

### Manual builds and comparisons

Run the following only on the Debian CUDA server, after transferring the local branch. No dependencies or models are downloaded. These commands configure CUDA 12.8 for sm_86, keep FA enabled, and limit build parallelism to two. Build directories are separate so variants can be compared without changing binaries between runs.

```bash
git switch feature/turboquant-prefill-cuda
export BONSAI_MODEL='/absolute/path/Ternary-Bonsai-2-27B-Uncensored-Heretic-PTQ1_0.gguf'
mkdir -p turboquant-logs
set -o pipefail

TBQ_PREFILL_FA_PAIRS='f16-f16;q4_1-q4_1;q5_0-q5_0;tbq3_0-tbq3_0;tbq3_0-tbq4_0;tbq4_0-tbq3_0;tbq4_0-tbq4_0;tbqp3_0-tbq3_0;tbqp3_0-tbq4_0;turbo2_0-turbo2_0;turbo2_0-tbq3_0;turbo2_0-tbq4_0;tbq3_0-turbo2_0;tbq4_0-turbo2_0;tbqp3_0-turbo2_0'

configure_tbq_variant() {
  local tq_name=$1 tq_rows=$2 tq_defer=$3 tq_rotate=$4 tq_split=${5:-0}
  cmake -S . -B "build-tbq-${tq_name}" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_NCCL=OFF -DLLAMA_BUILD_TESTS=ON \
    -DGGML_CUDA_FA_QUANTS="$TBQ_PREFILL_FA_PAIRS" \
    -DGGML_CUDA_TBQ_PREFILL_ROWS="$tq_rows" \
    -DGGML_CUDA_TBQ_DEFER_V="$tq_defer" \
    -DGGML_CUDA_TBQ_PRE_ROTATE_Q="$tq_rotate" \
    -DGGML_CUDA_TBQ_SPLIT_KV="$tq_split" \
    2>&1 | tee "turboquant-logs/configure-${tq_name}.log" || return
  cmake --build "build-tbq-${tq_name}" --parallel 2 \
    --target llama-bench llama-server test-backend-ops test-quantize-fns test-arg-parser test-gguf \
    2>&1 | tee "turboquant-logs/build-${tq_name}.log"
}

configure_tbq_variant legacy2 2 OFF OFF
configure_tbq_variant rows8 8 OFF OFF
# Compare these next after validating the first two variants:
configure_tbq_variant defer-v 8 ON OFF
configure_tbq_variant all 8 ON ON
# Exercise Q preparation with the earlier V placement as well:
configure_tbq_variant q-only 8 OFF ON
# Optional occupancy comparison:
configure_tbq_variant rows4 4 OFF OFF
# Deterministic coverage of single-partition and split-prefill paths:
configure_tbq_variant split1 8 ON ON 1
configure_tbq_variant split2 8 ON ON 2
```

If a build fails, retain the first full compiler error and stop that variant. The CUDA/glibc declaration conflict discussed above still applies; this branch does not alter the installed toolkit. `legacy2` reproduces the earlier vector policy on the current codecs; it is not a claim of binary identity with `ef52091d3`.

Run regressions on each successfully built variant. Expected TurboQuant-specific counts are 16 writes, 89 attention comparisons, and 6 rejection cases. All comparisons must execute rather than report unsupported. In particular, `split1` and `split2` must both pass the same-byte F32 reference at the existing `5e-4` NMSE threshold. Keep the original Bonsai cases:

```bash
for tq_variant in legacy2 rows8 defer-v all q-only rows4 split1 split2; do
  "build-tbq-${tq_variant}/bin/test-quantize-fns" -v \
    2>&1 | tee "turboquant-logs/quantize-${tq_variant}.log"
  "build-tbq-${tq_variant}/bin/test-arg-parser" \
    2>&1 | tee "turboquant-logs/args-${tq_variant}.log"
  "build-tbq-${tq_variant}/bin/test-gguf" 12345 \
    2>&1 | tee "turboquant-logs/gguf-${tq_variant}.log"
  "build-tbq-${tq_variant}/bin/test-backend-ops" test -b CUDA0 \
    -o SET_ROWS,FLASH_ATTN_EXT -p 'type_k=(turbo2|tbq|f16)' \
    2>&1 | tee "turboquant-logs/attention-${tq_variant}.log"
  "build-tbq-${tq_variant}/bin/test-backend-ops" test -b CUDA0 \
    -o MUL_MAT,GET_ROWS -p 'ptq1_0|pq2_0' \
    2>&1 | tee "turboquant-logs/bonsai-${tq_variant}.log"
done

GGML_CUDA_BATCH_INVARIANT=1 build-tbq-all/bin/test-backend-ops test -b CUDA0 \
  -o FLASH_ATTN_EXT -p 'type_k=(turbo2|tbq)' \
  2>&1 | tee turboquant-logs/attention-batch-invariant.log
```

Select only built and passing variants for performance measurements. Keep model, driver, clock/power conditions, batch/microbatch settings, and other workload activity identical. Preserve full output, including the bench header and actual batch defaults. Stop the server first. The following matches the supplied benchmark arguments; it does not silently add different batch sizes:

```bash
git rev-parse HEAD | tee turboquant-logs/prefill-commit.log
for tq_variant in legacy2 rows8 defer-v all q-only rows4; do
  for tq_pair in q4_1:q4_1 tbq3:tbq3; do
    tq_k=${tq_pair%:*}
    tq_v=${tq_pair#*:}
    "build-tbq-${tq_variant}/bin/llama-bench" -m "$BONSAI_MODEL" \
      -ngl 99 -fa 1 -p 4096 -n 256 -r 3 -ctk "$tq_k" -ctv "$tq_v" \
      2>&1 | tee "turboquant-logs/pp4096-${tq_variant}-${tq_k}-${tq_v}.log"
  done
done
```

Repeat passing variants with `tbq4:tbq4`, `tbqp3:tbq3`, `turbo2:tbq3`, and `turbo2:turbo2`, and at shorter/longer prompt sizes to distinguish fixed preparation overhead from context-scaling cost. Retain the earlier server endpoint, repeated-request, RAM/VRAM, and long-context quality checks. The first acceptance question is whether pp4096 materially improves over the fresh `legacy2` run without a material tg256 regression; no improvement is assumed in advance.

If target profiling tools are already available, record the selected `flash_attn_ext_vec` query-row count, actual KV partition count, registers/thread, spills/local-memory traffic, occupancy, DRAM traffic, WHT barrier stalls, and time in `k_turboquant_prepare_q` and `flash_attn_combine_results`. This determines whether to retain 4 or 8 rows, enable Q preparation, pursue packed readers, or start native MMA/GQA tiling. Record compiler/runtime failures and return their logs for targeted corrections. All current optimization results remain **PENDING USER VALIDATION**.

## Changed files

The prefill addition changes these seven existing files relative to `57e1458b35a18be2307fc407bd79f8d007e20868`: this document, `ggml/CMakeLists.txt`, `ggml/src/ggml-cuda/CMakeLists.txt`, `ggml/src/ggml-cuda/fattn-common.cuh`, `ggml/src/ggml-cuda/fattn-vec.cuh`, `ggml/src/ggml-cuda/turboquant.cuh`, and `tests/test-backend-ops.cpp`. No new test file is added.

The 2-bit addition changes these 29 files relative to parent `ef52091d3f54cece74b07115cdec9a27742f106c`:

```text
TURBOQUANT_INTEGRATION.md
common/arg.cpp
ggml/CMakeLists.txt
ggml/cmake/common.cmake
ggml/include/ggml.h
ggml/src/ggml-common.h
ggml/src/ggml-cuda/fattn-common.cuh
ggml/src/ggml-cuda/fattn-vec.cuh
ggml/src/ggml-cuda/fattn.cu
ggml/src/ggml-cuda/set-rows.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq3_0-turbo2_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq4_0-turbo2_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbqp3_0-turbo2_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-tbq3_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-tbq4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-turbo2_0.cu
ggml/src/ggml-cuda/template-instances/generate_cu_files.py
ggml/src/ggml-cuda/turboquant.cuh
ggml/src/ggml-quants.c
ggml/src/ggml-quants.h
ggml/src/ggml-turboquant.h
ggml/src/ggml.c
gguf-py/gguf/constants.py
src/llama-context.cpp
tests/test-arg-parser.cpp
tests/test-backend-ops.cpp
tests/test-gguf.cpp
tests/test-quantize-fns.cpp
tools/llama-bench/llama-bench.cpp
```

The following inventory is relative to the PrismML base commit and includes regression files. No new file was added under `tests/`.

```text
TURBOQUANT_INTEGRATION.md
common/arg.cpp
ggml/CMakeLists.txt
ggml/cmake/common.cmake
ggml/include/ggml.h
ggml/src/CMakeLists.txt
ggml/src/ggml-common.h
ggml/src/ggml-cpu/ggml-cpu.cpp
ggml/src/ggml-cuda/CMakeLists.txt
ggml/src/ggml-cuda/cpy-utils.cuh
ggml/src/ggml-cuda/fattn-common.cuh
ggml/src/ggml-cuda/fattn-vec.cuh
ggml/src/ggml-cuda/fattn.cu
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml-cuda/set-rows.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq3_0-tbq3_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq3_0-tbq4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq4_0-tbq3_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq4_0-tbq4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbqp3_0-tbq3_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbqp3_0-tbq4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-turbo2_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-tbq3_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-tbq4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq3_0-turbo2_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq4_0-turbo2_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbqp3_0-turbo2_0.cu
ggml/src/ggml-cuda/template-instances/generate_cu_files.py
ggml/src/ggml-cuda/turboquant.cuh
ggml/src/ggml-quants.c
ggml/src/ggml-quants.h
ggml/src/ggml-turboquant.h
ggml/src/ggml.c
gguf-py/gguf/constants.py
src/llama-context.cpp
src/llama-kv-cache.cpp
src/llama-model-loader.cpp
src/llama-quant.cpp
tests/test-arg-parser.cpp
tests/test-backend-ops.cpp
tests/test-gguf.cpp
tests/test-quantize-fns.cpp
tools/llama-bench/llama-bench.cpp
```
