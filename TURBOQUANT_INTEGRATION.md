# TurboQuant KV cache integration

Status: **PENDING USER VALIDATION** for compilation, regression execution, model inference, numerical correctness, benchmarks, and GPU compatibility. Development performed source inspection and diff checks only. No builds, tests, project binaries, or inference were executed.

## Scope and architecture

This is a Bonsai-first integration for a personal PrismML fork. The implementation starts at PrismML `e3de715222b2667c54a267441159a5729f11155e`. It keeps the model loader, Bonsai weight representations, and existing CPU/CUDA weight kernels as the source of truth.

Reference: [jarkevithwlad/turboquant-prismml-cuda](https://github.com/jarkevithwlad/turboquant-prismml-cuda), pinned to `f7d1a4381104ed72cca090f8f0b56633fbebd620`. Relevant history includes `45a2ef747c3b8bb56bd44787ae39af1d59508519` (reference rebase), `1bc46807d98df97d94ce37a7934e70a314c72eac` (normalization and V rotation corrections), and `0bfe33411f7af381964510d083a0d8ee047201ef` (external/internal rotation interaction). These are provenance references, not commits merged into this branch. The reference's attention-sharpening heuristic, AMX types, other head dimensions, FP8 type remapping, and unrelated server changes are excluded.

Cache writes use the existing CUDA SET_ROWS interface. Each 256-element block receives deterministic sign flips, a Walsh-Hadamard transform, and Lloyd-Max scalar quantization. TBQP3 stores a 2-bit centroid index and a 1-bit QJL residual projection. Shared internal codec routines keep the CPU reference and CUDA writer tables and packing consistent.

Attention uses the existing vector Flash Attention kernel for both prefill and decode. It transforms Q once per query, evaluates compressed K directly, accumulates V in the transformed domain, and applies an inverse transform to the result. TBQP3 additionally transforms Q for its QJL correction. Normalization matches the reference: the centroid query uses `scale/256`, the QJL query uses `scale*1.2533/(256*256)`, and the V output uses `1/256`. The scalar TBQP3 decoder reconstructs this same linear estimator.

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

`GGML_TYPE_COUNT` is 147. No existing GGUF tensor or model file-type identifiers are reassigned. Rebuild the libraries and tools together after changing the enum. The reference's IDs and Q2_0 loader remapping are not used. The model loader and model quantizer reject TurboQuant weight types.

| K argument | V arguments |
|---|---|
| `tbq3` or `tbq3_0` | `tbq3`, `tbq3_0`, `tbq4`, `tbq4_0` |
| `tbq4` or `tbq4_0` | `tbq3`, `tbq3_0`, `tbq4`, `tbq4_0` |
| `tbqp3` or `tbqp3_0` | `tbq3`, `tbq3_0`, `tbq4`, `tbq4_0` |

Requirements: NVIDIA CUDA backend, Flash Attention enabled, 256-dimensional K and V attention heads, and CUDA KV buffers for every attention layer. `-fa auto` enables FA through the existing quantized-V rule; use `-fa on` for validation. CPU codecs are provided for serialization and numerical references, not CPU TurboQuant inference.

Not supported: TBQP3 V, mixed TurboQuant/conventional cache pairs, other head dimensions, MLA cache layouts, non-CUDA KV placement, and context shifting. The normal memory capability API reports shifting unavailable, allowing the server to disable it. Cache reuse without position changes remains available. Existing conventional cache types keep their behavior. TurboQuant state files are specific to this fork's type mapping.

The reference-style vector path can be slower than the existing MMA attention path, especially during prefill. Memory savings, output quality, and speed on the RTX 3070 are unmeasured. A 32K cache fitting in 8 GB cannot be guaranteed before measuring the complete model, cache, and compute buffers.

## Manual compilation on Debian 13

All commands below are for the user's compilation machine only. No dependencies are installed by this guide. Have CUDA 12.8, a CUDA-supported host compiler, CMake, and the dependencies in `docs/build.md` available. These commits are local and have not been pushed; first transfer the branch using your chosen Git workflow.

Run from the repository root in Bash. Adjust only the CUDA installation path if necessary. Two build jobs limit peak compiler memory on the approximately 16 GB RAM machine.

```bash
git switch feature/turboquant-kv-cache
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
  -DGGML_CUDA_FA_QUANTS='f16-f16;bf16-bf16;q4_0-q4_0;q4_1-q4_1;q5_0-q5_0;q8_0-q8_0;tbq3_0-tbq3_0;tbq3_0-tbq4_0;tbq4_0-tbq3_0;tbq4_0-tbq4_0;tbqp3_0-tbq3_0;tbqp3_0-tbq4_0' \
  2>&1 | tee turboquant-logs/configure.log

cmake --build build-turboquant --parallel 2 \
  --target llama-server llama-bench llama-perplexity \
  test-quantize-fns test-arg-parser test-gguf test-backend-ops \
  2>&1 | tee turboquant-logs/build.log
```

Fresh CUDA builds include all six TurboQuant pairs by default. Existing CMake caches keep their previous selection, so the explicit selection above is intentional. `GGML_CUDA_FA_QUANTS=all` includes the existing conventional pairs and the six supported TurboQuant pairs; it does not instantiate unsupported cross-products. A selective build may omit pairs, in which case context initialization reports the missing pair. HIP and MUSA keep their conventional pair set and ignore the CUDA-only TurboQuant entries.

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
  -o SET_ROWS,FLASH_ATTN_EXT -p 'type_k=(tbq|f16)' \
  2>&1 | tee turboquant-logs/turboquant-ops.log

build-turboquant/bin/test-backend-ops test -b CUDA0 \
  -o MUL_MAT,GET_ROWS -p 'ptq1_0|pq2_0' \
  2>&1 | tee turboquant-logs/bonsai-ops.log

build-turboquant/bin/test-backend-ops test -b CUDA0 \
  -o SET_ROWS,FLASH_ATTN_EXT -p 'q4_1|q5_0' \
  2>&1 | tee turboquant-logs/baseline-ops.log
```

The new cases cover codec block layouts, fixed IDs, zero input, golden packed bytes, multiple blocks, chunk offsets, nonfinite scale rejection, QJL equivalence against a direct Walsh matrix calculation, CLI aliases and rejected pairs, and a synthetic PTQ1_0 `output.weight` GGUF round trip with tensor data.

CUDA cases cover I32/I64 row indices, multiple blocks per row, untouched rows, strided source views, sequences, all six cache pairs, grouped-query attention, masked prefill, odd query counts, longer-cache decode, and rejected head sizes/types. They compare cache writes to scalar encoding and attention to a separate F32 CPU graph decoded from the same encoded cache bytes. The test harness uses a dedicated evaluation override because the production CPU backend deliberately rejects TurboQuant operators.

Check that TurboQuant cases actually executed. A green exit with every case marked unsupported is not validation. With the complete pair list above, expect 12 CUDA cache-write cases, 24 CUDA attention comparisons, and 3 expected-rejection cases. Existing Bonsai regressions remain in the suite. Missing-pair selective builds should report unsupported cases and reject the corresponding server configuration.

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
for pair in q4_1:q4_1 tbq4:tbq4 tbqp3:tbq3; do
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
for pair in q4_1:q4_1 tbq4:tbq4 tbqp3:tbq3; do
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

- `invalid ggml type 143`: verify the executable came from this branch and its freshly rebuilt libraries. This branch keeps PTQ1_0=143 and COUNT=147. Do not convert or relabel model tensors to work around a stale executable.
- CUDA pair unavailable: inspect the CMake pair list and rebuild with `GGML_CUDA_FA=ON` and the required `_0` pair. Existing build caches do not pick up changed defaults automatically.
- CUDA KV buffers required: keep every attention layer's KV cache on CUDA. Remove `--no-kv-offload`; inspect `-ngl` and actual placement. CPU-resident recurrent layers do not by themselves require TurboQuant CPU support.
- Unsupported head size: this release requires 256 for K and V in attention layers. It does not silently choose another block format or F16.
- Out of memory: retain allocation sizes from the log. Reduce context or microbatch for diagnosis and rerun the conventional baseline with identical settings. Reducing offload can make an attention layer CPU-resident, which is unsupported for TurboQuant.
- Context full: shifting is disabled for TurboQuant. Start a new conversation or reduce the retained prompt instead of relying on position shifts.
- Bad outputs or numerical mismatches: retain the cache pair, prompt, seed, batch settings, complete log, and backend reference results. Avoid simultaneous unrelated tuning changes.
- Compiler failure: retain configure/build logs, the first complete error, commit hash, host compiler version, CUDA Toolkit version, and driver information. No automatic build retry is performed on the development machine.

Report the exact commit, commands, model filename, successful baseline results, regression counts, and relevant logs. Corrections will be limited to the reported source issue and returned for another user-managed build.

## Changed files

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
