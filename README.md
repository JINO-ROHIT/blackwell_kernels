# Blackwell Kernels

A collection of Blackwell kernels that can be run on Modal.

## Modal setup

```console
uv pip install modal
python3 -m modal setup
```

## Profiling

```console
modal run main.py --action profile
```

This writes a Chrome trace to `trace.json.gz`.

## Benchmark results

Configuration: `M = 4096`, `N = 4096`, `K = 4096`

| Kernel | Time (ms) | Throughput (TFLOPS) |
| --- | ---: | ---: |
| CuBLAS | 0.0904 | 1520.34 |
| v0 | 0.5426 | 253.30 |
| v1 | 0.5448 | 252.29 |
| v3 | 0.2027 | 678.08 |
| v4 (3D TMA, 128 B) | 0.2027 | 678.19 |
| v5 (pipelining) | 0.1352 | 1016.80 |
| v6 (warp specialization) | 0.1372 | 1002.09 |

## References

1. [Outperforming cuBLAS on B200](https://www.paulwillchan.com/articles/outperforming-cublas-b200#b200-specifications)
2. [tcgen05](https://gau-nernst.github.io/tcgen05/)
3. [Acquire and Release Semantics](https://davekilian.com/acquire-release.html)
