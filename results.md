# Benchmark Results

Mean ± std over 20 runs, in seconds. Columns are GPU models, rows are frameworks.

## OpenFLR v1

| Framework | NVIDIA A100-SXM4-80GB | NVIDIA H100 80GB HBM3 | NVIDIA L40S | Tesla V100-PCIE-32GB |
|---|---|---|---|---|
| jax | 0.0172 ± 0.000244 | 0.00973 ± 2.14e-05 | 0.0402 ± 0.000109 | 0.0357 ± 4.2e-05 |
| numpy | 21.52 ± 5.302 | 5.285 ± 4.052 | 25.14 ± 0.232 | 6.461 ± 0.00791 |
| torch | 0.0196 ± 0.0134 | 0.0118 ± 0.0109 | 0.0369 ± 0.003 | 0.0341 ± 0.00386 |

## OpenFLR v2

| Framework | NVIDIA A100-SXM4-80GB | NVIDIA H100 80GB HBM3 | NVIDIA L40S | Tesla V100-PCIE-32GB |
|---|---|---|---|---|
| jax | 0.0112 ± 0.000231 | 0.00631 ± 2.92e-05 | 0.0252 ± 7.09e-05 | 0.0236 ± 5.41e-05 |
| numpy | 15.67 ± 4.474 | 2.806 ± 0.00797 | 20.15 ± 0.302 | 4.911 ± 0.00843 |
| torch | 0.0112 ± 0.00474 | 0.00668 ± 0.00446 | 0.0242 ± 0.00347 | 0.0224 ± 0.00289 |
