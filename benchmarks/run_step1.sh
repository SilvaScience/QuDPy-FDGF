#!/bin/sh
# Step 1 of the benchmark plan: run the four benchmarks one after another (single BLAS thread).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in bench_time_fft_ex3 bench_gmres_eta bench_example2_dense_sparse bench_scaling; do
  echo "=== $s start $(date +%H:%M:%S)"
  python $s.py > logs/$s.log 2>&1
  echo "=== $s end $(date +%H:%M:%S) exit $?"
done
echo "=== ALL DONE"
