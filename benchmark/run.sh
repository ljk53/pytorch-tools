#!/bin/bash

set -ue -o pipefail

make

echo "============================================================"
echo "MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 C++"
echo "============================================================"
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ./microbench
echo

echo "============================================================"
echo "C++"
echo "============================================================"
./microbench
echo

echo "============================================================"
echo "MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 Python"
echo "============================================================"
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python microbench.py
echo

echo "============================================================"
echo "Python"
echo "============================================================"
python microbench.py
echo