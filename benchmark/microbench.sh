#!/bin/bash

set -ue -o pipefail

ROOT="$( cd "$(dirname "$0")"; pwd -P)"

cd $ROOT

make microbench

echo "=========================================================================================="
echo "C++"
echo "=========================================================================================="
./microbench
echo

echo "=========================================================================================="
echo "Python ($(python -c 'import torch; print(torch.version.__version__)'))"
echo "=========================================================================================="
python microbench.py
echo

echo "=========================================================================================="
echo "MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 C++"
echo "=========================================================================================="
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ./microbench
echo

echo "=========================================================================================="
echo "MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 Python ($(python -c 'import torch; print(torch.version.__version__)'))"
echo "=========================================================================================="
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python microbench.py
echo
