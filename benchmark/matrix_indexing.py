# https://github.com/pytorch/pytorch/issues/29973

# Benchmark with `timeit`:
# python -m timeit 'import matrix_indexing; matrix_indexing.index_over_matrix(matrix_indexing.NUMPY_MATRIX);'
# python -m timeit 'import matrix_indexing; matrix_indexing.index_over_matrix(matrix_indexing.TORCH_MATRIX);'
#
# or with the ad-hoc tool:
# python matrix_indexing.py


import torch
import numpy as np
from common import *

BATCH_SIZE = 32
SEQUENCE_LENGTH = 512

COUNT = 20
TOTAL_COUNT = COUNT * BATCH_SIZE * SEQUENCE_LENGTH

TORCH_MATRIX = torch.full(
    size = (BATCH_SIZE, SEQUENCE_LENGTH),
    fill_value = 0,
    dtype = int,
)

NUMPY_MATRIX = np.full(
    shape = (BATCH_SIZE, SEQUENCE_LENGTH),
    fill_value = 0,
    dtype = int,
)


def index_over_matrix(matrix):
    for row_index in range(BATCH_SIZE):
        for column_index in range(SEQUENCE_LENGTH):
            matrix[row_index][column_index]


def index_over_matrix_numpy():
    def run():
        for _ in range(COUNT):
            index_over_matrix(NUMPY_MATRIX)

    benchmark("index_over_matrix_numpy", run, TOTAL_COUNT)


def index_over_matrix_torch():
    def run():
        for _ in range(COUNT):
            index_over_matrix(TORCH_MATRIX)

    benchmark("index_over_matrix_torch", run, TOTAL_COUNT)


if __name__ == "__main__":
    report_header()
    index_over_matrix_numpy()
    index_over_matrix_torch()
