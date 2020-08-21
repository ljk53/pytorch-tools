## Benchmark Matrix Indexing

Debug the issue discussed at: https://github.com/pytorch/pytorch/issues/29973

```
==========================================================================================================================================
Python (1.7.0a0+410d5b9)
==========================================================================================================================================
                                                   name        samples/sec (avg)                 ns (min)                    stdev
                                index_over_matrix_numpy               3088372.15                   293.69                    21.12
                                index_over_matrix_torch                244448.22                  3894.44                   110.80

==========================================================================================================================================
C++
==========================================================================================================================================
                                                   name        samples/sec (avg)                 ns (min)                    stdev
                                index_over_matrix_torch                562570.80                  1737.36                    23.27
                         index_over_matrix_torch_nograd                542979.46                  1739.20                    77.57
                   index_over_matrix_torch_nograd_novar               1429841.96                   671.70                    14.53
            index_over_matrix_torch_nograd_novar_select               1429094.91                   649.61                    47.21
     index_over_matrix_torch_nograd_novar_select_inline               1965405.45                   485.92                    17.48
```

Compared to the equivalent C++ implementation, Python overhead is 3894 - 1737 = 2157 ns. It calls indexing twice, so each op call's overhead is ~1 us.

Compared before/after applying AutoNonVariableTypeMode guard, the VariableType overhead is (1737 - 671) / 2 = 533 ns.

The at::indexing::get_item() dispatches to torch::select() function, which in turn calls Tensor::as_strided().

Directly calling Torch::as_strided() can save (649 - 485) / 2 = 82 ns, which seems to be c10 dispatching cost.

If we exclude the dispatching cost from Tensor::as_strided(), then the expected remaining cost is 485 - 82 * 2 = 321 ns, which is similar to numpy.
