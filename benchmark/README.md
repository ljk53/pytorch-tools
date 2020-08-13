## Quick Start

### Run on Linux
```bash
./run.sh
```

### Run on MacOS
```bash
LIBTORCH=macos ./run.sh
```

### Run on Raspberry Pi
```bash
LIBTORCH=rpi CXX11_ABI=1 ./run.sh
```

### Run with locally built LibTorch
```bash
LIBTORCH=local ./run.sh
```

### Aggregate benchmark results in this document
```bash
./aggregates.sh
```

## Metric Keywords

The name of each row is the concatenation of keywords in the following table.

| Keyword  | Meaning                                            |
|----------|----------------------------------------------------|
| sN       | Input size, e.g.: "s1" - size-1-tensor             |
| scripted | TorchScript version                                |
| nograd   | Skip autograd functionality (with torch.no_grad()) |
| novar    | Skip autograd dispatching (only togglable in C++)  |
| outplace | Output result into pre-allocated tensor.           |

For example, "mm_s64_nograd_outplace" measures the performance of multiplying two 64x64 matrices, skip autograd, write result into pre-allocated tensor.

## Observations

### Autograd Dispatching Cost (simple add case)

This is a type of PyTorch framework cost. Even with the Autograd mode turned off, it still dispatches operator calls into "Autograd Kernel", which checks the Autograd mode, only to find out it's disabled and does nothing in most cases.

This cost can only be measured in C++ code, using the "AutoNonVariableTypeMode" guard.
We measure it by comparing the latency of "XXX_novar" with that of "XXX".
It varies from 48ns to 181ns, except for Raspberry Pi:
```
Autograd Dispatching Cost     	   MacBookPro:cpp:add_s1_nograd_outplace	     48.67 ns	        10%
Autograd Dispatching Cost     	        Linux:cpp:add_s1_nograd_outplace	     69.11 ns	        10%
Autograd Dispatching Cost     	          WSL:cpp:add_s1_nograd_outplace	    110.01 ns	        13%
Autograd Dispatching Cost     	       Ubuntu:cpp:add_s1_nograd_outplace	    181.13 ns	        16%
Autograd Dispatching Cost     	 RaspberryPi4:cpp:add_s1_nograd_outplace	    732.65 ns	        21%
```

### Output Allocation Cost (simple add case)

We measure the cost of allocating result tensor by comparing the latency of the simple add case "torch.add(a, b)" with that of the outplace version "torch.add(a, b, out=c)".
Most samples fall into the range of 700ns ~ 1000ns, which seems to be the cost of allocating a size-1 tensor in PyTorch.
```
Output Allocation Cost        	                   WSL:cpp:add_s1_nograd	    409.17 ns	        43%
Output Allocation Cost        	                 Linux:cpp:add_s1_nograd	    614.69 ns	        80%
Output Allocation Cost        	    MacBookPro:py:add_s1_nograd_scripted	    708.29 ns	       103%
Output Allocation Cost        	                Ubuntu:cpp:add_s1_nograd	    745.84 ns	        56%
Output Allocation Cost        	        Ubuntu:py:add_s1_nograd_scripted	    747.36 ns	        44%
Output Allocation Cost        	            MacBookPro:cpp:add_s1_nograd	    792.54 ns	       147%
Output Allocation Cost        	             MacBookPro:py:add_s1_nograd	    798.71 ns	        80%
Output Allocation Cost        	                    WSL:py:add_s1_nograd	    839.67 ns	        48%
Output Allocation Cost        	           WSL:py:add_s1_nograd_scripted	    935.60 ns	        69%
Output Allocation Cost        	                 Ubuntu:py:add_s1_nograd	    990.51 ns	        41%
Output Allocation Cost        	         Linux:py:add_s1_nograd_scripted	   1206.43 ns	       114%
Output Allocation Cost        	                  Linux:py:add_s1_nograd	   1427.27 ns	        74%
Output Allocation Cost        	          RaspberryPi4:cpp:add_s1_nograd	   3054.49 ns	        74%
Output Allocation Cost        	  RaspberryPi4:py:add_s1_nograd_scripted	   3272.29 ns	        52%
Output Allocation Cost        	           RaspberryPi4:py:add_s1_nograd	   5453.58 ns	        63%
```

### Autograd Cost (simple add case)

When input tensors have "requires_grad" set to true, it will do some extra work for each operator call to record the computation graph (for backprop). 

We measure the cost by comparing the latency of `_nograd` version with that of `_grad` version.
It varies from 1000ns to 2000ns for most cases.
For some reason, TorchScript seems to be performing visibly worse in this case (2500ns ~ 2900ns).
```
Autograd Cost                 	              MacBookPro:cpp:add_s1_grad	   1150.06 ns	        86%
Autograd Cost                 	                   Linux:cpp:add_s1_grad	   1309.20 ns	        95%
Autograd Cost                 	               MacBookPro:py:add_s1_grad	   1344.34 ns	        75%
Autograd Cost                 	                     WSL:cpp:add_s1_grad	   1354.22 ns	        99%
Autograd Cost                 	                  Ubuntu:cpp:add_s1_grad	   1431.45 ns	        69%
Autograd Cost                 	                    Linux:py:add_s1_grad	   1634.59 ns	        49%
Autograd Cost                 	             WSL:py:add_s1_grad_scripted	   1795.96 ns	        78%
Autograd Cost                 	                   Ubuntu:py:add_s1_grad	   1872.47 ns	        55%
Autograd Cost                 	                      WSL:py:add_s1_grad	   1949.63 ns	        75%
Autograd Cost                 	           Linux:py:add_s1_grad_scripted	   2584.05 ns	       114%
Autograd Cost                 	          Ubuntu:py:add_s1_grad_scripted	   2769.63 ns	       113%
Autograd Cost                 	      MacBookPro:py:add_s1_grad_scripted	   2868.69 ns	       205%
Autograd Cost                 	            RaspberryPi4:cpp:add_s1_grad	   7328.31 ns	       102%
Autograd Cost                 	             RaspberryPi4:py:add_s1_grad	   8580.48 ns	        61%
Autograd Cost                 	    RaspberryPi4:py:add_s1_grad_scripted	  11237.98 ns	       118%
```

### TorchScript v.s. Python (simple add case)

TorchScript is executed by PyTorch's own JIT interpreter.
For most cases, TorchScript is about 300ns ~ 1000ns faster than Python, except for the Autograd case.
```
TorchScript v.s. Python       	               MacBookPro:py:add_s1_grad	  -1129.40 ns	       -26%
TorchScript v.s. Python       	                   Ubuntu:py:add_s1_grad	     74.73 ns	         1%
TorchScript v.s. Python       	                    Linux:py:add_s1_grad	    149.49 ns	         3%
TorchScript v.s. Python       	                    WSL:py:add_s1_nograd	    298.64 ns	        13%
TorchScript v.s. Python       	    MacBookPro:py:add_s1_nograd_outplace	    304.53 ns	        44%
TorchScript v.s. Python       	           WSL:py:add_s1_nograd_outplace	    394.57 ns	        29%
TorchScript v.s. Python       	             MacBookPro:py:add_s1_nograd	    394.95 ns	        28%
TorchScript v.s. Python       	                      WSL:py:add_s1_grad	    452.31 ns	        11%
TorchScript v.s. Python       	        Ubuntu:py:add_s1_nograd_outplace	    728.74 ns	        43%
TorchScript v.s. Python       	         Linux:py:add_s1_nograd_outplace	    878.11 ns	        83%
TorchScript v.s. Python       	                 Ubuntu:py:add_s1_nograd	    971.89 ns	        40%
TorchScript v.s. Python       	                  Linux:py:add_s1_nograd	   1098.95 ns	        49%
TorchScript v.s. Python       	             RaspberryPi4:py:add_s1_grad	   1848.14 ns	         9%
TorchScript v.s. Python       	  RaspberryPi4:py:add_s1_nograd_outplace	   2324.35 ns	        37%
TorchScript v.s. Python       	           RaspberryPi4:py:add_s1_nograd	   4505.65 ns	        47%
```

### CPP v.s. Python (simple add case)

For simple add case, PyTorch code seems to be 500ns ~ 2000ns slower than C++ code per operator call.
```
CPP v.s. Python               	    MacBookPro:py:add_s1_nograd_outplace	    455.78 ns	        85%
CPP v.s. Python               	             MacBookPro:py:add_s1_nograd	    461.95 ns	        35%
CPP v.s. Python               	               MacBookPro:py:add_s1_grad	    656.22 ns	        26%
CPP v.s. Python               	           WSL:py:add_s1_nograd_outplace	    797.38 ns	        83%
CPP v.s. Python               	        Ubuntu:py:add_s1_nograd_outplace	   1100.70 ns	        83%
CPP v.s. Python               	         Linux:py:add_s1_nograd_outplace	   1168.37 ns	       153%
CPP v.s. Python               	                    WSL:py:add_s1_nograd	   1227.89 ns	        90%
CPP v.s. Python               	                 Ubuntu:py:add_s1_nograd	   1345.36 ns	        65%
CPP v.s. Python               	                   Ubuntu:py:add_s1_grad	   1786.38 ns	        51%
CPP v.s. Python               	                      WSL:py:add_s1_grad	   1823.30 ns	        67%
CPP v.s. Python               	                  Linux:py:add_s1_nograd	   1980.95 ns	       144%
CPP v.s. Python               	                    Linux:py:add_s1_grad	   2306.35 ns	        86%
CPP v.s. Python               	  RaspberryPi4:py:add_s1_nograd_outplace	   4456.77 ns	       107%
CPP v.s. Python               	           RaspberryPi4:py:add_s1_nograd	   6855.86 ns	        95%
CPP v.s. Python               	             RaspberryPi4:py:add_s1_grad	   8108.03 ns	        56%
```

### Costs relative to larger workload

For larger workload (e.g.: multiplying two 256x256 matrices) above mentioned costs are relatively small.
And the variance of the runtime measurements seems to be larger than the actual difference.
```
Autograd Cost                 	                   Linux:py:mm_s256_grad	  -8531.67 ns	        -3%
Autograd Cost                 	              MacBookPro:py:mm_s256_grad	   9586.67 ns	         4%
Autograd Cost                 	                  Ubuntu:py:mm_s256_grad	  17376.81 ns	         2%
Autograd Cost                 	                     WSL:py:mm_s256_grad	  38468.28 ns	        10%
Autograd Cost                 	            RaspberryPi4:py:mm_s256_grad	1076407.36 ns	         4%
CPP v.s. Python               	              MacBookPro:py:mm_s256_grad	 -70904.67 ns	       -23%
CPP v.s. Python               	                  Ubuntu:py:mm_s256_grad	   4571.44 ns	         0%
CPP v.s. Python               	                   Linux:py:mm_s256_grad	  18334.70 ns	         7%
CPP v.s. Python               	                     WSL:py:mm_s256_grad	  30935.23 ns	         8%
CPP v.s. Python               	            RaspberryPi4:py:mm_s256_grad	1207404.31 ns	         4%
Output Allocation Cost        	          RaspberryPi4:py:mm_s256_nograd	 -65425.60 ns	        -0%
Output Allocation Cost        	                 Linux:py:mm_s256_nograd	  -8185.77 ns	        -3%
Output Allocation Cost        	         Ubuntu:cpp:mm_s256_nograd_novar	  -1429.31 ns	        -0%
Output Allocation Cost        	                   WSL:py:mm_s256_nograd	   -857.56 ns	        -0%
Output Allocation Cost        	            MacBookPro:py:mm_s256_nograd	   1855.56 ns	         1%
Output Allocation Cost        	     MacBookPro:cpp:mm_s256_nograd_novar	   7929.29 ns	         3%
Output Allocation Cost        	                Ubuntu:py:mm_s256_nograd	   8032.09 ns	         1%
Output Allocation Cost        	          Linux:cpp:mm_s256_nograd_novar	   8051.73 ns	         3%
Output Allocation Cost        	   RaspberryPi4:cpp:mm_s256_nograd_novar	  11570.45 ns	         0%
Output Allocation Cost        	            WSL:cpp:mm_s256_nograd_novar	  35201.01 ns	        10%
```

## Benchmark Results

### MacBookPro 10.15.6 | CPU: Topology: 8-Core model: Intel Core i7-7920HQ bits: 64 type: MCP L2 cache: 256 KiB  Speed: 3100 MHz
```
==========================================================================================
C++
==========================================================================================
                               name              samples/sec                       ns
       add_s1_nograd_outplace_novar               2036069.21                   491.14
             add_s1_nograd_outplace               1850750.95                   540.32
                      add_s1_nograd                747880.19                  1337.11
                        add_s1_grad                399360.50                  2504.00

       mm_s64_nograd_novar_outplace                188230.50                  5312.64
                mm_s64_nograd_novar                141872.22                  7048.60
                        mm_s64_grad                 82426.89                 12131.96

      mm_s256_nograd_novar_outplace                  5670.87                176339.79
               mm_s256_nograd_novar                  5575.22                179365.10
                       mm_s256_grad                  4770.12                209638.26

==========================================================================================
Python (1.6.0)
==========================================================================================
                               name              samples/sec                       ns
    add_s1_nograd_outplace_scripted               1482667.76                   674.46
             add_s1_nograd_scripted                733659.85                  1363.03
               add_s1_grad_scripted                237733.93                  4206.38

             add_s1_nograd_outplace                998601.01                  1001.40
                      add_s1_nograd                568463.28                  1759.13
                        add_s1_grad                325688.47                  3070.42

             mm_s64_nograd_outplace                220956.13                  4525.79
                      mm_s64_nograd                171112.53                  5844.11
                        mm_s64_grad                110945.02                  9013.47

            mm_s256_nograd_outplace                  8883.44                112568.96
                     mm_s256_nograd                  8562.18                116792.69
                       mm_s256_grad                  7939.65                125950.20

==========================================================================================
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 C++
==========================================================================================
                               name              samples/sec                       ns
       add_s1_nograd_outplace_novar               2049649.63                   487.89
             add_s1_nograd_outplace               1865506.30                   536.05
                      add_s1_nograd                755092.85                  1324.34
                        add_s1_grad                406905.30                  2457.57

       mm_s64_nograd_novar_outplace                171864.32                  5818.54
                mm_s64_nograd_novar                146877.07                  6808.41
                        mm_s64_grad                 89800.45                 11135.80

      mm_s256_nograd_novar_outplace                  2281.57                438294.27
               mm_s256_nograd_novar                  2216.67                451127.54
                       mm_s256_grad                  2481.96                402907.32

==========================================================================================
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 Python (1.6.0)
==========================================================================================
                               name              samples/sec                       ns
    add_s1_nograd_outplace_scripted               1419646.84                   704.40
             add_s1_nograd_scripted                698127.97                  1432.40
               add_s1_grad_scripted                231137.36                  4326.43

             add_s1_nograd_outplace               1013655.57                   986.53
                      add_s1_nograd                547582.14                  1826.21
                        add_s1_grad                312149.82                  3203.59

             mm_s64_nograd_outplace                149058.74                  6708.76
                      mm_s64_nograd                130582.24                  7658.01
                        mm_s64_grad                 87050.06                 11487.64

            mm_s256_nograd_outplace                  2982.56                335282.83
                     mm_s256_nograd                  2987.12                334770.23
                       mm_s256_grad                  2900.35                344786.05
```

### Ubuntu Linux 4.15.0-101 | CPU(s): 4 Single core AMD EPYC 7601s (-SMP-) cache: 2048 KB clock speeds: max: 2199 MHz 1: 2199 MHz 2: 2199 MHz 3: 2199 MHz 4: 2199 MHz
```
==========================================================================================
C++
==========================================================================================
                               name              samples/sec                       ns
       add_s1_nograd_outplace_novar                866349.03                  1154.27
             add_s1_nograd_outplace                749789.30                  1333.71
                      add_s1_nograd                485856.68                  2058.22
                        add_s1_grad                286375.32                  3491.92

       mm_s64_nograd_novar_outplace                 51364.60                 19468.66
                mm_s64_nograd_novar                 44717.00                 22362.86
                        mm_s64_grad                 36911.43                 27091.88

      mm_s256_nograd_novar_outplace                  2060.93                485217.55
               mm_s256_nograd_novar                  2020.76                494863.94
                       mm_s256_grad                  2021.77                494615.53

==========================================================================================
Python (1.6.0)
==========================================================================================
                               name              samples/sec                       ns
    add_s1_nograd_outplace_scripted                584490.52                  1710.89
             add_s1_nograd_scripted                419872.11                  2381.68
               add_s1_grad_scripted                196997.72                  5076.20

             add_s1_nograd_outplace                428790.91                  2332.14
                      add_s1_nograd                289262.15                  3457.07
                        add_s1_grad                197444.98                  5064.70

             mm_s64_nograd_outplace                 49377.53                 20252.13
                      mm_s64_nograd                 44141.79                 22654.27
                        mm_s64_grad                 29265.69                 34169.70

            mm_s256_nograd_outplace                  2051.08                487548.49
                     mm_s256_nograd                  1978.58                505412.37
                       mm_s256_grad                  1873.15                533860.55

==========================================================================================
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 C++
==========================================================================================
                               name              samples/sec                       ns
       add_s1_nograd_outplace_novar                878419.54                  1138.41
             add_s1_nograd_outplace                756864.89                  1321.24
                      add_s1_nograd                478831.40                  2088.42
                        add_s1_grad                284283.19                  3517.62

       mm_s64_nograd_novar_outplace                 28646.56                 34908.20
                mm_s64_nograd_novar                 27757.51                 36026.28
                        mm_s64_grad                 24472.36                 40862.42

      mm_s256_nograd_novar_outplace                   573.51               1743635.50
               mm_s256_nograd_novar                   577.66               1731130.49
                       mm_s256_grad                   562.06               1779173.63

==========================================================================================
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 Python (1.6.0)
==========================================================================================
                               name              samples/sec                       ns
    add_s1_nograd_outplace_scripted                592432.55                  1687.96
             add_s1_nograd_scripted                398104.73                  2511.90
               add_s1_grad_scripted                186684.16                  5356.64

             add_s1_nograd_outplace                396164.45                  2524.20
                      add_s1_nograd                295833.73                  3380.28
                        add_s1_grad                181238.62                  5517.59

             mm_s64_nograd_outplace                 26334.61                 37972.84
                      mm_s64_nograd                 26059.60                 38373.57
                        mm_s64_grad                 23438.10                 42665.58

            mm_s256_nograd_outplace                   573.21               1744565.74
                     mm_s256_nograd                   573.80               1742766.05
                       mm_s256_grad                   571.73               1749071.48
```

### Linux 4.16.18-210 | CPU: 2x 12-Core Intel Xeon E5-2678 v3 (-MT MCP SMP-) speed/min/max: 3003/1200/2501 MHz
```
==========================================================================================
C++
==========================================================================================
                               name              samples/sec                       ns
       add_s1_nograd_outplace_novar               1428819.33                   699.88
             add_s1_nograd_outplace               1309092.13                   763.89
                      add_s1_nograd                763146.70                  1310.36
                        add_s1_grad                396468.33                  2522.27

       mm_s64_nograd_novar_outplace                 83170.10                 12023.55
                mm_s64_nograd_novar                 74686.30                 13389.34
                        mm_s64_grad                 57967.58                 17251.02

      mm_s256_nograd_novar_outplace                 19225.90                 52013.16
               mm_s256_nograd_novar                 18422.48                 54281.52
                       mm_s256_grad                 17164.74                 58258.98

==========================================================================================
Python (1.6.0)
==========================================================================================
                               name              samples/sec                       ns
    add_s1_nograd_outplace_scripted                853625.10                  1171.47
             add_s1_nograd_scripted                439249.79                  2276.61
               add_s1_grad_scripted                178332.74                  5607.50

             add_s1_nograd_outplace                423141.35                  2363.28
                      add_s1_nograd                238036.15                  4201.04
                        add_s1_grad                161691.84                  6184.60

             mm_s64_nograd_outplace                 62103.29                 16102.21
                      mm_s64_nograd                 54260.06                 18429.76
                        mm_s64_grad                 48884.43                 20456.41

            mm_s256_nograd_outplace                 19873.36                 50318.61
                     mm_s256_nograd                 19044.19                 52509.46
                       mm_s256_grad                 17754.49                 56323.78

==========================================================================================
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 C++
==========================================================================================
                               name              samples/sec                       ns
       add_s1_nograd_outplace_novar               1448606.44                   690.32
             add_s1_nograd_outplace               1307993.63                   764.53
                      add_s1_nograd                690876.02                  1447.44
                        add_s1_grad                350395.05                  2853.92

       mm_s64_nograd_novar_outplace                126462.11                  7907.51
                mm_s64_nograd_novar                114767.64                  8713.26
                        mm_s64_grad                 79083.59                 12644.85

      mm_s256_nograd_novar_outplace                  2334.72                428316.69
               mm_s256_nograd_novar                  2261.67                442151.79
                       mm_s256_grad                  2320.53                430936.19

==========================================================================================
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 Python (1.6.0)
==========================================================================================
                               name              samples/sec                       ns
    add_s1_nograd_outplace_scripted               1066693.79                   937.48
             add_s1_nograd_scripted                445395.63                  2245.19
               add_s1_grad_scripted                244953.90                  4082.40

             add_s1_nograd_outplace                665832.29                  1501.88
                      add_s1_nograd                397035.98                  2518.66
                        add_s1_grad                262861.45                  3804.29

             mm_s64_nograd_outplace                111155.70                  8996.39
                      mm_s64_nograd                 88353.77                 11318.14
                        mm_s64_grad                 72184.87                 13853.32

            mm_s256_nograd_outplace                  1964.71                508980.83
                     mm_s256_nograd                  2039.08                490418.45
                       mm_s256_grad                  2129.74                469540.80
```

### WSL Linux 4.19.104-microsoft-standard | CPU: Topology: Dual Core model: Intel Core m3-7Y30 bits: 64 type: MT MCP L2 cache: 4096 KiB Speed: 1608 MHz
```
==========================================================================================
C++
==========================================================================================
                               name              samples/sec                       ns
       add_s1_nograd_outplace_novar               1157832.75                   863.68
             add_s1_nograd_outplace               1037407.89                   963.94
                      add_s1_nograd                743160.75                  1345.60
                        add_s1_grad                380720.31                  2626.60

       mm_s64_nograd_novar_outplace                128542.21                  7779.55
                mm_s64_nograd_novar                110591.42                  9042.29
                        mm_s64_grad                 61585.19                 16237.67

      mm_s256_nograd_novar_outplace                  3773.14                265030.94
               mm_s256_nograd_novar                  2958.10                338054.84
                       mm_s256_grad                  3221.84                310381.72

==========================================================================================
Python (1.6.0)
==========================================================================================
                               name              samples/sec                       ns
    add_s1_nograd_outplace_scripted                720470.25                  1387.98
             add_s1_nograd_scripted                347123.82                  2880.82
               add_s1_grad_scripted                246266.58                  4060.64

             add_s1_nograd_outplace                553081.42                  1808.05
                      add_s1_nograd                381161.76                  2623.56
                        add_s1_grad                220348.33                  4538.27

             mm_s64_nograd_outplace                111672.67                  8954.74
                      mm_s64_nograd                 71603.25                 13965.85
                        mm_s64_grad                 58694.64                 17037.33

            mm_s256_nograd_outplace                  3323.33                300902.87
                     mm_s256_nograd                  3529.01                283365.70
                       mm_s256_grad                  2657.08                376353.41

==========================================================================================
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 C++
==========================================================================================
                               name              samples/sec                       ns
       add_s1_nograd_outplace_novar               1207753.68                   827.98
             add_s1_nograd_outplace               1055150.39                   947.73
                      add_s1_nograd                722329.73                  1384.41
                        add_s1_grad                355638.87                  2811.84

       mm_s64_nograd_novar_outplace                121615.82                  8222.61
                mm_s64_nograd_novar                107868.96                  9270.51
                        mm_s64_grad                 79957.66                 12506.62

      mm_s256_nograd_novar_outplace                  2120.34                471622.19
               mm_s256_nograd_novar                  2132.19                469000.31
                       mm_s256_grad                  2086.07                479370.78

==========================================================================================
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 Python (1.6.0)
==========================================================================================
                               name              samples/sec                       ns
    add_s1_nograd_outplace_scripted                752265.96                  1329.32
             add_s1_nograd_scripted                585588.09                  1707.68
               add_s1_grad_scripted                242731.54                  4119.78

             add_s1_nograd_outplace                588793.24                  1698.39
                      add_s1_nograd                390284.35                  2562.23
                        add_s1_grad                219935.95                  4546.78

             mm_s64_nograd_outplace                102837.63                  9724.07
                      mm_s64_nograd                 87392.58                 11442.62
                        mm_s64_grad                 65288.36                 15316.66

            mm_s256_nograd_outplace                  2103.06                475498.66
                     mm_s256_nograd                  2035.33                491320.71
                       mm_s256_grad                  2104.07                475269.56
```

### RaspberryPi4 Linux 5.4.51-v7l+ #1327 | CPU: Topology: Quad Core model: ARMv7 v7l variant: cortex-a72 bits: 32 type: MCP Speed: 1500 MHz min/max: 600/1500 MHz
```
==========================================================================================
C++
==========================================================================================
                               name              samples/sec                       ns
       add_s1_nograd_outplace_novar                290512.88                  3442.19
             add_s1_nograd_outplace                236918.15                  4220.87
                      add_s1_nograd                137170.65                  7290.19
                        add_s1_grad                 69173.50                 14456.40

       mm_s64_nograd_novar_outplace                  2277.92                438996.33
                mm_s64_nograd_novar                  2254.80                443498.63
                        mm_s64_grad                  2177.88                459161.82

      mm_s256_nograd_novar_outplace                    34.83              28706885.04
               mm_s256_nograd_novar                    34.81              28726573.32
                       mm_s256_grad                    35.11              28479562.97

==========================================================================================
Python (1.6.0a0+b31f58d)
==========================================================================================
                               name              samples/sec                       ns
    add_s1_nograd_outplace_scripted                148738.03                  6723.23
             add_s1_nograd_scripted                103788.81                  9634.95
               add_s1_grad_scripted                 47433.26                 21082.25

             add_s1_nograd_outplace                118752.27                  8420.89
                      add_s1_nograd                 70153.88                 14254.38
                        add_s1_grad                 45805.02                 21831.67

             mm_s64_nograd_outplace                  2266.38                441232.06
                      mm_s64_nograd                  2231.85                448058.12
                        mm_s64_grad                  2159.71                463026.15

            mm_s256_nograd_outplace                    34.77              28762320.43
                     mm_s256_nograd                    34.90              28649732.47
                       mm_s256_grad                    34.70              28815683.72

==========================================================================================
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 C++
==========================================================================================
                               name              samples/sec                       ns
       add_s1_nograd_outplace_novar                294777.29                  3392.39
             add_s1_nograd_outplace                245157.30                  4079.01
                      add_s1_nograd                140475.52                  7118.68
                        add_s1_grad                 68450.60                 14609.08

       mm_s64_nograd_novar_outplace                  2229.72                448486.08
                mm_s64_nograd_novar                  2208.35                452827.33
                        mm_s64_grad                  2137.29                467881.74

      mm_s256_nograd_novar_outplace                    34.91              28647558.56
               mm_s256_nograd_novar                    34.90              28651011.17
                       mm_s256_grad                    35.04              28539716.73

==========================================================================================
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 Python (1.6.0a0+b31f58d)
==========================================================================================
                               name              samples/sec                       ns
    add_s1_nograd_outplace_scripted                171189.37                  5841.48
             add_s1_nograd_scripted                105548.20                  9474.34
               add_s1_grad_scripted                 48773.34                 20503.00

             add_s1_nograd_outplace                113732.86                  8792.53
                      add_s1_nograd                 72117.80                 13866.20
                        add_s1_grad                 42644.15                 23449.87

             mm_s64_nograd_outplace                  2268.44                440832.54
                      mm_s64_nograd                  2232.76                447875.20
                        mm_s64_grad                  2168.17                461219.39

            mm_s256_nograd_outplace                    34.90              28649804.37
                     mm_s256_nograd                    34.93              28631541.13
                       mm_s256_grad                    32.66              30618404.60
```
