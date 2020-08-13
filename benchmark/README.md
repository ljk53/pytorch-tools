## Quick Start

### Run on Linux
```bash
./run.sh
```

### Run on MacOS
```bash
LIBTORCH=macos ./run.sh
```

### Run with locally built LibTorch
```bash
LIBTORCH=local ./run.sh
```

## Measures

### MacBook Pro 10.15.6 | CPU: Topology: 8-Core model: Intel Core i7-7920HQ bits: 64 type: MCP L2 cache: 256 KiB  Speed: 3100 MHz
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
