import statistics
import time


def report_header():
    print("{:>55}{:>25}{:>25}{:>25}".format(
        "name", "samples/sec (avg)", "ns (min)", "stdev"))


def report(name, duration_avg, duration_min, stdev):
    print("{:>55}{:>25.2f}{:>25.2f}{:>25.2f}".format(
        name, 1 / duration_avg, duration_min * 1e9, stdev * 1e9))


def benchmark(name, fn, count):
    fn()  # warm up

    samples = []
    for _ in range(10):
        start = time.time()
        fn()
        samples.append((time.time() - start) / count)

    report(name,
           sum(samples) / len(samples),
           min(samples),
           statistics.stdev(samples))
