import time
import torch


def report(name, start, count):
    duration = time.time() - start
    print("{:>35}{:>25.2f}{:>25.2f}".format(
        name, count / duration, duration / count * 1e9))


def benchmark(name, fn, count):
    fn()  # warm up
    start = time.time()
    fn()
    report(name, start, count)


def add_s1_nograd_outplace(count):
    with torch.no_grad():
        a = torch.ones(1, requires_grad=False)
        b = torch.ones(1, requires_grad=False)
        c = torch.empty(1, requires_grad=False)

        def run():
            nonlocal a, b, c
            for _ in range(count):
                torch.add(a, b, out=c)

        benchmark("add_s1_nograd_outplace", run, count)


if __name__ == "__main__":
    add_s1_nograd_outplace(100000000)
