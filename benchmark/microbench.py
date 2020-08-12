import time
import torch


def report(name, start, count):
    duration = time.time() - start
    print("{:>30}\tthroughput: {:>10.2f} samples/sec\tduration: {:>15.2f} ns".format(
        name, count / duration, duration / count * 1e9))


def add_s1_grad(count):
    a = torch.ones(1, requires_grad=True)
    b = torch.ones(1, requires_grad=True)

    start = time.time()
    c = a
    for _ in range(count):
        c = torch.add(c, b)
    report("add_s1_grad", start, count)


def add_s1_grad_scripted(count):
    a = torch.ones(1, requires_grad=True)
    b = torch.ones(1, requires_grad=True)

    @torch.jit.script
    def f(a, b, count: int):
        c = a
        for _ in range(count):
            c = torch.add(c, b)
        return c

    start = time.time()
    f(a, b, count)
    report("add_s1_grad_scripted", start, count)


def add_s1_nograd(count):
    with torch.no_grad():
        a = torch.ones(1, requires_grad=False)
        b = torch.ones(1, requires_grad=False)

        start = time.time()
        c = a
        for _ in range(count):
            c = torch.add(c, b)
        report("add_s1_nograd", start, count)


def add_s1_nograd_scripted(count):
    with torch.no_grad():
        a = torch.ones(1, requires_grad=False)
        b = torch.ones(1, requires_grad=False)

        @torch.jit.script
        def f(a, b, count: int):
            c = a
            for _ in range(count):
                c = torch.add(c, b)
            return c

        start = time.time()
        f(a, b, count)
        report("add_s1_nograd_scripted", start, count)


def mm_s1000_grad(count):
    a = torch.ones(1000, 1000, requires_grad=True)
    b = torch.ones(1000, 1000, requires_grad=True)

    start = time.time()
    c = a
    for _ in range(count):
        c = torch.mm(c, b)
    report("mm_s1000_grad", start, count)


if __name__ == "__main__":
    add_s1_grad(100000)
    add_s1_nograd(100000)
    add_s1_grad_scripted(100000)
    add_s1_nograd_scripted(100000)
    mm_s1000_grad(10)
