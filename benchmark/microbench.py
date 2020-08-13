import time
import torch


def report_header():
    print("{:>35}\t{:>15}\t{:>15}".format(
        "name", "samples/sec", "ns"))


def report(name, start, count):
    duration = time.time() - start
    print("{:>35}\t{:>15.2f}\t{:>15.2f}".format(
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


def add_s1_nograd_outplace(count):
    with torch.no_grad():
        a = torch.ones(1, requires_grad=False)
        b = torch.ones(1, requires_grad=False)
        c = torch.empty(1, requires_grad=False)

        start = time.time()
        for _ in range(count):
            torch.add(a, b, out=c)
            a, c = c, a
        report("add_s1_nograd_outplace", start, count)


def add_s1_nograd_outplace_scripted(count):
    with torch.no_grad():
        a = torch.ones(1, requires_grad=False)
        b = torch.ones(1, requires_grad=False)
        c = torch.empty(1, requires_grad=False)

        @torch.jit.script
        def f(a, b, c, count: int):
            for _ in range(count):
                torch.add(a, b, out=c)
                a, c = c, a
            return a

        start = time.time()
        f(a, b, c, count)
        report("add_s1_nograd_outplace_scripted", start, count)


def mm_sN_grad(count, N):
    a = torch.ones(N, N, requires_grad=True)
    b = torch.ones(N, N, requires_grad=True)
    c = torch.empty(N, N)

    start = time.time()
    for _ in range(count):
        c = torch.mm(a, b)
    report("mm_s{}_grad".format(N), start, count)


def mm_sN_nograd(count, N):
    with torch.no_grad():
        a = torch.ones(N, N, requires_grad=False)
        b = torch.ones(N, N, requires_grad=False)
        c = torch.empty(N, N)

        start = time.time()
        for _ in range(count):
            c = torch.mm(a, b)
        report("mm_s{}_nograd".format(N), start, count)


def mm_sN_nograd_outplace(count, N):
    with torch.no_grad():
        a = torch.ones(N, N, requires_grad=False)
        b = torch.ones(N, N, requires_grad=False)
        c = torch.empty(N, N)

        start = time.time()
        for _ in range(count):
            torch.mm(a, b, out=c)
        report("mm_s{}_nograd_outplace".format(N), start, count)


if __name__ == "__main__":
    report_header()

    add_s1_nograd_outplace_scripted(100000)
    add_s1_nograd_scripted(100000)
    add_s1_grad_scripted(100000)

    print()
    add_s1_nograd_outplace(100000)
    add_s1_nograd(100000)
    add_s1_grad(100000)

    for N in [64, 256]:
        print()
        count = int(1024 / N) ** 3 * 10
        mm_sN_nograd_outplace(count, N)
        mm_sN_nograd(count, N)
        mm_sN_grad(count, N)
