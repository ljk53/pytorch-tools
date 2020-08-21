import time
import torch
from common import *


def add_s1_grad(count):
    a = torch.ones(1, requires_grad=True)
    b = torch.ones(1, requires_grad=True)
    c = a

    def run():
        nonlocal b, c
        for _ in range(count):
            c = torch.add(c, b)

    benchmark("add_s1_grad", run, count)


def add_s1_grad_scripted(count):
    a = torch.ones(1, requires_grad=True)
    b = torch.ones(1, requires_grad=True)
    c = a

    @torch.jit.script
    def f(a, b, c, count: int):
        for _ in range(count):
            c = torch.add(c, b)
        return c

    def run():
        f(a, b, c, count)

    benchmark("add_s1_grad_scripted", run, count)


def add_s1_nograd(count):
    with torch.no_grad():
        a = torch.ones(1, requires_grad=False)
        b = torch.ones(1, requires_grad=False)
        c = a

        def run():
            nonlocal b, c
            for _ in range(count):
                c = torch.add(c, b)

        benchmark("add_s1_nograd", run, count)


def add_s1_nograd_scripted(count):
    with torch.no_grad():
        a = torch.ones(1, requires_grad=False)
        b = torch.ones(1, requires_grad=False)
        c = a

        @torch.jit.script
        def f(a, b, c, count: int):
            for _ in range(count):
                c = torch.add(c, b)
            return c

        def run():
            f(a, b, c, count)

        benchmark("add_s1_nograd_scripted", run, count)


def add_s1_nograd_outplace(count):
    with torch.no_grad():
        a = torch.ones(1, requires_grad=False)
        b = torch.ones(1, requires_grad=False)
        c = torch.empty(1, requires_grad=False)

        def run():
            nonlocal a, b, c
            for _ in range(count):
                torch.add(a, b, out=c)
                a, c = c, a

        benchmark("add_s1_nograd_outplace", run, count)


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

        def run():
            f(a, b, c, count)

        benchmark("add_s1_nograd_outplace_scripted", run, count)


def mm_sN_grad(count, N):
    a = torch.ones(N, N, requires_grad=True)
    b = torch.ones(N, N, requires_grad=True)
    c = torch.empty(N, N)

    def run():
        nonlocal a, b, c
        for _ in range(count):
            c = torch.mm(a, b)

    benchmark("mm_s{}_grad".format(N), run, count)


def mm_sN_nograd(count, N):
    with torch.no_grad():
        a = torch.ones(N, N, requires_grad=False)
        b = torch.ones(N, N, requires_grad=False)
        c = torch.empty(N, N)

        def run():
            nonlocal a, b, c
            for _ in range(count):
                c = torch.mm(a, b)

        benchmark("mm_s{}_nograd".format(N), run, count)


def mm_sN_nograd_outplace(count, N):
    with torch.no_grad():
        a = torch.ones(N, N, requires_grad=False)
        b = torch.ones(N, N, requires_grad=False)
        c = torch.empty(N, N)

        def run():
            nonlocal a, b, c
            for _ in range(count):
                torch.mm(a, b, out=c)

        benchmark("mm_s{}_nograd_outplace".format(N), run, count)


if __name__ == "__main__":
    report_header()

    add_s1_nograd_outplace_scripted(500000)
    add_s1_nograd_scripted(500000)
    add_s1_grad_scripted(500000)

    print()
    add_s1_nograd_outplace(500000)
    add_s1_nograd(500000)
    add_s1_grad(500000)

    for N in [64, 256]:
        print()
        count = int(1024 / N) ** 3 * 10
        mm_sN_nograd_outplace(count, N)
        mm_sN_nograd(count, N)
        mm_sN_grad(count, N)
