import time
import torch
from common import *


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
    add_s1_nograd_outplace(10000)
