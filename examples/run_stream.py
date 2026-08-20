#!/usr/bin/env python

from affinity import GNU

from bmt import benchmark, report
from env import module_purge, module_load, module_list
from stream import StreamConfig

module_purge()
module_load(['gcc/15.2.0'])
module_list()

cfg = StreamConfig(
    bin = '../stream-gcc_O3_free+cpu+omp.x',
    outdir = './output',
    bind = 'spread',
    nthreads = [8, 16, 32]
)

results = benchmark(cfg, repeats=3)

report(results)
