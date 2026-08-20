#!/usr/bin/env python

from affinity import GNU

from bmt import benchmark, report
from env import module_purge, module_load, module_list
from babelstream import BabelStreamConfig

module_purge()
module_load(['gcc/15.2.0'])
module_list()

cfg = BabelStreamConfig(
    bin = '../stream-cuda.x',
    model = 'CUDA',
    device = 0, 
    outdir = './output',
)

results = benchmark(cfg, repeats=3)

report(results)
