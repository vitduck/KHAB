#!/usr/bin/env python3

from mpi import OpenMPI
from hpl import HplConfig, HplLauncher, Nvidia, NvidiaLegacy
from bmt import benchmark, report

from env import module_purge, module_load, module_list

module_purge()
module_load(['singularity/4.3.4', 'gcc/11.5.0', 'mpi/openmpi-4.1.8'])
module_list()

cfg = HplConfig(
    outdir = './output',
    nnodes = 1,
    ntasks = 8,  
    ngpus = 8, 
    nthreads = 4,
    mem = ['90%'],
    launcher = HplLauncher(
        mpi = OpenMPI(),
        backend = NvidiaLegacy(ngc='hpc-benchmarks_21.04.sif'),
        #  backend = Nvidia(ngc='hpc-benchmarks_24.09.sif')
    ),
)

results = benchmark(cfg, repeats=3)

report(results)
