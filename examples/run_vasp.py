#!/usr/bin/env python3

from bmt import benchmark, report
from mpi import OpenMPI
from vasp import VaspConfig, VaspLauncher

from env import module_purge, module_load, module_list

module_purge()
# module_load(['aocc/5.1', 'aocl/5.2', 'mpi/openmpi-4.1.8'])
module_load(['nvhpc/25.11_cuda12', 'mkl/2025.3'])
module_list()

cfg = VaspConfig(
    bin='/scratch/optpar01/apps/build/vasp/6.5.1/nvhpc_hpc_sdk/25.11/hpc-x/2.25.1/cuda/12.9/zen3/bin/vasp_gam',
    outdir='./output',
    nnodes = 1, 
    ntasks = [1, 2, 4, 8], 
    ngpus = [1, 2, 4, 8], 
    ncores = 1, 
    kpars = 1,
    launcher=VaspLauncher(
        mpi=OpenMPI(),
    ),
)

results = benchmark(cfg, repeats=1)
report(results)
