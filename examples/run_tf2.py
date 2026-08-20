#!/usr/bin/env python3

from bmt import benchmark, report
from env import module_purge, module_load, module_list
from mpi import OpenMPI
from tensorflow2 import TensorFlow2Config, TensorFlow2Launcher

module_purge()
module_load(['gcc/15.2.0', 'mpi/openmpi-4.1.8', 'singularity/4.3.4', 'git/2.52.0'])
module_list()

cfg = TensorFlow2Config(
    outdir='./output',
    nnodes=1,
    ntasks=8,
    ngpus=8,
    nthreads=[1, 2, 4, 8], 
    launcher=TensorFlow2Launcher(
        mpi=OpenMPI(),
        ngc='/scratch/optpar01/singularity/x86_64/tensorflow-23.07-tf2-py3.sif',
        tf_cnn_benchmark='/scratch/optpar01/work/2026/02-os_update/03-test/04-resnet50/benchmarks-c8e97df',
        imagenet_dir='/scratch/optpar01/inputs/tensorflow/imagenet_tfrecord',
        num_batches=100,
    ),
)

results = benchmark(cfg, repeats=1)
report(results)
