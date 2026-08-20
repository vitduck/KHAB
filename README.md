# KISTI HPC-AI Benchmark (KHAB)
## Overview 
KHAB offers a declarative approach to automations of HPC and AI applications benchmarks. 
- Each benchmark is declared with a set of sweepable parameters that affect the performance. 
- The framework serializes each combination into individual runs, steps through every parameter combination automatically, and summarizes the results

List of supported benchmark

| Benchmark     |  Target                           | Sweepable parameters                                                                             |
| ------------- | ----------------------------------| ------------------------------------------------------------------------------------------------ |
| STREAM        | CPU memory bandwidth              | `nthreads`, `place`, `bind`                                                                      |
| BabelStream   | GPU memory bandwidth              | `device`, `size`, `ntimes`                                                                       |
| HPL           | Dense linear algebra              | `nnodes`, `ntasks`,`ngpus`, `sizes`, `blocksizes`, `mem`, `broadcast`                            |
| TensorFlow2   | Training throughput               | `nnodes`, `ntasks`,`ngpus`,`nthreads`, `models`, `batch_sizes`                                   |
| VASP          | Mterials science                  | `nnodes`, `ntasks`,`ngpus`,`ntasks`,`ngpus`, `ncores`, `kpars`, `nsims`                          |
| TensorRT-LLM  | LLM inference throughput/latency  | `sharding` (`nnodes`,`pp`,`pt`), `max_batch_sizes`, `max_num_tokens`, `workloads`, `concurrency` |

## Architecture 
```
khab/
├── env.sh                     # PYTHONPATH modification
├── .gitignore
├── LICENSE
├── khab/                      # Core framework package
│   ├── bmt.py                 # Base class  
│   ├── device.py              # Device class
│   ├── cpu.py                 # CPU info 
│   ├── gpu.py                 # GPU info 
│   ├── mpi.py                 # MPI abstract class and launcher 
│   ├── launcher.py            # Abstract bmt launcher base class
│   ├── affinity.py            # OpenMP affinity presets (GNU, Intel)
│   ├── env.py                 # Module load/unload helpers
│   ├── util.py                # Helper utilities 
│   ├── report.py              # Output format and progress bar 
│   └── benchmarks/            # Derived benchmark classes 
│       ├── stream.py
│       ├── babelstream.py
│       ├── hpl.py
│       ├── tensorflow2.py
│       ├── vasp.py
│       └── llama3.py
└──examples/                   # Benchmark scripts 
    ├── run_stream.py
    ├── run_babelstream.py
    ├── run_hpl.py
    ├── run_tf2.py
    ├── run_vasp.py
    └── llama3/                  
        ├── run_llama3.py
        ├── a100.yaml
        └── v100.yaml
```
## Environmental setup 
```
conda create -n khab python=3.12
conda activate khab

pip install pyyaml rich py-cpuinfo nvidia-ml-py ClusterShell
```
## Benchmarks

### STREAM

`StreamConfig` sweeps thread count and OpenMP affinity placement.

It wraps the classic [STREAM benchmark](https://www.cs.virginia.edu/stream/), the standard tool for measuring sustained memory bandwidth.
- `nthreads`: number of OpenMP threads
- `place`: which hardware units threads are pinned to (cores vs hardware threads)
- `bind`: how threads are distributed
    -  `spread` maximizes memory channel utilization across NUMA domains
    -  `close` clusters threads together and reduces cross-NUMA traffic
    -  For Zen architectures, one thread per CCX typically gives highest bandwidth due to localized L3 cache

Compiler and array size choices play a crucial role in archive optimal bandwidth

- Array size must be sufficiently large with respect to the CPU's L3 cache, e.g. 4x total cache size
- GCC does not automatically emit non-temporal (write-through) store instructions, leading to understated bandwidth. 
   - On Intel CPUs, Intel (R) oneAPI compilers emit non-temporal stores by default. 
   - On AMD CPUs, AMD Optimizing C/C++ and Fortran Compilers (AOCC) can also non-temporal stores 

```python
from affinity import GNU

from bmt import benchmark, report
from env import module_purge, module_load, module_list
from stream import StreamConfig

module_purge()
module_load(['gcc/15.2.0'])
module_list()

cfg = StreamConfig(
    bin = '<path-to-stream-binary>',
    outdir = './output',
    bind = 'spread',
    nthreads = [8, 16, 32]
)

results = benchmark(cfg, repeats=3)

report(results)
```

### BabelStream

`BabelStreamConfig` measures GPU (or CPU) memory bandwidth on a single device. 

It wraps [BabelStream](https://github.com/UoB-HPC/BabelStream), a STREAM-style benchmark reimplemented across many parallel programming models for cross-vendor comparison.
- `device`: which device to run the test
- `size`: array size in elements; larger arrays better saturate GPU memory bandwidth but need more device memory
- `ntimes`: kernel repetitions; more repetitions reduce measurement noise at the cost of longer wall time

BabelStream itself supports many backends (CUDA, HIP, SYCL, OpenMP, Kokkos, RAJA, TBB, Thrust).

`BabelStreamConfig` currently only support NVIDIA/CUDA backend. But adding another is straightforward
- The launcher's flags (`--device`, `--arraysize`, `--numtimes`) are common across BabelStream's backends
- Only the output parsing need to be changed to accommodate new backend 

```python
from affinity import GNU

from bmt import benchmark, report
from env import module_purge, module_load, module_list
from babelstream import BabelStreamConfig

module_purge()
module_load(['gcc/15.2.0'])
module_list()

cfg = BabelStreamConfig(
    bin = '<path-to-babelstream-binary>',
    model = 'CUDA',
    device = 0,
    outdir = './output',
)

results = benchmark(cfg, repeats=3)

report(results)
```

### HPL

`HplConfig` auto-selects its backend (`CpuHPL`, `NvidiaLegacy`, `Nvidia`) from the CPU/GPU dection.

- `nnodes` / `ntasks` / `ngpus`: MPI parametes  
- `sizes`: problem size N 
- `blocksizes`: tile size for the distributed matrix, affecting cache reuse and communication granularity. Optimal value is hardware-dependent
- `mem`: memory fraction used to auto-derive N when `sizes` isn't given directly
- `broadcast`: MPI algorithm for panel broadcast

The two NVIDIA backends target different container generations and GPUs.

| Backend | Container | Targets | Notable difference |
|---|---|---|---|
| `NvidiaLegacy` | NGC HPC-Benchmarks v21.04 | V100 | Always passes `--cpu-cores-per-rank` (from `nthreads`); requires `nthreads` to be set |
| `Nvidia` | NGC HPC-Benchmarks v24.09 | A100, H100, H200, GH200 | No `--cpu-cores-per-rank` flag; auto-adds `--no-multinode` for single-node runs |

Block size (NB) defaults are auto-selected per backend and, for `Nvidia`, per GPU generation.

| Backend | GPU | Default NB |
|---|---|---|
| `CpuHPL` | any | 232 |
| `NvidiaLegacy` | V100 | 288 |
| `Nvidia` | A100 | 384 |
| `Nvidia` | H100 / H200 / GH200 | 1024 |
| `Nvidia` | other | 256 |

`blocksizes` stays sweepable; these are just the framework's starting point, override with an explicit value or list to tune further.

```python
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
    mem = ['70%', '80%', '90%'],
    launcher = HplLauncher(
        mpi = OpenMPI(),
        backend = NvidiaLegacy(ngc='<path-to-hpc-benchmarks-sif>'),
    ),
)

results = benchmark(cfg, repeats=3)

report(results)
```

### TensorFlow2

`TensorFlow2Config` runs training throughput via Horovod/MPI inside a Singularity container, with NUMA-aware rank mapping.

- `nnodes` / `ntasks` / `ngpus`: MPI parametes  
- `nthreads`: CPU threads feeding the data pipeline; too few can leave the GPU data-starved
- `models`: which CNN architecture to benchmark
- `batch_sizes`: images per GPU per step; larger batches raise GPU utilization up to memory limits

The benchmark requires a specific combination: 
- Tensorflow Singularity container:  `nvcr.io/nvidia/tensorflow:23.07-tf2-py3`
- `tf_cnn_benchmark` commit: `c8e97df0d4d3d0c1020b98391c526df12371fc30`

Default batch size is auto-selected per detected GPU when `batch_sizes` is left at `0`.

| GPU | Default batch size |
|---|---|
| V100 | 256 |
| A100 | 512 |
| H100 | 768 |
| H200 | 768 |
| other | 64 |

```python
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
        ngc='<path-to-tensorflow-sif>',
        tf_cnn_benchmark='<path-to-tf_cnn_benchmarks-repo>',
        imagenet_dir='<path-to-imagenet-tfrecord>',
        num_batches=100,
    ),
)

results = benchmark(cfg, repeats=1)
report(results)
```

### VASP

`VaspConfig` runs VASP with `ntasks`/`ngpus` zipped together, writing `NCORE`/`KPAR`/`NSIM` into `INCAR` before each run.

- `ntasks` / `ngpus`: MPI rank and GPU count (zipped together), e.g. one gpu per rank. 
- `ncores` (NCORE): cores grouped per orbital. For GPU, `NCORE=1` is automatically enforced. 
- `kpars` (KPAR): k-point groups run in parallel; higher KPAR reduces communication at the cost of duplicated memory
- `nsims` (NSIM): bands optimized simultaneously in RMM-DIIS

```python
from bmt import benchmark, report
from mpi import OpenMPI
from vasp import VaspConfig, VaspLauncher

from env import module_purge, module_load, module_list

module_purge()
module_load(['nvhpc/25.11_cuda12', 'mkl/2025.3'])
module_list()

cfg = VaspConfig(
    bin='<path-to-vasp-binary>',
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
```
