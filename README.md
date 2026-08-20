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
