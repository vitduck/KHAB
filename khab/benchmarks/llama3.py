#!/usr/bin/env python3

import itertools
import json
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

from bmt import (
    BmtConfig, RunSpec, _as_list,
    benchmark, _run,
)
from launcher import Launcher
from util import Command, _flatten_cmd, fmt_cmd, sys_cmd

log = logging.getLogger(__name__)

QUANT_FLAGS = {
    'FP16': [],
    'FP8': [],
    'INT8': ['--use_weight_only', '--weight_only_precision', 'int8'],
    'INT4': ['--use_weight_only', '--weight_only_precision', 'int4'],
}

def _infer_quant(model: str) -> str:
    match = re.search(
        r'(?i)(?:^|[-_/])(FP8|FP16|INT8|INT4)(?:$|[-_/])',
        model,
    )
    return match.group(1).upper() if match else 'FP16'

def _in_slurm() -> bool:
    return bool(os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_NNODES'))

def _derive_ngpus(nnodes: int, pp: int, tp: int) -> int:
    world_size = pp * tp
    if world_size % nnodes != 0:
        raise ValueError(
            f'Invalid topology: pp * tp = {world_size} is not divisible by nnodes = {nnodes}'
        )
    return world_size // nnodes

# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LlamaSpec(RunSpec):
    model: str = ''
    tp: int = 1
    pp: int = 1
    quant: str = 'FP16'
    max_seq_len: int = 8192
    max_batch_size: int = 2048
    batch_size: int = 1
    concurrency: int | str = 'auto'
    max_num_tokens: int = 8192
    input_mean: int = 128
    output_mean: int = 128
    num_requests: int = 30000
    dataset_num_requests: int = 30000
    warmup: int = 10

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    model: str

    def __post_init__(self):
        self.model_fs = self.model.replace('/', '__')

    def hf_dir(self) -> Path:
        return Path('hf-models') / self.model

    def dataset_file(self, spec: LlamaSpec) -> Path:
        return (
            Path('datasets') /
            f'synthetic-isl{spec.input_mean}-osl{spec.output_mean}'
            f'-req{spec.dataset_num_requests}.txt'
        )

    def ckpt_dir(self, spec: LlamaSpec) -> Path:
        return Path('ckpts') / self.model_fs / spec.quant / f'pp{spec.pp}-tp{spec.tp}'

    def engine_dir(self, spec: LlamaSpec) -> Path:
        return (
            Path('engines') / self.model_fs / spec.quant /
            f'pp{spec.pp}-tp{spec.tp}-sl{spec.max_seq_len}'
            f'-tk{spec.max_num_tokens}-bs{spec.max_batch_size}'
        )

# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

@dataclass
class Backend(ABC):
    metric_headers: list = field(default_factory=lambda: list(_THROUGHPUT_METRIC_HEADERS))

    @abstractmethod
    def throughput_cmd(self, spec: LlamaSpec, sif: str, kv_cache_fraction: float, paths: Paths) -> Command:
        pass

    @abstractmethod
    def latency_cmd(self, spec: LlamaSpec, sif: str, kv_cache_fraction: float, paths: Paths) -> Command:
        pass

    @abstractmethod
    def parse_throughput(self, text: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def parse_latency(self, text: str) -> dict[str, Any]:
        pass

# ---------------------------------------------------------------------------
# PytorchBackend
# ---------------------------------------------------------------------------

@dataclass
class PytorchBackend(Backend):
    metric_headers: list = field(default_factory=lambda: [
        'throughput (tps)',
        'throughput (tps/user)',
        'speed (tps/user)',
        'throughput (tps/gpu)',
        'ttft avg (ms)',
        'itl avg (ms)',
    ])

    def throughput_cmd(self, spec: LlamaSpec, sif: str, kv_cache_fraction: float, paths: Paths) -> Command:
        bench = [
            'trtllm-bench',
            ['--model', spec.model],
            ['--model_path', str(paths.hf_dir())],
            'throughput',
            ['--tp', str(spec.tp)],
            ['--pp', str(spec.pp)],
            ['--kv_cache_free_gpu_mem_fraction', str(kv_cache_fraction)],
            ['--max_batch_size', str(spec.max_batch_size)],
            ['--max_num_tokens', str(spec.max_num_tokens)],
            ['--dataset', str(paths.dataset_file(spec))],
            ['--backend', 'pytorch'],
            ['--warmup', str(spec.warmup)],
            ['--num_requests', str(spec.num_requests)],
            '--streaming',
        ]
        if spec.concurrency != 'auto':
            bench.append(['--concurrency', str(spec.concurrency)])

        if _in_slurm():
            return [
                [
                    'srun',
                    ['--nodes', str(spec.nnodes)],
                    ['--ntasks', str(spec.ntasks)],
                    ['--ntasks-per-node', str(spec.ngpus)],
                    ['--cpus-per-task', str(spec.nthreads)],
                ],
                ['singularity', 'run', '--nv', sif],
                ['trtllm-llmapi-launch'],
                bench,
            ]

        return [
            ['singularity', 'run', '--nv', '--cleanenv', sif],
            ['mpirun', ['-np', '1']],
            bench,
        ]

    def latency_cmd(self, spec: LlamaSpec, sif: str, kv_cache_fraction: float, paths: Paths) -> Command:
        bench = [
            'trtllm-bench',
            ['--model', spec.model],
            ['--model_path', str(paths.hf_dir())],
            'latency',
            ['--tp', str(spec.tp)],
            ['--pp', str(spec.pp)],
            ['--kv_cache_free_gpu_mem_fraction', str(kv_cache_fraction)],
            ['--num_requests', str(spec.num_requests)],
            ['--dataset', str(paths.dataset_file(spec))],
        ]

        if _in_slurm():
            return [
                [
                    'srun',
                    ['--nodes', str(spec.nnodes)],
                    ['--ntasks', str(spec.ntasks)],
                    ['--ntasks-per-node', str(spec.ngpus)],
                    ['--cpus-per-task', str(spec.nthreads)],
                ],
                ['singularity', 'run', '--nv', sif],
                ['trtllm-llmapi-launch'],
                bench,
            ]

        return [
            ['singularity', 'run', '--nv', '--cleanenv', sif],
            ['mpirun', ['-np', '1']],
            bench,
        ]

    def parse_throughput(self, text: str) -> dict[str, Any]:
        patterns = {
            'throughput (tps)': re.compile(r'Total Output Throughput \(tokens/sec\):\s+([\d.]+)'),
            'throughput (tps/user)': re.compile(r'Per User Output Throughput \[w/ ctx\] \(tps/user\):\s+([\d.]+)'),
            'speed (tps/user)': re.compile(r'Per User Output Speed \(tps/user\):\s+([\d.]+)'),
            'throughput (tps/gpu)': re.compile(r'Per GPU Output Throughput \(tps/gpu\):\s+([\d.]+)'),
            'ttft avg (ms)': re.compile(r'Average time-to-first-token \[TTFT\] \(ms\):\s+([\d.]+)'),
            'itl avg (ms)': re.compile(r'Average time-per-output-token \[TPOT\] \(ms\):\s+([\d.]+)'),
        }
        return {k: round(float(m.group(1)), 2)
                for k, p in patterns.items()
                if (m := p.search(text))}

    def parse_latency(self, text: str) -> dict[str, Any]:
        patterns = {
            'ttft avg (ms)': re.compile(r'Average time-to-first-token \[TTFT\] \(ms\):\s+([\d.]+)'),
            'itl avg (ms)': re.compile(r'Average time-per-output-token \[TPOT\] \(ms\):\s+([\d.]+)'),
            'latency avg (ms)': re.compile(r'Average request latency \(ms\):\s+([\d.]+)'),
            'ttft p99 (ms)': re.compile(r'\[TTFT\] P99\s+:\s+([\d.]+)'),
            'itl p99 (ms)': re.compile(r'\[TPOT\] P99\s+:\s+([\d.]+)'),
            'latency p99 (ms)': re.compile(r'\[Latency\] P99\s+:\s+([\d.]+)'),
            'request tput (req/s)': re.compile(r'Request Throughput \(req/sec\):\s+([\d.]+)'),
            'output tput (tok/s)': re.compile(r'Total Output Throughput \(tokens/sec\):\s+([\d.]+)'),
        }
        return {k: round(float(m.group(1)), 2)
                for k, p in patterns.items()
                if (m := p.search(text))}

# ---------------------------------------------------------------------------
# LegacyBackend (ABC) -- shared convert/build for all legacy backends
# ---------------------------------------------------------------------------

@dataclass
class LegacyBackend(Backend, ABC):

    def convert_cmd(self, spec: LlamaSpec, sif: str, paths: Paths) -> Command:
        return [
            ['singularity', 'run', '--nv', '--cleanenv', sif],
            ['python', '/opt/TensorRT-LLM/examples/llama/convert_checkpoint.py',
                ['--model_dir', str(paths.hf_dir())],
                ['--output_dir', str(paths.ckpt_dir(spec))],
                ['--pp_size', str(spec.pp)],
                ['--tp_size', str(spec.tp)],
                ['--workers', str(spec.pp * spec.tp)],
                *QUANT_FLAGS.get(spec.quant, []),
            ]
        ]

    def build_cmd(self, spec: LlamaSpec, sif: str, paths: Paths) -> Command:
        #['--multiple_profiles', 'enable'],
        #['--gather_all_token_logits'],
        return [
            ['singularity', 'run', '--nv', '--cleanenv', sif],
            ['trtllm-build',
                ['--checkpoint_dir', str(paths.ckpt_dir(spec))],
                ['--output_dir', str(paths.engine_dir(spec))],
                ['--max_seq_len', str(spec.max_seq_len)],
                ['--max_num_tokens', str(spec.max_num_tokens)],
                ['--max_batch_size', str(spec.max_batch_size)],
                ['--gemm_plugin', 'auto'],
                ['--reduce_fusion', 'enable'],
                ['--context_fmha', 'enable'],
                ['--kv_cache_type', 'paged'],
                ['--use_paged_context_fmha', 'enable'],
                ['--gpus_per_node', str(spec.ngpus)],
                ['--workers', str(spec.pp * spec.tp)],
            ]
        ]

    def parse_throughput(self, text: str) -> dict[str, Any]:
        result = {}
        m = re.search(r'Token Throughput \(tokens/sec\):\s+([\d.]+)', text)
        if m:
            result['throughput (tps)'] = round(float(m.group(1)))
        streaming = {
            'ttft avg (ms)': re.compile(r'Average time-to-first-token \(ms\):\s+([\d.]+)'),
            'itl avg (ms)': re.compile(r'Average inter-token latency \(ms\):\s+([\d.]+)'),
        }
        for k, p in streaming.items():
            m = p.search(text)
            if m:
                result[k] = round(float(m.group(1)), 2)
        return result

    def parse_latency(self, text: str) -> dict[str, Any]:
        patterns = {
            'ttft avg (ms)': re.compile(r'Average time-to-first-token \(ms\):\s+([\d.]+)'),
            'itl avg (ms)': re.compile(r'Average inter-token latency \(ms\):\s+([\d.]+)'),
            'latency avg (ms)': re.compile(r'Average request latency \(ms\):\s+([\d.]+)'),
            'ttft p99 (ms)': re.compile(r'TTFT.*P99.*?:\s+([\d.]+)'),
            'itl p99 (ms)': re.compile(r'(?:TPOT|ITL).*P99.*?:\s+([\d.]+)'),
            'latency p99 (ms)': re.compile(r'(?:\[Latency\] P99|P99 \(ms\))\s+:?\s+([\d.]+)'),
            'request tput (req/s)': re.compile(r'Request Throughput \(req/sec\):\s+([\d.]+)'),
            'output tput (tok/s)': re.compile(r'Generation Token Throughput \(tokens/sec\):\s+([\d.]+)'),
        }
        return {k: round(float(m.group(1)), 2)
                for k, p in patterns.items()
                if (m := p.search(text))}

# ---------------------------------------------------------------------------
# LegacyCppBackend
# ---------------------------------------------------------------------------

@dataclass
class LegacyCppBackend(LegacyBackend):
    def throughput_cmd(self, spec: LlamaSpec, sif: str, kv_cache_fraction: float, paths: Paths) -> Command:
        return [
            ['singularity', 'run', '--nv', sif],
            ['mpirun', ['-np', '1']],
            ['trtllm-bench',
                ['--model', spec.model],
                'throughput',
                ['--kv_cache_free_gpu_mem_fraction', str(kv_cache_fraction)],
                ['--engine_dir', str(paths.engine_dir(spec))],
                ['--dataset', str(paths.dataset_file(spec))],
                ['--num_requests', str(spec.num_requests)],
                '--streaming',
            ]
        ]

    def latency_cmd(self, spec: LlamaSpec, sif: str, kv_cache_fraction: float, paths: Paths) -> Command:
        return [
            ['singularity', 'run', '--nv', sif],
            ['mpirun', ['-np', '1']],
            ['trtllm-bench',
                ['--model', spec.model],
                'latency',
                ['--kv_cache_free_gpu_mem_fraction', str(kv_cache_fraction)],
                ['--engine_dir', str(paths.engine_dir(spec))],
                ['--dataset', str(paths.dataset_file(spec))],
                ['--num_requests', str(spec.num_requests)],
            ]
        ]

# ---------------------------------------------------------------------------
# LegacyPythonBackend
# ---------------------------------------------------------------------------

@dataclass
class LegacyPythonBackend(LegacyBackend):

    def throughput_cmd(self, spec: LlamaSpec, sif: str, kv_cache_fraction: float, paths: Paths) -> Command:
        return [[
            'singularity', 'run', '--nv', sif,
            'mpirun', ['-np', str(spec.pp * spec.tp)],
            'python', '/opt/TensorRT-LLM/benchmarks/python/benchmark.py',
            ['--model', 'llama'],
            ['--engine_dir', str(paths.engine_dir(spec))],
            ['--batch_size', str(spec.max_batch_size)],
            ['--kv_cache_free_gpu_mem_fraction', str(kv_cache_fraction)],
            ['--input_output_len', f'{spec.input_mean},{spec.output_mean}'],
        ]]

    def latency_cmd(self, spec: LlamaSpec) -> Command:
        raise NotImplementedError('LegacyPythonBackend does not support latency benchmark')

    def parse_latency(self, text: str) -> dict[str, Any]:
        raise NotImplementedError('LegacyPythonBackend does not support latency benchmark')

# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------

@dataclass
class LlamaLauncher(Launcher):
    backend: Backend = field(default=None)
    sif: str = ''
    kv_cache_fraction: float = 0.95
    model: str = ''
    paths: Paths = field(default=None)

    def build_cmd(self, spec: LlamaSpec) -> Command:
        raise NotImplementedError(f'{type(self).__name__} does not implement build_cmd')

    def download_cmd(self, spec: LlamaSpec) -> Command:
        return [[
            'singularity', 'run', '--nv', self.sif,
            'hf', 'download', self.model,
            ['--local-dir', str(self.paths.hf_dir())],
            ['--exclude', '*.pth'],
            ['--max-workers', '16'],
        ]]

    def download(self, spec: LlamaSpec) -> None:
        """Download HF model weights -- shared across all backends."""
        cmd = self.download_cmd(spec)
        rc = sys_cmd(cmd)
        if rc != 0:
            raise RuntimeError(
                f'download failed with exit code {rc}: {self.paths.hf_dir()}\n\n'
                f'Command:\n{fmt_cmd(cmd)}'
            )

    def dataset_cmd(self, spec: LlamaSpec) -> Command:
        return [
            ['singularity', 'run', '--nv', self.sif],
            [
                'python', '/opt/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py',
                ['--tokenizer', self.model],
                '--stdout',
                'token-norm-dist',
                ['--num-requests', str(spec.dataset_num_requests)],
                ['--input-mean', str(spec.input_mean)],
                ['--output-mean', str(spec.output_mean)],
                ['--input-stdev', '0'],
                ['--output-stdev', '0'],
            ]
        ]

    def dataset(self, spec: LlamaSpec) -> None:
        """Generate synthetic dataset -- shared across all backends.

        Filters stdout to JSON-only lines, discarding NVIDIA container noise
        (driver banners, warnings, etc.) that would corrupt the dataset file.
        Skips silently if the output file already exists.
        """
        out = self.paths.dataset_file(spec)
        if out.exists():
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.dataset_cmd(spec)
        flat = _flatten_cmd(cmd)
        stderr_tmp = f'{out}.err.tmp'
        stderr_path = f'{out}.err'
        n_written = n_skipped = 0

        try:
            with open(stderr_tmp, 'w', buffering=1) as err, out.open('w') as f:
                with subprocess.Popen(
                    flat,
                    stdout=subprocess.PIPE,
                    stderr=err,
                    text=True,
                    bufsize=1,
                ) as proc:
                    for line in proc.stdout:
                        line = line.rstrip('\n')
                        try:
                            obj = json.loads(line)
                            if not isinstance(obj, dict):
                                n_skipped += 1
                                continue
                            f.write(line + '\n')
                            n_written += 1
                        except json.JSONDecodeError:
                            n_skipped += 1
                    rc = proc.wait()
        except KeyboardInterrupt:
            if os.path.exists(stderr_tmp):
                os.replace(stderr_tmp, stderr_path)
            raise

        if rc == 0:
            if os.path.exists(stderr_tmp):
                os.remove(stderr_tmp)
        else:
            if os.path.exists(stderr_tmp):
                os.replace(stderr_tmp, stderr_path)
            out.unlink(missing_ok=True)
            raise RuntimeError(
                f'prepare_dataset failed with exit code {rc}.\n\n'
                f'stdout: {out}\n'
                f'stderr: {stderr_path}\n\n'
                f'Inspect the stderr file for runtime error details.\n\n'
                f'Command:\n{fmt_cmd(cmd)}'
            )

        log.debug('dataset %s: %d records written, %d noise lines dropped', out, n_written, n_skipped)

    def convert(self, spec: LlamaSpec) -> None:
        """Convert HF checkpoint to TRT-LLM format -- legacy backends only."""
        assert isinstance(self.backend, LegacyBackend)
        out = self.paths.ckpt_dir(spec)
        if (out / 'config.json').exists():
            return
        out.mkdir(parents=True, exist_ok=True)
        cmd = self.backend.convert_cmd(spec, self.sif, self.paths)
        rc = sys_cmd(cmd)
        if rc != 0:
            raise RuntimeError(
                f'convert failed with exit code {rc}: {out}\n\n'
                f'Command:\n{fmt_cmd(cmd)}'
            )

    def build(self, spec: LlamaSpec) -> None:
        """Build TRT-LLM engine -- legacy backends only."""
        assert isinstance(self.backend, LegacyBackend)
        out = self.paths.engine_dir(spec)
        if (out / 'config.json').exists():
            return
        out.mkdir(parents=True, exist_ok=True)
        cmd = self.backend.build_cmd(spec, self.sif, self.paths)
        rc = sys_cmd(cmd)
        if rc != 0:
            raise RuntimeError(
                f'build failed with exit code {rc}: {out}\n\n'
                f'Command:\n{fmt_cmd(cmd)}'
            )

@dataclass
class LlamaThroughputLauncher(LlamaLauncher):
    def build_cmd(self, spec: LlamaSpec) -> Command:
        return self.backend.throughput_cmd(spec, self.sif, self.kv_cache_fraction, self.paths)

@dataclass
class LlamaLatencyLauncher(LlamaLauncher):
    def build_cmd(self, spec: LlamaSpec) -> Command:
        return self.backend.latency_cmd(spec, self.sif, self.kv_cache_fraction, self.paths)

# ---------------------------------------------------------------------------
# Config headers
# ---------------------------------------------------------------------------

_THROUGHPUT_METRIC_HEADERS = ['throughput (tps)', 'ttft avg (ms)', 'itl avg (ms)']

_THROUGHPUT_KEY_HEADERS = [
    'model', 'nnodes', 'ngpus', 'pp', 'tp', 'quant',
    'input_mean', 'output_mean', 'num_requests',
    'max_num_tokens', 'max_batch_size', 'concurrency',
]

_LATENCY_KEY_HEADERS = [
    'model', 'pp', 'tp', 'quant',
    'input_mean', 'output_mean', 'num_requests',
    'max_num_tokens', 'batch_size',
]

_LATENCY_METRIC_HEADERS = [
    'ttft avg (ms)', 'itl avg (ms)', 'latency avg (ms)',
    #  'ttft p99 (ms)', 'itl p99 (ms)', 'latency p99 (ms)',
    'request tput (req/s)', 'output tput (tok/s)',
]

LLAMA_HEADER_ALIASES = {
    'nnodes': 'n',
    'ngpus': 'g',
    'quant': 'qnt',
    'input_mean': 'isl',
    'output_mean': 'osl',
    'num_requests': 'req',
    'max_num_tokens': 'mnt',
    'max_batch_size': 'mbs',
    'batch_size': 'bs',
    'concurrency': 'cc',
    'throughput (tps)': 'tps',
    'throughput (tps/user)': 'tps/u',
    'speed (tps/user)': 'spd/u',
    'throughput (tps/gpu)': 'tps/g',
    'ttft avg (ms)': 'ttft',
    'itl avg (ms)': 'itl',
    'latency avg (ms)': 'lat',
    'ttft p99 (ms)': 'ttft-p99',
    'itl p99 (ms)': 'itl-p99',
    'latency p99 (ms)': 'lat-p99',
    'request tput (req/s)': 'req/s',
    'output tput (tok/s)': 'tok/s',
}

LLAMA_HEADER_NOTES = {
    'n': 'number of nodes',
    'g': 'number of GPUs per node',
    'qnt': 'quantization',
    'isl': 'input sequence length mean',
    'osl': 'output sequence length mean',
    'req': 'number of requests',
    'mnt': 'maximum number of tokens',
    'mbs': 'maximum batch size',
    'bs': 'batch size',
    'cc': 'concurrency',
    'tps': 'output token throughput',
    'tps/u': 'output token throughput per user',
    'spd/u': 'decode-only output speed per user',
    'tps/g': 'output token throughput per GPU',
    'ttft': 'average time to first token in ms',
    'itl': 'average inter-token latency in ms',
    'lat': 'average request latency in ms',
    'ttft-p99': 'p99 time to first token in ms',
    'itl-p99': 'p99 inter-token latency in ms',
    'lat-p99': 'p99 request latency in ms',
    'req/s': 'request throughput',
    'tok/s': 'output token throughput',
}

# ---------------------------------------------------------------------------
# Config base
# ---------------------------------------------------------------------------

@dataclass
class LlamaConfig(BmtConfig):
    """Base LLaMA config — owns all shared fields and prep-step logic.

    Not intended for benchmarking directly. Subclass with
    LlamaThroughputConfig or LlamaLatencyConfig for benchmarking.
    """
    name: ClassVar[str] = 'LLAMA3'

    bin: str = ''
    outdir: str = './output'

    model: str = ''
    kv_cache_fraction: float = 0.95
    warmup: int = 10

    header_aliases: dict[str, str] = field(default_factory=lambda: dict(LLAMA_HEADER_ALIASES))
    header_notes: dict[str, str] = field(default_factory=lambda: dict(LLAMA_HEADER_NOTES))

    launcher: LlamaLauncher = field(default_factory=lambda: LlamaLauncher(backend=None))

    sharding: list[dict] = field(default_factory=lambda: [
        {'nnodes': 1, 'pp': 1, 'tp': 1}
    ])
    quant: str = ''
    max_seq_len: int = 8192
    max_batch_sizes: int | list[int] = field(default_factory=lambda: [2048])
    max_num_tokens: int | list[int] = field(default_factory=lambda: [8192])
    concurrency: int | list[int] | None = None
    workloads: list[dict] = field(default_factory=lambda: [
        {'input_mean': 128, 'output_mean': 128, 'num_requests': 30000}
    ])

    def _validate_bin(self) -> None:
        pass

    def _validate_config(self) -> None:
        if not self.launcher.backend:
            raise ValueError('LlamaLauncher requires a backend')
        if not self.model:
            raise ValueError("LlamaConfig: 'model' is required")
        if self.quant not in QUANT_FLAGS:
            raise ValueError(f'Unsupported quantization: {self.quant}')
        if not self.sharding:
            raise ValueError("LlamaConfig: 'sharding' must be a non-empty list")
        for entry in self.sharding:
            missing = [k for k in ('nnodes', 'pp', 'tp') if k not in entry]
            if missing:
                raise ValueError(f"sharding entry missing keys {missing}: {entry}")

    def _resolve_defaults(self) -> None:
        if not self.quant:
            self.quant = _infer_quant(self.model)
        else:
            self.quant = self.quant.upper()

        launcher = self.launcher
        backend = launcher.backend
        if backend is not None:
            launcher.model = self.model
            launcher.kv_cache_fraction = self.kv_cache_fraction
            launcher.paths = Paths(self.model)
        self.variant = type(backend).__name__ if backend else ''

    def _num_requests(self, workload: dict) -> int:
        return workload['num_requests']

    def _dataset_num_requests(self, workload: dict) -> int:
        return workload.get('dataset_num_requests', workload['num_requests'])

    def __iter__(self):
        for shard, workload, mbs, mnt in itertools.product(
            self.sharding,
            self.workloads,
            _as_list(self.max_batch_sizes),
            _as_list(self.max_num_tokens),
        ):
            nnodes = shard['nnodes']
            pp     = shard['pp']
            tp     = shard['tp']
            world_size = pp * tp
            ngpus = _derive_ngpus(nnodes, pp, tp)
            cpus_on_node = int(os.environ.get('SLURM_CPUS_ON_NODE', 0))
            nthreads = max(1, cpus_on_node // ngpus) if cpus_on_node else max(1, self._cpu.threads // ngpus)
            spec = LlamaSpec(
                nnodes=nnodes,
                ntasks=world_size,
                ngpus=ngpus,
                nthreads=nthreads,
                model=self.model,
                pp=pp,
                tp=tp,
                quant=self.quant,
                max_seq_len=self.max_seq_len,
                max_batch_size=mbs,
                max_num_tokens=mnt,
                input_mean=workload['input_mean'],
                output_mean=workload['output_mean'],
                num_requests=self._num_requests(workload),
                dataset_num_requests=self._dataset_num_requests(workload),
                warmup=self.warmup,
            )
            self._validate(spec)
            yield spec

    def _validate(self, spec: LlamaSpec) -> None:
        super()._validate(spec)
        world_size = spec.pp * spec.tp
        expected = spec.nnodes * spec.ngpus
        if world_size != expected:
            raise ValueError(
                f'Invalid topology: pp * tp = {world_size}, '
                f'but nnodes * ngpus = {expected}'
            )

    # Spec helpers
    def download_specs(self) -> list[LlamaSpec]:
        return [next(iter(self))]

    def dataset_specs(self) -> list[LlamaSpec]:
        seen = {}
        for spec in self:
            key = (spec.input_mean, spec.output_mean, spec.dataset_num_requests)
            if key not in seen:
                seen[key] = spec
        return list(seen.values())

    def ckpt_specs(self) -> list[LlamaSpec]:
        seen = {}
        for spec in self:
            key = (spec.pp, spec.tp)
            if key not in seen:
                seen[key] = spec
        return list(seen.values())

    def engine_specs(self) -> list[LlamaSpec]:
        seen = {}
        for spec in self:
            key = (spec.pp, spec.tp, spec.max_seq_len, spec.max_num_tokens, spec.max_batch_size)
            if key not in seen:
                seen[key] = spec
        return list(seen.values())

# ---------------------------------------------------------------------------
# Concrete benchmark configs
# ---------------------------------------------------------------------------

@dataclass
class LlamaThroughputConfig(LlamaConfig):
    key_headers: list = field(default_factory=lambda: list(_THROUGHPUT_KEY_HEADERS))
    metric_headers: list = field(default_factory=list)  # populated from backend in _resolve_defaults

    def _validate_config(self) -> None:
        super()._validate_config()
        if self.concurrency is not None and isinstance(self.launcher.backend, LegacyPythonBackend):
            raise ValueError('LegacyPythonBackend does not support concurrency')

    def _resolve_defaults(self) -> None:
        super()._resolve_defaults()
        self.metric_headers = list(self.launcher.backend.metric_headers)
        self.variant = f'{type(self.launcher.backend).__name__}/throughput'

    def __iter__(self):
        values = ['auto'] if self.concurrency is None else _as_list(self.concurrency)
        for spec, concurrency in itertools.product(super().__iter__(), values):
            yield replace(spec, concurrency=concurrency)

    def parse(self, output_path: str) -> dict[str, Any]:
        try:
            text = Path(output_path).read_text()
        except FileNotFoundError:
            log.warning('output not found: %s', output_path)
            return {}
        return self.launcher.backend.parse_throughput(text)

    def _output_path(self, spec: LlamaSpec, n: int, outdir: str) -> str:
        tag = (
            f'LLAMA3-{self.host}-throughput'
            f'-n{spec.nnodes}'
            f'-pp{spec.pp}-tp{spec.tp}'
            f'-isl{spec.input_mean}-osl{spec.output_mean}-req{spec.num_requests}'
            f'-mtk{spec.max_num_tokens}-mbs{spec.max_batch_size}'
        )
        if spec.concurrency != 'auto':
            tag += f'-cc{spec.concurrency}'
        return os.path.join(outdir, f'{tag}.out.{n}')

@dataclass
class LlamaLatencyConfig(LlamaConfig):
    key_headers: list = field(default_factory=lambda: list(_LATENCY_KEY_HEADERS))
    metric_headers: list = field(default_factory=lambda: list(_LATENCY_METRIC_HEADERS))

    def _validate_config(self) -> None:
        super()._validate_config()
        if isinstance(self.launcher.backend, LegacyPythonBackend):
            raise ValueError('LegacyPythonBackend does not support latency mode')

    def _resolve_defaults(self) -> None:
        super()._resolve_defaults()
        self.variant = f'{type(self.launcher.backend).__name__}/latency'

    def parse(self, output_path: str) -> dict[str, Any]:
        try:
            text = Path(output_path).read_text()
        except FileNotFoundError:
            log.warning('output not found: %s', output_path)
            return {}
        return self.launcher.backend.parse_latency(text)

    def _output_path(self, spec: LlamaSpec, n: int, outdir: str) -> str:
        tag = (
            f'LLAMA3-{self.host}-latency'
            f'-n{spec.nnodes}'
            f'-pp{spec.pp}-tp{spec.tp}'
            f'-isl{spec.input_mean}-osl{spec.output_mean}-req{spec.num_requests}'
            f'-mtk{spec.max_num_tokens}-mbs{spec.max_batch_size}'
        )
        return os.path.join(outdir, f'{tag}.out.{n}')

# ---------------------------------------------------------------------------
# Free step functions
# ---------------------------------------------------------------------------

def _desc(config: LlamaConfig, step: str) -> str:
    backend_name = type(config.launcher.backend).__name__
    return f'{config.name} ({backend_name}/{step})'

def download(config: LlamaConfig) -> None:
    launcher = config.launcher
    _run(
        specs=config.download_specs(),
        work_fn=launcher.download,
        desc=_desc(config, 'download'),
        cmd_fn=launcher.download_cmd,
        path_fn=lambda spec: launcher.paths.hf_dir(),
    )

def dataset(config: LlamaConfig) -> None:
    launcher = config.launcher
    _run(
        specs=config.dataset_specs(),
        work_fn=launcher.dataset,
        desc=_desc(config, 'dataset'),
        cmd_fn=launcher.dataset_cmd,
        path_fn=launcher.paths.dataset_file,
    )

def convert(config: LlamaConfig) -> None:
    launcher = config.launcher
    if not isinstance(launcher.backend, LegacyBackend):
        raise TypeError(f'{type(launcher.backend).__name__} does not support convert step')
    _run(
        specs=config.ckpt_specs(),
        work_fn=launcher.convert,
        desc=_desc(config, 'convert'),
        cmd_fn=lambda spec: launcher.backend.convert_cmd(spec, launcher.sif, launcher.paths),
        path_fn=launcher.paths.ckpt_dir,
    )

def build(config: LlamaConfig) -> None:
    launcher = config.launcher
    if not isinstance(launcher.backend, LegacyBackend):
        raise TypeError(f'{type(launcher.backend).__name__} does not support build step')
    _run(
        specs=config.engine_specs(),
        work_fn=launcher.build,
        desc=_desc(config, 'build'),
        cmd_fn=lambda spec: launcher.backend.build_cmd(spec, launcher.sif, launcher.paths),
        path_fn=launcher.paths.engine_dir,
    )
