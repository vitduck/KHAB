#!/usr/bin/env python3

import argparse
import platform
import yaml

from bmt import benchmark, report
from env import module_load, module_list, module_purge
from llama3 import (
    LlamaConfig,
    LlamaLauncher,
    LlamaLatencyConfig,
    LlamaLatencyLauncher,
    LlamaThroughputConfig,
    LlamaThroughputLauncher,
    LegacyCppBackend,
    LegacyPythonBackend,
    PytorchBackend,
    build,
    convert,
    dataset,
    download,
)

ARCH = platform.machine()

SIFS = {
    #'cpp': '../00-sif/tensorrt_llm_v0.14.0_x86.sif',
    'cpp': '../00-sif/test2.sif',
    'python': '../00-sif/tensorrt_llm_v0.14.0_x86.sif',
    'pytorch': {
        'x86_64': '../00-sif/tensorrt_llm_v1.2.0_x86.sif',
        'aarch64': '../00-sif/tensorrt_llm_v1.2.0_aarch64.sif',
    },
}

RUN_STEPS = ['throughput', 'latency']
PREP_STEPS = ['download', 'data', 'convert', 'build']
ALL_STEPS = PREP_STEPS + RUN_STEPS

BACKEND_CLASSES = {
    'cpp': LegacyCppBackend,
    'python': LegacyPythonBackend,
    'pytorch': PytorchBackend,
}


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}

def resolve_backend(backend_name):
    if backend_name not in BACKEND_CLASSES:
        raise ValueError(
            f'Unsupported backend: {backend_name}. '
            f'Available: {list(BACKEND_CLASSES)}'
        )

    sif = SIFS[backend_name]
    if isinstance(sif, dict):
        if ARCH not in sif:
            raise RuntimeError(
                f'No SIF for backend={backend_name}, arch={ARCH}. '
                f'Available: {list(sif)}'
            )
        sif = sif[ARCH]

    return BACKEND_CLASSES[backend_name](), sif


def merge_config(base_config, mode_config):
    run_config = dict(base_config)
    run_config.update(mode_config)
    return run_config


def _override_num_requests(base_config, mode_config):
    """Shared helper: pop num_requests from mode_config and override workloads.

    Preserves dataset_num_requests from the original workload num_requests so
    the dataset filename and generation size are unaffected by the override.
    """
    run_config = dict(base_config)
    mode_config = dict(mode_config)
    num_requests = mode_config.pop('num_requests', None)
    run_config.update(mode_config)

    if num_requests is not None:
        workloads = []
        for workload in run_config['workloads']:
            item = dict(workload)
            item['dataset_num_requests'] = item.get(
                'dataset_num_requests',
                item['num_requests'],
            )
            item['num_requests'] = num_requests
            workloads.append(item)
        run_config['workloads'] = workloads

    return run_config


def throughput_config(base_config, mode_config):
    return _override_num_requests(base_config, mode_config)


def latency_config(base_config, mode_config):
    return _override_num_requests(base_config, mode_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRT-LLM LLaMA benchmark')
    parser.add_argument('steps', metavar='STEP', nargs='+', choices=ALL_STEPS,
                        help=f'one or more of: {", ".join(ALL_STEPS)}')
    parser.add_argument('--config', required=True, help='YAML config file')
    args = parser.parse_args()

    yaml_config = load_yaml(args.config)

    if 'config' not in yaml_config:
        raise KeyError("YAML must contain a top-level 'config' block")
    base_config = dict(yaml_config['config'])

    try:
        backend_name = base_config.pop('backend')
    except KeyError as exc:
        raise KeyError("YAML 'config' block must define 'backend'") from exc

    backend, sif = resolve_backend(backend_name)

    module_purge()
    module_load(['singularity/4.3.4'])
    module_list()

    throughput_cfg = throughput_config(base_config, yaml_config.get('throughput', {}))

    if 'download' in args.steps:
        cfg = LlamaConfig(
            **throughput_cfg,
            launcher=LlamaLauncher(backend=backend, sif=sif),
        )
        download(cfg)

    if 'data' in args.steps:
        cfg = LlamaConfig(
            **base_config,
            launcher=LlamaLauncher(backend=backend, sif=sif),
        )
        dataset(cfg)

    if 'convert' in args.steps:
        cfg = LlamaConfig(
            **throughput_cfg,
            launcher=LlamaLauncher(backend=backend, sif=sif),
        )
        convert(cfg)

    if 'build' in args.steps:
        cfg = LlamaConfig(
            **throughput_cfg,
            launcher=LlamaLauncher(backend=backend, sif=sif),
        )
        build(cfg)

    if 'throughput' in args.steps:
        cfg = LlamaThroughputConfig(
            **throughput_cfg,
            launcher=LlamaThroughputLauncher(backend=backend, sif=sif),
        )
        report(benchmark(cfg, repeats=1))

    if 'latency' in args.steps:
        cfg = LlamaLatencyConfig(
            **latency_config(base_config, yaml_config.get('latency', {})),
            launcher=LlamaLatencyLauncher(backend=backend, sif=sif),
        )
        report(benchmark(cfg, repeats=1))
