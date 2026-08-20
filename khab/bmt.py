#!/usr/bin/env python3
from __future__ import annotations

import datetime
import itertools
import os
import time
import yaml

from dataclasses import dataclass, field
from socket import gethostname
from typing import Any, ClassVar, Iterator

from cpu import CPU
from gpu import GPU
from launcher import Launcher
from report import report
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, ProgressColumn, Task, TextColumn
from rich.text import Text
from util import fmt_cmd, sys_cmd

__all__ = [
    'RunSpec', 'BmtResult', 'BmtConfig',
    'Launcher',
    'benchmark', '_run', 'report',
]

_loglevel = os.environ.get('LOGLEVEL', '').upper()

def _as_list(v: Any) -> list:
    """Normalize a scalar-or-list config field to a list.
    Supports both scalar and list in YAML fields, e.g. `threads: 8` and `threads: [8, 16, 32]`
    """
    return v if isinstance(v, list) else [v]

@dataclass(frozen=True)
class RunSpec:
    nnodes: int = 1
    ntasks: int = 1
    ngpus: int = 0
    nthreads: int = 0
    extra_env: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    @property
    def env(self) -> dict[str, str]:
        base = {}
        if self.nthreads:
            base['OMP_NUM_THREADS'] = str(self.nthreads)
        if self.ngpus:
            base['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(self.ngpus))
        return {**base, **self.extra_env}

@dataclass
class BmtResult:
    spec: RunSpec
    output: str
    walltime: float
    config: Any = field(default=None, repr=False)
    metrics: dict[str, Any] = field(default_factory=dict)

@dataclass
class BmtConfig:
    name: ClassVar[str] = 'KHIB'
    version: ClassVar[str] = '0.6'

    bin: str
    outdir: str
    variant: str = ''

    key_headers: list = field(default_factory=list)
    metric_headers: list = field(default_factory=list)

    nnodes: int | list[int] = 1
    ntasks: int | list[int] = 1
    ngpus: int | list[int] = 0
    nthreads: int | list[int] = 0

    launcher: Launcher = field(default_factory=Launcher)
    host: str = field(default_factory=lambda: os.environ.get('SLURM_JOB_PARTITION') or gethostname())

    _cpu: CPU = field(init=False, repr=False)
    _gpu: GPU = field(init=False, repr=False)

    def __post_init__(self):
        self._cpu = CPU()
        self._gpu = GPU()

        self._cpu.info()
        self._gpu.info()

        self.outdir = os.path.join(
            self.outdir,
            datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S"),
        )
        os.makedirs(self.outdir, exist_ok=True)

        self._resolve_defaults()

        self.launcher.bin = self.bin

        self._validate_bin()
        self._validate_config()

    def _validate_bin(self) -> None:
        """Check binary exists. Override to skip for container-based benchmarks."""
        if not os.path.isfile(self.bin):
            raise FileNotFoundError(f"Binary not found: {self.bin}")

    def _validate_config(self) -> None:
        """Config-level validation hook — runs once at init after _resolve_defaults.

        Override in subclasses to validate config-level invariants.
        Call super() to chain validation up the hierarchy.
        """

    def _resolve_defaults(self) -> None:
        pass

    def __iter__(self) -> Iterator[RunSpec]:
        for nnodes, ntasks, ngpus, nthreads in itertools.product(
            _as_list(self.nnodes),
            _as_list(self.ntasks),
            _as_list(self.ngpus),
            _as_list(self.nthreads),
        ):
            spec = RunSpec(nnodes=nnodes, ntasks=ntasks, ngpus=ngpus, nthreads=nthreads)
            self._validate(spec)
            yield spec

    def _validate(self, spec: RunSpec) -> None:
        if spec.ngpus > self._gpu.count:
            raise ValueError(f"Requested {spec.ngpus} GPUs but only {self._gpu.count} available")

    def to_yaml(self, path: str) -> None:
        data = {
            'benchmark': type(self).__name__,
            'bin': self.bin,
            'outdir': self.outdir,
            'nnodes': self.nnodes,
            'ntasks': self.ntasks,
            'ngpus': self.ngpus,
            'nthreads': self.nthreads,
        }

        data.update(self._yaml_extra())

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _yaml_extra(self) -> dict:
        return {}

    @classmethod
    def from_yaml(cls, path: str) -> BmtConfig:
        with open(path) as f:
            data = yaml.safe_load(f)

        data.pop('benchmark', None)

        return cls(**data)

    def parse(self, output_path: str) -> dict[str, Any]:
        return {}

    def _pre_run(self, spec: RunSpec, n: int, output_path: str) -> None:
        """Pre-run hook called just before each repeat. Override in subclasses
        for benchmark-specific setup (e.g. writing INCAR for VASP).
        """

    def _post_run(self, spec: RunSpec, n: int, output_path: str) -> None:
        """Post-run hook called after each repeat. Override in subclasses for
        benchmark-specific archiving (e.g. copying OUTCAR, OSZICAR for VASP).
        """

    def _output_path(self, spec: RunSpec, n: int, outdir: str) -> str:
        tag = f"n{spec.nnodes}_t{spec.ntasks}_c{spec.nthreads}_g{spec.ngpus}"
        return os.path.join(outdir, f"{self.name}_{tag}.out.{n}")

_console = Console(stderr=True, no_color=True, highlight=False)

class _BlockBarColumn(ProgressColumn):
    """Neutral block-style progress bar using the old tqdm visual style."""

    def __init__(self, width: int = 33):
        super().__init__()
        self.width = width

    def render(self, task: Task) -> Text:
        if task.total:
            ratio = min(1.0, max(0.0, task.completed / task.total))
        else:
            ratio = 0.0
        done = int(self.width * ratio)
        return Text('█' * done + '░' * (self.width - done))

def _make_progress() -> Progress:
    return Progress(
        _BlockBarColumn(width=33),
        TextColumn('{task.completed}/{task.total}'),
        TextColumn('{task.description}'),
        console=_console,
    )

def _render_status(progress: Progress, command: str = '', output: str = '') -> Group:
    blocks = [progress]
    if command:
        blocks.extend([
            Text(''),
            Text('[command]'),
            Text(command),
        ])
    if output:
        blocks.extend([
            Text(''),
            Text('[output]'),
            Text(output),
        ])
    return Group(*blocks)

def _failure_message(rc: int, output_path: str, command: str) -> str:
    return (
        f'Command failed with exit code {rc}.\n\n'
        f'stdout: {output_path}\n'
        f'stderr: {output_path}.err\n\n'
        f'Inspect the stderr file for runtime error details.'
    )

def _run(
    specs: list,
    work_fn,
    desc: str,
    cmd_fn=None,
    path_fn=None,
) -> None:
    """Progress-tracked executor for fire-and-forget steps (prep steps).

    Args:
        specs:   list of spec objects to iterate over.
        work_fn: callable(spec) — executes the step; return value ignored.
        desc:    progress bar label.
        cmd_fn:  optional callable(spec) -> Command, for [command] display.
        path_fn: optional callable(spec) -> str, for [output] display.
    """
    progress = _make_progress()
    task = progress.add_task(desc, total=len(specs))
    command = ''
    output_path = ''

    with Live(
        _render_status(progress, command, output_path),
        console=_console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        for spec in specs:
            if cmd_fn is not None:
                command = fmt_cmd(cmd_fn(spec))
            if path_fn is not None:
                output_path = str(path_fn(spec))
            live.update(_render_status(progress, command, output_path))

            work_fn(spec)

            progress.advance(task)
            live.update(_render_status(progress, command, output_path))

def benchmark(config: BmtConfig, repeats: int = 1, desc: str = None) -> list[BmtResult]:
    """Run a benchmark sweep, collecting timed results.

    Args:
        config:  BmtConfig instance defining the sweep.
        repeats: number of times to repeat each spec.
        desc:    progress bar label; defaults to 'name (variant)'.
    """
    results: list[BmtResult] = []
    specs = list(config)
    total = len(specs) * repeats

    outdir = config.outdir
    if not os.access(outdir, os.W_OK):
        raise PermissionError(f"Output directory not writable: {outdir}")

    if desc is None:
        desc = f'{config.name} ({config.variant})' if config.variant else config.name

    def _work(spec):
        cmd = config.launcher.build_cmd(spec)
        for n in range(1, repeats + 1):
            output_path = config._output_path(spec, n, outdir)
            live.update(_render_status(progress, fmt_cmd(cmd), output_path))

            config._pre_run(spec, n, output_path)
            start = time.perf_counter()
            rc = sys_cmd(cmd, output_path)
            walltime = time.perf_counter() - start

            if rc != 0:
                live.update(_render_status(progress, fmt_cmd(cmd), output_path))
                raise RuntimeError(_failure_message(rc, output_path, fmt_cmd(cmd)))

            results.append(BmtResult(
                spec=spec,
                output=output_path,
                walltime=walltime,
                config=config,
                metrics=config.parse(output_path),
            ))
            config._post_run(spec, n, output_path)
            progress.advance(task)
            live.update(_render_status(progress, fmt_cmd(cmd), output_path))

            if n < repeats:
                time.sleep(5)

    progress = _make_progress()
    task = progress.add_task(desc, total=total)

    with Live(
        _render_status(progress),
        console=_console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        for spec in specs:
            _work(spec)

    return results
